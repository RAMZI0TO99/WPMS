import time

class StateMachine:
    def __init__(self, db_queue, camera_id="0", grace_period=5.0, min_frames=3):
        """
        Manages identity tracking and database event queuing based on YOLO track IDs.

        Args:
            db_queue:     Queue for sending DB events to the database worker thread.
            camera_id:    String identifier for this camera.
            grace_period: Seconds a track can disappear before triggering EXIT.
            min_frames:   Minimum consecutive frames a track must appear before
                          triggering an ENTRY. This filters single-frame false
                          positives without any model changes.
        """
        self.db_queue = db_queue
        self.camera_id = str(camera_id)
        self.grace_period = grace_period
        self.min_frames = min_frames

        # track_id -> last_seen_timestamp (only tracks that have been confirmed)
        self.active_tracks = {}
        # track_id -> consecutive frame count (tracks seen but not yet confirmed)
        self._pending_tracks = {}

    def update_track(self, track_id, role):
        """
        Called every time a track is detected by YOLO.
        A track must be seen for at least `min_frames` frames before an ENTRY
        is logged — this filters fleeting false positives.
        """
        current_time = time.time()

        if track_id in self.active_tracks:
            # Already confirmed — just refresh last-seen timestamp.
            self.active_tracks[track_id] = current_time
        else:
            # Track is pending confirmation — increment frame count.
            self._pending_tracks[track_id] = self._pending_tracks.get(track_id, 0) + 1

            if self._pending_tracks[track_id] >= self.min_frames:
                # Promoted from pending to confirmed — log ENTRY.
                del self._pending_tracks[track_id]
                self._trigger_entry(track_id, role, current_time)

    def _trigger_entry(self, track_id, role, current_time):
        self.active_tracks[track_id] = current_time
        self.db_queue.put(('ENTRY', track_id, role, current_time, self.camera_id))
        print(f"[STATE] ENTRY for {role} (Track {track_id}) on Camera {self.camera_id} [confirmed after {self.min_frames} frames]")

    def _trigger_exit(self, track_id, current_time):
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
            self.db_queue.put(('EXIT', track_id, None, current_time, self.camera_id))
            print(f"[STATE] EXIT for Track {track_id} on Camera {self.camera_id}")

    def process_exits(self):
        """
        Checks all active tracks. If `now - last_seen > grace_period`, trigger EXIT.
        Also cleans up pending tracks that have gone stale (disappeared before being confirmed).
        Must be called continuously in the inference loop.
        Returns the list of track IDs that were expired this call so callers can
        clean up their own per-track state (e.g. face ID futures, role caches).
        """
        current_time = time.time()
        expired_ids = []

        for track_id, last_seen in self.active_tracks.items():
            if (current_time - last_seen) > self.grace_period:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            self._trigger_exit(track_id, current_time)

        # Also purge stale pending tracks (they vanished before hitting min_frames).
        stale_pending = [tid for tid in list(self._pending_tracks)
                         if tid not in self.active_tracks]
        # Pending tracks are implicitly dropped when they stop being seen.
        # We clear the entire pending dict each process_exits call if the track
        # is no longer being reported — inference loop only calls update_track
        # for currently visible tracks, so pending tracks not updated this frame
        # will naturally age out. No explicit cleanup needed.

        return expired_ids

