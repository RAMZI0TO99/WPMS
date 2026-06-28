import time

class StateMachine:
    def __init__(self, db_queue, grace_period=5.0):
        """
        Manages identity tracking and database event queuing based on YOLO track IDs.
        """
        self.db_queue = db_queue
        self.grace_period = grace_period
        
        # track_id -> last_seen_timestamp
        self.active_tracks = {}

    def update_track(self, track_id, role):
        """
        Called every time a track is detected by YOLO.
        """
        current_time = time.time()
        
        if track_id not in self.active_tracks:
            self._trigger_entry(track_id, role, current_time)
            
        self.active_tracks[track_id] = current_time

    def _trigger_entry(self, track_id, role, current_time):
        self.active_tracks[track_id] = current_time
        self.db_queue.put(('ENTRY', track_id, role, current_time))
        print(f"[STATE] Instant ENTRY for {role} (Track {track_id})")

    def _trigger_exit(self, track_id, current_time):
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
            self.db_queue.put(('EXIT', track_id, None, current_time))
            print(f"[STATE] EXIT for Track {track_id}")

    def process_exits(self):
        """
        Checks all active tracks. If `now - last_seen > grace_period`, trigger EXIT.
        Must be called continuously in the inference loop.
        """
        current_time = time.time()
        expired_ids = []
        for track_id, last_seen in self.active_tracks.items():
            if (current_time - last_seen) > self.grace_period:
                expired_ids.append(track_id)
                
        for track_id in expired_ids:
            self._trigger_exit(track_id, current_time)
