import unittest
from queue import Queue
from unittest.mock import patch

from src.tracker_state import StateMachine


class TrackerStateTests(unittest.TestCase):
    def setUp(self):
        self.events = Queue()
        self.state = StateMachine(self.events, camera_id="camera-a", min_frames=3)
        clock_patch = patch("src.tracker_state.time.time", return_value=100.0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def frame(self, *track_ids):
        for track_id in track_ids:
            self.state.update_track(track_id, "Manager")
        return self.state.process_exits()

    def test_missed_frames_reset_confirmation(self):
        self.frame(1)
        self.frame()
        self.frame(1)
        self.frame()
        self.frame(1)
        self.assertTrue(self.events.empty())

    def test_disappeared_pending_track_is_removed(self):
        self.frame(1)
        self.frame()
        self.assertEqual(self.state._pending_tracks, {})

    def test_three_consecutive_frames_emit_one_entry(self):
        self.frame(1)
        self.frame(1)
        self.assertTrue(self.events.empty())
        self.frame(1)
        self.frame(1)
        self.assertEqual(
            self.events.get_nowait(), ("ENTRY", 1, "Manager", 100.0, "camera-a")
        )
        self.assertTrue(self.events.empty())

    def test_confirmation_is_independent_for_each_track(self):
        self.frame(1, 2)
        self.frame(2)
        self.frame(1, 2)
        self.assertEqual(self.events.get_nowait()[1], 2)
        self.assertTrue(self.events.empty())

    def test_duplicate_observations_in_one_frame_do_not_confirm(self):
        self.frame(1, 1, 1)
        self.assertTrue(self.events.empty())
        self.frame(1)
        self.assertTrue(self.events.empty())
        self.frame(1)
        self.assertEqual(self.events.get_nowait()[0], "ENTRY")

    def test_confirmation_restarts_after_a_gap(self):
        self.frame(1)
        self.frame(1)
        self.frame()
        self.frame(1)
        self.assertTrue(self.events.empty())
        self.frame(1)
        self.frame(1)
        self.assertEqual(self.events.get_nowait()[1], 1)

    def test_confirmed_tracks_keep_the_exit_grace_period(self):
        for _ in range(3):
            self.frame(1)
        self.events.get_nowait()
        self.clock.return_value = 104.0
        self.assertEqual(self.frame(), [])
        self.frame(1)
        self.clock.return_value = 109.0
        self.assertEqual(self.frame(), [])
        self.clock.return_value = 110.0
        self.assertEqual(self.frame(), [1])
        self.assertEqual(self.events.get_nowait(), ("EXIT", 1, None, 110.0, "camera-a"))
        self.assertTrue(self.events.empty())


if __name__ == "__main__":
    unittest.main()
