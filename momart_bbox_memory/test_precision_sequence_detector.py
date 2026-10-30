import unittest

import numpy as np

from .detector import Detection
from .precision_detector import PrecisionSequenceDetector


class FakeDetector:
    def __init__(self):
        self.calls = []
        self.resets = []
        self.metadata = {"source": "test"}

    def detect(self, image, stream_id=0):
        self.calls.append((stream_id, image.copy()))
        return Detection(np.array([0.1, 0.2, 0.8, 0.9], np.float32),
                         0.9, True, False, 1)

    def reset(self, stream_id=0):
        self.resets.append(stream_id)


class PrecisionSequenceDetectorTests(unittest.TestCase):
    def make_wrapper(self):
        wrapper = PrecisionSequenceDetector.__new__(PrecisionSequenceDetector)
        wrapper.detector = FakeDetector()
        wrapper.num_streams = 2
        wrapper.cache = [None, None]
        wrapper.processed_frames = wrapper.valid_frames = wrapper.reset_count = 0
        return wrapper

    def test_overlapping_history_is_detected_once_and_episode_reset_clears_cache(self):
        wrapper = self.make_wrapper()
        a, b, c, d = [np.full((90, 120, 3), i, np.uint8) for i in range(4)]
        first = wrapper([a, b, c, d])
        second = wrapper([b, np.full_like(b, 5), d, np.full_like(d, 6)])
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertEqual(wrapper.processed_frames, 6)
        self.assertEqual(wrapper.valid_frames, 6)
        wrapper.reset_all()
        self.assertEqual(wrapper.detector.resets, [0, 1])
        wrapper([b, np.full_like(b, 5), d, np.full_like(d, 6)])
        self.assertEqual(wrapper.processed_frames, 10)
        self.assertEqual(wrapper.reset_count, 2)


if __name__ == "__main__":
    unittest.main()
