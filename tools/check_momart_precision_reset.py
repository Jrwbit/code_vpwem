#!/usr/bin/env python3
"""Regression tests for the real sequence adapter, without loading GDINO.

Extract only the adapter class from the supplied source file; its detector is
a stateful test double. This exercises the production class body unchanged.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


def load_adapter(path):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name == 'PrecisionSequenceDetector']
    if len(nodes) != 1:
        raise ValueError('Expected exactly one PrecisionSequenceDetector')
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['PrecisionSequenceDetector']


class FakeDetector:
    def __init__(self):
        self.calls = []
        self.resets = []

    def reset(self, stream_id):
        self.resets.append(stream_id)

    def detect(self, image, stream_id):
        self.calls.append((stream_id, int(image[0, 0, 0])))
        # Different outputs on successive calls expose duplicate state updates.
        return SimpleNamespace(valid=True, call=len(self.calls), stream=stream_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    adapter = load_adapter(args.source)

    class ResetTests(unittest.TestCase):
        def make(self, streams=2):
            result = adapter.__new__(adapter)
            result.detector = FakeDetector()
            result.num_streams = streams
            result.cache = [None] * streams
            result.processed_frames = result.valid_frames = result.reset_count = 0
            return result

        def images(self, *values):
            return [np.full((90, 120, 3), value, np.uint8) for value in values]

        def test_reset_padding_is_one_time_per_environment(self):
            wrapper = self.make()
            wrapper.reset_all()
            result = wrapper(self.images(10, 10, 20, 20))
            self.assertEqual(wrapper.detector.calls, [(0, 10), (1, 20)])
            self.assertIs(result[0], result[1])
            self.assertIs(result[2], result[3])
            self.assertIsNot(result[0], result[2])
            self.assertEqual(wrapper.processed_frames, 2)

        def test_identical_later_frames_still_advance_time(self):
            wrapper = self.make(1)
            wrapper(self.images(10, 10))
            before = wrapper.processed_frames
            # First entry overlaps the last observation; second is a new time.
            result = wrapper(self.images(10, 10))
            self.assertEqual(wrapper.processed_frames - before, 1)
            self.assertIsNot(result[0], result[1])

        def test_reset_clears_cache_even_for_identical_episode_image(self):
            wrapper = self.make(1)
            first = wrapper(self.images(10, 10))
            wrapper.reset_all()
            before = wrapper.processed_frames
            second = wrapper(self.images(10, 10))
            self.assertEqual(wrapper.processed_frames - before, 1)
            self.assertIsNot(first[-1], second[0])

        def test_distinct_initial_frames_are_not_deduplicated(self):
            wrapper = self.make()
            wrapper(self.images(10, 11, 20, 21))
            self.assertEqual(wrapper.detector.calls, [(0, 10), (0, 11), (1, 20), (1, 21)])

        def test_only_leading_initial_padding_is_deduplicated(self):
            wrapper = self.make(1)
            wrapper(self.images(10, 10, 11, 11))
            self.assertEqual(wrapper.detector.calls, [(0, 10), (0, 11), (0, 11)])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResetTests))
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
