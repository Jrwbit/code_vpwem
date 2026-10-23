"""RRT behavior checks using configuration-space obstacles, without robot hardware."""
import unittest

import numpy as np

from connect_fixed_camera_home import rrt_connect


class FixedCameraRrtTests(unittest.TestCase):
    def test_bidirectional_search_detours_around_obstacle_and_rechecks_edges(self):
        strict_calls = []

        def check(a, b, strict):
            if strict:
                strict_calls.append((a.copy(), b.copy()))
            count = max(3, int(np.ceil(np.linalg.norm(b-a) / (.01 if strict else .08))) + 1)
            points = np.linspace(a, b, count)
            blocked = np.any((np.abs(points[:, 0]) <= .35) & (np.abs(points[:, 1]) <= .5))
            return {'ok': not blocked}

        start, goal = np.array([-1., 0.]), np.array([1., 0.])
        result = rrt_connect(start, goal, np.array([-2., -2.]), np.array([2., 2.]),
                             check, np.random.default_rng(12), seconds=2, step_rad=.16,
                             seed_rows=[[-.8, .8], [0, .8], [.8, .8]])
        self.assertTrue(result['ok'])
        rows = np.asarray(result['arm_rows'])
        np.testing.assert_allclose(rows[0], start)
        np.testing.assert_allclose(rows[-1], goal)
        self.assertLessEqual(np.linalg.norm(np.diff(rows, axis=0), axis=1).max(), .160000001)
        self.assertTrue(any(abs(q[1]) > .5 for q in rows))
        self.assertEqual(len(result['checks']), len(rows)-1)
        self.assertTrue(all(check(a, b, True)['ok'] for a, b in zip(rows, rows[1:])))
        self.assertGreaterEqual(len(strict_calls), len(rows)-1)

    def test_coarse_route_is_never_returned_when_dense_checks_reject_it(self):
        # An impenetrable barrier in one dimension, deliberately invisible to
        # the coarse search callback. Every purported solution must be rejected.
        def check(a, b, strict):
            crossing = min(a[0], b[0]) <= 0 <= max(a[0], b[0])
            return {'ok': not (strict and crossing)}

        result = rrt_connect(np.array([-1.]), np.array([1.]), np.array([-2.]), np.array([2.]),
                             check, np.random.default_rng(3), seconds=.04, step_rad=.2)
        self.assertFalse(result['ok'])
        self.assertGreater(result['strict_rejected_routes'], 0)
        self.assertEqual(result['strict_rejected_routes'], result['restarts'])
        self.assertNotIn('arm_rows', result)

    def test_invalid_start_is_rejected_before_search(self):
        def check(a, b, strict):
            self.assertTrue(strict)
            return {'ok': False, 'reason': 'fixture_collision'}

        result = rrt_connect(np.zeros(2), np.ones(2), -np.ones(2), np.ones(2)*2,
                             check, np.random.default_rng(1), seconds=1)
        self.assertFalse(result['ok'])
        self.assertEqual(result['reason'], 'start_is_not_a_legal_adjustment_pose')
        self.assertEqual(result['iterations'], 0)


if __name__ == '__main__':
    unittest.main()
