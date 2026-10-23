"""Physics-backed regression tests for offline cycle validation.

Run with the local planner venv and simulation package on PYTHONPATH. The small
MuJoCo scene deliberately has valid endpoints with a colliding midpoint.
"""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_validation import (ClearancePolicy, _mating_exemption, check_segment,
                              pair_clearance, pair_contact_constraints, temporary_offline_policy, validate_cycle)
from sim.helios_insert import corridor, scene, swap


class SliderRobot:
    def __init__(self, xml):
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.values = {name: 0. for name in scene.WAIST + scene.LEFT}
        self.lim = {name: (-1., 1.) for name in self.values}
        self.dadr = {name: 0 for name in self.values}

    def set(self, group, q):
        self.values.update(zip(group, q))
        self.d.qpos[0] = self.values[scene.LEFT[0]]

    def get(self, group):
        return np.array([self.values[n] for n in group])

    def fk(self):
        mujoco.mj_forward(self.m, self.d)

    def tcp_pose(self, tcp):
        T = np.eye(4); T[0, 3] = self.d.qpos[0]
        return T

    def _gname(self, gid):
        return self.m.geom(gid).name

    def body_pad(self, c):
        return 0.

    def limit_margin(self, group):
        return min(min(self.values[n] - self.lim[n][0], self.lim[n][1] - self.values[n]) for n in group)


def context():
    robot = SliderRobot('''<mujoco><worldbody>
    <geom name="box_wall" type="sphere" size=".02" margin=".03"/>
    <body><joint name="slider" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
    <geom name="tool_disc_test" type="sphere" size=".01" mass="1" margin=".03"/></body>
    </worldbody></mujoco>''')
    target = np.eye(4); target[0, 3] = .1
    return SimpleNamespace(robot=robot, tcp='FLANGE', zones={}, geoms=[],
                           cfg={'box_T_base_link': np.eye(4).tolist(), 'elbow': {'max_rel_m': .1},
                                'targets': [{'id': 'JM160', 'T': target.tolist()}]})


class PairPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ClearancePolicy()
        self.elbow = patch.object(swap, 'elbow_rel', return_value=0.)
        self.zone = patch.object(corridor, 'adjust_violation', return_value='')
        self.elbow.start(); self.zone.start()
        self.addCleanup(self.elbow.stop); self.addCleanup(self.zone.stop)

    def test_only_tool_workpiece_pairs_taper(self):
        for fraction, expected in [(1., .008), (.5, .0055), (0., .003)]:
            self.assertAlmostEqual(pair_clearance(('tool_pin_JM160', 'box_wall'), 'insert', fraction, self.policy), expected)
            for pair in [('AR5-5_07L_link7', 'box_wall'), ('tool_disc_1', 'table'),
                         ('tool_pin_JM160', 'hik_camera'), ('tool_board', 'box_wall')]:
                self.assertEqual(pair_clearance(pair, 'insert', fraction, self.policy), .008)

    def test_interior_collision_rejected_despite_clear_endpoints(self):
        ctx = context(); a = np.zeros(11); b = a.copy(); a[4] = -.1; b[4] = .1
        self.assertTrue(check_segment(ctx, a, a, 'insert', 'JM160')['ok'])
        self.assertTrue(check_segment(ctx, b, b, 'insert', 'JM160')['ok'])
        result = check_segment(ctx, a, b, 'insert', 'JM160')
        self.assertFalse(result['ok'])
        self.assertLess(result['min_clearance_mm'], -29.)
        self.assertGreater(result['samples'], 3)
        self.assertEqual(set(result['worst_pair']), {'tool_disc_test', 'box_wall'})

    def test_non_mating_solid_penetration_never_exempt(self):
        ctx = context(); q = np.zeros(11)
        ctx.robot.ignore_disc_box = True  # legacy flag must not bypass this gate
        result = check_segment(ctx, q, q, 'insert', 'JM160', ClearancePolicy(local_box_m=0.))
        self.assertFalse(result['ok'])
        self.assertIn('solid_penetration', result['failures'][0]['reasons'])

    def test_waist_motion_rejected(self):
        ctx = context(); q = np.zeros(11); q[4] = .1; b = q.copy(); b[0] = math.radians(.01)
        result = check_segment(ctx, q, b, 'insert', 'JM160')
        self.assertFalse(result['ok'])
        self.assertAlmostEqual(result['max_waist_change_deg'], .01)

    def test_cartesian_error_not_only_end_collision(self):
        ctx = context(); a = np.zeros(11); a[4] = .1; b = a.copy(); b[4] = .2
        Ta = np.eye(4); Tb = Ta.copy(); Ta[:3, 3] = [.1, .004, 0]; Tb[:3, 3] = [.2, .004, 0]
        result = check_segment(ctx, a, b, 'insert', 'JM160', Ta=Ta, Tb=Tb)
        self.assertFalse(result['ok']); self.assertAlmostEqual(result['max_path_deviation_mm'], 4.)

    def test_cycle_requires_target_and_return_and_no_seam_jump(self):
        ctx = context(); q = np.zeros(11); q[4] = .1; b = q.copy(); b[4] = .2
        valid = [{'waypoints': [q, b], 'phase': 'insert'}, {'waypoints': [b, q], 'phase': 'retract'}]
        self.assertTrue(validate_cycle(ctx, valid, 'JM160', np.zeros(4))['ok'])
        invalid = [{'waypoints': [q, b], 'phase': 'insert'}]
        self.assertIn('cycle_did_not_return_to_start', validate_cycle(ctx, invalid, 'JM160', np.zeros(4))['failures'])
        jumped = [{'waypoints': [q, b], 'phase': 'insert'}, {'waypoints': [q, q], 'phase': 'retract'}]
        self.assertIn('segment_1_discontinuous_seam', validate_cycle(ctx, jumped, 'JM160', np.zeros(4))['failures'])

    def test_declared_cartesian_target_must_be_reached(self):
        ctx = context(); q = np.zeros(11); q[4] = .1; b = q.copy(); b[4] = .2
        Ta = np.eye(4); Ta[0, 3] = .1; Tb = Ta.copy(); Tb[0, 3] = .3
        parts = [{'waypoints': [q, b], 'phase': 'insert', 'kind': 'linear', 'Ta': Ta, 'Tb': Tb},
                 {'waypoints': [b, q], 'phase': 'retract'}]
        result = validate_cycle(ctx, parts, 'JM160', np.zeros(4))
        self.assertFalse(result['ok']); self.assertIn('segment_0_end_target_missed', result['failures'])

    def test_mating_pin_compression_bounded_and_boss_not_exempt(self):
        xml = '''<mujoco><worldbody><geom name="term_JM160_gnd" type="cylinder" pos="0 0 .004" size=".004 .004"/>
        <body pos="0 0 -.009"><joint name="slider" type="slide" axis="0 0 1"/>
        <geom name="tool_pin_JM160" type="cylinder" size=".0025 .011" mass="1"/></body></worldbody></mujoco>'''
        r = SliderRobot(xml); ctx = context(); ctx.robot = r
        r.fk(); c = SimpleNamespace(geom1=1, geom2=0)
        names = ('tool_pin_JM160', 'term_JM160_gnd')
        self.assertTrue(_mating_exemption(ctx, c, names, 'JM160', 'insert', self.policy))
        r.d.qpos[0] = .002; r.fk()
        self.assertFalse(_mating_exemption(ctx, c, names, 'JM160', 'insert', self.policy))
        self.assertFalse(_mating_exemption(ctx, c, ('tool_boss_JM160', names[1]), 'JM160', 'insert', self.policy))
        self.assertFalse(_mating_exemption(ctx, c, names, 'JM160', 'transit', self.policy))

    def test_generation_and_gate_share_pair_threshold(self):
        ctx = context(); q = np.zeros(11); q[4] = .034
        ctx.robot.set(scene.WAIST + scene.LEFT, q); ctx.robot.fk()
        _, h_insert = pair_contact_constraints(ctx, 'JM160', 'insert', self.policy)
        _, h_corridor = pair_contact_constraints(ctx, 'JM160', 'corridor', self.policy)
        self.assertAlmostEqual(float(h_insert[0]), .5 * (.004 - .003))
        self.assertAlmostEqual(float(h_corridor[0]), .5 * (.004 - .010))
        self.assertTrue(check_segment(ctx, q, q, 'insert', 'JM160')['ok'])
        self.assertFalse(check_segment(ctx, q, q, 'corridor', 'JM160')['ok'])

    def test_generator_adapters_restore_after_exception(self):
        ctx = context(); original_line, original_chunk = corridor.check_line, corridor._chunk_ok
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with temporary_offline_policy(ctx, 'JM160'):
                self.assertIsNot(corridor.check_line, original_line)
                self.assertIsNot(corridor._chunk_ok, original_chunk)
                raise RuntimeError('interrupted')
        self.assertIs(corridor.check_line, original_line)
        self.assertIs(corridor._chunk_ok, original_chunk)

    def test_contact_overflow_prevents_false_pass(self):
        ctx = context(); q = np.zeros(11); q[4] = .1
        ctx.robot.d.warning[int(mujoco.mjtWarning.mjWARN_CONTACTFULL)].number = 1
        result = check_segment(ctx, q, q, 'insert', 'JM160')
        self.assertFalse(result['ok'])
        self.assertIn('mujoco_collision_coverage_invalid', result['failures'][0]['reasons'])
        with self.assertRaisesRegex(RuntimeError, 'coverage invalid'):
            pair_contact_constraints(ctx, 'JM160', 'insert')

    def test_nan_and_missing_cartesian_path_rejected(self):
        ctx = context(); q = np.zeros(11); q[4] = np.nan
        with self.assertRaises(ValueError): check_segment(ctx, q, q, 'insert', 'JM160')
        with self.assertRaises(ValueError): ClearancePolicy(local_box_m=-.001)
        q[4] = .1
        with self.assertRaises(ValueError): validate_cycle(ctx, [{'waypoints': [q, q], 'kind': 'linear'}], 'JM160', np.zeros(4))


if __name__ == '__main__':
    unittest.main()
