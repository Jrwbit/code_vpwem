import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_controller_kinematics import audit
from independent_route_audit import DIMENSIONS, flange


class ControllerAuditTests(unittest.TestCase):
    def fixture(self):
        nominal = np.array(DIMENSIONS, float)
        effective = nominal.copy()
        effective[2, 2] += 1  # Synthetic geometry, not device calibration data.
        probe = {'q': [0]*7, 'T_base_flange': flange([0]*7, effective).ravel().tolist()}
        cfg = {'nominal_dh': {'alpha_a_d_theta_deg_mm': nominal.ravel().tolist()},
               'effective_dh': {'alpha_a_d_theta_deg_mm': effective.ravel().tolist()},
               'controller_fk_audit': {'samples': [probe]}}
        config = {'state': {'raw': {arm: {'controller_id': 'synthetic', 'kinematic_config': cfg}
                                    for arm in ('left_arm', 'right_arm')}}}
        target = [0]*7
        actual = [.01, -.02, .01, .02, 0, 0, 0]
        sample = {'local_time': 'synthetic',
                  'raw': {'left_arm': {'ok': True, 'controller_id': 'synthetic', 'q': actual,
                                      'qd_raw': [0]*7, 'T_base_flange_rowmajor': probe['T_base_flange']}},
                  'plan': {'name': 'test', 'segment': 0, 'step': 1},
                  'motion': {'busy': False, 'operation_id': 'new',
                             'last_result': {'ok': True, 'group': 'left_arm', 'operation_kind': 'position',
                                             'operation_id': 'new', 'target': target}}}
        return config, sample

    def test_command_and_actual_channels_are_distinct_and_old_operation_is_excluded(self):
        cfg, sample = self.fixture()
        old = copy.deepcopy(sample)
        old['motion']['last_result']['operation_id'] = 'before-recording'
        moving = copy.deepcopy(sample)
        moving['motion']['busy'] = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.jsonl'
            path.write_text('\n'.join(json.dumps(s) for s in [old, moving, sample]))
            report = audit(cfg, path, {'test'})
        self.assertEqual(report['settled_commands'], 1)
        self.assertLess(report['summary']['effective_command_vs_cartesian_getter']['position_mm']['max'], 1e-9)
        self.assertGreater(report['summary']['effective_actual_vs_cartesian_getter']['position_mm']['max'], .1)

    def test_archive_identity_must_match_the_parameter_source(self):
        cfg, sample = self.fixture()
        sample['raw']['left_arm']['controller_id'] = 'different robot'
        moving = copy.deepcopy(sample)
        moving['motion']['busy'] = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.jsonl'
            path.write_text('\n'.join(json.dumps(s) for s in [moving, sample]))
            with self.assertRaisesRegex(ValueError, 'identity'):
                audit(cfg, path)


if __name__ == '__main__':
    unittest.main()
