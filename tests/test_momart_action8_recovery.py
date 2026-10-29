import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/momart_boxed_250k_action8_recovery_20261029/run.py'
spec = importlib.util.spec_from_file_location('momart_action8_recovery', SOURCE)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.meta = self.root / 'recovery_20261029'
        self.meta.mkdir()
        self.task = recovery.TASKS[0]
        self.path = self.root / 'seed100_repeat3' / 'step_250000' / self.task / 'result.npy'
        self.path.parent.mkdir(parents=True)
        self.value = dict(method='boxed_no_history', checkpoint_step=250000, eval_seed=100,
            env_name=self.task.rsplit('_', 1)[0], env_type='suboptimal',
            episode_success=np.array([1] * 15 + [0] * 15), mean_success=.5,
            episode_steps=np.full(30, 1504))

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, existing):
        (self.meta / 'manifest.json').write_text(json.dumps(dict(existing_results_sha256=existing)))

    def test_accepts_original_eight_step_result_and_rejects_four_step(self):
        np.save(self.path, self.value)
        self.assertEqual(recovery.read_result(self.path, 100, self.task)['successes'], 15)
        np.save(self.path, dict(self.value, execution_action_steps=4, episode_steps=np.full(30, 1500)))
        with self.assertRaises(ValueError):
            recovery.read_result(self.path, 100, self.task)

    def test_partial_recovery_preserves_original_and_does_not_report_final_rate(self):
        np.save(self.path, self.value)
        self.manifest({str(self.path): recovery.sha(self.path)})
        with patch.object(recovery, 'ROOT', self.root), patch.object(recovery, 'RECOVERY', self.meta):
            result = recovery.summarize()
        self.assertFalse(result['complete'])
        self.assertEqual(result['runs']['seed100_repeat3']['completed_tasks'], 1)
        self.assertNotIn('success_rate', result['runs']['seed100_repeat3'])

    def test_live_summary_ignores_in_progress_numpy_write(self):
        self.path.write_bytes(b'incomplete write')
        self.manifest({})
        with patch.object(recovery, 'ROOT', self.root), patch.object(recovery, 'RECOVERY', self.meta):
            result = recovery.summarize()
        self.assertEqual(result['runs']['seed100_repeat3']['completed_tasks'], 0)

    def test_modified_original_result_is_detected(self):
        np.save(self.path, self.value)
        self.manifest({str(self.path): recovery.sha(self.path)})
        np.save(self.path, dict(self.value, episode_success=np.ones(30), mean_success=1.))
        with patch.object(recovery, 'ROOT', self.root), patch.object(recovery, 'RECOVERY', self.meta):
            with self.assertRaisesRegex(ValueError, 'Previously valid result was changed'):
                recovery.summarize()


if __name__ == '__main__':
    unittest.main()
