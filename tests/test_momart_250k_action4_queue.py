import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/momart_boxed_250k_action4_20261028/run.py'
spec = importlib.util.spec_from_file_location('momart_250k_action4_queue', SOURCE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class ResultContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task = queue.TASKS[0]
        self.value = dict(method='boxed_no_history', checkpoint_step=250000, eval_seed=100,
                          execution_action_steps=4, training_action_steps=8, horizon=9, obs_steps=2,
                          env_name=self.task.rsplit('_', 1)[0], env_type='suboptimal',
                          episode_success=np.array([1] * 18 + [0] * 12), mean_success=0.6,
                          episode_steps=np.full(30, 1500))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, value=None):
        path = self.root / 'result.npy'
        np.save(path, self.value if value is None else value)
        return path

    def test_valid_four_step_result(self):
        result = queue.read_result(self.write(), 100, self.task)
        self.assertEqual((result['successes'], result['episodes']), (18, 30))

    def test_old_protocol_or_wrong_seed_cannot_be_reused(self):
        for key, wrong in [('checkpoint_step', 150000), ('execution_action_steps', 8),
                           ('horizon', 5), ('eval_seed', 200)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    queue.read_result(self.write(dict(self.value, **{key: wrong})), 100, self.task)

    def test_partial_or_inconsistent_outcomes_are_rejected(self):
        for change in [dict(episode_success=np.ones(29)), dict(mean_success=0.9),
                       dict(episode_steps=np.full(30, 1496))]:
            with self.assertRaises(ValueError):
                queue.read_result(self.write(dict(self.value, **change)), 100, self.task)

    def test_partial_seed_does_not_get_final_success_rate(self):
        result = self.root / 'seed100' / 'step_250000' / self.task / 'result.npy'
        result.parent.mkdir(parents=True)
        np.save(result, self.value)
        with patch.object(queue, 'OUTPUT', self.root):
            summary = queue.summarize()
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['runs']['100']['completed_tasks'], 1)
        self.assertNotIn('success_rate', summary['runs']['100'])
        self.assertEqual(summary['runs']['200']['completed_tasks'], 0)


if __name__ == '__main__':
    unittest.main()
