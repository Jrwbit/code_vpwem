import math
import unittest

import torch

from momart_boxed_no_history.resume import extend_cosine_horizon


class CosineExtensionTest(unittest.TestCase):
    def make_scheduler(self, step=150000):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000)
        state = scheduler.state_dict()
        state.update(last_epoch=step, _last_lr=[0.0], _step_count=step + 1)
        scheduler.load_state_dict(state)
        optimizer.param_groups[0]['lr'] = 0.0
        return optimizer, scheduler

    def test_reopens_zero_lr_and_preserves_step(self):
        optimizer, scheduler = self.make_scheduler()
        extend_cosine_horizon(scheduler, 250000)
        expected = 1e-4 * (1 + math.cos(math.pi * 150000 / 250000)) / 2
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)
        self.assertEqual(scheduler.last_epoch, 150000)
        optimizer.step()
        scheduler.step()
        expected_next = 1e-4 * (1 + math.cos(math.pi * 150001 / 250000)) / 2
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_next)

    def test_restart_preserves_optimizer_moments_and_exact_saved_lr(self):
        optimizer, scheduler = self.make_scheduler()
        param = optimizer.param_groups[0]['params'][0]
        param.grad = torch.ones_like(param)
        optimizer.step()
        moments = optimizer.state[param]['exp_avg'].clone()
        extend_cosine_horizon(scheduler, 250000)
        saved = scheduler.get_last_lr()[0]
        self.assertFalse(extend_cosine_horizon(scheduler, 250000))
        self.assertEqual(scheduler.get_last_lr()[0], saved)
        torch.testing.assert_close(optimizer.state[param]['exp_avg'], moments)

    def test_rejects_shorter_or_finished_horizon(self):
        for horizon in (100000, 150000):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                extend_cosine_horizon(self.make_scheduler()[1], horizon)

    def test_reaches_zero_at_extended_end(self):
        optimizer, scheduler = self.make_scheduler(step=249999)
        extend_cosine_horizon(scheduler, 250000)
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.last_epoch, 250000)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.0)


if __name__ == '__main__':
    unittest.main()
