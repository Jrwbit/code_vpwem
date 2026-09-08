import importlib.util
import os
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


class _FakeRunner:
    def __init__(self):
        self.obs = deque(maxlen=4)
        self.n_obs_steps = 3


class _FakeModel:
    def __init__(self):
        self.runner = _FakeRunner()
        self.windows = []

    @staticmethod
    def _padded(values, count):
        values = list(values)[-count:]
        if len(values) < count:
            values = [values[0]] * (count - len(values)) + values
        return values

    def get_action(self, observation=None):
        if observation is not None:
            self.runner.obs.append(observation)
        window = self._padded(self.runner.obs, self.runner.n_obs_steps)
        self.windows.append(
            {
                "frame_ids": [int(item["agent_pos"][0]) for item in window],
                "head_cam": [item["head_cam"].copy() for item in window],
            }
        )
        return np.zeros((6, 1), dtype=np.float32)

    def update_obs(self, observation):
        self.runner.obs.append(observation)

    def reset_obs(self):
        self.runner.obs.clear()


class _FakeTask:
    def __init__(self):
        self.step = 0
        self.take_action_cnt = 0
        self.eval_success = False
        self.max_reward = 0.0

    def take_action(self, action):
        self.step += 1
        self.take_action_cnt += 1

    def get_obs(self):
        return _encoded_observation(self.step)


def _encoded_observation(frame_id):
    # Deliberately asymmetric channels make an accidental RGB/BGR reversal
    # visible to the assertion below.
    head_cam = np.asarray([[[frame_id + 1, frame_id + 11, frame_id + 21]]], dtype=np.uint8)
    return {
        "agent_pos": np.asarray([frame_id], dtype=np.float32),
        "head_cam": head_cam,
    }


def _load_corrected_module(fake_model):
    package = types.ModuleType("DP")
    deploy = types.ModuleType("DP.deploy_policy")
    deploy.encode_obs = lambda observation: observation
    deploy.get_model = lambda usr_args: fake_model
    package.deploy_policy = deploy

    sys.modules.pop("DPFixedQueue", None)
    sys.modules.pop("DPCorrected", None)
    with mock.patch.dict(sys.modules, {"DP": package, "DP.deploy_policy": deploy}):
        sys.path.insert(0, str(TOOLS_DIR))
        try:
            spec = importlib.util.spec_from_file_location("DPCorrected", TOOLS_DIR / "DPCorrected.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["DPCorrected"] = module
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(TOOLS_DIR))
    return module


class RMBenchCorrectedDPTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("DPFixedQueue", None)
        sys.modules.pop("DPCorrected", None)

    def test_continuous_queue_and_channel_identity(self):
        model = _FakeModel()
        module = _load_corrected_module(model)
        task = _FakeTask()

        module.eval(task, model, _encoded_observation(0))
        module.eval(task, model, task.get_obs())

        self.assertEqual(model.windows[0]["frame_ids"], [0, 0, 0])
        self.assertEqual(model.windows[1]["frame_ids"], [4, 5, 6])
        np.testing.assert_array_equal(
            model.windows[1]["head_cam"][-1], _encoded_observation(6)["head_cam"]
        )
        self.assertEqual(model._rollout_diagnostics["fixed_queue_dedup_count"], 1)

    def test_corrected_policy_defaults_to_oidn_and_rejects_optix(self):
        model = _FakeModel()
        module = _load_corrected_module(model)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(module.get_model({"ckpt_setting": "demo_clean"}), model)
            self.assertEqual(os.environ["RMBENCH_RT_DENOISER"], "oidn")
        with mock.patch.dict(os.environ, {"RMBENCH_RT_DENOISER": "optix"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires RMBENCH_RT_DENOISER=oidn"):
                module.get_model({"ckpt_setting": "demo_clean"})

    def test_formal_launcher_pins_corrected_protocol(self):
        launcher = (TOOLS_DIR / "run_rmbench_corrected_ood_eval.sh").read_text()
        self.assertIn("export RMBENCH_RT_DENOISER=oidn", launcher)
        self.assertIn("--policy_name DPCorrected", launcher)
        self.assertIn("--policy_name VPWEMCorrected", launcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=4", launcher)

        default_launcher = (TOOLS_DIR / "run_rmbench_online_eval.sh").read_text()
        self.assertIn("export RMBENCH_RT_DENOISER=oidn", default_launcher)
        self.assertIn("--policy_name DPCorrected", default_launcher)
        self.assertIn("--policy_name VPWEMCorrected", default_launcher)


if __name__ == "__main__":
    unittest.main()
