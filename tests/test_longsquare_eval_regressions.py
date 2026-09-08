import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DOMAINS = ("mikasa", "momart", "robomimic")


def parse_repo_file(relative_path):
    return ast.parse((REPO_ROOT / relative_path).read_text())


class LongSquareEvalRegressionTest(unittest.TestCase):
    def test_memory_inference_uses_selected_ema_or_online_model(self):
        tree = parse_repo_file("cleandiffuser/diffusion/longmem_ddpm.py")
        sample = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "sample"
        )

        inference_roots = []
        for node in ast.walk(sample):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "inference"
            ):
                continue
            memory_lookup = node.func.value
            if not isinstance(memory_lookup, ast.Subscript):
                continue
            if isinstance(memory_lookup.value, ast.Name):
                inference_roots.append(memory_lookup.value.id)
            elif isinstance(memory_lookup.value, ast.Attribute):
                inference_roots.append(ast.unparse(memory_lookup.value))

        self.assertGreaterEqual(len(inference_roots), 2)
        self.assertEqual(set(inference_roots), {"model"})

    def test_longsquare_qformer_receives_training_time_memory_settings(self):
        tree = parse_repo_file("pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py")
        qformer_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "QFormerMemory"
        )
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in qformer_call.keywords}

        self.assertEqual(keywords["use_pos_emb"], "args.memory_pos_emb")
        self.assertEqual(
            keywords["subsample_ratio"], "args.dataset_memory_subsample_ratio"
        )

    def test_training_and_longsquare_eval_share_configured_architecture(self):
        for relative_path in (
            "pipelines/longmemdpptp_robomimic_image_emb.py",
            "pipelines/longmemdpptp_robomimic_lhsq_image_emb_eval.py",
        ):
            tree = parse_repo_file(relative_path)
            diffusion_call = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "LongMemChiTransformerPTP"
            )
            keywords = {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in diffusion_call.keywords
            }
            self.assertEqual(
                keywords["num_layers"],
                "args.diffusion_num_layers",
                relative_path,
            )

    def test_paper_pipelines_do_not_hardcode_four_diffusion_layers(self):
        for path in (REPO_ROOT / "pipelines").glob("*.py"):
            if not any(domain in path.name for domain in PAPER_DOMAINS):
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if "Transformer" not in node.func.id:
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                if "num_layers" not in keywords:
                    continue
                self.assertEqual(
                    ast.unparse(keywords["num_layers"]),
                    "args.diffusion_num_layers",
                    str(path.relative_to(REPO_ROOT)),
                )

    def test_online_qformer_construction_has_explicit_memory_settings(self):
        for path in (REPO_ROOT / "pipelines").glob("longmemdpptp_*.py"):
            if not any(domain in path.name for domain in PAPER_DOMAINS):
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "QFormerMemory"
                ):
                    continue
                keywords = {
                    keyword.arg: ast.unparse(keyword.value)
                    for keyword in node.keywords
                }
                self.assertEqual(
                    keywords.get("use_pos_emb"),
                    "args.memory_pos_emb",
                    str(path.relative_to(REPO_ROOT)),
                )
                self.assertEqual(
                    keywords.get("subsample_ratio"),
                    "args.dataset_memory_subsample_ratio",
                    str(path.relative_to(REPO_ROOT)),
                )

    def test_paper_configs_use_eight_diffusion_layers(self):
        config_roots = (
            REPO_ROOT / "configs/dp/mikasa",
            REPO_ROOT / "configs/dp/momart",
            REPO_ROOT / "configs/dp/robomimic_multi_modal",
        )
        for config_root in config_roots:
            for path in config_root.glob("chi_transformer*/**/*.yaml"):
                config = yaml.safe_load(path.read_text())
                self.assertEqual(
                    config.get("diffusion_num_layers"),
                    8,
                    str(path.relative_to(REPO_ROOT)),
                )

    def test_longsquare_reward_checks_the_selected_target_peg(self):
        robomimic = types.ModuleType("robomimic")
        robomimic_envs = types.ModuleType("robomimic.envs")
        env_robosuite = types.ModuleType("robomimic.envs.env_robosuite")
        env_robosuite.EnvRobosuite = object
        module_name = "_longsquare_wrapper_under_test"
        module_path = REPO_ROOT / "cleandiffuser/env/robomimic/robomimic_image_wrapper.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                "robomimic": robomimic,
                "robomimic.envs": robomimic_envs,
                "robomimic.envs.env_robosuite": env_robosuite,
            },
        ):
            spec.loader.exec_module(module)
        LongSquareWrapper = module.LongSquareWrapper

        calls = []
        robosuite_env = SimpleNamespace(
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    body_xpos={"square_nut": np.zeros(3, dtype=np.float32)}
                )
            ),
            obj_body_id={"SquareNut": "square_nut"},
            on_peg=lambda peg_pos, peg_id: calls.append((peg_pos, peg_id)) or False,
        )
        wrapped_env = SimpleNamespace(
            env=SimpleNamespace(env=robosuite_env),
            step=lambda action: (
                {
                    "robot0_eef_pos": np.array([1.0, 1.0, 1.0], dtype=np.float32)
                },
                0,
                False,
                {},
            ),
        )
        wrapper = LongSquareWrapper.__new__(LongSquareWrapper)
        wrapper.env = wrapped_env
        wrapper.target_peg_id = 1
        wrapper.rew = 0

        wrapper.step(np.zeros(7, dtype=np.float32))

        self.assertEqual(calls[0][1], 1)


if __name__ == "__main__":
    unittest.main()
