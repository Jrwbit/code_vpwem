import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "summarize_ncm_crossbench.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_ncm_crossbench", MODULE_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def test_causal_shorthand_checkpoint_and_budget_class(tmp_path):
    result_path = tmp_path / "shell" / "scratch600" / "eval_5k" / "ncm_eval.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "success_gap_correct_minus_matched_wrong": 0.5,
                "modes": {
                    "correct": {"mean_success": 0.8, "branch_acc": 0.7},
                    "matched_wrong": {"mean_success": 0.3, "branch_acc": 0.2},
                },
            }
        )
    )

    rows = SUMMARY.collect_causal(tmp_path)

    assert len(rows) == 2
    assert {row["checkpoint"] for row in rows} == {5_000}
    assert {row["budget_class"] for row in rows} == {"diagnostic"}


def test_control_delta_requires_both_modes(tmp_path):
    work = tmp_path / "robomimic_controls" / "square_mh"
    work.mkdir(parents=True)
    for mode, success in (("none", 0.7), ("zero", 0.68)):
        np.save(
            work / f"result_1000000_memory-{mode}.npy",
            {"mean_success": success, "episode_success": np.ones(100)},
        )

    rows = SUMMARY.collect_controls(tmp_path)
    deltas = SUMMARY.control_deltas(rows)

    assert len(rows) == 2
    assert len(deltas) == 1
    assert deltas[0]["task"] == "square_mh"
    assert np.isclose(deltas[0]["normal_minus_zero"], 0.02)


def test_parity_diagnostics_keeps_swap_out_of_rollout_curves(tmp_path):
    work = tmp_path / "swap_blocks" / "corrected_ood10"
    work.mkdir(parents=True)
    (work / "episodes.tsv").write_text(
        "phase\tepisode\tsuccess\n"
        "dp\t0\t0\n"
        "dp\t1\t1\n"
        "vpwem\t0\t0\n",
        encoding="utf-8",
    )

    rows = SUMMARY.collect_parity_diagnostics(tmp_path)

    assert len(rows) == 2
    dp = next(row for row in rows if row["diagnostic"] == "corrected_ood_dp")
    assert dp["episodes"] == 2
    assert dp["successes"] == 1
    assert np.isclose(dp["success_rate"], 0.5)
    assert SUMMARY.collect_rollouts(tmp_path) == []
