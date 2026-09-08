#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("vpwem", "random_cf", "unmatched_branch_cf", "matched_cf")


def parse_args():
    parser = argparse.ArgumentParser()
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}", required=True)
        parser.add_argument(f"--{method.replace('_', '-')}-offline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    return parser.parse_args()


def paired_bootstrap(values, seed, samples):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975))]


def exact_mcnemar(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    b = int(np.sum(left & ~right))
    c = int(np.sum(~left & right))
    discordant = b + c
    if discordant == 0:
        return {"left_only": b, "right_only": c, "p_exact": 1.0}
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {"left_only": b, "right_only": c, "p_exact": float(p_value)}


def load_rollout(path):
    return np.load(Path(path).expanduser(), allow_pickle=True).item()


def main():
    args = parse_args()
    rollout_paths = {
        method: getattr(args, method) for method in METHODS
    }
    offline_paths = {
        method: getattr(args, f"{method}_offline") for method in METHODS
    }
    rollouts = {method: load_rollout(path)
                for method, path in rollout_paths.items()}
    reference_mode = rollouts["vpwem"]["modes"]["correct"]
    reference_keys = (
        reference_mode["episode_reset_seed"],
        reference_mode["episode_env_slot"],
        reference_mode["episode_gt_branch"],
    )
    reference_donor = rollouts["vpwem"]["modes"][
        "matched_wrong"]["episode_donor_branch"]
    for method, result in rollouts.items():
        for intervention, mode in result["modes"].items():
            if not (
                    np.array_equal(
                        mode["episode_reset_seed"], reference_keys[0])
                    and np.array_equal(
                        mode["episode_env_slot"], reference_keys[1])
                    and np.array_equal(
                        mode["episode_gt_branch"], reference_keys[2])):
                raise ValueError(
                    f"{method}/{intervention} did not use the paired "
                    "episode/branch table")
        if not np.array_equal(
                result["modes"]["matched_wrong"]["episode_donor_branch"],
                reference_donor):
            raise ValueError(
                f"{method} did not use the shared matched-wrong donor table")

    summary = {
        "methods": {},
        "paired_comparisons": {},
        "within_method_interventions": {},
        "reproducibility_checks": {
            "shared_episode_table": True,
            "shared_gt_branch_table": True,
            "shared_matched_wrong_donor_table": True,
        },
    }
    for method, result in rollouts.items():
        correct = result["modes"]["correct"]
        wrong = result["modes"]["matched_wrong"]
        entry = dict(result["summary"])
        entry.update({
            "overall_success": float(correct["mean_success"]),
            "overall_success_macro": float(correct.get(
                "mean_success_macro", correct["mean_success"])),
            "branch_acc": float(correct["branch_acc"]),
            "branch_acc_macro": float(correct.get(
                "branch_acc_macro", correct["branch_acc"])),
            "matched_wrong_success": float(wrong["mean_success"]),
            "matched_wrong_branch_acc": float(wrong["branch_acc"]),
            "wrong_branch_choice_rate": float(wrong["wrong_branch_choice_rate"]),
            "first_direction_to_wrong_rate": float(
                wrong["first_direction_to_wrong_rate"]),
        })
        if offline_paths[method]:
            with open(Path(offline_paths[method]).expanduser(), encoding="utf-8") as stream:
                entry["offline"] = json.load(stream)
        summary["methods"][method] = entry

        success_delta = (
            correct["episode_success"].astype(np.float64)
            - wrong["episode_success"].astype(np.float64))
        correct_branch = (
            correct["episode_branch_choice"] == correct["episode_gt_branch"])
        wrong_branch = (
            wrong["episode_branch_choice"] == wrong["episode_gt_branch"])
        branch_delta = (
            correct_branch.astype(np.float64)
            - wrong_branch.astype(np.float64))
        wrong_attraction = (
            (wrong["episode_branch_choice"]
             == wrong["episode_donor_branch"]).astype(np.float64)
            - (correct["episode_branch_choice"]
               == wrong["episode_donor_branch"]).astype(np.float64))
        summary["within_method_interventions"][method] = {
            "success_correct_minus_matched_wrong": float(
                success_delta.mean()),
            "success_gap_ci95": paired_bootstrap(
                success_delta, args.bootstrap_seed + 2,
                args.bootstrap_samples),
            "success_mcnemar": exact_mcnemar(
                correct["episode_success"], wrong["episode_success"]),
            "branch_correct_minus_matched_wrong": float(
                branch_delta.mean()),
            "branch_gap_ci95": paired_bootstrap(
                branch_delta, args.bootstrap_seed + 3,
                args.bootstrap_samples),
            "branch_mcnemar": exact_mcnemar(
                correct_branch, wrong_branch),
            "wrong_attraction_uplift": float(wrong_attraction.mean()),
            "wrong_attraction_uplift_ci95": paired_bootstrap(
                wrong_attraction, args.bootstrap_seed + 4,
                args.bootstrap_samples),
        }

    matched = rollouts["matched_cf"]["modes"]["correct"]
    for control in ("vpwem", "random_cf", "unmatched_branch_cf"):
        baseline = rollouts[control]["modes"]["correct"]
        success_delta = (
            matched["episode_success"].astype(np.float64)
            - baseline["episode_success"].astype(np.float64))
        matched_branch = (
            matched["episode_branch_choice"] == matched["episode_gt_branch"])
        baseline_branch = (
            baseline["episode_branch_choice"] == baseline["episode_gt_branch"])
        branch_delta = matched_branch.astype(float) - baseline_branch.astype(float)
        summary["paired_comparisons"][f"matched_cf_minus_{control}"] = {
            "success_delta": float(success_delta.mean()),
            "success_delta_ci95": paired_bootstrap(
                success_delta, args.bootstrap_seed, args.bootstrap_samples),
            "success_mcnemar": exact_mcnemar(
                matched["episode_success"], baseline["episode_success"]),
            "branch_acc_delta": float(branch_delta.mean()),
            "branch_acc_delta_ci95": paired_bootstrap(
                branch_delta, args.bootstrap_seed + 1,
                args.bootstrap_samples),
            "branch_mcnemar": exact_mcnemar(
                matched_branch, baseline_branch),
        }

    matched_entry = summary["methods"]["matched_cf"]
    matched_vs_vpwem = summary["paired_comparisons"][
        "matched_cf_minus_vpwem"]
    matched_vs_random = summary["paired_comparisons"][
        "matched_cf_minus_random_cf"]
    matched_vs_unmatched = summary["paired_comparisons"][
        "matched_cf_minus_unmatched_branch_cf"]
    offline_available = all(
        "offline" in summary["methods"][method] for method in METHODS)
    gap_dominates = None
    if offline_available:
        matched_gap = matched_entry["offline"]["delta_cf"]
        gap_dominates = all(
            matched_gap > summary["methods"][method]["offline"]["delta_cf"]
            for method in ("vpwem", "random_cf", "unmatched_branch_cf"))
    gates = {
        "branch_acc_gain_ge_10pp": (
            matched_vs_vpwem["branch_acc_delta"] >= 0.10),
        "correct_minus_matched_wrong_success_ge_15pp": (
            matched_entry["success_gap_correct_minus_matched_wrong"] >= 0.15),
        "overall_success_not_below_vpwem_by_5pp": (
            matched_vs_vpwem["success_delta"] >= -0.05),
        "branch_acc_exceeds_random_and_unmatched": (
            matched_vs_random["branch_acc_delta"] > 0
            and matched_vs_unmatched["branch_acc_delta"] > 0),
        "success_not_below_random_or_unmatched_by_5pp": (
            matched_vs_random["success_delta"] >= -0.05
            and matched_vs_unmatched["success_delta"] >= -0.05),
        "offline_gap_exceeds_all_controls": gap_dominates,
        "wrong_history_attracts_action": (
            matched_entry["wrong_attraction_uplift"] > 0),
        "wrong_history_attracts_first_direction": (
            matched_entry.get("first_direction_wrong_uplift", 0.0) > 0),
    }
    available = [value for value in gates.values() if value is not None]
    summary["preregistered_gates"] = gates
    summary["go_to_necessity_gate"] = bool(available and all(available))

    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    with open(output / "RESULTS.md", "w", encoding="utf-8") as stream:
        stream.write("# NCM-VPWEM ShellGameTouch Stage-1 results\n\n")
        stream.write(
            "| Method | Success | BranchAcc | Correct−MatchedWrong | "
            "Offline ΔCF | Wrong attraction | First-dir attraction |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for method in METHODS:
            result = summary["methods"][method]
            stream.write(
                f"| {method} | {result['overall_success']:.3f} | "
                f"{result['branch_acc']:.3f} | "
                f"{result['success_gap_correct_minus_matched_wrong']:.3f} | "
                f"{result.get('offline', {}).get('delta_cf', float('nan')):.3f} | "
                f"{result['wrong_attraction_uplift']:.3f} | "
                f"{result.get('first_direction_wrong_uplift', float('nan')):.3f} |\n")
        stream.write("\n## Pre-registered gates\n\n")
        for key, value in gates.items():
            stream.write(f"- {key}: {value}\n")
        stream.write(
            f"\nDecision: {'GO' if summary['go_to_necessity_gate'] else 'NO-GO'}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
