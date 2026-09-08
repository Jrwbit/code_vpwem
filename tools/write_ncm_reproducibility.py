#!/usr/bin/env python3
"""Write a compact provenance manifest for an NCM ShellGameTouch stage."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


METHODS = ("vpwem", "random_cf", "unmatched_branch_cf", "matched_cf")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def git_value(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    stage = Path(args.stage).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    source_files = [
        "cleandiffuser/dataset/ncm_pairing.py",
        "cleandiffuser/dataset/mikasa_dataset_longmem.py",
        "cleandiffuser/diffusion/longmem_ddpm.py",
        "cleandiffuser/env/gymnasium_wrapper.py",
        "pipelines/longmemdpptp_mikasa_image.py",
        "pipelines/ncm_shellgame_eval.py",
        "pipelines/ncm_shellgame_offline_eval.py",
        "tools/collect_shellgame_neutral_hold.py",
        "tools/build_ncm_shellgame_manifest.py",
        "tools/analyze_ncm_shellgame.py",
        "tools/run_ncm_shellgame_stage1.sh",
        "tools/run_ncm_shellgame_stage1_oracle.sh",
        "configs/dp/mikasa/chi_transformer_ptp_longmem/shell_game_touch_ncm.yaml",
    ]
    marker = Path(args.dataset).expanduser() / "NCM_PHYSICAL_HOLD.json"
    payload = {
        "schema_version": 1,
        "stage": str(stage),
        "git_head": git_value(repo, "rev-parse", "HEAD"),
        "git_status_porcelain_sha256": hashlib.sha256(
            (git_value(repo, "status", "--porcelain=v1") or "").encode()
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "base_checkpoint": file_entry(args.base_checkpoint),
        "pair_manifest": file_entry(args.manifest),
        "dataset_marker": file_entry(marker),
        "source_files": {
            relative: file_entry(repo / relative)
            for relative in source_files
        },
        "methods": {},
    }
    with open(marker, encoding="utf-8") as stream:
        payload["dataset_protocol"] = json.load(stream)
    for method in METHODS:
        payload["methods"][method] = {
            "checkpoint": file_entry(
                stage / method / "train/models/model_10000.pt"),
            "rollout": file_entry(
                stage / method / "eval/ncm_eval.npy"),
            "offline": file_entry(
                stage / method / "offline/ncm_offline_gap.json"),
            "train_metrics": file_entry(
                stage / method / "train/metrics.jsonl"),
        }
    output = stage / "analysis/reproducibility.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    print(output)


if __name__ == "__main__":
    main()
