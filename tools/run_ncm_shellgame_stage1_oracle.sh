#!/usr/bin/env bash
set -euo pipefail

# Scientifically preferred Stage-1 protocol: shared zero arm deltas with the
# gripper held open, followed by MIKASA's released state-oracle demonstrator in closed
# loop.  The generic runner still supports the marked hybrid smoke dataset,
# but this wrapper deliberately cannot fall back to it.

export NCM_ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
export NCM_PYTHON=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
export NCM_DATASET=${NCM_DATASET:-/home/jrw/vpwem_data/mikasa/ShellGameTouchNeutralHoldOracle-v0}
export NCM_OUTPUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_vpwem_shellgame_stage1_oracle_seed100}
export NCM_SOURCE_DATASET=${NCM_SOURCE_DATASET:-/home/jrw/vpwem_data/mikasa/ShellGameTouch-v0}
export NCM_ORACLE_CHECKPOINT=${NCM_ORACLE_CHECKPOINT:-/home/jrw/MIKASA-Robo/oracle_checkpoints/ppo_memtasks/state/normalized_dense/ShellGameTouch-v0/ppo-mlp-state-dense-shell-game-touch-v0__123__state__20250312_085649/20250312_085649/final_success_ckpt.pt}
export NCM_COLLECTION_GPU=${NCM_COLLECTION_GPU:-7}
export NCM_PAIRING_GPU=${NCM_PAIRING_GPU:-2}

if [[ ! -s "${NCM_ORACLE_CHECKPOINT}" ]]; then
  echo "Missing released MIKASA oracle checkpoint: ${NCM_ORACLE_CHECKPOINT}" >&2
  exit 2
fi

exec bash "${NCM_ROOT}/tools/run_ncm_shellgame_stage1.sh"
