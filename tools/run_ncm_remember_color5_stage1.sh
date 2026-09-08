#!/usr/bin/env bash
set -euo pipefail

ROOT=${NCM_ROOT:-/home/jrw/code_vpwem}
PY=${NCM_PYTHON:-/home/jrw/anaconda3/envs/vpwem/bin/python}
OUT=${NCM_OUTPUT:-/home/jrw/vpwem_outputs/ncm_crossbench_20260823}
# Run the four causal comparison arms concurrently.  Override with four
# space-separated physical GPU ids when the allocation changes.
read -r -a GPUS <<< "${NCM_RC5_GPUS:-2 3 5 6}"
DATA=/home/jrw/vpwem_data/mikasa/RememberColor5NeutralHoldOracle-v0
SOURCE=/home/jrw/vpwem_data/mikasa/RememberColor5-v0
BASE=/home/jrw/vpwem_outputs/per_vpwem_screen_seed100/fifo_controls/mikasa/RememberColor5-v0/100/models/model_400000.pt
MANIFEST=${OUT}/remember_color5/pair_manifest/ncm_shellgame_pairs.npz
GATE=${OUT}/remember_color5/pair_manifest/PAIR_GATE.json

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

test -s "${BASE}"
test -s "${MANIFEST}"
"${PY}" - "${GATE}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("passed", False):
    raise SystemExit("RC5 pair-quality gate did not pass")
PY

mkdir -p "${OUT}/logs"
methods=(vpwem random_cf unmatched_branch_cf matched_cf)
modes=(none random unmatched_branch matched)
coefficients=(0.0 0.5 0.5 0.5)

if (( ${#GPUS[@]} != ${#methods[@]} )); then
  echo "NCM_RC5_GPUS must contain exactly ${#methods[@]} ids" >&2
  exit 2
fi

run_method() {
  local index=$1 gpu=$2
  method=${methods[$index]}
  mode=${modes[$index]}
  coefficient=${coefficients[$index]}
  work=${OUT}/remember_color5/stage1/${method}/train
  checkpoint=${work}/models/model_20000.pt
  if test -s "${checkpoint}"; then
    echo "[$(date -Is)] SKIP complete ${method}" >&2
    return 0
  fi
  mkdir -p "${work}"
  echo "[$(date -Is)] START ${method} gpu=${gpu}" >&2
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" \
    "${ROOT}/pipelines/longmemdpptp_mikasa_image.py" \
    --config-name=remember_color_5 \
    dataset_path="${DATA}" \
    +normalizer_dataset_path="${SOURCE}" \
    +ncm_pair_manifest_path="${MANIFEST}" \
    +ncm_pair_batch_fraction=0.1 \
    +ncm_pairing_seed=100 \
    +init_model_path="${BASE}" \
    work_dir="${work}" \
    save_path="${OUT}" \
    exp_name="ncm_remember_color5/${method}" \
    mem_compress_method=qformer \
    mem_compress_length=2 \
    qformer_num_layers=2 \
    qformer_cache_compress=fifo \
    qformer_cache_max_length=8 \
    diffusion_num_layers=8 \
    +counterfactual_negative_mode="${mode}" \
    +counterfactual_memory_coeff="${coefficient}" \
    +counterfactual_memory_margin=0.1 \
    +counterfactual_memory_probability=1.0 \
    gradient_steps=20000 \
    lr=0.00001 \
    save_freq=20000 \
    ++checkpoint_save_early_steps='[5000,10000]' \
    ++keep_checkpoints=4 \
    ++num_workers=8 \
    +inference_action_steps=1 \
    +dataset_preserve_episode_anchor=true \
    seed=100 \
    device=cuda:0 \
    wandb_mode=offline \
    >>"${OUT}/logs/remember_color5_${method}_stage1.log" 2>&1
  echo "[$(date -Is)] DONE ${method} gpu=${gpu}" >&2
}

pids=()
for index in "${!methods[@]}"; do
  run_method "${index}" "${GPUS[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || exit 1

date -Is >"${OUT}/remember_color5/stage1/TRAINING_COMPLETE"
