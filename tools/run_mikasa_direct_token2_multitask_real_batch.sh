#!/usr/bin/env bash
set -u

repo=/home/jrw/code_vpwem_fair_history
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
output_root=/home/jrw/vpwem_outputs/direct_token2_real_batch/mikasa_multitask
supervisor_log=${output_root}/supervisor_gpu3.log
mkdir -p "$output_root"
cd "$repo"

is_oom() {
  local log=$1
  grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$log"
}

for batch in 256 192 128; do
  work=${output_root}/real_bs${batch}_grad1
  log=${work}/train.log
  mkdir -p "$work"
  printf '[%s] START benchmark=mikasa_multitask tasks=4 method=direct_token2 real_bs=%s grad_accum=1 gpu=3\n' \
    "$(date -Is)" "$batch" | tee -a "$supervisor_log"
  CUDA_VISIBLE_DEVICES=3 "$python_bin" pipelines/fair_history_mikasa_multitask.py \
    history_method=direct_token2 batch_size="$batch" gradient_steps=600000 \
    keep_checkpoints=20 work_dir="$work" hydra.run.dir="$work" \
    >"$log" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    printf '%s\n' "$batch" >"$output_root/SELECTED_REAL_BATCH"
    printf '[%s] DONE benchmark=mikasa_multitask tasks=4 method=direct_token2 real_bs=%s grad_accum=1\n' \
      "$(date -Is)" "$batch" | tee -a "$supervisor_log"
    exit 0
  fi
  if [ "$status" -eq 137 ] || is_oom "$log"; then
    printf '[%s] OOM benchmark=mikasa_multitask tasks=4 method=direct_token2 real_bs=%s grad_accum=1 status=%s\n' \
      "$(date -Is)" "$batch" "$status" | tee -a "$supervisor_log"
    continue
  fi
  printf '[%s] FAILED benchmark=mikasa_multitask tasks=4 method=direct_token2 real_bs=%s grad_accum=1 status=%s\n' \
    "$(date -Is)" "$batch" "$status" | tee -a "$supervisor_log"
  exit "$status"
done

printf '[%s] OOM_ALL benchmark=mikasa_multitask tasks=4 method=direct_token2 tried=256,192,128 grad_accum=1\n' \
  "$(date -Is)" | tee -a "$supervisor_log"
exit 2
