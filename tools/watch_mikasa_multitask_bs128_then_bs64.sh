#!/usr/bin/env bash
set -u

watched_pid=${1:?usage: $0 PID}
repo=/home/jrw/code_vpwem_fair_history
root=/home/jrw/vpwem_outputs/direct_token2_real_batch/mikasa_multitask
bs128_log=${root}/real_bs128_grad1/train_bs128.log
bs64_work=${root}/real_bs64_grad1
watch_log=${root}/bs128_to_bs64_watch.log

printf '[%s] WATCH pid=%s condition=cuda_oom fallback=real_bs64_grad1 gpu=3\n' \
  "$(date -Is)" "$watched_pid" >>"$watch_log"
while kill -0 "$watched_pid" 2>/dev/null; do
  sleep 30
done

if grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$bs128_log"; then
  printf '[%s] OOM_DETECTED pid=%s launching=real_bs64_grad1 gpu=3\n' \
    "$(date -Is)" "$watched_pid" >>"$watch_log"
  mkdir -p "$bs64_work"
  cd "$repo"
  nohup bash tools/run_mikasa_direct_token2_multitask_bs64_gpu3.sh \
    >"$bs64_work/train_bs64.log" 2>&1 &
  printf '[%s] BS64_PID=%s\n' "$(date -Is)" "$!" >>"$watch_log"
  backup=/home/jrw/vpwem_checkpoint_copies/direct_token2_mikasa_multitask_bs64
  mkdir -p "$backup"
  nohup bash -c 'while true; do if test -d /home/jrw/vpwem_outputs/direct_token2_real_batch/mikasa_multitask/real_bs64_grad1/models; then find /home/jrw/vpwem_outputs/direct_token2_real_batch/mikasa_multitask/real_bs64_grad1/models -maxdepth 1 -type f -mmin +1 \( -name "model_*.pt" -o -name "scheduler_*.pt" \) -exec cp --no-clobber --preserve=timestamps -t /home/jrw/vpwem_checkpoint_copies/direct_token2_mikasa_multitask_bs64 {} +; fi; date -Is > /home/jrw/vpwem_checkpoint_copies/direct_token2_mikasa_multitask_bs64/LAST_SCAN; sleep 30; done' \
    >"$backup/copy_watcher.log" 2>&1 &
else
  printf '[%s] EXIT_NO_OOM pid=%s fallback_not_started=true\n' \
    "$(date -Is)" "$watched_pid" >>"$watch_log"
fi
