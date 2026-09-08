#!/usr/bin/env bash

# Shared GPU admission policy for the remote experiment supervisors.
#
# A supervisor must first acquire an idle card. After the queue owns its turn,
# later stages continue unconditionally even if another user joins the card.
# GPU usage after acquisition is monitoring information, not a new admission
# gate. This keeps a multi-stage train/eval queue contiguous.

GPU_POLICY_INITIAL_MAX_USED_MIB=${GPU_POLICY_INITIAL_MAX_USED_MIB:-1024}
GPU_POLICY_INITIAL_MAX_UTILIZATION=${GPU_POLICY_INITIAL_MAX_UTILIZATION:-20}
GPU_POLICY_CONTINUE_WARN_USED_MIB=${GPU_POLICY_CONTINUE_WARN_USED_MIB:-${GPU_POLICY_CONTINUE_MAX_USED_MIB:-10240}}
GPU_POLICY_INITIAL_STABLE_SAMPLES=${GPU_POLICY_INITIAL_STABLE_SAMPLES:-3}
GPU_POLICY_POLL_SECONDS=${GPU_POLICY_POLL_SECONDS:-60}
GPU_POLICY_OWNS_TURN=${GPU_POLICY_START_OWNED:-0}
GPU_POLICY_ALLOWED_GPUS=${GPU_POLICY_ALLOWED_GPUS:-}

validate_pinned_gpu() {
  local gpu=$1 item
  [[ -z "$GPU_POLICY_ALLOWED_GPUS" ]] && return 0
  IFS=',' read -ra allowed_items <<< "$GPU_POLICY_ALLOWED_GPUS"
  for item in "${allowed_items[@]}"; do
    [[ "$gpu" == "${item// /}" ]] && return 0
  done
  printf '%s GPU_REJECTED gpu=%s allowed=%s reason=outside-pinned-set\n' \
    "$(date -Is)" "$gpu" "$GPU_POLICY_ALLOWED_GPUS" >&2
  return 2
}

wait_for_gpu_turn() {
  local gpu=$1 used utilization limit mode stable required_stable
  validate_pinned_gpu "$gpu" || return $?
  stable=0
  while true; do
    IFS=, read -r used utilization < <(nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)
    used=${used// /}
    utilization=${utilization// /}

    if (( GPU_POLICY_OWNS_TURN )); then
      mode=owned-continuation
      if (( used > GPU_POLICY_CONTINUE_WARN_USED_MIB )); then
        printf '%s GPU_CONTINUE_PRESSURE gpu=%s used_mib=%s utilization=%s warn_mib=%s action=continue\n' \
          "$(date -Is)" "$gpu" "$used" "$utilization" \
          "$GPU_POLICY_CONTINUE_WARN_USED_MIB"
      else
        printf '%s GPU_READY gpu=%s mode=%s used_mib=%s utilization=%s action=continue\n' \
          "$(date -Is)" "$gpu" "$mode" "$used" "$utilization"
      fi
      return 0
    else
      mode=initial
      limit=$GPU_POLICY_INITIAL_MAX_USED_MIB
      required_stable=$GPU_POLICY_INITIAL_STABLE_SAMPLES
      if (( used <= limit && utilization <= GPU_POLICY_INITIAL_MAX_UTILIZATION )); then
        stable=$((stable + 1))
      else
        stable=0
      fi
    fi

    if (( stable >= required_stable )); then
      GPU_POLICY_OWNS_TURN=1
      printf '%s GPU_READY gpu=%s mode=%s used_mib=%s utilization=%s limit_mib=%s\n' \
        "$(date -Is)" "$gpu" "$mode" "$used" "$utilization" "$limit"
      return 0
    fi
    printf '%s WAIT_GPU gpu=%s mode=%s used_mib=%s utilization=%s limit_mib=%s stable=%s/%s\n' \
      "$(date -Is)" "$gpu" "$mode" "$used" "$utilization" "$limit" \
      "$stable" "$required_stable"
    sleep "$GPU_POLICY_POLL_SECONDS"
  done
}
