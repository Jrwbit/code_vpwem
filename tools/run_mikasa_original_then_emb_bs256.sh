#!/usr/bin/env bash
set -euo pipefail

repo=/home/jrw/code_vpwem_momart_fair
python_bin=/home/jrw/anaconda3/envs/vpwem/bin/python
gpu=${GPU:-2}
raw_work=/home/jrw/vpwem_outputs/dpptp_mikasa_multitask/mikasa_all_balanced_original_dpptp_bs256/100
raw_ckpt=$raw_work/models/model_600000.pt
emb_work=/home/jrw/vpwem_outputs/dpptp_mikasa_multitask_emb/mikasa_all_balanced_embedding_dpptp_bs256/100
log_root=/home/jrw/vpwem_outputs/dpptp_mikasa_multitask_emb/logs

mkdir -p "$log_root" "$emb_work"
exec 9>/home/jrw/.mikasa-original-then-emb-bs256.lock
flock -n 9 || exit 0

while [[ ! -s "$raw_ckpt" ]]; do
    sleep 60
done

cd "$repo"
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$repo PYTHONUNBUFFERED=1 \
    "$python_bin" tools/build_mikasa_multitask_embeddings.py \
    --checkpoint "$raw_ckpt" --device cuda:0 \
    >> "$log_root/build_embeddings_gpu${gpu}.log" 2>&1

CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$repo WANDB_MODE=offline \
    PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8 \
    "$python_bin" pipelines/dpptp_mikasa_multitask_emb.py \
    --config-name=mikasa_multitask_emb batch_size=256 \
    work_dir="$emb_work" hydra.run.dir="$emb_work" \
    >> "$log_root/embedding_dpptp_bs256_gpu${gpu}_train.log" 2>&1
