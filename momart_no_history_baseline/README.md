# MoMaRT No-History baseline

This folder owns the raw-image No-History training and evaluation entries,
configuration, task-conditioned encoder, and observation-only dataset path. Training
and evaluation read persistent caches from `/home/jrw/vpwem_data/momart_zarr`,
including the `env_args` environment metadata. Rerunning the converter adds
metadata to existing complete caches without rewriting observation arrays.
Old HDF5 evaluation paths remain supported for compatibility. It uses the
repository's installed `cleandiffuser` and `robomimic` libraries for the
ResNet observation encoder, PTP decoder, DDPM training, and HDF5 support.
Run it from a checkout of this repository with its normal Python environment.

The model has two recent observations, nine action positions, no long-term
memory (`Tm=0`), and the same task embeddings and `LongMemChiTransformerPTP`
as the previous 1M raw-image No-History experiment. Physical and effective
batch sizes are both 64. The ten task datasets and seed remain unchanged.

The dataset returns only the two observations consumed by the model, while
retaining the nine-action target and the previous task and sample scheduling.
It skips all history image reads, processing, collation, and GPU copies. This
changes data loading cost, so compare resulting scores with the old run only
after evaluation. No speedup figure has been measured yet.

## Run

On BHR, from this repository checkout:

```bash
GPU=7 VPWEM_PYTHON=/home/jrw/anaconda3/envs/vpwem/bin/python \
  ./momart_no_history_baseline/run.sh \
  work_dir=/home/jrw/vpwem_outputs/fair_history_momart_no_history_baseline/seed100
```

Override `data_root=/path/to/momart_zarr` if needed. Run in a fresh `work_dir`:
the trainer resumes from the newest complete checkpoint pair in that directory.
Every 100k steps it saves `model_STEP.pt` and `scheduler_STEP.pt`, retaining all
ten milestones from 100k through 1M. Model files are about 836 MB each; allow
at least 9 GB of free space for checkpoint pairs plus logs.

For one task evaluation, use the saved Hydra config and normalizer statistics:

```bash
GPU=2 VPWEM_PYTHON=/home/jrw/anaconda3/envs/vpwem/bin/python \
  ./momart_no_history_baseline/evaluate.sh \
  --checkpoint /path/to/seed100/models/model_1000000.pt \
  --train-config /path/to/seed100/.hydra/config.yaml \
  --dataset-path /home/jrw/vpwem_data/momart_zarr/table_cleanup_to_sink_expert.zarr \
  --env-name table_cleanup_to_sink --env-type expert \
  --work-dir /path/to/eval/table_cleanup_to_sink_expert \
  --result-path /path/to/eval/table_cleanup_to_sink_expert/result.npy
```

For all ten tasks, set `TRAIN_ROOT=/path/to/seed100` and run
`./momart_no_history_baseline/evaluate_all.sh`. The evaluator uses the same
30 episodes per task and seed 20260917 as the previous 1M assessment.

No dataset or checkpoint is included in this code folder. The existing
four-method pipeline and completed 1M run remain separate.
