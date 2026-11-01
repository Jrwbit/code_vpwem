# Single-target box selection pilot

This isolated experiment learns a pointer over up to eight detected object
instances plus a null option. It does not modify the production A1/A2/A3 model,
main RGB path, or completed shared caches.

`export_targets.py` restores stored simulator states without advancing physics.
It finds bowl translation and door/rack/drawer joint motion, retaining an event
only when a gripper surface is within 5 cm near onset. Supervision is the
currently moving object, otherwise the next event within 40 physical steps
(2 seconds). Overlapping objects and unknown periods are unlabelled. These are
privileged weak supervision proxies, **not recorded contact forces, human
intent, successful-grasp labels, or independent evaluation ground truth**.
Drawer collision geometry cannot distinguish a handle from its parent panel.

`export_features.py` has no target/GT/event arguments. Its inputs are current
and t-8 original RGB, current/delta proprioception, task ID, and the frozen
production detector. Per candidate it exports ROI features, box, confidence,
and class. Public feature files are separate from privileged target files.

`train.py` uses episode-disjoint train/dev/test splits from the detector corpus.
Normalizers and task priors use train only. Checkpoint selection uses dev only;
all 200 epochs are run and earliest tied dev-best checkpoint is retained. The
fixed pilot seeds are 100/200/300. The target is a specific instance matched by
class and IoU >= 0.5, rather than just a class label. Metrics separate detector
availability, selection conditional on availability, and visible-target recall.
Unknown intent periods are excluded, never counted as correct null predictions.
Null is trained for unavailable proxy targets and is not yet a calibrated
"no manipulation intent" prediction.

`evaluate_dense.py` predicts from public inputs and writes predictions before
reading privileged labels for scoring/visualization. It replays original
observations every 8 physical steps (0.4 seconds). Videos run at 2.5 fps, with
cyan predicted boxes and yellow offline proxy reference boxes. This is recorded
observation replay, not closed-loop robot control. The optional IoU/margin lock
never emits stale boxes on a missing observation; its parameters are a fixed
prototype, not a validated stage-completion mechanism.

## Reproduction on BHR

Use the existing `detector_compare` Python environment from the repository root.
All datasets, features, checkpoints, figures, videos and logs stay outside Git.

```bash
python experiments/momart_target_selector_20261101/export_targets.py \
  --corpus /home/jrw/vpwem_outputs/momart_detector_drawer_augmented_20261030 \
  --output /path/to/new/targets.json --render-gpu 0

CUDA_VISIBLE_DEVICES=3 python experiments/momart_target_selector_20261101/export_features.py \
  --corpus /home/jrw/vpwem_outputs/momart_detector_drawer_augmented_20261030 \
  --config /home/jrw/vpwem_outputs/momart_mask_cache_gpu3_6_20261101/gpu3.yaml \
  --output /path/to/new/features.npz

python experiments/momart_target_selector_20261101/train.py \
  --corpus /home/jrw/vpwem_outputs/momart_detector_drawer_augmented_20261030 \
  --targets /path/to/new/targets.json --features /path/to/new/features.npz \
  --output /path/to/new/seed100 --seed 100

python -m pytest -q experiments/momart_target_selector_20261101/test_selector.py
```

For a subset of task types, pass `--task-vocabulary` pointing to the original
feature JSON and `--stride 8` when exporting dense replay inputs. Do not refit
normalizers, vocabulary, or thresholds on the test trajectories. A source
manifest mismatch, reused output directory, or empty label split is an error.

Pilot artifacts: `/home/jrw/vpwem_outputs/momart_target_selector_20261101`.
`pilot_seed100` is the prespecified primary run; `pilot_seed200_actual` and
`pilot_seed300_actual` are explicit seed repetitions. Earlier directories named
`pilot_seed200`/`pilot_seed300` inadvertently used the default seed100 and are
excluded from repetitions. The saved result's `seed`, not its directory name,
is authoritative.

Before policy integration: review true intent/stage boundaries including failed
attempts and no-target periods, reserve fresh episodes, validate switching and
occlusion, then compare one-target-box conditioning against the baseline.
