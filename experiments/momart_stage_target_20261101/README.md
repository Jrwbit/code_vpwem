# Stage/role constrained target selection: an unsuccessful offline pilot

This isolated experiment tests whether predicted interaction roles can prevent
a pointer from selecting a rack when a bowl should be acquired. The tested hard
gate does not improve full target recall and introduces new wrong-role choices;
it is not enabled in the production policy.

`export_new_episodes.py` freezes eight valid-source episodes (seed20261102,
two per available task) before training. It excludes all 26 episodes in the
previous detector/selector corpus. This is new to that corpus, not a claim that
the data was never used anywhere else. Sampling is every16 physical steps.
The existing CAD annotation exporter rejects RGB replay MAE>12 using its
pre-existing rule: 364 candidate frames, nine rejected, 355 retained. Labels
are simulator-assisted reference boxes, not independent human GT.

`export_phases.py` reads simulator states only to make offline labels. A weak
holding proxy requires the recorded grasp heuristic to be positive now and
8 steps earlier, gripper-to-bowl surface distance <=3.5cm at both times, and
bowl pose relative to the gripper changing <=2.5cm. The recorded `proprio[2]`
is a filtered finger-width heuristic, not contact force or object identity.
Training-only pre-grasp labels may inspect the next16 steps. Geometry, GT and
future frames are never passed to the model's feature exporter.

Five roles are acquisition of bowl, door-handle operation, rack operation,
drawer-handle operation, and carry/place with unavailable recipient region.
Holding/recently released bowls keep the subject identity but have no labelled
recipient box. This null output means unavailable conditioning, not solved
placement, no manipulation intent, or confirmed task success. Other intervals
without proxy evidence remain unlabelled. The result remains weak supervision.

`run.py` reuses the previous causal RGB/proprio/task/detector features and adds
the masked mean of current candidate features to the context for both new
methods. The pointer and role head train on276 old train samples. Each method's
checkpoint is selected separately by old-dev role-macro decision accuracy;
normalizers use train only. No new episode enters training, checkpoint choice,
or reject calibration. The source passes predicted role logits to the class
gate, never the GT role. The gate allows only classes0/2/3/5 for the four
supported roles and only null for unsupported carry/place. Null remains valid
when the matching class has no detection.

The flat comparison uses the new label protocol but ignores predicted-role
gating. The previous saved pointer is evaluated without retraining, with its
original context and normalizers. Therefore previous-vs-new also changes label
supervision; the comparison does not isolate architecture alone. New-stage
acceptance scores multiply conditional pointer confidence by role confidence,
then choose a threshold on old-dev only. The target is at least90% selected-box
precision with >=10 selections; failure to meet this constraint is recorded,
never relabelled as a passing calibration.

Metrics report all visible-target recall, output coverage, wrong-role outputs,
wrong non-bowl boxes during acquisition, and outputs on unsupported recipient
frames. Rejecting everything yields zero recall, not100% accuracy. Unknown
intervals are excluded, and unsupported recipient nulls do not count as correct
object localizations.

## Run

Use the existing BHR `detector_compare` environment and fresh output paths.
Source data, frozen features, GT/proxies, checkpoints, logs and figures stay
outside Git. The completed shared cache and production model are untouched.

```bash
python experiments/momart_stage_target_20261101/export_new_episodes.py \
 --previous /home/jrw/vpwem_outputs/momart_detector_drawer_augmented_20261030 \
 --output /path/to/new_corpus

# Use existing detector_comparison_20261030/annotate.py with --reject-unaligned,
# then previous export_targets.py and causal export_features.py on new_corpus.
# Reuse the old feature JSON's task vocabulary for the new feature exporter.

python experiments/momart_stage_target_20261101/export_phases.py \
 --corpus /path/to/corpus --events /path/to/motion_events.json \
 --output /path/to/phases.json --gpu 0

python experiments/momart_stage_target_20261101/run.py \
 --old-corpus /path/to/previous_corpus --old-features /path/to/old_features.npz \
 --old-labels /path/to/old_phases.json --new-corpus /path/to/new_corpus \
 --new-features /path/to/new_features.npz --new-labels /path/to/new_phases.json \
 --previous-model /path/to/previous_selector.pt --output /path/to/seed100 --seed 100

python -m pytest -q experiments/momart_stage_target_20261101/test_stage.py \
 experiments/momart_target_selector_20261101/test_selector.py
```

Artifacts: `/home/jrw/vpwem_outputs/momart_stage_target_20261101`.
Seeds100/200/300 use the same frozen recipe. Primary analysis is seed100;
subsequent seeds are replications and not selected by new-test performance.
Figure selection is first two examples per predeclared error/coverage category,
including added gate errors and missed correct boxes, not just improvements.

Next evidence needed is reviewed stage/role transitions, failed attempts and
recipient regions. This pilot does not justify replacing the one-box pointer
or the eight-object production conditioning.
