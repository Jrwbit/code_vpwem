# MoMaRT detector pilot

This experiment compares external perception routes without changing the
No-History policy or starting A1/A2/A3 training. Data and weights belong under
the ignored `artifacts/` tree or the server output directory, never in Git.

The reference is assistant-reviewed, simulation-assisted visible annotation,
not independent human GT. Six classes are red bowl, dishwasher door, dishwasher
door handle, dishwasher rack, drawer, and drawer handle. A drawer pull and its
two support ends form **one** instance. Parent boxes include their visible
handle pixels. Maximum x/y box coordinates are exclusive.

The pilot selects 20 episodes from four dataset files with seed 20261031:
three train, one development, and one held-out episode per file. Each episode
contributes eight spread anchors; development/test also contain eight
consecutive frames for causal tracking. Source PNG hashes and episode IDs are
stored in the manifest. Nine of 224 candidate frames failed whole-image replay
alignment (MAE > 12) and were rejected before any detector training. The frozen
reference contains 215 images and 1,114 instances: 90/62/63 train/dev/test images.

`corpus.py` reads original HDF5 RGB. `annotate.py` uses stored simulator state
and curated CAD parts **only to produce labels**. It restores the state without
advancing physics, forces static/sleeping renderer transforms to update, and
checks every image against the original RGB. The first CAD split included part
of the drawer front panel; visual review led to a narrower geometric selection.
`merge_handles.py` records the subsequent correction from two support ends to
one physical drawer pull. Current `annotate.py` emits this corrected taxonomy
directly; the migration script is only needed for the archived v2 output.

`labels.py` produces YOLO labels and review sheets. Acceptance requires a
separate `review.json` recording the actual review, limitations and SHA256 of
both annotation JSON and manifest. Automatically exported per-frame `pending`
fields describe the pre-review state; the signed aggregate review is recorded
separately to preserve the original export bytes. `export_package.py` bundles
the exact RGB, visible masks, COCO boxes, YOLO labels, provenance and review.

`train_yolo.py` fine-tunes YOLOv8n for up to 100 epochs at 320 pixels, choosing
the checkpoint using the development split. A second run uses `--imgsz 640
--name yolo_n_640` for native 640 training. `train_mask.py` fine-tunes an actual
COCO-pretrained ResNet50-FPN Mask R-CNN for 25 epochs, including visible-mask
supervision, choosing its checkpoint by development F1. This tests the domain
adaptation route associated with VIMA; it does not reproduce the VIMA policy.

`benchmark.py` compares the original six-prompt GroundingDINO, strict returned
phrases, joint explicit token spans, YOLO-World at 320/640, fine-tuned YOLO at
320/640, Mask R-CNN and the old bowl-only `combined_blend` reference. All use
the same clipping, class NMS (0.5), bowl red-support gate and 100-detection cap.
The detection benchmark cap differs from the old memory bank's eight slots;
integrating a selected detector still requires a shared capacity decision for
A1/A2/A3. Evaluation uses class-aware one-to-one IoU matching. Per-class
thresholds are selected only on development: prefer precision >= .95 with at
least five predictions, otherwise report the best F1 operating point and flag
that the target was not achieved. Empty output never means perfect precision.

Inference timing is batch one, FP32, with preloaded original RGB, two warm-up
frames and CUDA synchronization. Low-score proposal generation is separated
from a second timing pass using the frozen deployment thresholds. Model/text
initialization, disk I/O, ROI features, memory and policy are excluded. The
reported model latency is therefore not the end-to-end control loop latency.

`track_sam.py` tests eight-frame forward-only SAM2 propagation after first-frame
initialization. The actual deployment probe uses the best static detector as
selected by development F1; an explicitly separate oracle-first-box experiment
measures annotation/tracking headroom and is never treated as an online result.
There is no future correction, backward pass, re-detection or new-object
discovery in these short probes. Parent/part masks can overlap. Source video
is encoded losslessly and checked against the original RGB before inference.

On BHR, the environment is `/home/jrw/vpwem_envs/detector_compare`, inheriting
PyTorch 2.6/cu118 from the existing Python 3.9 environment, with Ultralytics
8.3.241, CLIP and Polars installed separately. The baseline environment was not
modified. Runtime outputs are in
`/home/jrw/vpwem_outputs/momart_detector_comparison_20261030/`.

Run `test_metrics.py` locally and `test_adapters.py` in the detector environment.
Then run `run_suite.py --output OUTPUT --stage detectors`, `--stage mask`, and
`--stage tracking` sequentially. Use the repository's `tools/gpu_gate.py` to
wait for a free device. The initial GPU5 timing attempt was terminated after
an external training process began sharing that card. Controlled comparisons
restart on GPU3 of the same RTX 4090 D model; no external process was stopped.
