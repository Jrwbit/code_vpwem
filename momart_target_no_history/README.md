# Replace the historical No-History boxes

Two modes consume the same new frozen single-target boxes:

- `box`: draw a red, one-pixel outline on both recent main RGB observations; retain the original policy architecture.
- `box_coordinates_roi`: draw those same outlines and retain the historical latest-frame position and ROI tokens. This uses the original 16-frequency Fourier position encoder, 64×64 unpainted ROI, shared main RGB ResNet and masked current-object transformer. It is not coordinate-only.

The selector reads main-camera sensors only. The **policy retains all observations in the chosen historical config**, including wrist RGB/depth, as explicitly confirmed by the user on 2026-11-03. No history bank, multi-object sequence or new ROI design is introduced. `legacy_tokens.py` is a verbatim copy of the original server model; see `legacy_source.json`.

## Paired configurations

```bash
python -m momart_target_no_history.configure \
  --base-config momart_boxed_no_history/config_bs256_150k.yaml \
  --target-metadata /path/to/completed_target_cache_metadata.json \
  --cache-root /path/to/new_target_sidecars \
  --output-root /path/to/new_policy_outputs \
  --config-dir /path/to/new_config_directory
python -m momart_target_no_history.train --config /path/to/new_config_directory/box.yaml
python -m momart_target_no_history.train --config /path/to/new_config_directory/box_coordinates_roi.yaml
```

Only input mode/provenance, sidecar location, method name and output directory change. All task sources, modalities, batch, learning rate, seed, optimizer settings, crop, horizon, DDPM steps and training budget come from the base. This example uses historical **batch256 / 150k**; the archived BS64/1M config instead preserves that recipe. Old policy weights are not loaded. Both variants share the base scientific settings. The old token config's inactive long-memory dropout was 0 rather than 0.3; there is no long-memory branch here.

## Cache and online inputs

Each source has `SOURCE.bbox.zarr`: float32 `boxes[N,4]` normalized xyxy and boolean `valid[N]`, in logical training-frame order. Attributes must contain:

- `complete=true`, `completed_frames=N`;
- `signature=source_signature(dataset, source_path)`, binding source, frame count, episode boundaries, demo order and source-frame map;
- identical `target_metadata` across both modes/all sources, containing `annotation_standard=causal_selected_target_v1`, `bundle_manifest_sha256`, `sensor_protocol_sha256`, and the exact `supported_sources` list. SHA values identify the actual frozen selector bundle and causal raw-sensor/frontend protocol used to produce the cache.

Invalid boxes leave the image unchanged and mask both extra tokens. Each observation uses its own box; only the latest supplies coordinate/ROI inputs. Tail padding repeats the last physical frame and selection. Batched reads preserve this rule.

`replace_target_inputs(obs, boxes, valid, mode)` is the common cached/online supplied-box bridge. Its online caller must obtain frozen-selector results for each observation's exact episode/sensor step, keep separate selector state per environment and reuse duplicated reset observations without advancing the selector again. The bridge itself does not run a detector or choose a target. Do not use the old red-bowl evaluation entry point for this new protocol.

## Readiness

Replacement input paths, configuration generator and training entry point are implemented. Both complete original multimodal policies passed CPU forward/backward checks. Tests cover cached/online supplied-box parity, unchanged wrist/depth/proprio/actions, missing boxes, duplicates/tail padding, cache identity and refusal to drop unsupported sources.

**Formal training has not started.** Full causal target caches are not generated, and the live simulator/selector observation bridge still needs integration. The frozen selector covers only `table_cleanup_to_dishwasher_expert`, `table_setup_from_dishwasher_expert`, `table_setup_from_dresser_expert`, `unload_dishwasher_to_dresser_suboptimal`. The old recipe contains ten sources; startup rejects the other six instead of falling back to old frames/labels or quietly reducing the suite. Complete ten-source replacement needs vocabulary extension and causal replay verification.

The earlier 4.62× benchmark was a main-camera-only shared skeleton, not a measured speedup for either complete multimodal variant.

## Unified ten-source update (2026-11-03)

The user explicitly rejected routing sink to a separate specialist. The current replacement is **one jointly trained five-task MLP64 pointer**, shared across all ten expert/suboptimal sources. `selector.UnifiedTargetSelector` constructs exactly one runtime and maps quality suffixes to the five actual task IDs. No original-model fallback or per-task model head is used. The four-task incumbent is an offline comparator only.

`prepare_sink.sh` reproduces the fixed episode-level sink corpus and the training-only weak labels. `train_unified.py` jointly fits original training data plus new sink training episodes; it selects weights using equal-weight five-task dev macro F1, and a single global calibration using dev only. Newly sampled sink source-valid test episodes are reported only after freezing. The new bundle contains exactly one weight file. Its 732 public-feature queries matched the original checkpoint's probabilities/slots exactly. Existing confirmation10 remains unopened.

`inventory.py` verified all ten read-only perception stores: 654,460 original logical training frames. `prepare_cache.py` reuses only those matching frozen detections/ROI features, batches frozen main-image global features on GPU1, then computes current/t−8 own-robot geometry and unified selection on four CPU workers. Every frame is included, even frames excluded from weak-label supervision. Each source has one writer, a committed prefix, an exclusive lock, and checksummed final arrays. Future globals may exist on disk, but each query only accesses its own current and same-episode t−8 entries.

```bash
python -m momart_target_no_history.train_unified --root /home/jrw/vpwem_outputs \
  --output /path/to/new_joint_model --device cuda:0
python -m momart_target_no_history.prepare_cache --config momart_object_memory/config.yaml \
  --perception-root /path/to/verified/perception_stores \
  --selector-bundle /path/to/frozen/five_task_bundle \
  --urdf /path/to/fetch_gripper.urdf --output /path/to/new/target_cache \
  --device cuda:0 --batch-size 128 --workers 4
```

See the latest unified-cache report for measured final completion; the earlier four-source readiness paragraph records the state before this update. Neither these proxy scores nor cache completion establishes policy closed-loop success.
