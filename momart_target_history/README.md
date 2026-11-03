# Three matched new-box experiments

User-authorized launch: physical GPUs5/6/7, one arm per GPU. Every arm uses physical/effective BS256, seed100, exactly250000 successful optimizer updates, cosine LR horizon250000, and saves model/EMA/optimizer plus scheduler/AMP/RNG every25000 updates. All ten milestones are retained. No historical weights are loaded. Existing other GPU jobs are untouched.

| GPU | Variant | Extra historical tokens |
|---|---|---|
| 5 | box | none |
| 6 | recent8 | image256 + position256 from the previous8 actual frames |
| 7 | stride20_8 | image256 + position256 from the latest8 saved episode-clock frames0,20,40,... strictly before the current query |

All use the completed unified five-task target cache (654460 logical training frames), the original ten data sources/task IDs, original six observation modalities, two observations, nine predicted actions/eight executed actions, 50DDPM steps,8-layer PTP, shared training hyperparameters. Only current main RGB is painted. History reads only raw main RGB; wrist/depth/proprio/scan current observations remain inputs as before. Historical ROI pixels have no painted outline.

The newest observation is the query. History ends strictly before it, including at most the older current observation; no future frame, future selected target or reference label is read. Early/short episodes use invalid padding, never repeated observations. Sparse history is saved on the episode clock, not sampled backwards relative to arbitrary query phases. Each arm keeps the legacy length-grouping rule, with dataset/epoch-local seeds and an identical balanced task schedule so sampling is independent of model RNG. First20 batch-index hashes are logged for cross-arm comparison. Resume seeks past committed batches rather than prefetched batches.

All three arms share one sample-level main crop across t0,t1 and main depth. Historical frames use that SAME crop. Train follows the old random offset distribution; eval uses the old center crop. Thus augmentation sharing changes relative to the old per-image randomizer, consistently for all three new arms. Crop RNG depends on the committed optimizer step, independently of history/dropout/model RNG. Coordinates are translated, clipped and normalized in the cropped image; partial area is recorded, fully cropped targets are invalid. Historical pixels are cropped with the same transform, then ROIAlign resized to75x100, normalized identically to main RGB, and encoded by the SAME trainable main ResNet18 (not cached/frozen policy features).

Four-corner coordinates use sin/cos frequencies1,2,4,8 and a256D MLP; actual frame ages use low-frequency sin/cos and a256D MLP. A learned query conditioned on the known task scores aligned ROI/coordinate/time/visible-fraction records. The SAME temporal weights aggregate image and position values into two tokens, preserving correspondence. Both are attention-masked when history is empty. ROI activation checkpointing/chunks128 bound memory while retaining physical BS256 and visual gradients.

The two-history variants are initialized identically. All arms share identical initial current-observation parameters; common diffusion weights and time/current positional embeddings are copied from the same freshly initialized baseline, while two history slots are new. History tokens are packed in the two observation rows for the existing DDPM condition cache; the transformer unpacks `[diffusion_step,history_image,history_position,obs_t0,obs_t1]`. Both history tokens are causal for all action positions (they only use frames<=t0), and PTP still masks obs_t1 from the earliest action. Short-condition dropout0.3 drops the whole packed condition, including validity; the old no-memory long dropout remains inactive.

AMP overflow retries the same physical batch without advancing the scheduler, EMA or committed step. Milestone writes use temporary files and atomic rename. Normalizers and the concrete run config/source commit are recorded before training. No separate test/benchmark study is launched; syntax compilation and the actual formal run's first successful updates establish startup.

Deploy the committed packages, write the pushed commit to `.source_commit`, then:

```bash
bash momart_target_history/run_three.sh /path/to/deployed/project
```

Outputs: `/home/jrw/vpwem_outputs/momart_new_target_three_arms_bs256_250k_20261103/{box,recent8,stride20_8}/seed100/` and sibling variant logs. Raw data, caches, weights and generated logs are not committed.
