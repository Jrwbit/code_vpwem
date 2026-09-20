# sh2 VPWEM persistent layout

All durable VPWEM state is rooted at `/oss/sh2`:

- Code: `/oss/sh2/vpwem`
- Datasets: `/oss/sh2/vpwem_data`
- Checkpoints: `/oss/sh2/vpwem_checkpoints`
- Experiment outputs: `/oss/sh2/vpwem_outputs`
- Runtime environment: `/oss/sh2/vpwem_runtime/envs/vpwem`
- Reproducible environment image: `/oss/sh2/vpwem_runtime/env_images/vpwem-py39-cu128-complete.tar.gz`
- Simulator sources: `/oss/sh2/vpwem_runtime/sources`
- Simulator assets: `/oss/sh2/vpwem_runtime/assets`
- Package and compiler caches: `/oss/sh2/vpwem_runtime/cache`
- Legacy environment backups: `/oss/sh2/vpwem_runtime/env_images/legacy`

Activate the unified runtime with:

```bash
source /oss/sh2/vpwem_runtime/activate.sh
```

`TMPDIR` points to instance-local `/tmp` because OSSFS does not support every
POSIX operation needed by compilers and simulators. It is disposable scratch;
all code, environments, datasets, assets, checkpoints, caches, logs, and
experiment outputs remain under `/oss/sh2`.
