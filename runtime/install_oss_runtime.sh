#!/usr/bin/env bash
set -euo pipefail

ROOT=/oss/sh2/vpwem_runtime
FINAL_ENV=$ROOT/envs/vpwem
BUILD_ROOT=/tmp/vpwem_micromamba_build
BUILD_ENV=$BUILD_ROOT/env
RELOCATED=$BUILD_ROOT/relocated
LOG=$ROOT/logs/install.log
STATUS=$ROOT/status
MICROMAMBA=$ROOT/tools/micromamba
ARCHIVE=$ROOT/env_images/vpwem-py39-cu128-complete.tar.gz

mkdir -p "$ROOT"/{tools,envs,env_images,pkgs,cache/pip,cache/torch,cache/huggingface,cache/triton,cache/cuda,cache/pycache,tmp,env_specs,logs,status}
exec >>"$LOG" 2>&1
printf '%s installer start\n' "$(date -Is)"

if [[ -f "$STATUS/ENV_READY" && -x "$FINAL_ENV/bin/python" ]]; then
    export PYTHONPATH=/oss/sh2/vpwem:/oss/sh2/vpwem_runtime/sources/iGibson-momart${PYTHONPATH:+:$PYTHONPATH}
    if "$FINAL_ENV/bin/python" "$ROOT/env_specs/validate.py" >"$ROOT/logs/final-validation.json" 2>&1; then
        printf '%s existing environment is already valid\n' "$(date -Is)"
        exit 0
    fi
fi

cleanup() {
    local status=$?
    if [[ $status -eq 0 ]]; then
        rm -rf "$BUILD_ROOT"
    else
        printf '%s installer failed status=%s; preserving build for resume\n' \
            "$(date -Is)" "$status"
    fi
    rmdir "$STATUS/install.lock" 2>/dev/null || true
}
trap cleanup EXIT

if ! mkdir "$STATUS/install.lock" 2>/dev/null; then
    printf '%s another installer owns the lock\n' "$(date -Is)"
    exit 0
fi

retry() {
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        printf '%s attempt=%s command=%q\n' "$(date -Is)" "$attempt" "$1"
        if "$@"; then
            return 0
        fi
        printf '%s failed; retrying in 30 seconds\n' "$(date -Is)"
        sleep 30
    done
}

if [[ ! -x "$MICROMAMBA" ]] || ! "$MICROMAMBA" --version >/dev/null 2>&1; then
    rm -f "$ROOT/tmp/micromamba.tar.bz2.part"
    retry curl -fL --retry 20 --retry-all-errors --retry-delay 5 \
        -o "$ROOT/tmp/micromamba.tar.bz2.part" \
        https://micro.mamba.pm/api/micromamba/linux-64/latest
    rm -rf "$ROOT/tmp/micromamba-extract"
    mkdir -p "$ROOT/tmp/micromamba-extract"
    tar -xjf "$ROOT/tmp/micromamba.tar.bz2.part" -C "$ROOT/tmp/micromamba-extract"
    cp "$ROOT/tmp/micromamba-extract/bin/micromamba" "$MICROMAMBA"
    chmod 755 "$MICROMAMBA"
fi
"$MICROMAMBA" --version

mkdir -p "$BUILD_ROOT"
export MAMBA_ROOT_PREFIX=$BUILD_ROOT/mamba
export XDG_CACHE_HOME=$ROOT/cache
export PIP_CACHE_DIR=$ROOT/cache/pip
export TORCH_HOME=$ROOT/cache/torch
export HF_HOME=$ROOT/cache/huggingface
export TRITON_CACHE_DIR=$ROOT/cache/triton
export CUDA_CACHE_PATH=$ROOT/cache/cuda
export PYTHONPYCACHEPREFIX=$ROOT/cache/pycache
mkdir -p "$BUILD_ROOT/tmp"
# OSSFS cannot safely host compiler object files (ftruncate/symlink semantics).
# Only disposable build scratch uses the instance disk; cleanup removes it.
export TMPDIR=$BUILD_ROOT/tmp
export MUJOCO_GL=egl

if [[ ! -x "$BUILD_ENV/bin/python" ]]; then
    retry "$MICROMAMBA" create -y -p "$BUILD_ENV" -f "$ROOT/env_specs/environment.yml"
fi

PYTHON=$BUILD_ENV/bin/python
PIP=("$PYTHON" -m pip)
if ! "$PYTHON" -c 'import torch, torchvision, torchaudio; assert torch.__version__.startswith("2.8.0")' \
    >/dev/null 2>&1; then
    retry "${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
fi
retry "${PIP[@]}" install --no-build-isolation -r "$ROOT/env_specs/pip-requirements.txt"
retry "${PIP[@]}" install --no-build-isolation \
    'robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz'

if [[ -f /oss/sh2/vpwem_runtime/sources/ManiSkill/pyproject.toml ]]; then
    MANISKILL_SOURCE=$BUILD_ROOT/maniskill-source
    rm -rf "$MANISKILL_SOURCE"
    mkdir -p "$MANISKILL_SOURCE"
    cp -a /oss/sh2/vpwem_runtime/sources/ManiSkill/. "$MANISKILL_SOURCE/"
    retry "${PIP[@]}" install --no-build-isolation --no-deps "$MANISKILL_SOURCE"
else
    printf '%s\n' 'persistent ManiSkill source is missing' >&2
    exit 1
fi

# Build package metadata and wheels on the local scratch disk.  OSSFS is the
# persistent source of truth, but is too slow and lacks filesystem semantics
# expected by setuptools' build directory.
VPWEM_SOURCE=$BUILD_ROOT/vpwem-source
rm -rf "$VPWEM_SOURCE"
mkdir -p "$VPWEM_SOURCE"
cp -a /oss/sh2/vpwem/cleandiffuser "$VPWEM_SOURCE/"
cp -a /oss/sh2/vpwem/pyproject.toml /oss/sh2/vpwem/setup.py \
    /oss/sh2/vpwem/README.md /oss/sh2/vpwem/requirements.txt "$VPWEM_SOURCE/"
if [[ -d /oss/sh2/vpwem/.git ]]; then
    cp -a /oss/sh2/vpwem/.git "$VPWEM_SOURCE/"
fi
retry "${PIP[@]}" install --no-build-isolation --no-deps "$VPWEM_SOURCE"

if [[ -f /oss/sh2/bhr_20260916/MIKASA-Robo/setup.py ]]; then
    MIKASA_SOURCE=$BUILD_ROOT/mikasa-source
    rm -rf "$MIKASA_SOURCE"
    mkdir -p "$MIKASA_SOURCE"
    cp -a /oss/sh2/bhr_20260916/MIKASA-Robo/. "$MIKASA_SOURCE/"
    retry "${PIP[@]}" install --no-build-isolation --no-deps "$MIKASA_SOURCE"
else
    # The published package pins torch==2.2.1, which would replace the
    # Blackwell-capable torch 2.8/cu128 build installed above.  Runtime
    # dependencies are installed separately; keep the selected CUDA stack.
    retry "${PIP[@]}" install --no-deps 'mikasa-robo-suite==0.0.2'
fi

export PYTHONPATH=/oss/sh2/vpwem:/oss/sh2/vpwem_runtime/sources/iGibson-momart${PYTHONPATH:+:$PYTHONPATH}
"$PYTHON" "$ROOT/env_specs/validate.py" >"$ROOT/logs/build-validation.json"
"$MICROMAMBA" list -p "$BUILD_ENV" --explicit >"$ROOT/env_specs/conda-explicit.txt"
"${PIP[@]}" freeze >"$ROOT/env_specs/pip-freeze.txt"

rm -f "$ARCHIVE.part"
"$BUILD_ENV/bin/conda-pack" --ignore-missing-files --format tar.gz -p "$BUILD_ENV" \
    -o "$ARCHIVE.part" --dest-prefix "$FINAL_ENV"
mv "$ARCHIVE.part" "$ARCHIVE"

rm -rf "$RELOCATED"
mkdir -p "$RELOCATED"
tar -xzf "$ARCHIVE" -C "$RELOCATED"

rm -rf "$FINAL_ENV"
mkdir -p "$FINAL_ENV"
cp -RL "$RELOCATED"/. "$FINAL_ENV"/

export VPWEM_ENV=$FINAL_ENV
export PATH=$FINAL_ENV/bin:$PATH
"$FINAL_ENV/bin/python" "$ROOT/env_specs/validate.py" >"$ROOT/logs/final-validation.json"
PYTHONPATH=/oss/sh2/vpwem "$FINAL_ENV/bin/python" -m pytest -q /oss/sh2/vpwem/tests \
    >"$ROOT/logs/final-pytest.log" 2>&1
printf '%s\n' "$(date -Is)" >"$STATUS/ENV_READY.part"
mv "$STATUS/ENV_READY.part" "$STATUS/ENV_READY"
printf '%s installer complete\n' "$(date -Is)"
