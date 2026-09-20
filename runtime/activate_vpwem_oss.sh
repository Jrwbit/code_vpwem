#!/usr/bin/env bash

# Persistent VPWEM runtime on the sh2 OSS mount.
export VPWEM_ROOT=/oss/sh2/vpwem
export VPWEM_RUNTIME_ROOT=/oss/sh2/vpwem_runtime
export VPWEM_ENV=${VPWEM_RUNTIME_ROOT}/envs/vpwem
export VPWEM_DATA_ROOT=/oss/sh2/vpwem_data
export VPWEM_CHECKPOINT_ROOT=/oss/sh2/vpwem_checkpoints
export VPWEM_OUTPUT_ROOT=/oss/sh2/vpwem_outputs
export VPWEM_ASSET_ROOT=${VPWEM_RUNTIME_ROOT}/assets

export MAMBA_ROOT_PREFIX=${VPWEM_RUNTIME_ROOT}/micromamba
export CONDA_PKGS_DIRS=${VPWEM_RUNTIME_ROOT}/pkgs
export XDG_CACHE_HOME=${VPWEM_RUNTIME_ROOT}/cache
export PIP_CACHE_DIR=${VPWEM_RUNTIME_ROOT}/cache/pip
export TORCH_HOME=${VPWEM_RUNTIME_ROOT}/cache/torch
export HF_HOME=${VPWEM_RUNTIME_ROOT}/cache/huggingface
export TRITON_CACHE_DIR=${VPWEM_RUNTIME_ROOT}/cache/triton
export CUDA_CACHE_PATH=${VPWEM_RUNTIME_ROOT}/cache/cuda
export PYTHONPYCACHEPREFIX=${VPWEM_RUNTIME_ROOT}/cache/pycache
export PYTHONDONTWRITEBYTECODE=1

# OSSFS does not implement all POSIX operations used for temporary files.
export TMPDIR=/tmp/vpwem-runtime-${UID}
mkdir -p "${TMPDIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MS_ASSET_DIR=${VPWEM_ASSET_ROOT}/maniskill/data
export IGIBSON_SOURCE=${VPWEM_RUNTIME_ROOT}/sources/iGibson-momart
export __EGL_VENDOR_LIBRARY_FILENAMES=${VPWEM_ASSET_ROOT}/momart/nvidia_egl_vendor.json
export PYTHONPATH="${VPWEM_ROOT}:${IGIBSON_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${VPWEM_ENV}/bin:${PATH}"

cd "${VPWEM_ROOT}"
