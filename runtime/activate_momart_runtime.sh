#!/usr/bin/env bash

source /oss/sh2/vpwem_runtime/activate.sh
export VPWEM_DATA_ROOT=/oss/sh2/vpwem_data
export VPWEM_CHECKPOINT_ROOT=/oss/sh2/vpwem_checkpoints
export VPWEM_OUTPUT_ROOT=/oss/sh2/vpwem_outputs
export IGIBSON_SOURCE=/oss/sh2/vpwem_runtime/sources/iGibson-momart
export PYTHONPATH="${IGIBSON_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export __EGL_VENDOR_LIBRARY_FILENAMES=/oss/sh2/vpwem_runtime/assets/momart/nvidia_egl_vendor.json
cd /oss/sh2/vpwem
