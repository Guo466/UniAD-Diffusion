#!/usr/bin/env bash

set -x

# CHECKPOINT=$1
# MODEL_NAME=$2

# WORK_DIRS=$(dirname "$CHECKPOINT")
# EXP_NAME=$(basename "$WORK_DIRS")

# cp configs/${EXP_NAME}.py configs/tmp.py

PYTHONPATH="./":$PYTHONPATH \
DEPLOY=True python convert_onnx_fix_distance_path.py --ckpt "tools/last.ckpt" --onnx "tools/DLP.onnx" --batch 8 --model_version "dlp_v009a_occ"