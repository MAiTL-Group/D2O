#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
TESTSETS="${TESTSETS:-imagenetv2/imagenet_a/imagenet_r/imagenet_sketch}"
MODEL_ARCH="${MODEL_ARCH:-ViT-B/16}"

python -u Pre_extract_class_emb_default.py \
  --data "${DATA_ROOT}" \
  --test_set "${TESTSETS}" \
  --arch "${MODEL_ARCH}" \
  --class_type Custom \
  --GPT \
  --descriptor_path ./descriptions
