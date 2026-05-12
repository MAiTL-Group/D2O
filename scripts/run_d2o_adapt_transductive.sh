#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
TESTSETS="${TESTSETS:-pug_cpitch/pug_cyaw/pug_croll/pug_opitch/pug_oroll/pug_oscale/pug_otexture/pug_oyaw/pug_slight/pug_worlds}"
MODEL_ARCH="${MODEL_ARCH:-ViT-B/16}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/d2o_adapt_transductive}"
GAMMA_ENV="${GAMMA_ENV:-0.8}"
CPEN_LAMBDA_PROJ="${CPEN_LAMBDA_PROJ:-0.4}"

python -u ADAPT_transductive_ecw_adapt_probe.py \
  --data "${DATA_ROOT}" \
  --test_set "${TESTSETS}" \
  --arch "${MODEL_ARCH}" \
  --bank_size 6 \
  --alpha 0.9 \
  --bt 64 \
  --class_type Custom \
  --GPT \
  --cpen \
  --cpen_max_samples 128 \
  --cpen_lambda_proj "${CPEN_LAMBDA_PROJ}" \
  --cpen_style_dim 8 \
  --cpen_max_clusters 16 \
  --cpen_min_cluster_count 5 \
  --gamma_env "${GAMMA_ENV}" \
  --env_rho 0.05 \
  --env_max_abs 2.0 \
  --prior logits \
  --seed 0 \
  --output_dir "${OUTPUT_DIR}"
