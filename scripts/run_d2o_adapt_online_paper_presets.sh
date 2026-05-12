#!/usr/bin/env bash
set -euo pipefail

# Reproduce the dataset-specific online D2O+ADAPT hyperparameters
# recorded in configs/d2o_paper_hparams.yaml.

DATA_ROOT="${DATA_ROOT:-./data}"
MODEL_ARCH="${MODEL_ARCH:-ViT-B/16}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-./outputs/d2o_adapt_online_paper_presets}"
PRESET_GROUP="${PRESET_GROUP:-all}"
PRESET_TESTSETS="${PRESET_TESTSETS:-}"

run_group() {
  local name="$1"
  local testsets="$2"
  local gamma_env="$3"
  local lambda_proj="$4"

  if [[ "${PRESET_GROUP}" != "all" && "${PRESET_GROUP}" != "${name}" ]]; then
    return
  fi

  if [[ -n "${PRESET_TESTSETS}" ]]; then
    testsets="${PRESET_TESTSETS}"
  fi

  echo "[online:${name}] TESTSETS=${testsets} gamma_env=${gamma_env} cpen_lambda_proj=${lambda_proj}"
  DATA_ROOT="${DATA_ROOT}" \
  MODEL_ARCH="${MODEL_ARCH}" \
  TESTSETS="${testsets}" \
  GAMMA_ENV="${gamma_env}" \
  CPEN_LAMBDA_PROJ="${lambda_proj}" \
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${name}" \
  bash scripts/run_d2o_adapt_online.sh
}

run_group "imagenetv2" "imagenetv2" "0.1" "0.4"
run_group "imagenet_variants_other" "imagenet_a/imagenet_r/imagenet_sketch" "0.5" "0.4"
run_group "imagenet_c" "imagenet_c" "0.15" "0.4"
run_group "pug" "pug_cpitch/pug_cyaw/pug_croll/pug_opitch/pug_oroll/pug_oscale/pug_otexture/pug_oyaw/pug_slight/pug_worlds" "0.8" "0.4"
run_group "fine_grained_special" "oxford_flowers/fgvc_aircraft/ucf101" "0.15" "0.4"
run_group "fine_grained_default" "caltech101/stanford_cars/dtd/eurosat/food101/oxford_pets/sun397" "0.1" "0.4"
