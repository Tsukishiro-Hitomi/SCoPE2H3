#!/usr/bin/env bash
# One-video H100 smoke test before launching the full step-3000 evaluation.
set -euo pipefail

ROOT=/inspire/hdd/project/wuliqifa/public/zxr
PROJECT_ROOT="${ROOT}/scope2h3"
RUN_ROOT="${PROJECT_ROOT}/local/evaluations/step-3000-pose-200"
CONDA_ROOT=/inspire/hdd/project/wuliqifa/public/miniconda3
MODEL_ROOT="${ROOT}/models/MiniMax-H3-Ref2VA-BF16"
ADAPTER="${PROJECT_ROOT}/local/runs/scope2h3-formal-10k-h100-20261013/step-3000.safetensors"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate h3
export DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/torchinductor-cache"
mkdir -p "${RUN_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${PROJECT_ROOT}"

python apps/generate_camera_eval.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" --limit 1 \
  --model-dir "${MODEL_ROOT}" --adapter-path "${ADAPTER}" \
  --output-dir "${RUN_ROOT}/generated" --num-frames 124 --steps 50 --seed 42 \
  2>&1 | tee -a "${RUN_ROOT}/preflight.log"

test -s "${RUN_ROOT}/generated/1324-spatialvid-0000.mp4"
