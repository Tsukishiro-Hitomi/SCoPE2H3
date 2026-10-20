#!/usr/bin/env bash
# Generate the 200-video SCoPE camera-control evaluation set on one GPU.
# The generator atomically writes each MP4 and skips completed files, so a
# restarted/preempted job resumes safely.
set -euo pipefail

ROOT=/inspire/hdd/project/wuliqifa/public/zxr
PROJECT_ROOT="${ROOT}/scope2h3"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/local/evaluations/step-3000-pose-200}"
MANIFEST="${1:-${RUN_ROOT}/manifest.jsonl}"
CONDA_ROOT=/inspire/hdd/project/wuliqifa/public/miniconda3
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/models/MiniMax-H3-Ref2VA-BF16}"
ADAPTER="${ADAPTER_PATH:-${PROJECT_ROOT}/local/runs/scope2h3-formal-10k-h100-20261013/step-3000.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/generated}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate h3
export DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUN_ROOT}/torchinductor-cache}"
mkdir -p "${RUN_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${PROJECT_ROOT}"

python apps/generate_camera_eval.py \
  --manifest "${MANIFEST}" \
  --model-dir "${MODEL_ROOT}" --adapter-path "${ADAPTER}" \
  --output-dir "${OUTPUT_DIR}" --num-frames 124 --steps 50 --seed 42 \
  2>&1 | tee -a "${RUN_ROOT}/generation.log"
