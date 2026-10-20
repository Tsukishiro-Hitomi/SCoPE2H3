#!/usr/bin/env bash
# One-video runtime and artifact-format validation for the pinned ViPE stack.
set -euo pipefail

ROOT=/inspire/hdd/project/wuliqifa/public/zxr
PROJECT_ROOT="${ROOT}/scope2h3"
VIPE_ROOT="${PROJECT_ROOT}/local/third_party/vipe"
RUN_ROOT="${PROJECT_ROOT}/local/evaluations/step-3000-pose-200"
VIPE_CACHE="${PROJECT_ROOT}/local/vipe-cache"
OUTPUT="${RUN_ROOT}/vipe-preflight"

export TORCH_HOME="${VIPE_CACHE}/torch"
export HF_HOME="${VIPE_CACHE}/huggingface"
export TORCH_CUDA_ARCH_LIST=9.0
rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"
cd "${VIPE_ROOT}"
"${VIPE_ROOT}/.venv/bin/vipe" infer \
  "${RUN_ROOT}/generated/1324-spatialvid-0000.mp4" \
  --output "${OUTPUT}" --pipeline pose_only_long

test -s "${OUTPUT}/pose/1324-spatialvid-0000.npz"
"${VIPE_ROOT}/.venv/bin/python" - "${OUTPUT}/pose/1324-spatialvid-0000.npz" <<'PY'
import sys
import numpy as np

archive = np.load(sys.argv[1], allow_pickle=False)
poses, indices = archive["data"], archive["inds"]
assert poses.ndim == 3 and poses.shape[1:] == (4, 4), poses.shape
assert len(poses) == len(indices) and len(poses) > 1
print(f"ViPE pose archive: poses={poses.shape}, indices={indices.shape}")
PY
