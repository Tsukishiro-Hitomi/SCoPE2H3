#!/usr/bin/env bash
set -euo pipefail
PROJECT=/inspire/hdd/project/wuliqifa/public/zxr/scope2h3
VIPE="${PROJECT}/local/third_party/vipe"
RUN="${PROJECT}/local/evaluations/step-500-pose-200"
export TORCH_HOME="${PROJECT}/local/vipe-cache/torch"
export HF_HOME="${PROJECT}/local/vipe-cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTHONHOME="${PROJECT}/local/python312"
export PYTHONPATH="${VIPE}/.venv/lib/python3.12/site-packages:${VIPE}"
cd "${VIPE}"
"${PROJECT}/local/python312/bin/python3.12" "${PROJECT}/tools/check_vipe_offline.py"
for mode in generated gt; do
  extra=()
  if [[ "$mode" == gt ]]; then extra=(--ground-truth); fi
  "${PROJECT}/local/python312/bin/python3.12" "${PROJECT}/tools/run_vipe_pose_estimation.py" \
    --manifest "$1" --video-root "${RUN}/generated" \
    --output-root "${RUN}/vipe-${mode}" \
    --limit "${VIPE_LIMIT:-0}" \
    --vipe "${PROJECT}/tools/vipe_shared_runtime.sh" "${extra[@]}"
done
unset PYTHONHOME PYTHONPATH
PYTHON=/inspire/hdd/project/wuliqifa/public/miniconda3/envs/h3/bin/python
if "$PYTHON" "${PROJECT}/tools/build_vipe_metric_pairs.py" \
  --manifest "${RUN}/manifest-available.jsonl" \
  --gt-vipe-root "${RUN}/vipe-gt" --generated-vipe-root "${RUN}/vipe-generated" \
  --output "${RUN}/pairs-95.jsonl"; then
  PYTHONPATH="${PROJECT}/src" "$PYTHON" "${PROJECT}/apps/evaluate_camera_control.py" \
    --pairs "${RUN}/pairs-95.jsonl" --output "${RUN}/metrics-95.json"
else
  echo 'Other shards still have pending poses; the last shard will aggregate metrics.'
fi
