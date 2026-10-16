#!/usr/bin/env bash
# Restart-safe formal SCoPE-to-H3 adapter training.  The platform may restart
# this script after a pre-emption; it always resumes from the latest adapter.
set -euo pipefail

# SCOPE2H3_ROOT may point at a cross-project shared copy when the original
# project's compute budget is exhausted.
ROOT="${SCOPE2H3_ROOT:-/inspire/hdd/project/wuliqifa/public/zxr}"
PROJECT_ROOT="${SCOPE2H3_PROJECT_ROOT:-${ROOT}/scope2h3}"
DATA_ROOT="${SCOPE2H3_DATA_ROOT:-${ROOT}/scope_h3_ref2va/data}"
MODEL_ROOT="${SCOPE2H3_MODEL_ROOT:-${ROOT}/models/MiniMax-H3-Ref2VA-BF16/Ref2VA}"
# Set RUN_TAG per submission to keep independently restarted formal runs apart.
# A retry of the same Job reuses this directory and resumes from its latest
# adapter checkpoint.
RUN_TAG="${RUN_TAG:-scope2h3-formal-10k-20261011}"
RUN_ROOT="${PROJECT_ROOT}/local/runs/${RUN_TAG}"
CONDA_ROOT=/inspire/hdd/project/wuliqifa/public/miniconda3
TARGET_STEPS=10000
STAGE_STEPS=500
SAVE_STEPS=250
# Capture a short, phase-level timing sample at the start of every resumed
# stage.  The profiler synchronizes ranks and is therefore deliberately
# limited to ten optimizer steps; it excludes checkpoint serialization.
PROFILE_SAMPLE_STEPS=10

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate h3
export DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/torchinductor-cache"
mkdir -p "${RUN_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${RUN_ROOT}/metadata" "${RUN_ROOT}/validation"
cd "${PROJECT_ROOT}"

MODEL_PATHS="$(python - "${MODEL_ROOT}" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
paths = [sorted(glob.glob(os.path.join(root, 'text_encoder', 'model*.safetensors'))), sorted(glob.glob(os.path.join(root, 'transformer', 'model*.safetensors'))), os.path.join(root, 'video_vae', 'source', 'model.safetensors'), os.path.join(root, 'audio_vae', 'model.safetensors')]
if not paths[0] or not paths[1] or any(not os.path.isfile(path) for path in paths[2:]):
    raise SystemExit('incomplete Ref2VA checkpoint')
print(json.dumps(paths))
PY
)"

METADATA="${RUN_ROOT}/metadata/train_spatialvid_waymo_balanced.json"
python - "${DATA_ROOT}" "${METADATA}" <<'PY'
import copy, json, random, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
def load(name):
    return json.loads((root / name / 'train_metadata.json').read_text())
def rebase(record, prefix):
    record = copy.deepcopy(record)
    for key in ('video', 'input_audio', 'camera_poses'):
        if key in record: record[key] = f'{prefix}/{record[key]}'
    for ref in record.get('references', []):
        for key in ('image', 'video'):
            if key in ref: ref[key] = f'{prefix}/{ref[key]}'
    return record
spatial = [rebase(x, '1324-spatialvid') for x in load('1324-spatialvid')]
waymo = [rebase(x, '2800-waymo') for x in load('2800-waymo')]
rng = random.Random(20261011)
records = spatial + rng.sample(waymo, len(spatial))
rng.shuffle(records)
output.write_text(json.dumps(records, ensure_ascii=False) + '\n')
print(f'balanced training records: {len(records)}')
PY

latest_step() {
  find "${RUN_ROOT}" -maxdepth 1 -type f -name 'step-*.safetensors' -printf '%f\n' 2>/dev/null | sed -E 's/^step-([0-9]+)\.safetensors$/\1/' | sort -n | tail -n1
}

validate() {
  local checkpoint=$1 step=$2 source=$3
  local base metadata
  case "${source}" in
    spatialvid) base="${DATA_ROOT}/1324-spatialvid"; metadata="${DATA_ROOT}/1324-spatialvid/val_metadata.json" ;;
    waymo) base="${DATA_ROOT}/2800-waymo"; metadata="${DATA_ROOT}/2800-waymo/val_metadata.json" ;;
  esac
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 apps/train_scope_adapter.py \
    --task sft --dataset_base_path "${base}" --dataset_metadata_path "${metadata}" \
    --data_file_keys video,input_audio,references,camera_poses --height 480 --width 832 --num_frames 124 \
    --dataset_repeat 1 --dataset_num_workers 4 --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/" \
    --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0 \
    --gradient_accumulation_steps 1 --use_gradient_checkpointing --find_unused_parameters \
    --output_path "${RUN_ROOT}" --resume_adapter_path "${checkpoint}" --eval_only \
    --eval_output_path "${RUN_ROOT}/validation/step-${step}-${source}.json" 2>&1 | tee -a "${RUN_ROOT}/validation.log"
}

completed="$(latest_step)"
completed="${completed:-0}"
while (( completed < TARGET_STEPS )); do
  next=$((completed + STAGE_STEPS))
  (( next > TARGET_STEPS )) && next=${TARGET_STEPS}
  profile_until=$((completed + PROFILE_SAMPLE_STEPS))
  (( profile_until > next )) && profile_until=${next}
  resume_args=()
  if (( completed > 0 )); then resume_args=(--resume_adapter_path "${RUN_ROOT}/step-${completed}.safetensors" --initial_step "${completed}"); fi
  echo "training global steps ${completed} -> ${next}"
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 apps/train_scope_adapter.py \
    --task sft --dataset_base_path "${DATA_ROOT}" --dataset_metadata_path "${METADATA}" \
    --data_file_keys video,input_audio,references,camera_poses --height 480 --width 832 --num_frames 124 \
    --dataset_repeat 20 --dataset_num_workers 4 --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/" \
    --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0 \
    --num_epochs 1 --max_steps "${next}" --gradient_accumulation_steps 1 --use_gradient_checkpointing --find_unused_parameters \
    --save_steps "${SAVE_STEPS}" --output_path "${RUN_ROOT}" --enable_tensorboard_log --enable_csv_log \
    --enable_profiling --profile_steps "${profile_until}" "${resume_args[@]}" \
    2>&1 | tee -a "${RUN_ROOT}/train.log"
  test -s "${RUN_ROOT}/step-${next}.safetensors"
  validate "${RUN_ROOT}/step-${next}.safetensors" "${next}" spatialvid
  validate "${RUN_ROOT}/step-${next}.safetensors" "${next}" waymo
  completed=${next}
done

python tools/legacy/plot_training_metrics.py --run-root "${RUN_ROOT}" || true
