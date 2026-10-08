#!/usr/bin/env bash
# Interruption-tolerant 1,000-step SCoPE H3 Ref2VA run on 124_dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_ROOT="${OVERLAY_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OVERLAY_PARENT="${OVERLAY_PARENT:-$(dirname "${OVERLAY_ROOT}")}" 
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-${OVERLAY_PARENT}/DiffSynth-Studio}"
DATA_ROOT="${DATA_ROOT:-${OVERLAY_ROOT}/data/124_dataset}"
MODEL_ROOT="${MODEL_ROOT:-${OVERLAY_PARENT}/models/MiniMax-H3-Ref2VA-BF16/Ref2VA}"
RUN_NAME="${RUN_NAME:-scope-h3-124dataset-1000step}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OVERLAY_ROOT}/checkpoints/${RUN_NAME}}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
PROFILE_STEPS="${PROFILE_STEPS:-10}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/project/wuliqifa/public/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate h3
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH="${OVERLAY_PARENT}:${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for required in "${DIFFSYNTH_ROOT}/diffsynth" "${DATA_ROOT}/metadata_train.json" "${DATA_ROOT}/metadata_val.json" "${MODEL_ROOT}/processor"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"
INITIAL_STEP="${INITIAL_STEP:-0}"
if [[ -z "${RESUME_ADAPTER}" ]]; then
  latest_adapter=$(find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name 'step-*.safetensors' -printf '%f\n' | sort -V | tail -n 1)
  if [[ -n "${latest_adapter}" ]]; then
    RESUME_ADAPTER="${OUTPUT_ROOT}/${latest_adapter}"
    INITIAL_STEP="${latest_adapter#step-}"
    INITIAL_STEP="${INITIAL_STEP%.safetensors}"
  fi
fi

MODEL_PATHS="$(python - "${MODEL_ROOT}" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
paths = [
    sorted(glob.glob(os.path.join(root, "text_encoder", "model*.safetensors"))),
    sorted(glob.glob(os.path.join(root, "transformer", "model*.safetensors"))),
    os.path.join(root, "video_vae", "source", "model.safetensors"),
    os.path.join(root, "audio_vae", "model.safetensors"),
]
if not paths[0] or not paths[1] or any(not os.path.isfile(path) for path in paths[2:]):
    raise FileNotFoundError("MODEL_ROOT does not contain a complete Ref2VA checkpoint")
print(json.dumps(paths))
PY
)"

args=(
  --task sft --dataset_base_path "${DATA_ROOT}" --dataset_metadata_path "${DATA_ROOT}/metadata_train.json"
  --data_file_keys "video,input_audio,references,camera_poses"
  # Four FSDP ranks consume four samples per optimizer step.  Ten repeats
  # yield 4,440 sample presentations, enough for the requested 1,000 steps.
  --height 480 --width 832 --num_frames 124 --dataset_repeat 10 --dataset_num_workers 4
  --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/"
  --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0
  --num_epochs 1 --max_steps "${MAX_STEPS}" --initial_step "${INITIAL_STEP}"
  --gradient_accumulation_steps 1 --use_gradient_checkpointing --find_unused_parameters
  --save_steps "${SAVE_STEPS}" --output_path "${OUTPUT_ROOT}" --enable_tensorboard_log --enable_csv_log
  --enable_profiling --profile_steps "${PROFILE_STEPS}"
)
if [[ -n "${RESUME_ADAPTER}" ]]; then
  args+=(--resume_adapter_path "${RESUME_ADAPTER}")
fi

echo "Output: ${OUTPUT_ROOT}"
echo "Target step: ${MAX_STEPS}; resume step: ${INITIAL_STEP}"
accelerate launch --config_file "${OVERLAY_ROOT}/scripts/accelerate_manual_fsdp_h3_dit_text_4gpu.yaml" --num_processes "${NUM_PROCESSES}" "${OVERLAY_ROOT}/train_scope_ref2va.py" "${args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/train.log"
