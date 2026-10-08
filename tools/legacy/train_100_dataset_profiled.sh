#!/usr/bin/env bash
# End-to-end launcher for the prepared 100-dataset H3/SCoPE training set.
# Example:
# MODEL_ROOT=/models/MiniMax-H3/Ref2VA bash scripts/train_100_dataset_profiled.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_ROOT="${OVERLAY_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OVERLAY_PARENT="${OVERLAY_PARENT:-$(dirname "${OVERLAY_ROOT}")}" 
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-${OVERLAY_PARENT}/DiffSynth-Studio}"
DATA_ROOT="${DATA_ROOT:-${OVERLAY_ROOT}/data/100_dataset}"
# Override MODEL_ROOT when the checkpoint is mounted elsewhere.
MODEL_ROOT="${MODEL_ROOT:-/inspire/hdd/project/wuliqifa/public/zxr/models/MiniMax-H3-Ref2VA-BF16/Ref2VA}"
RUN_NAME="${RUN_NAME:-scope-h3-100dataset-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OVERLAY_ROOT}/checkpoints/${RUN_NAME}}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
PROFILE_STEPS="${PROFILE_STEPS:-0}"

for required in "${DIFFSYNTH_ROOT}/diffsynth" "${DATA_ROOT}/metadata_train.json" "${DATA_ROOT}/metadata_val.json" "${DATA_ROOT}/camera_poses_24fps" "${MODEL_ROOT}/processor"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done
command -v accelerate >/dev/null || { echo "accelerate is not installed or not on PATH" >&2; exit 2; }
command -v tensorboard >/dev/null || echo "Note: tensorboard CLI not found; logs will still be written."

export PYTHONPATH="${OVERLAY_PARENT}:${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}"

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

echo "Output: ${OUTPUT_ROOT}"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_ROOT}/tensorboard_log --bind_all"
START_SECONDS=${SECONDS}
accelerate launch --config_file "${OVERLAY_ROOT}/scripts/accelerate_manual_fsdp_h3_dit_text_4gpu.yaml" --num_processes "${NUM_PROCESSES}" "${OVERLAY_ROOT}/train_scope_ref2va.py" \
  --task sft --dataset_base_path "${DATA_ROOT}" --dataset_metadata_path "${DATA_ROOT}/metadata_train.json" \
  --data_file_keys "video,input_audio,references,camera_poses" \
  --height 480 --width 832 --num_frames 124 --dataset_repeat "${DATASET_REPEAT}" --dataset_num_workers "${NUM_WORKERS}" \
  --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/" \
  --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0 \
  --num_epochs "${NUM_EPOCHS}" --gradient_accumulation_steps 1 --use_gradient_checkpointing --find_unused_parameters \
  --save_steps "${SAVE_STEPS}" --output_path "${OUTPUT_ROOT}" --enable_tensorboard_log --enable_csv_log \
  --enable_profiling --profile_steps "${PROFILE_STEPS}" 2>&1 | tee "${OUTPUT_ROOT}/train.log"

python "${OVERLAY_ROOT}/scripts/plot_training_metrics.py" "${OUTPUT_ROOT}" --elapsed-seconds "$((SECONDS - START_SECONDS))" | tee "${OUTPUT_ROOT}/metrics_summary.log"
echo "Training complete. Metrics: ${OUTPUT_ROOT}/metrics/summary.json"
