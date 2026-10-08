#!/usr/bin/env bash
# Distributed validation loss for the 124_dataset step-250 adapter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_ROOT="${OVERLAY_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OVERLAY_PARENT="${OVERLAY_PARENT:-$(dirname "${OVERLAY_ROOT}")}" 
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-${OVERLAY_PARENT}/DiffSynth-Studio}"
DATA_ROOT="${DATA_ROOT:-${OVERLAY_ROOT}/data/124_dataset}"
MODEL_ROOT="${MODEL_ROOT:-${OVERLAY_PARENT}/models/MiniMax-H3-Ref2VA-BF16/Ref2VA}"
RUN_ROOT="${RUN_ROOT:-${OVERLAY_ROOT}/checkpoints/scope-h3-124dataset-1000step-v2}"
ADAPTER_PATH="${ADAPTER_PATH:-${RUN_ROOT}/step-250.safetensors}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/project/wuliqifa/public/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate h3
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PYTHONPATH="${OVERLAY_PARENT}:${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for required in "${DATA_ROOT}/metadata_val.json" "${MODEL_ROOT}/processor" "${ADAPTER_PATH}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

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

accelerate launch --config_file "${OVERLAY_ROOT}/scripts/accelerate_manual_fsdp_h3_dit_text_4gpu.yaml" --num_processes 4 "${OVERLAY_ROOT}/train_scope_ref2va.py" \
  --task sft --dataset_base_path "${DATA_ROOT}" --dataset_metadata_path "${DATA_ROOT}/metadata_val.json" \
  --data_file_keys "video,input_audio,references,camera_poses" \
  --height 480 --width 832 --num_frames 124 --dataset_repeat 1 --dataset_num_workers 0 \
  --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/" \
  --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0 \
  --num_epochs 1 --gradient_accumulation_steps 1 --use_gradient_checkpointing --find_unused_parameters \
  --output_path "${RUN_ROOT}" --resume_adapter_path "${ADAPTER_PATH}" --eval_only \
  --eval_output_path "${RUN_ROOT}/validation_loss.json" 2>&1 | tee "${RUN_ROOT}/validation.log"
