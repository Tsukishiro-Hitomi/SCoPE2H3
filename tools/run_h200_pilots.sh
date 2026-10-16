#!/usr/bin/env bash
# Restart-safe H200 pilot: performance preflight plus two 250-step comparisons.
set -euo pipefail

ROOT=/inspire/hdd/project/wuliqifa/public/zxr
PROJECT_ROOT="${ROOT}/scope2h3"
DATA_ROOT="${ROOT}/scope_h3_ref2va/data"
MODEL_ROOT="${ROOT}/models/MiniMax-H3-Ref2VA-BF16/Ref2VA"
RUN_ROOT="${PROJECT_ROOT}/local/runs/h200-pilots-20261009"
CONDA_ROOT=/inspire/hdd/project/wuliqifa/public/miniconda3

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate h3
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/torchinductor-cache"
mkdir -p "${RUN_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${PROJECT_ROOT}"

MODEL_PATHS="$(python - "${MODEL_ROOT}" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
paths = [
    sorted(glob.glob(os.path.join(root, "text_encoder", "model*.safetensors"))),
    sorted(glob.glob(os.path.join(root, "transformer", "model*.safetensors"))),
    os.path.join(root, "video_vae", "source", "model.safetensors"),
    os.path.join(root, "audio_vae", "model.safetensors"),
]
if not paths[0] or not paths[1] or any(not os.path.isfile(path) for path in paths[2:]):
    raise SystemExit("incomplete Ref2VA checkpoint")
print(json.dumps(paths))
PY
)"

# The balanced manifest has the same number of records from each source and
# fixed sampling, so a platform restart cannot silently change the experiment.
# Its base path is DATA_ROOT, whereas the source metadata files are rooted at
# their own dataset directories; make every asset path relative to DATA_ROOT.
BALANCED_METADATA="${RUN_ROOT}/metadata/train_spatialvid_waymo_balanced.json"
mkdir -p "$(dirname "${BALANCED_METADATA}")"
python - "${DATA_ROOT}" "${BALANCED_METADATA}" <<'PY'
import copy
import json
import random
import sys
from pathlib import Path

data_root = Path(sys.argv[1])
output = Path(sys.argv[2])
spatial = json.loads((data_root / "1324-spatialvid/train_metadata.json").read_text())
waymo = json.loads((data_root / "2800-waymo/train_metadata.json").read_text())

def rebase(record, source_dir):
    record = copy.deepcopy(record)
    for key in ("video", "input_audio", "camera_poses"):
        if key in record:
            record[key] = f"{source_dir}/{record[key]}"
    for reference in record.get("references", []):
        for key in ("image", "video"):
            if key in reference:
                reference[key] = f"{source_dir}/{reference[key]}"
    return record

spatial = [rebase(record, "1324-spatialvid") for record in spatial]
waymo = [rebase(record, "2800-waymo") for record in waymo]
rng = random.Random(20261009)
waymo = rng.sample(waymo, len(spatial))
records = spatial + waymo
rng.shuffle(records)
output.write_text(json.dumps(records, ensure_ascii=False) + "\n")
print(f"wrote {len(records)} balanced records to {output}")
PY

latest_step() {
  local stage=$1
  find "${stage}" -maxdepth 1 -type f -name 'step-*.safetensors' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^step-([0-9]+)\.safetensors$/\1/' | sort -n | tail -n 1
}

run_stage() {
  local name=$1
  local dataset_base=$2
  local metadata=$3
  local target_steps=$4
  local profiling=$5
  local stage="${RUN_ROOT}/${name}"
  local completed resume_args=()
  mkdir -p "${stage}"
  completed="$(latest_step "${stage}")"
  if [[ -n "${completed}" && "${completed}" -ge "${target_steps}" ]]; then
    echo "${name}: already has step ${completed}; skipping training."
    return
  fi
  if [[ -n "${completed}" ]]; then
    resume_args=(--resume_adapter_path "${stage}/step-${completed}.safetensors" --initial_step "${completed}")
    echo "${name}: resuming from step ${completed}."
  fi
  profile_args=()
  if [[ "${profiling}" == true ]]; then
    profile_args=(--enable_profiling --profile_steps 25)
  fi
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 apps/train_scope_adapter.py \
    --task sft \
    --dataset_base_path "${dataset_base}" \
    --dataset_metadata_path "${metadata}" \
    --data_file_keys video,input_audio,references,camera_poses \
    --height 480 --width 832 --num_frames 124 --dataset_repeat 1 --dataset_num_workers 4 \
    --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/" \
    --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0 \
    --num_epochs 1 --max_steps "${target_steps}" --gradient_accumulation_steps 1 \
    --use_gradient_checkpointing --find_unused_parameters --save_steps 25 \
    --output_path "${stage}" --enable_tensorboard_log --enable_csv_log \
    "${resume_args[@]}" "${profile_args[@]}" 2>&1 | tee -a "${stage}/train.log"
  test -s "${stage}/step-${target_steps}.safetensors"
}

evaluate() {
  local name=$1
  local split_name=$2
  local dataset_base=$3
  local metadata=$4
  local checkpoint=$5
  local output="${RUN_ROOT}/${name}/eval-${split_name}.json"
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 apps/train_scope_adapter.py \
    --task sft \
    --dataset_base_path "${dataset_base}" \
    --dataset_metadata_path "${metadata}" \
    --data_file_keys video,input_audio,references,camera_poses \
    --height 480 --width 832 --num_frames 124 --dataset_repeat 1 --dataset_num_workers 4 \
    --model_paths "${MODEL_PATHS}" --processor_path "${MODEL_ROOT}/processor/" \
    --scale_gate_hidden 64 --learning_rate 2e-5 --weight_decay 1e-2 --audio_loss_weight 0.0 \
    --gradient_accumulation_steps 1 --use_gradient_checkpointing --find_unused_parameters \
    --output_path "${RUN_ROOT}/${name}" --resume_adapter_path "${checkpoint}" --eval_only \
    --eval_output_path "${output}" 2>&1 | tee -a "${RUN_ROOT}/${name}/validation.log"
  test -s "${output}"
}

if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
  run_stage preflight-spatialvid "${DATA_ROOT}/1324-spatialvid" "${DATA_ROOT}/1324-spatialvid/train_metadata.json" 25 true
fi
run_stage pilot-spatialvid "${DATA_ROOT}/1324-spatialvid" "${DATA_ROOT}/1324-spatialvid/train_metadata.json" 250 false
run_stage pilot-balanced "${DATA_ROOT}" "${BALANCED_METADATA}" 250 false

for stage in pilot-spatialvid pilot-balanced; do
  checkpoint="${RUN_ROOT}/${stage}/step-250.safetensors"
  evaluate "${stage}" spatialvid "${DATA_ROOT}/1324-spatialvid" "${DATA_ROOT}/1324-spatialvid/val_metadata.json" "${checkpoint}"
  evaluate "${stage}" waymo "${DATA_ROOT}/2800-waymo" "${DATA_ROOT}/2800-waymo/val_metadata.json" "${checkpoint}"
done

python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for stage in ("pilot-spatialvid", "pilot-balanced"):
    summary[stage] = {}
    for path in sorted((root / stage).glob("eval-*.json")):
        summary[stage][path.name] = json.loads(path.read_text())
(root / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
