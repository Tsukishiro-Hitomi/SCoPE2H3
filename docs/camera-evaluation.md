# Camera-control evaluation

This is the four pose metrics protocol used for SCoPE-to-H3 checkpoints:

1. Generate from the held-out video's reference image, prompt, and commanded
   trajectory.
2. Run the **same pinned ViPE version** on the ground-truth and generated
   videos.  Do not compare generated ViPE poses directly to the input poses:
   the published protocol estimates both trajectories from video.
3. Pair the ViPE pose artifacts by their stored source-frame indices, anchor
   both paths to frame zero, and report raw-scale metrics.  No per-clip
   normalisation, Sim(3), or Procrustes alignment is applied.

`RotErr` is mean geodesic SO(3) error in **radians**; `TransErr` is mean
translation L2 error; `CamMC` is mean 3x4 camera-matrix Frobenius error; and
`ATE` is RMS translation error. Results are macro-averaged over videos.

## Weights and versions

The H3 base model is loaded only from `--model-dir`. The adapter is a separate
`safetensors` file passed by `--adapter-path`; loading validates that its exact
set of `scope_pe` keys matches the installed H3 overlay before inference.

ViPE is intentionally a separate environment: its dependency stack contains
CUDA extensions and several auxiliary models. Pin the ViPE checkout or package
version used for the run, record `vipe --version`/git commit in the run folder,
and let ViPE manage its own first-use model cache. The pose-only pipeline is
recommended; current ViPE writes OpenCV camera-to-world matrices to
`pose/<video>.npz` as `data`, with original frame numbers in `inds`.

## Step-3000, 200-video evaluation

```bash
source scripts/activate.sh
export PYTHONPATH="$PWD/src"
RUN=local/evaluations/step-3000-pose-200
CKPT=local/runs/scope2h3-formal-10k-h100-20261013/step-3000.safetensors

python tools/build_camera_eval_manifest.py \
  --data-root ../scope_h3_ref2va/data --output "$RUN/manifest.jsonl" --per-source 100

python apps/generate_camera_eval.py \
  --manifest "$RUN/manifest.jsonl" --model-dir ../models/MiniMax-H3-Ref2VA-BF16 \
  --adapter-path "$CKPT" --output-dir "$RUN/generated" --num-frames 124 --steps 50 --seed 42
```

Activate the separately installed, pinned ViPE environment before each of the
next commands (the `vipe` executable must be on `PATH`):

```bash
python tools/run_vipe_pose_estimation.py --manifest "$RUN/manifest.jsonl" \
  --video-root "$RUN/generated" --output-root "$RUN/vipe-generated"
python tools/run_vipe_pose_estimation.py --manifest "$RUN/manifest.jsonl" \
  --video-root "$RUN/generated" --ground-truth --output-root "$RUN/vipe-gt"
```

Return to the H3 environment for scoring:

```bash
python tools/build_vipe_metric_pairs.py --manifest "$RUN/manifest.jsonl" \
  --gt-vipe-root "$RUN/vipe-gt" --generated-vipe-root "$RUN/vipe-generated" \
  --output "$RUN/vipe_pairs.jsonl"
python apps/evaluate_camera_control.py --pairs "$RUN/vipe_pairs.jsonl" --output "$RUN/metrics.json"
```
