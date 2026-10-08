# scope2h3

`scope2h3` is a single, installable repository for MiniMax H3 Ref2VA inference
and SCoPE camera-conditioned adapter training.

* `src/diffsynth/` is a controlled DiffSynth runtime copy. Keep changes there
  minimal and document upstream synchronization in `docs/upstream-sync.md`.
* `src/scope2h3/` is the project-owned layer: camera geometry, SCoPE adapter,
  training orchestration, checkpoint validation, and command-line APIs.
* Models, datasets, outputs, and adapter checkpoints belong under `local/` and
  are intentionally excluded from Git.

## Install

Use the existing `h3` Conda environment:

```bash
source scripts/activate.sh
python -m pip install -e '.[audio,training,test]'
```

Or create a fresh, minimal environment first:

```bash
conda env create -f environment.yml
conda activate scope2h3
python -m pip install -e '.[audio,training,test]'
```

`DIFFSYNTH_SKIP_DOWNLOAD=True` is deliberately set by `scripts/activate.sh`:
all checkpoint paths must be supplied locally.

## Verify

```bash
python -m pytest
python -m unittest discover -s tests -t .
```

## Inference

```bash
scope2h3-infer --model-dir local/models/MiniMax-H3-Ref2VA-BF16 \
  --prompt-file local/inputs/prompt.txt --reference-image local/inputs/ref.jpg
```

See [architecture](docs/architecture.md) for ownership boundaries and
[upstream synchronization](docs/upstream-sync.md) before modifying the runtime.
