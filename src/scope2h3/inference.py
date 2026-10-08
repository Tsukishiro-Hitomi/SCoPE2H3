"""Command-line and library entry point for MiniMax H3 Ref2VA inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


FPS = 24
REQUIRED_REF2VA_PATHS = (
    "Ref2VA/text_encoder/model.safetensors.index.json",
    "Ref2VA/transformer/model.safetensors.index.json",
    "Ref2VA/video_vae/source/model.safetensors",
    "Ref2VA/audio_vae/model.safetensors",
    "Ref2VA/processor",
)


def aligned_frames(seconds: float, fps: int = FPS) -> int:
    """Return the smallest H3-valid (17n+5) frame count for a duration."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = max(1, round(seconds * fps))
    return ((frames - 5 + 16) // 17) * 17 + 5


def validate_ref2va_model_dir(model_dir: Path) -> Path:
    """Resolve and validate a local Ref2VA checkpoint without loading it."""
    root = model_dir.expanduser().resolve()
    missing = [str(root / item) for item in REQUIRED_REF2VA_PATHS if not (root / item).exists()]
    if missing:
        raise FileNotFoundError("incomplete Ref2VA checkpoint:\n  " + "\n  ".join(missing))
    return root


def load_pipeline(model_dir: Path, *, vram_reserve_gb: float = 5.0) -> Any:
    """Load a local BF16 Ref2VA pipeline with CPU offload."""
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax H3 Ref2VA inference requires a CUDA GPU")
    root = validate_ref2va_model_dir(model_dir)
    config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    return MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=sorted(str(p) for p in (root / "Ref2VA/text_encoder").glob("model*.safetensors")), **config),
            ModelConfig(path=sorted(str(p) for p in (root / "Ref2VA/transformer").glob("model*.safetensors")), **config),
            ModelConfig(path=str(root / "Ref2VA/video_vae/source/model.safetensors"), **config),
            ModelConfig(path=str(root / "Ref2VA/audio_vae/model.safetensors"), **config),
        ],
        processor_config=ModelConfig(path=str(root / "Ref2VA/processor")),
        vram_limit=max(1.0, torch.cuda.mem_get_info()[1] / (1024**3) - vram_reserve_gb),
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Ref2VA video with local MiniMax H3 weights.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vram-reserve-gb", type=float, default=5.0)
    return parser


def main() -> None:
    args = parser().parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("height and width must be positive multiples of 32")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    prompt_path, image_path = args.prompt_file.expanduser().resolve(), args.reference_image.expanduser().resolve()
    if not prompt_path.is_file() or not image_path.is_file():
        raise FileNotFoundError("prompt-file and reference-image must be existing files")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("prompt-file is empty")
    from PIL import Image
    from diffsynth.utils.data.audio_video import write_video_audio

    pipe = load_pipeline(args.model_dir, vram_reserve_gb=args.vram_reserve_gb)
    frames = aligned_frames(args.seconds)
    output = (args.output or Path("local/outputs") / f"ref2va_{time.strftime('%Y%m%d_%H%M%S')}.mp4").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    video, audio = pipe(
        prompt=prompt, height=args.height, width=args.width, num_frames=frames,
        num_inference_steps=args.steps, seed=args.seed,
        references=[{"type": "image", "image": Image.open(image_path).convert("RGB")}],
    )
    write_video_audio(video=video, audio=audio, output_path=str(output), fps=FPS, audio_sample_rate=pipe.audio_vae.sample_rate)
    print(output)
