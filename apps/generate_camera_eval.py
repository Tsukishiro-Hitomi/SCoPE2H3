"""Generate a fixed camera-control evaluation manifest with one loaded H3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scope2h3.adapter import set_scope_camera
from scope2h3.inference import load_scope_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vram-reserve-gb", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N manifest records (preflight use)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_frames % 17 != 5:
        raise ValueError("--num-frames must be 17n+5, matching the H3 VAE")
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be multiples of 32")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from diffsynth.utils.data.audio_video import write_video_audio

    pipe = load_scope_pipeline(args.model_dir, args.adapter_path, vram_reserve_gb=args.vram_reserve_gb)
    for position, record in enumerate(records, start=1):
        output = args.output_dir / f"{record['id']}.mp4"
        if output.is_file() and not args.overwrite:
            print(f"[{position}/{len(records)}] already exists: {output.name}")
            continue
        references = record.get("references", [])
        if len(references) != 1 or references[0].get("type") != "image" or "image" not in references[0]:
            raise ValueError(f"{record['id']}: exactly one image reference is required")
        poses = np.load(record["camera_poses"], allow_pickle=False)
        if len(poses) != args.num_frames:
            raise ValueError(f"{record['id']}: pose frames={len(poses)}, expected {args.num_frames}")
        set_scope_camera(pipe, torch.from_numpy(poses).to(torch.float32), float(record["camera_x_fov"]))
        print(f"[{position}/{len(records)}] generating {output.name}")
        video, audio = pipe(
            prompt=record["prompt"], height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.steps, seed=args.seed,
            references=[{"type": "image", "image": Image.open(references[0]["image"]).convert("RGB")}],
        )
        temporary = output.with_name(f".{output.stem}.partial.mp4")
        write_video_audio(video=video, audio=audio, output_path=str(temporary), fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
        temporary.replace(output)


if __name__ == "__main__":
    main()
