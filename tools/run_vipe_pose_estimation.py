#!/usr/bin/env python3
"""Run one pinned ViPE executable over videos listed in a camera-eval manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True, help="contains <id>.mp4; ignored with --ground-truth")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vipe", default="vipe", help="ViPE CLI from its separately pinned environment")
    parser.add_argument("--pipeline", default="pose_only_long", help="ViPE pipeline name")
    parser.add_argument("--ground-truth", action="store_true", help="estimate manifest video rather than generated <id>.mp4")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    executable = shutil.which(args.vipe)
    if executable is None:
        raise FileNotFoundError(f"ViPE executable not found: {args.vipe}; install/pin nvidia-vipe in its own environment")
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        records = records[:args.limit]
    pending = []
    for record in records:
        video = Path(record["video"]) if args.ground_truth else args.video_root / f"{record['id']}.mp4"
        pose = args.output_root / "pose" / f"{video.stem}.npz"
        if pose.is_file() and not args.force:
            continue
        if not video.is_file():
            raise FileNotFoundError(f"{record['id']}: video not found: {video}")
        pending.append((video.resolve(), pose))
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inputs-", dir=args.output_root) as directory:
        for video, _ in pending:
            link = Path(directory) / video.name
            if link.exists():
                raise ValueError(f"duplicate video basename: {video.name}")
            link.symlink_to(video)
        if pending:
            subprocess.run([executable, "infer", directory, "--output", str(args.output_root.resolve()), "--pipeline", args.pipeline], check=True)
    for video, pose in pending:
        if not pose.is_file():
            raise RuntimeError(f"{video}: ViPE completed without expected pose artifact {pose}")


if __name__ == "__main__":
    main()
