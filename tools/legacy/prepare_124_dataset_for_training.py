"""Derive frame-rate-aligned camera poses for 124_dataset training.

DiffSynth samples target videos at 24 FPS.  The source clips retain their
native rates, so their 124-frame raw pose arrays must be sampled with exactly
the same frame-count and index rule before camera-token construction.
"""

from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


TARGET_FPS, MAX_FRAMES, TIME_FACTOR, TIME_REMAINDER = 24, 124, 17, 5


def stream_info(path: Path) -> tuple[float, int, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames,duration", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream")
    stream = streams[0]
    numerator, denominator = (float(value) for value in stream["avg_frame_rate"].split("/"))
    return numerator / denominator, int(stream["nb_frames"]), float(stream["duration"])


def sampled_frame_count(duration: float) -> int:
    count = min(MAX_FRAMES, math.floor(duration * TARGET_FPS))
    while count > 1 and count % TIME_FACTOR != TIME_REMAINDER:
        count -= 1
    if count <= 0:
        raise ValueError(f"duration {duration} is too short for H3")
    return count


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "124_dataset"
    records = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    # ffprobe is I/O-bound; probing clips concurrently keeps preparation short
    # while preserving metadata order below.
    with ThreadPoolExecutor(max_workers=24) as executor:
        stream_infos = list(executor.map(lambda record: stream_info(root / record["video"]), records))
    derived: list[dict] = []
    report: list[dict] = []
    for record, (fps, video_frames, duration) in zip(records, stream_infos):
        sample_id = Path(record["video"]).stem
        source = Path(record["video"]).parts[0]
        poses = np.load(root / record["camera_poses"], allow_pickle=False)
        if poses.dtype != np.float32 or poses.shape != (video_frames, 4, 4):
            raise ValueError(f"{sample_id}: video/pose mismatch: {video_frames} vs {poses.dtype} {poses.shape}")
        frames = sampled_frame_count(duration)
        indices = np.rint(np.arange(frames) * fps / TARGET_FPS).astype(np.int64)
        indices = np.minimum(indices, video_frames - 1)
        destination = root / "camera_poses_24fps" / source / f"{sample_id}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, poses[indices].astype(np.float32, copy=False))
        derived_record = dict(record)
        derived_record["camera_poses"] = str(destination.relative_to(root)).replace("\\", "/")
        derived.append(derived_record)
        report.append({"sample_id": sample_id, "source_fps": fps, "duration_s": duration, "training_frames": frames})

    by_id = {Path(record["video"]).stem: record for record in derived}
    for split in ("train", "val"):
        ids = [entry["sample_id"] for entry in json.loads((root / f"{split}_index.json").read_text(encoding="utf-8"))]
        selected = [by_id[sample_id] for sample_id in ids]
        (root / f"metadata_{split}.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "metadata_24fps.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "training_frame_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(derived)} frame-rate-aligned pose arrays")


if __name__ == "__main__":
    main()
