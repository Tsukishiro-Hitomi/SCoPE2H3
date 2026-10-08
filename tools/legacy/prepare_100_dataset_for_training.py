"""Validate and combine the 100-dataset packages into H3 training manifests.

Run from the repository root:

    python scope_h3_ref2va/scripts/prepare_100_dataset_for_training.py

The generated paths are relative to ``data/100_dataset``.  Use that directory
as both ``--dataset_base_path`` and the parent of the selected metadata file.
No source image, video, or pose asset is copied or modified.  It writes
small, derived 24-FPS pose arrays under ``camera_poses_24fps/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np


REQUIRED_KEYS = {
    "video", "input_audio", "references", "camera_poses", "camera_x_fov", "prompt",
}
SOURCES = (("spatialvid", "s"), ("waymo", "w"))
TARGET_FPS, TARGET_FRAMES, TIME_FACTOR, TIME_REMAINDER = 24, 124, 17, 5


def video_stream_info(path: Path) -> tuple[float, int, float]:
    """Read the same frame-rate inputs used by DiffSynth's video loader."""
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
    fps = numerator / denominator
    frame_count = int(stream["nb_frames"])
    duration = float(stream["duration"])
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: invalid video stream metadata")
    return fps, frame_count, duration


def training_frame_count(duration: float) -> int:
    """Mirror ``FrameSamplerByRateMixin.get_num_frames`` exactly."""
    count = min(TARGET_FRAMES, math.floor(duration * TARGET_FPS))
    while count > 1 and count % TIME_FACTOR != TIME_REMAINDER:
        count -= 1
    if count <= 0:
        raise ValueError("video is too short for an H3-compatible frame count")
    return count


def read_entries(root: Path, source: str, prefix: str) -> tuple[list[dict], list[dict]]:
    package = root / source
    entries = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) != 100:
        raise ValueError(f"{source}: expected exactly 100 metadata records")

    records: list[dict] = []
    stats: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        if set(entry) != REQUIRED_KEYS:
            raise ValueError(f"{source}[{index}]: expected keys {sorted(REQUIRED_KEYS)}, got {sorted(entry)}")
        if not isinstance(entry["references"], list) or len(entry["references"]) != 1:
            raise ValueError(f"{source}[{index}]: expected exactly one image reference")
        reference = entry["references"][0]
        if set(reference) != {"type", "image"} or reference["type"] != "image":
            raise ValueError(f"{source}[{index}]: invalid reference block")
        if not isinstance(entry["prompt"], str) or not entry["prompt"].strip():
            raise ValueError(f"{source}[{index}]: prompt is empty")
        fov = float(entry["camera_x_fov"])
        if not math.isfinite(fov) or not 0.0 < fov < math.pi:
            raise ValueError(f"{source}[{index}]: invalid camera_x_fov {fov!r}")

        relative_paths = (entry["video"], entry["input_audio"], reference["image"], entry["camera_poses"])
        for value in relative_paths:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or not (package / path).is_file():
                raise FileNotFoundError(f"{source}[{index}]: invalid or missing asset {value!r}")

        source_pose_path = package / entry["camera_poses"]
        poses = np.load(source_pose_path, allow_pickle=False)
        if poses.dtype != np.float32 or poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"{source}[{index}]: expected float32 poses [F,4,4], got {poses.dtype} {poses.shape}")
        if not np.isfinite(poses).all():
            raise ValueError(f"{source}[{index}]: poses contain non-finite values")
        if not np.allclose(poses[0], np.eye(4, dtype=np.float32), atol=1e-4):
            raise ValueError(f"{source}[{index}]: first pose is not identity")

        video_path = package / entry["video"]
        raw_fps, raw_frames, duration = video_stream_info(video_path)
        if raw_frames != poses.shape[0]:
            raise ValueError(
                f"{source}[{index}]: video has {raw_frames} frames but pose array has {poses.shape[0]} frames"
            )
        aligned_frames = training_frame_count(duration)
        aligned_indices = np.rint(np.arange(aligned_frames) * raw_fps / TARGET_FPS).astype(np.int64)
        aligned_indices = np.minimum(aligned_indices, raw_frames - 1)
        aligned_poses = poses[aligned_indices]
        aligned_path = root / "camera_poses_24fps" / source / f"{prefix}_{index}.npy"
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(aligned_path, aligned_poses.astype(np.float32, copy=False))

        # Preserve the portable six-field H3 schema, changing only the base
        # directory of file-valued fields for the combined dataset root.
        records.append({
            "video": f"{source}/{entry['video']}",
            "input_audio": f"{source}/{entry['input_audio']}",
            "references": [{"type": "image", "image": f"{source}/{reference['image']}"}],
            "camera_poses": str(aligned_path.relative_to(root)).replace("\\", "/"),
            "camera_x_fov": fov,
            "prompt": entry["prompt"].strip(),
        })
        stats.append({
            "id": f"{prefix}_{index}", "source": source, "raw_frames": raw_frames,
            "training_frames": aligned_frames, "raw_fps": raw_fps, "fov_rad": fov,
        })
    return records, stats


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "100_dataset"
    all_records: list[dict] = []
    all_stats: list[dict] = []
    for source, prefix in SOURCES:
        records, stats = read_entries(root, source, prefix)
        all_records.extend(records)
        all_stats.extend(stats)

    # A deterministic 90/10 split stratified by source; every tenth numbered
    # sample is held out.  Keep the combined manifest for full-data training.
    validation_ids = {f"{prefix}_{number}" for _, prefix in SOURCES for number in range(10, 101, 10)}
    train_records = [record for record, stat in zip(all_records, all_stats) if stat["id"] not in validation_ids]
    val_records = [record for record, stat in zip(all_records, all_stats) if stat["id"] in validation_ids]

    write_json(root / "metadata.json", all_records)
    write_json(root / "metadata_train.json", train_records)
    write_json(root / "metadata_val.json", val_records)
    write_json(root / "training_data_report.json", {
        "status": "passed",
        "schema": "MiniMax H3 Ref2VA six-field metadata schema",
        "dataset_base_path": ".",
        "records": len(all_records),
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "split_rule": "per source: samples whose numeric ID is divisible by 10 are validation",
        "sources": {
            source: {
                "records": sum(stat["source"] == source for stat in all_stats),
                "source_frame_count_min": min(stat["raw_frames"] for stat in all_stats if stat["source"] == source),
                "source_frame_count_max": max(stat["raw_frames"] for stat in all_stats if stat["source"] == source),
                "training_frame_count_min": min(stat["training_frames"] for stat in all_stats if stat["source"] == source),
                "training_frame_count_max": max(stat["training_frames"] for stat in all_stats if stat["source"] == source),
            }
            for source, _ in SOURCES
        },
        "checks": [
            "required metadata schema", "relative asset paths", "non-empty prompts",
            "finite FOV in (0, pi)", "finite float32 OpenCV c2w poses [F,4,4]", "identity first pose",
            "video/pose source frame-count match", "24 FPS pose sampling matched to DiffSynth frame selection",
        ],
    })
    print(f"Validated {len(all_records)} records; wrote metadata.json, metadata_train.json, metadata_val.json")


if __name__ == "__main__":
    main()
