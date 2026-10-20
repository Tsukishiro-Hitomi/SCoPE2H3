"""Raw-scale camera-control metrics used by the SCoPE paper.

The paper estimates poses for *both* reference and generated videos with ViPE,
anchors each estimate at frame zero, and compares them without trajectory
rescaling.  This module implements only that deterministic comparison; it does
not confuse the commanded input poses with the poses inferred from a video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_NAMES = ("RotErr", "TransErr", "CamMC", "ATE")


def _homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"poses must be [T,3,4] or [T,4,4], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("poses contain non-finite values")
    if poses.shape[1:] == (4, 4):
        return poses.copy()
    bottom = np.zeros((len(poses), 1, 4), dtype=poses.dtype)
    bottom[..., 0, 3] = 1
    return np.concatenate((poses, bottom), axis=1)


def _anchor_to_first(poses: np.ndarray) -> np.ndarray:
    poses = _homogeneous(poses)
    return np.linalg.inv(poses[:1]) @ poses


def load_vipe_pose_archive(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ViPE's ``pose/<video>.npz`` artifact as ``(frame_indices, c2w)``.

    ViPE stores OpenCV camera-to-world matrices under ``data`` and the source
    frame numbers under ``inds``.  Retaining indices is essential: failed or
    skipped frames must not be accidentally paired by array position.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ViPE pose archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive or "inds" not in archive:
            raise ValueError(f"{path}: expected ViPE keys 'data' and 'inds', found {archive.files}")
        indices = np.asarray(archive["inds"], dtype=np.int64)
        poses = _homogeneous(archive["data"])
    if len(indices) != len(poses) or len(indices) == 0:
        raise ValueError(f"{path}: invalid ViPE pose/index lengths: {len(poses)}, {len(indices)}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{path}: ViPE frame indices are not unique")
    return indices, poses


def align_vipe_poses(
    target_indices: np.ndarray,
    target_c2w: np.ndarray,
    predicted_indices: np.ndarray,
    predicted_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair two ViPE trajectories by original frame number and anchor them."""
    target_by_index = {int(index): pose for index, pose in zip(target_indices, target_c2w)}
    predicted_by_index = {int(index): pose for index, pose in zip(predicted_indices, predicted_c2w)}
    common = np.array(sorted(set(target_by_index).intersection(predicted_by_index)), dtype=np.int64)
    if len(common) < 2:
        raise ValueError("fewer than two common ViPE frames; cannot evaluate a trajectory")
    if common[0] != 0:
        raise ValueError(f"frame 0 is absent from paired ViPE estimates (first common frame is {common[0]})")
    target = np.stack([target_by_index[int(index)] for index in common])
    predicted = np.stack([predicted_by_index[int(index)] for index in common])
    return common, _anchor_to_first(target), _anchor_to_first(predicted)


def evaluate_camera_poses(target_c2w: np.ndarray, predicted_c2w: np.ndarray) -> dict[str, float]:
    """Compute paper-compatible raw-scale pose metrics for aligned c2w paths.

    ``RotErr`` is the mean SO(3) geodesic angle in radians. ``TransErr`` is
    the mean per-frame translation L2 error, ``CamMC`` is the mean Frobenius
    error of the 3x4 camera matrices, and ``ATE`` is the RMS translation error.
    No Sim(3), Procrustes, or max-norm alignment is applied.
    """
    target = _anchor_to_first(target_c2w)
    predicted = _anchor_to_first(predicted_c2w)
    if target.shape != predicted.shape:
        raise ValueError(f"trajectory shapes differ: target={target.shape}, predicted={predicted.shape}")
    relative_rotation = np.swapaxes(target[:, :3, :3], 1, 2) @ predicted[:, :3, :3]
    trace = np.trace(relative_rotation, axis1=1, axis2=2)
    angles = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    translation_delta = predicted[:, :3, 3] - target[:, :3, 3]
    return {
        "RotErr": float(angles.mean()),
        "TransErr": float(np.linalg.norm(translation_delta, axis=1).mean()),
        "CamMC": float(np.linalg.norm(predicted[:, :3, :4] - target[:, :3, :4], axis=(1, 2)).mean()),
        "ATE": float(np.sqrt(np.square(translation_delta).sum(axis=1).mean())),
    }


def evaluate_vipe_archives(target_path: str | Path, predicted_path: str | Path) -> dict[str, Any]:
    """Compare ViPE artifacts and retain pairing provenance in the result."""
    target_indices, target = load_vipe_pose_archive(target_path)
    predicted_indices, predicted = load_vipe_pose_archive(predicted_path)
    common, target, predicted = align_vipe_poses(target_indices, target, predicted_indices, predicted)
    return {
        **evaluate_camera_poses(target, predicted),
        "paired_frames": int(len(common)),
        "frame_indices": common.tolist(),
        "target_pose_path": str(Path(target_path).resolve()),
        "predicted_pose_path": str(Path(predicted_path).resolve()),
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average per-video metrics; no frame-rich clips can dominate."""
    if not records:
        raise ValueError("cannot aggregate an empty evaluation")
    missing = [name for name in METRIC_NAMES if any(name not in record for record in records)]
    if missing:
        raise ValueError(f"records do not all contain metrics: {missing}")
    return {
        "videos": len(records),
        "mean": {name: float(np.mean([record[name] for record in records])) for name in METRIC_NAMES},
        "median": {name: float(np.median([record[name] for record in records])) for name in METRIC_NAMES},
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
