import tempfile
import unittest
from pathlib import Path

import numpy as np

from scope2h3.camera_evaluation import (
    aggregate_metrics,
    evaluate_camera_poses,
    evaluate_vipe_archives,
)


def _trajectory() -> np.ndarray:
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, 0, 3] = [3.0, 4.0, 5.0]
    return poses


class CameraEvaluationTests(unittest.TestCase):
    def test_global_coordinate_frame_is_removed_but_scale_is_retained(self):
        target = _trajectory()
        predicted = target.copy()
        # A global world-frame translation changes neither trajectory after
        # frame-zero anchoring.
        predicted[:, 1, 3] += 7.0
        metrics = evaluate_camera_poses(target, predicted)
        self.assertEqual(metrics, {"RotErr": 0.0, "TransErr": 0.0, "CamMC": 0.0, "ATE": 0.0})

        predicted = target.copy()
        predicted[1:, 0, 3] *= 2.0
        scaled = evaluate_camera_poses(target, predicted)
        self.assertGreater(scaled["TransErr"], 0.0)
        self.assertGreater(scaled["ATE"], 0.0)

    def test_vipe_archives_are_paired_by_frame_indices(self):
        poses = _trajectory()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target, predicted = root / "target.npz", root / "predicted.npz"
            np.savez(target, data=poses, inds=np.array([0, 1, 2]))
            np.savez(predicted, data=poses[[2, 0, 1]], inds=np.array([2, 0, 1]))
            result = evaluate_vipe_archives(target, predicted)
        self.assertEqual(result["paired_frames"], 3)
        self.assertEqual(result["frame_indices"], [0, 1, 2])
        self.assertEqual(result["ATE"], 0.0)

    def test_aggregate_is_macro_mean(self):
        result = aggregate_metrics([
            {"RotErr": 1.0, "TransErr": 2.0, "CamMC": 3.0, "ATE": 4.0},
            {"RotErr": 3.0, "TransErr": 4.0, "CamMC": 5.0, "ATE": 6.0},
        ])
        self.assertEqual(result["videos"], 2)
        self.assertEqual(result["mean"]["RotErr"], 2.0)
