import unittest

import torch

from scope2h3.camera import build_target_plucker, h3_vae_latent_frame_indices


class CameraTests(unittest.TestCase):
    def test_h3_vae_temporal_anchors_match_chunks_and_final_token_drop(self):
        anchors = h3_vae_latent_frame_indices(num_frames=124, target_t=37)
        self.assertEqual(anchors.tolist(), [
            0, 4, 8, 12, 16, 17, 21, 25, 29, 33, 34, 38, 42, 46, 50,
            51, 55, 59, 63, 67, 68, 72, 76, 80, 84, 85, 89, 93, 97, 101,
            102, 106, 110, 114, 118, 119, 123,
        ])

    def test_plucker_is_relative_to_the_first_camera(self):
        poses = torch.eye(4).repeat(22, 1, 1)[:, :3]
        poses[:, 0, 3] = torch.linspace(2, 5, 22)
        plucker = build_target_plucker(poses, 1.0, target_t=7, token_h=2, token_w=4, height=64, width=128)
        self.assertEqual(tuple(plucker.shape), (1, 56, 6))
        torch.testing.assert_close(plucker[0, :8, 3:], torch.zeros_like(plucker[0, :8, 3:]), atol=1e-6, rtol=0)
