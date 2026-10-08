import unittest

import torch
from torch import nn

from scope2h3.adapter import install_scope_h3, set_scope_camera


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim = 2, 4
        self.qkv_proj = nn.Linear(8, 24, bias=False)
        self.q_norm, self.k_norm, self.out_proj = nn.Identity(), nn.Identity(), nn.Linear(8, 8, bias=False)
        self.softmax_scale = 1.0


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


class _Pipeline:
    def __init__(self):
        self.dit = _DiT()
        self.model_fn = lambda **inputs: (inputs["video_latents"], torch.zeros(1))


class AdapterTests(unittest.TestCase):
    def test_install_registers_only_scope_modules_and_camera_state(self):
        pipe = install_scope_h3(_Pipeline(), scale_gate_hidden=3)
        self.assertTrue(pipe._scope_h3_installed)
        for block in pipe.dit.blocks:
            self.assertIn("scope_pe", block.attn._modules)
            self.assertIsNone(block.attn._scope_target_positions)
        poses = torch.eye(4).repeat(5, 1, 1)[:, :3]
        set_scope_camera(pipe, poses, 1.0)
        self.assertEqual(tuple(pipe._scope_h3_camera[0].shape), (5, 3, 4))

    def test_duplicate_install_is_rejected(self):
        pipe = install_scope_h3(_Pipeline())
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            install_scope_h3(pipe)
