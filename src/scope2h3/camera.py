"""Camera-to-Pluecker conversion for H3 target video tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batch(poses: torch.Tensor) -> torch.Tensor:
    # Pose inversion and ray construction are geometry operations; keep them
    # float32 even while H3 itself trains in bf16.
    poses = poses.to(torch.float32)
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)
    if poses.ndim != 4 or poses.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("camera_poses must be [F,3|4,4] or [B,F,3|4,4]")
    if poses.shape[-2:] == (3, 4):
        bottom = torch.zeros((*poses.shape[:-2], 1, 4), dtype=poses.dtype, device=poses.device)
        bottom[..., 0, 3] = 1
        poses = torch.cat((poses, bottom), dim=-2)
    return poses


def _relative_to_first(poses: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(poses[:, :1]) @ poses


def h3_vae_latent_frame_indices(
    num_frames: int,
    target_t: int,
    *,
    clip_length: int = 17,
    temporal_ratio: int = 4,
    token_drop: int = 3,
) -> torch.Tensor:
    """Return the causal source-frame anchor of every released-H3 VAE time token.

    `encode_temporal` pads to 17-frame clips, encodes each clip to five tokens
    (the two temporal stride-2 layers give a ratio of four), concatenates all
    clips, then drops the final three tokens. A surviving token `j` of clip
    `c` is causally anchored at frame `17*c + 4*j`.
    """
    if num_frames <= 0 or target_t <= 0:
        raise ValueError("num_frames and target_t must be positive")
    if (clip_length, temporal_ratio, token_drop) != (17, 4, 3):
        raise ValueError("this adapter supports the released H3 VAE only: clip=17, ratio=4, drop=3")
    tokens_per_clip = (clip_length + temporal_ratio - 1) // temporal_ratio
    expected_t = ((num_frames + clip_length - 1) // clip_length) * tokens_per_clip - token_drop
    if target_t != expected_t:
        raise ValueError(
            f"pose length {num_frames} implies {expected_t} H3 VAE tokens, but target has {target_t}; "
            "poses must describe the frame-rate-aligned target video"
        )
    token = torch.arange(target_t, dtype=torch.long)
    return ((token // tokens_per_clip) * clip_length + (token % tokens_per_clip) * temporal_ratio).clamp_max(num_frames - 1)


def _resample_like_h3_vae(poses: torch.Tensor, target_t: int) -> torch.Tensor:
    return poses.index_select(1, h3_vae_latent_frame_indices(poses.shape[1], target_t).to(poses.device))


def build_target_plucker(
    camera_poses: torch.Tensor,
    x_fov: float | torch.Tensor,
    *,
    target_t: int,
    token_h: int,
    token_w: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return `[B, target_t * token_h * token_w, 6]` in patchify-video order.

    Temporal coordinates follow the released `MiniMaxH3VideoVAE.encode_temporal`
    rule exactly. Geometry is computed at target token centres.
    """
    poses = _relative_to_first(_as_batch(camera_poses))
    poses = _resample_like_h3_vae(poses, target_t)
    batch, frames = poses.shape[:2]
    fov = torch.as_tensor(x_fov, dtype=poses.dtype, device=poses.device).reshape(-1)
    if fov.numel() == 1:
        fov = fov.expand(batch)
    if fov.numel() != batch:
        raise ValueError("camera_x_fov must be scalar or have one value per batch item")
    if not torch.isfinite(poses).all() or not torch.isfinite(fov).all() or (fov <= 0).any() or (fov >= torch.pi).any():
        raise ValueError("camera poses/fov must be finite and fov must lie in (0, pi)")

    patch_w, patch_h = width / token_w, height / token_h
    u = torch.linspace(0.5 * patch_w, width - 0.5 * patch_w, token_w, device=poses.device, dtype=poses.dtype)
    v = torch.linspace(0.5 * patch_h, height - 0.5 * patch_h, token_h, device=poses.device, dtype=poses.dtype)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    focal = (width * 0.5) / torch.tan(fov * 0.5)
    ray_cam = torch.stack(((grid_u[None, None] - width * 0.5) / focal[:, None, None, None], (grid_v[None, None] - height * 0.5) / focal[:, None, None, None], torch.ones((batch, 1, token_h, token_w), device=poses.device, dtype=poses.dtype)), dim=-1)
    ray_cam = F.normalize(ray_cam.expand(batch, frames, -1, -1, -1), dim=-1, eps=1e-8)
    direction = torch.einsum("btij,bthwj->bthwi", poses[..., :3, :3], ray_cam)
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    origin = poses[..., :3, 3][..., None, None, :].expand_as(direction)
    moment = torch.cross(origin, direction, dim=-1)
    return torch.cat((direction, moment), dim=-1).reshape(batch, frames * token_h * token_w, 6)

