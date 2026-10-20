"""Runtime adapter for DiffSynth's packed-sequence MiniMax H3 attention.

This module deliberately patches only loaded Python objects.  It does not edit
the DiffSynth source tree or its checkpoint files.
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Callable

import torch

from .camera import build_target_plucker
from .encoding import H3SightlineCoordinatePE


def _scope_attention_forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None):
    """DiffSynth H3 attention forward with an optional target-video-only PE."""
    source = importlib.import_module(self.__class__.__module__)
    total = x.shape[0]
    qkv = self.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim)
    query, key, value = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
    query, key = self.q_norm(query), self.k_norm(key)
    if rope_freqs is not None:
        query, key = source._apply_rope(query, rope_freqs), source._apply_rope(key, rope_freqs)

    positions = getattr(self, "_scope_target_positions", None)
    plucker = getattr(self, "_scope_target_plucker", None)
    if positions is not None or plucker is not None:
        if positions is None or plucker is None:
            raise RuntimeError("incomplete SCoPE condition on H3 attention")
        positions = positions.to(query.device, dtype=torch.long)
        if positions.numel() != plucker.shape[0]:
            raise RuntimeError("SCoPE target positions and Pluecker rows differ")
        selected_q = query.index_select(0, positions)
        selected_k = key.index_select(0, positions)
        # DiffSynth's inference offloader knows the original H3 modules but
        # cannot know about this runtime-added adapter.  Move this small
        # (~58 MiB total) adapter on first use with its owning attention
        # block; without this explicit handoff its weights remain on CPU while
        # the packed Q/K rows are on CUDA.
        scope_parameter = next(self.scope_pe.parameters())
        if scope_parameter.device != query.device or scope_parameter.dtype != query.dtype:
            self.scope_pe.to(device=query.device, dtype=query.dtype)
        selected_q, selected_k = self.scope_pe(selected_q, selected_k, plucker.to(query.device))
        query, key = query.clone(), key.clone()
        query.index_copy_(0, positions, selected_q)
        key.index_copy_(0, positions, selected_k)

    out = source._sdpa_varlen_attention(query, key, value, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
    return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))


def _set_condition(dit: Any, positions: torch.Tensor, plucker: torch.Tensor) -> None:
    if plucker.shape[0] != positions.numel():
        raise ValueError("target token count must equal Pluecker coordinate count")
    # Keep this non-differentiable metadata on every block. It must survive
    # activation-checkpoint recomputation in backward; the next invocation
    # atomically replaces it.
    for block in dit.blocks:
        block.attn._scope_target_positions = positions.detach()
        block.attn._scope_target_plucker = plucker.detach()


def scope_model_fn(
    base_model_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    dit: Any,
    video_latents: torch.Tensor,
    packed: dict[str, torch.Tensor],
    keyframe_cond_anchor: torch.Tensor | None = None,
    ref_visual_anchor: torch.Tensor | None = None,
    camera_poses: torch.Tensor | None = None,
    camera_x_fov: float | torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set one target-video condition then delegate to the untouched H3 model fn."""
    if (camera_poses is None) != (camera_x_fov is None):
        raise ValueError("camera_poses and camera_x_fov must be supplied together")
    if camera_poses is not None:
        if video_latents.shape[0] != 1:
            raise NotImplementedError("MVP supports H3 batch size 1, matching the base packed pipeline")
        latent_t, latent_h, latent_w = (int(v) for v in video_latents.shape[2:])
        token_h, token_w = latent_h // 2, latent_w // 2
        plucker = build_target_plucker(
            camera_poses,
            camera_x_fov,
            target_t=latent_t,
            token_h=token_h,
            token_w=token_w,
            height=latent_h * 16,
            width=latent_w * 16,
        )[0]
        anchor = ref_visual_anchor if ref_visual_anchor is not None else keyframe_cond_anchor
        conditioned_rows = 0 if anchor is None else int(anchor.shape[0])
        target_positions = packed["img_pos"][conditioned_rows:]
        _set_condition(dit, target_positions, plucker)
    else:
        # An omitted camera must make the patched model exactly equivalent to base H3.
        for block in dit.blocks:
            block.attn._scope_target_positions = None
            block.attn._scope_target_plucker = None

    return base_model_fn(
        dit=dit,
        video_latents=video_latents,
        packed=packed,
        keyframe_cond_anchor=keyframe_cond_anchor,
        ref_visual_anchor=ref_visual_anchor,
        **kwargs,
    )


def install_scope_h3(pipe: Any, *, scale_gate_hidden: int = 64) -> Any:
    """Attach SCoPE modules and wrap a loaded DiffSynth H3 pipeline's model fn.

    Camera metadata is supplied with :func:`set_scope_camera` before calling
    the ordinary `pipe(...)`; this keeps the upstream pipeline unmodified.
    """
    if getattr(pipe, "_scope_h3_installed", False):
        raise RuntimeError("SCoPE H3 is already installed on this pipeline")
    for block in pipe.dit.blocks:
        attn = block.attn
        inner_dim = int(attn.num_heads * attn.head_dim)
        # H3 is normally loaded in BF16.  Creating the new module in the
        # projection's dtype avoids an otherwise easy-to-miss FP32/BF16 linear
        # mismatch during standalone inference (training casts these weights
        # before FSDP construction as well).
        scope_pe = H3SightlineCoordinatePE(inner_dim, scale_gate_hidden)
        weight = attn.qkv_proj.weight
        scope_pe.to(device=weight.device, dtype=weight.dtype)
        attn.add_module("scope_pe", scope_pe)
        # Make the inactive state explicit. Besides documenting the no-op
        # invariant, this prevents stale attributes should callers inspect or
        # reuse a pipeline before its first denoising call.
        attn._scope_target_positions = None
        attn._scope_target_plucker = None
        attn.forward = types.MethodType(_scope_attention_forward, attn)
    base_model_fn = pipe.model_fn

    def wrapped_model_fn(**inputs: Any):
        # Training keeps camera metadata in ``inputs_shared`` until the loss
        # invokes model_fn.  The runtime condition below is the single source
        # of truth; remove the metadata copy so Python never receives the same
        # keyword twice and the stock H3 model fn never sees unknown fields.
        inputs.pop("camera_poses", None)
        inputs.pop("camera_x_fov", None)
        camera = getattr(pipe, "_scope_h3_camera", None)
        return scope_model_fn(base_model_fn, camera_poses=None if camera is None else camera[0], camera_x_fov=None if camera is None else camera[1], **inputs)

    pipe.model_fn = wrapped_model_fn
    pipe._scope_h3_installed = True
    pipe._scope_h3_camera = None
    return pipe


def set_scope_camera(pipe: Any, camera_poses: torch.Tensor, camera_x_fov: float | torch.Tensor) -> None:
    """Set the camera trajectory used by the next and subsequent `pipe(...)` calls."""
    if not getattr(pipe, "_scope_h3_installed", False):
        raise RuntimeError("call install_scope_h3(pipe) before setting a camera")
    pipe._scope_h3_camera = (camera_poses, camera_x_fov)


def load_scope_adapter_weights(pipe: Any, adapter_path: str) -> None:
    """Load an adapter-only checkpoint exported by :func:`save_scope_adapter`.

    This deliberately accepts a plain inference pipeline, not the training
    wrapper.  It validates the exact set of SCoPE tensors before loading so a
    checkpoint from a different H3 layout cannot be silently accepted.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    if not getattr(pipe, "_scope_h3_installed", False):
        raise RuntimeError("call install_scope_h3(pipe) before loading an adapter")
    path = Path(adapter_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"adapter checkpoint does not exist: {path}")
    state_dict = load_file(str(path), device="cpu")
    expected = {name for name, _ in pipe.dit.named_parameters() if ".scope_pe." in name}
    provided = set(state_dict)
    if expected != provided:
        raise ValueError(
            f"{path}: adapter keys do not match this H3 model; "
            f"missing={len(expected - provided)}, unexpected={len(provided - expected)}"
        )
    pipe.dit.load_state_dict(state_dict, strict=False)
