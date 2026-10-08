"""Parameter-efficient SCoPE Q/K injection for H3's packed attention."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class H3SightlineCoordinatePE(nn.Module):
    """SCoPE's Normalize-Gate-Inject PE, adapted to H3's flattened Q/K width.

    `inner_dim` is 7168 for the released H3 (56 heads × 128 channels).  The
    scale gate deliberately uses a small bottleneck: it outputs `inner_dim`
    values but learns them through a low-rank `1 -> gate_hidden -> inner_dim`
    mapping.  Zero-initialised Q/K output projections give exact zero delta.
    """

    def __init__(self, inner_dim: int, gate_hidden: int = 64) -> None:
        super().__init__()
        if inner_dim <= 0 or gate_hidden <= 0:
            raise ValueError("inner_dim and gate_hidden must be positive")
        self.inner_dim = inner_dim
        self.eq = nn.Linear(7, inner_dim, bias=False)
        self.ek = nn.Linear(7, inner_dim, bias=False)
        self.norm_q = RMSNorm(inner_dim)
        self.norm_k = RMSNorm(inner_dim)
        self.scale_gate = nn.Sequential(
            nn.Linear(1, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, inner_dim), nn.Sigmoid()
        )
        nn.init.zeros_(self.eq.weight)
        nn.init.zeros_(self.ek.weight)
        nn.init.zeros_(self.scale_gate[0].bias)
        nn.init.zeros_(self.scale_gate[2].bias)

    @staticmethod
    def _features(plucker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction, moment = plucker[..., :3], plucker[..., 3:]
        moment_norm = moment.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        moment_hat = moment / moment_norm
        log_scale = moment_norm.log()
        return torch.cat((direction, moment_hat, log_scale), -1), torch.cat((moment_hat, direction, log_scale), -1), log_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, plucker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject PE into selected rows only.

        query/key: `[N, heads, head_dim]`; plucker: `[N, 6]` in the same row order.
        """
        if query.shape != key.shape or query.ndim != 3:
            raise ValueError("query and key must have matching [N, heads, head_dim] shapes")
        if plucker.shape != (query.shape[0], 6):
            raise ValueError("plucker must be [N, 6] and align with query rows")
        if query.shape[1] * query.shape[2] != self.inner_dim:
            raise ValueError("query width does not match this PE module")
        feat_q, feat_k, log_scale = self._features(plucker)
        gate = self.scale_gate(log_scale.to(query.dtype))
        q_delta = gate * self.norm_q(self.eq(feat_q.to(query.dtype)))
        k_delta = gate * self.norm_k(self.ek(feat_k.to(key.dtype)))
        return query + q_delta.view_as(query), key + k_delta.view_as(key)

