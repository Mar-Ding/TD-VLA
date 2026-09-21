"""Global-to-Spatial Coord-MLP Decoder (explicit position-conditioned).

From a frozen WeMM global embedding z [..., 1, 2048] produce P explicit spatial
slots [..., P, 512] through a SINGLE shared MLP over (z, coordinate Fourier PE):

    t_i = f_theta( concat(z, FourierPE(x_i, y_i)) ),   f shared across all slots

Grid: row-major, x,y in [-1, 1].  P=256 -> 16x16, P=64 -> 8x8.
PE:  fixed deterministic Fourier features, pe_dim=64, x/y symmetric, 0 params:
     PE = [sin(2π f_k x), cos(2π f_k x), sin(2π f_k y), cos(2π f_k y)], f_k = 2^k
Architecture (per experiment spec, ~1-2M params):
     concat(z, PE_i) [2112] -> Linear(2112,512) -> GELU -> Linear(512,512) -> t_i
     (1,081,856 + 262,656 = 1,344,512 params)

These are position-conditioned DECODED SLOTS — not native patch tokens, not
cropped features. The decoder is the ONLY interface change; WeMM / GPT /
Patch Policy / VQ-BeT / dataset / rollout protocol stay untouched.

Prohibited (per spec): per-position MLP, cross-attention, transformer decoder,
conv decoder, U-Net, extra supervision, geometry labels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CoordFourierPE(nn.Module):
    """Deterministic Fourier features on a row-major 2D grid ([-1,1]^2)."""

    def __init__(self, n_patches: int, pe_dim: int = 64):
        super().__init__()
        rows = int(round(n_patches ** 0.5))
        assert rows * rows == n_patches, f"n_patches must be a perfect square, got {n_patches}"
        n_freq = pe_dim // 4
        assert pe_dim == 4 * n_freq, f"pe_dim must be divisible by 4, got {pe_dim}"
        self.n_patches = n_patches
        self.pe_dim = pe_dim
        self.rows = self.cols = rows
        self.n_freq = n_freq

        freqs = 2.0 ** torch.arange(n_freq)                # [n_freq], f_k = 2^k
        xs = torch.linspace(-1.0, 1.0, self.cols)          # columns
        ys = torch.linspace(-1.0, 1.0, self.rows)          # rows
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")     # [rows, cols], row-major
        xxf = xx.reshape(-1, 1)                            # [P, 1]
        yyf = yy.reshape(-1, 1)
        argx = 2.0 * math.pi * freqs.view(1, -1) * xxf     # [P, n_freq]
        argy = 2.0 * math.pi * freqs.view(1, -1) * yyf
        pe = torch.cat(
            [torch.sin(argx), torch.cos(argx),
             torch.sin(argy), torch.cos(argy)],
            dim=1,
        )                                                  # [P, 4*n_freq] = [P, pe_dim]
        assert pe.shape == (n_patches, pe_dim)
        self.register_buffer("pe", pe)                     # deterministic, no grad
        self.register_buffer("grid_x", xs)
        self.register_buffer("grid_y", ys)

    def config_dict(self) -> dict:
        return {
            "pe": "deterministic_fourier",
            "pe_dim": self.pe_dim,
            "n_freq_per_axis": self.n_freq,
            "grid_rows": self.rows,
            "grid_cols": self.cols,
            "coordinate_range": "[-1,1]",
            "order": "row_major (i = r*cols + c)",
            "axes": "x,y symmetric",
            "trainable_params": 0,
        }

    def forward(self) -> torch.Tensor:
        return self.pe


class CoordSpatialDecoder(nn.Module):
    """Shared position-conditioned decoder: z [..., 1, D] -> [..., P, out_dim]."""

    def __init__(
        self,
        n_embed_in: int = 2048,
        pe_dim: int = 64,
        hidden: int = 512,
        out_dim: int = 512,
        n_patches: int = 256,
    ):
        super().__init__()
        self.pe = CoordFourierPE(n_patches, pe_dim)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed_in + pe_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim
        self.n_patches = n_patches

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [..., 1, D] -> [..., P, out_dim] (shared MLP, no per-position params)."""
        assert z.shape[-2] == 1, f"coord decoder expects a single global token, got P={z.shape[-2]}"
        assert z.shape[-1] == self.mlp[0].in_features - self.pe.pe_dim, (
            f"z dim {z.shape[-1]} != decoder input D={self.mlp[0].in_features - self.pe.pe_dim}"
        )
        lead = z.shape[:-2]                              # e.g. [B,T]
        zp = z.squeeze(-2).unsqueeze(-2).expand(*lead, self.n_patches, z.shape[-1])  # [...,P,D]
        pe = self.pe.pe.expand(*lead, self.n_patches, self.pe.pe_dim)               # [...,P,PE]
        inp = torch.cat([zp, pe], dim=-1)                # [...,P, D+PE]
        return self.mlp(inp)                             # [...,P,out_dim]
