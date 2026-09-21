"""Per-dimension affine action normaliser (limits mode) for the TD arm.

Ported from Patch policy/upstream/utils/normalizer.py (_fit / _normalize /
SingleFieldLinearNormalizer), minus the zarr dependency and the dict-of-fields
machinery — this arm normalises exactly one thing: action chunks.

Why this exists at all: PushT actions in this repo are RAW pixel coordinates
(dim0 12..511, dim1 25..511) with no normalisation anywhere in the data layer
(see doc/2026-09-16_P1_action_anchor_audit.md). A truncated-diffusion head needs its
anchors, its noise scale and its L1 target to live in a bounded space, so we fit
an affine map onto the TRAIN split only and squash to exactly [-1, 1].

CRITICAL — params are BUFFERS, not Parameters.
`train.py` -> `configure_optimizers` classifies every entry of
`model.named_parameters()` into a decay set or a no_decay set and then asserts
    set(param_dict) == (decay | no_decay)
The classification only recognises names ending in "bias", or "weight" on a
Linear/LayerNorm/Embedding. A ParameterDict named scale/offset/input_stats.*
matches NONE of those rules and would land in neither set -> hard assertion
failure at model construction. Buffers are also simply the honest container:
these quantities are fit once and never trained. They still round-trip through
state_dict(), so a checkpoint carries its own normaliser.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ActionNormalizer", "fit_limits", "normalize_with", "unnormalize_with"]


def fit_limits(
    data: torch.Tensor,
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
) -> dict:
    """Port of Patch policy/upstream/utils/normalizer.py::_fit with mode="limits".

    `data` is [N, D] flattened chunks. Returns plain tensors (scale, offset,
    input stats) so the caller can decide whether to buffer or inspect them.
    """
    assert output_max > output_min
    data = data.to(torch.float32)
    data = data.reshape(-1, data.shape[-1])

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    input_range = input_max - input_min
    # Constant dimensions (range < eps) are not scaled to [-1,1] — they are mapped
    # to the CENTRE of the output range instead, via offset below. Ported verbatim.
    ignore_dim = input_range < range_eps
    input_range = input_range.clone()
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return {
        "scale": scale,
        "offset": offset,
        "input_min": input_min,
        "input_max": input_max,
        "input_mean": input_mean,
        "input_std": input_std,
        "ignore_dim": ignore_dim,
    }


def normalize_with(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Port of _normalize(forward=True). Flattens trailing dims to scale.shape[0]."""
    scale = scale.to(device=x.device, dtype=torch.float32)
    offset = offset.to(device=x.device, dtype=torch.float32)
    src_shape = x.shape
    x = x.to(torch.float32).reshape(-1, scale.shape[0])
    x = x * scale + offset
    return x.reshape(src_shape)


def unnormalize_with(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Port of _normalize(forward=False)."""
    scale = scale.to(device=x.device, dtype=torch.float32)
    offset = offset.to(device=x.device, dtype=torch.float32)
    src_shape = x.shape
    x = x.to(torch.float32).reshape(-1, scale.shape[0])
    x = (x - offset) / scale
    return x.reshape(src_shape)


class ActionNormalizer(nn.Module):
    """[..., W, A] chunk <-> normalised space. Fit on the TRAIN split only.

    Deliberately flat: the affine params are per-element of the flattened (W*A)
    chunk vector, so a chunk of [5,2] maps through a 10-vector. That matches how
    the reference linear normaliser is used on chunked data and keeps the
    anchor-space distance metric (used to pick the positive mode) meaningful.
    """

    def __init__(self, n_window: int, act_dim: int, output_max: float = 1.0,
                 output_min: float = -1.0, range_eps: float = 1e-4):
        super().__init__()
        self.n_window = int(n_window)
        self.act_dim = int(act_dim)
        self.dim = self.n_window * self.act_dim
        self.output_max = float(output_max)
        self.output_min = float(output_min)
        self.range_eps = float(range_eps)
        self.fitted = False

        # Buffers, NOT Parameters — see module docstring.
        self.register_buffer("scale", torch.ones(self.dim), persistent=True)
        self.register_buffer("offset", torch.zeros(self.dim), persistent=True)
        self.register_buffer("input_min", torch.zeros(self.dim), persistent=True)
        self.register_buffer("input_max", torch.zeros(self.dim), persistent=True)
        self.register_buffer("input_mean", torch.zeros(self.dim), persistent=True)
        self.register_buffer("input_std", torch.ones(self.dim), persistent=True)

    @torch.no_grad()
    def fit(self, chunks: torch.Tensor) -> "ActionNormalizer":
        """`chunks` is [N, W, A] RAW pixel actions from the TRAIN split only."""
        assert chunks.ndim >= 2, f"expected [...,W,A], got {tuple(chunks.shape)}"
        flat = chunks.reshape(-1, self.dim)
        p = fit_limits(flat, output_max=self.output_max,
                       output_min=self.output_min, range_eps=self.range_eps)
        self.scale.copy_(p["scale"])
        self.offset.copy_(p["offset"])
        self.input_min.copy_(p["input_min"])
        self.input_max.copy_(p["input_max"])
        self.input_mean.copy_(p["input_mean"])
        self.input_std.copy_(p["input_std"])
        self.fitted = True
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "ActionNormalizer.normalize called before fit()"
        return normalize_with(x, self.scale, self.offset)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "ActionNormalizer.unnormalize called before fit()"
        return unnormalize_with(x, self.scale, self.offset)

    def stats(self) -> dict:
        """Everything needed to reproduce / audit this fit, JSON-serialisable."""
        return {
            "n_window": self.n_window,
            "act_dim": self.act_dim,
            "dim": self.dim,
            "mode": "limits",
            "output_min": self.output_min,
            "output_max": self.output_max,
            "range_eps": self.range_eps,
            "scale": [round(float(v), 8) for v in self.scale],
            "offset": [round(float(v), 8) for v in self.offset],
            "input_min": [round(float(v), 6) for v in self.input_min],
            "input_max": [round(float(v), 6) for v in self.input_max],
            "input_mean": [round(float(v), 6) for v in self.input_mean],
            "input_std": [round(float(v), 6) for v in self.input_std],
        }
