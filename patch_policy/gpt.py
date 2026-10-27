"""Causal GPT over [B, T, P, D] visual tokens with block-causal attention.

All dimensions (P, T, block_size) are parameterized — nothing hard-codes P=256
(build.md §8-§10). WeMM runs with P=1.
"""

from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """GPT-2 tanh-approximation GELU — bit-identical to patch_policy native
    models/vq_behavior_transformer/gpt.py `new_gelu`. Native uses this form, NOT
    nn.GELU() (exact erf). For native_parity forks the transformer MLP must use it
    so that loading native weights gives tensor-level parity."""
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


def generate_block_causal_mask(n_patches: int, block_size: int) -> torch.Tensor:
    """[S, S] bool mask, S = n_patches*block_size.

    Tokens within one timestep's n_patches are fully connected; across timesteps
    strictly causal (block-causal).
    """
    S = n_patches * block_size
    mask = torch.zeros(S, S, dtype=torch.bool)
    for t in range(block_size):
        mask[t * n_patches : (t + 1) * n_patches, : (t + 1) * n_patches] = True
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, n_patches: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.register_buffer("mask", generate_block_causal_mask(n_patches, block_size))
        # ---- attention kernel selector -------------------------------------------
        # "current" = the hand-written q@k^T -> masked_fill(-inf) -> softmax -> @v chain
        # and is the DEFAULT, so every run recorded in doc/ stays bit-identical.
        # "sdpa" routes the SAME math through F.scaled_dot_product_attention: same mask
        # object (bool, True == may attend -- exactly the convention
        # generate_block_causal_mask builds), same n_head/head_dim, same dropout
        # probability applied at the same point in the chain, same residual path.
        # Optional because the materialized [B, nh, T, T] score matrix is the largest
        # intermediate in this model (T = block_size*n_patches = 1280, nh = 8
        # -> 419 MB fp32, plus a second copy for masked_fill); the fused kernels never
        # build it. Set it with `set_attn_impl` / `train.py --attn sdpa`, not by hand.
        self.attn_impl = "current"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.attn_impl == "sdpa":
            # Mask is passed as a [T, T] bool and broadcast over [B, nh]. The kernel
            # applies dropout at the identical point (post-softmax weights, before @v),
            # so dropout_p carries the same semantics as self.attn_dropout below.
            # is_causal must stay False: the block-causal structure is already in the
            # mask, and a True here would mask a second time (strictly lower-triangular)
            # and silently change the token interaction.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self.mask[:T, :T],
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )  # [B, nh, T, hd]
        else:
            att = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)  # [B, nh, T, T]
            att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


ATTN_IMPLS = ("current", "sdpa")


def set_attn_impl(model: nn.Module, impl: str) -> int:
    """Set the attention-kernel selector on every attention module. Returns how many.

    Covers all THREE attention sites in this repo, which is why it lives here rather
    than being copied per module:

      * ``gpt.CausalSelfAttention``        -- block-causal trunk self-attention
      * ``global_policy.GlobalSelfAttention`` -- global-policy variant (custom mask)
      * ``td_head.CrossAttention``         -- denoiser cross-attention, no mask at all

    Duck-typed on the ``attn_impl`` attribute rather than isinstance(): attention here is
    reached through three different classes and through subclasses, and everything that
    has no selector (embeddings, MLPs, the cross-attention's Linear projections) must be
    skipped silently -- including ``policy_mode == "global"``, which builds a different
    trunk class. Verified against both arms, not just the TD one: ``vqbet.BehaviorTransformer``
    owns no attention of its own, it calls the same ``gpt.GPT`` trunk (vqbet.py:163), so
    ``--attn`` covers the VQ-BeT arm identically. A caller that asked for a NON-default
    kernel should still check the return value -- ``train.py --attn`` treats ``0`` as an
    error instead of pretending the flag did something.

    Kept in sync with the identical helper on the sibling Patch Policy + Flow-Matching
    line, where the same switch was benchmarked (see doc/2026-10-27_attn_sdpa.md for the
    numbers and their provenance).
    """
    if impl not in ATTN_IMPLS:
        raise ValueError(f"unknown attn impl {impl!r} (expected one of {ATTN_IMPLS})")
    n = 0
    for m in model.modules():
        if hasattr(m, "attn_impl"):
            m.attn_impl = impl
            n += 1
    return n


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, gelu: str = "exact"):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = gelu  # "exact" = nn.GELU() | "tanh" = GPT-2 new_gelu (native parity)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gelu == "tanh":
            return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))
        return self.dropout(self.c_proj(nn.functional.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, n_patches: int, dropout: float, gelu: str = "exact"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, n_patches, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    def __init__(
        self,
        input_dim: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        block_size: int,
        n_patches: int,
        dropout: float = 0.0,
        output_dim: int | None = None,  # native parity: lm_head n_embd->output_dim readout
        gelu: str = "exact",            # "exact" = nn.GELU | "tanh" = GPT-2 new_gelu (native parity)
    ):
        self.input_dim = input_dim
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.n_patches = n_patches
        self.dropout = dropout
        self.output_dim = output_dim
        self.gelu = gelu


class GPT(nn.Module):
    """x: [B, T, P, D] -> logits [B, T, n_embd] (last patch token per timestep).

    Optional AdaLN conditioning (WeMM-DINO Fusion): when a `adaln` module is
    attached, the WeMM global semantic stream modulates the projected DINO
    tokens between wte and wpe (spec §14):  X_fused = X + gamma*LN(X)+beta
    (residual AdaLN, zero-inited -> X_fused == X at init). With adaln=None
    (baseline) the forward is bit-for-bit unchanged.
    """

    def __init__(self, config: GPTConfig, adaln: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.wte = nn.Linear(config.input_dim, config.n_embd)  # visual projection (D -> n_embd)
        self.adaln = adaln                                       # optional gated-residual AdaLN
        self.wpe = nn.Embedding(config.block_size * config.n_patches, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [
                Block(config.n_embd, config.n_head, config.block_size, config.n_patches, config.dropout, config.gelu)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        # ---- native-parity lm_head readout (patch_policy models/.../gpt.py GPT) ----
        # Native applies lm_head Linear(n_embd -> output_dim, bias=False) over ALL
        # positions after ln_f, then keeps the last patch token per timestep
        # ([b,t,p,out] -> [b,t,out]). output_dim=None (default) = no lm_head -> read
        # last patch in n_embd directly (the pre-fork patch_policy layout). output_dim=256
        # restores the HIST DINO-P256 readout so head inputs are 256-d.
        self.lm_head = (
            nn.Linear(config.n_embd, config.output_dim, bias=False)
            if config.output_dim is not None
            else None
        )
        self.readout_dim = config.output_dim if config.output_dim is not None else config.n_embd
        # ---- GPT-2 style weight init (patch_policy GPT parity) ----
        # Baseline protocol: wte/wpe/h/ln_f all ~N(0,0.02), residual c_proj scaled by
        # 1/sqrt(2*n_layer) (nanoGPT). WITHOUT this, nn.Embedding/wpe falls back to the
        # PyTorch default N(0,1) (per-vector norm ~sqrt(n_embd) ~ 22.6), which swamps the
        # visual token stream ~40x -> policy collapses to a position-only, input-
        # independent action (DINO-P256 baseline audit, 2026-09-03).
        # The optional AdaLN fusion branch (self.adaln) is EXCLUDED: it must keep its own
        # zero-init identity start (fusion spec hard requirement).
        for _name, child in self._modules.items():
            if _name == "adaln":
                continue
            child.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """nanoGPT init (identical to patch_policy models/gpt.py GPT._init_weights)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        B, T, P, D = x.shape
        assert T * P <= self.config.block_size * self.config.n_patches, (
            f"seq len {T*P} exceeds block size {self.config.block_size * self.config.n_patches}"
        )
        x = self.wte(x)  # [B, T, P, n_embd]
        if self.adaln is not None:
            assert cond is not None, "AdaLN enabled but no conditioning stream provided"
            x = self.adaln(x, cond)  # [B, T, P, n_embd] gated residual (alpha=0 -> == X)
        pos = torch.arange(0, T * P, device=x.device)
        x = einops.rearrange(x, "b t p d -> b (t p) d") + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        if self.lm_head is not None:
            x = self.lm_head(x)  # native parity: project ALL positions, then read last patch
        x = einops.rearrange(x, "b (t p) d -> b t p d", t=T)
        return x[:, :, -1]  # [B, T, readout_dim] — the (only) visual token per timestep
