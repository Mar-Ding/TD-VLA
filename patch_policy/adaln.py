"""WeMM-DINO AdaLN Fusion — residual adaptive LayerNorm conditioning (spec §3-§4).

Fusion equation (residual AdaLN — NOT (1+gamma)*LN(X)+beta, NOT an alpha gate):

    X_fused = X + gamma(LN_g(g)) * LN_local(X) + beta(LN_g(g))    (= X + DeltaX)

  * X       : DINO-P256 dense tokens after the existing projection  [B,T,P,n_embd]
  * g       : frozen WeMM official final global embedding           [B,T,cond_dim]
  * LN_g    : LayerNorm over g (per-frame vector normalization)
  * gamma,beta = split(Linear(cond_dim, 2*n_embd)(LN_g(g)))         [B,T,1,n_embd]
  * LN_local: LayerNorm over X's last channel dim (per patch-vector), no affine
              (AdaLN-style: gamma/beta ARE the affine — ln_local_affine=False)
  * identity init (HARD requirement): the final Linear is zero-initialized
    (weight=0, bias=0) => gamma=0, beta=0 => X_fused == X exactly at init.

Residual form (X + ΔX) instead of in-place affine ((1+γ)·LN(X)+β) keeps the
DINO-P256 baseline input distribution AND the identity initialization exactly
(spec §3).

Gradient flow: dX_fused/dW_proj = dΔX/dW_proj is NONZERO at init (LN_local(X)
is nonzero and dL/dX_fused flows back through the residual), so the AdaLN Linear
gets gradient at step 0 and gamma/beta leave zero immediately — no cold start.
(This is why the spec moved from an alpha gate to AdaLN-only.)

DINO keeps the real dense spatial structure; WeMM never fabricates spatial
patches — it only adaptively normalizes/modulates the existing DINO tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _stats(t: torch.Tensor) -> dict:
    """Per-tensor summary: mean/std/abs_mean/max_abs (float32, syncs once each)."""
    t = t.detach().float()
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "abs_mean": float(t.abs().mean()),
        "max_abs": float(t.abs().max()),
    }


class AdaLN(nn.Module):
    """Residual AdaLN: X_fused = X + gamma*LN_local(X) + beta."""

    def __init__(
        self,
        cond_dim: int = 2048,
        n_embd: int = 512,
        hidden_dim: int | None = None,
        zero_init: bool = True,
        ln_local_affine: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 2 * n_embd
        assert hidden_dim == 2 * n_embd, (
            f"spec: Linear(cond_dim, hidden_dim) must split into gamma,beta of n_embd each; "
            f"got hidden_dim={hidden_dim}, 2*n_embd={2*n_embd}"
        )
        self.cond_dim = cond_dim
        self.n_embd = n_embd
        self.ln_g = nn.LayerNorm(cond_dim)                        # LN_global(g)
        self.proj = nn.Linear(cond_dim, hidden_dim)               # -> gamma,beta split
        if zero_init:
            # identity init (HARD requirement): gamma=0, beta=0 => X_fused == X exactly.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.ln_local = nn.LayerNorm(n_embd, elementwise_affine=ln_local_affine)
        self._diag = None  # last-forward diagnostics (detached; for train/audit logging)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x [B,T,P,n_embd]; cond [B,T,cond_dim] -> fused [B,T,P,n_embd]."""
        g = self.ln_g(cond)                       # [B,T,cond_dim]
        h = self.proj(g)                          # [B,T,2*n_embd]
        gamma, beta = h.chunk(2, dim=-1)          # [B,T,n_embd]
        gamma = gamma.unsqueeze(-2)               # [B,T,1,n_embd]
        beta = beta.unsqueeze(-2)
        delta = gamma * self.ln_local(x) + beta   # [B,T,P,n_embd]
        fused = x + delta                         # residual AdaLN
        with torch.no_grad():
            self._diag = {
                "gamma": gamma.detach(),
                "beta": beta.detach(),
                "delta_norm": float(delta.detach().float().norm()),
                "x_norm": float(x.detach().float().norm()),
            }
        return fused

    def diag_summary(self) -> dict:
        """Compact scalar summary of the last forward (gamma/beta stats, mod ratio)."""
        d = self._diag or {}
        mod = (d.get("delta_norm", 0.0) / (d.get("x_norm", 0.0) + 1e-6))
        return {
            "mod_ratio": mod,
            "delta_norm": d.get("delta_norm"),
            "x_norm": d.get("x_norm"),
            "gamma": _stats(d["gamma"]) if "gamma" in d else None,
            "beta": _stats(d["beta"]) if "beta" in d else None,
        }
