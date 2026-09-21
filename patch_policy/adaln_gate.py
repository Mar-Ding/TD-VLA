"""WeMM-DINO residual AdaLN + scalar alpha gate from 0 (WeMM-AdaLN-zero, v1 semantics).

Variants of the same residual-AdaLN conditioning branch:

  A1  AdaLN (patch_policy/adaln.py):            X_fused = X + γ·LN_local(X) + β
        proj zero-init -> identity start; proj WARM from step 0 (dL/dW_proj != 0 at init).
  Z0  AdaLNGate (THIS module, the experiment):  X_fused = X + α·(γ·LN_local(X) + β)   (= X + α·M)
        proj DEFAULT nn.Linear init (M != 0 at init, so γ/β already 'live');
        α: ONE learnable scalar, init 0 -> identity start comes from α==0, NOT from
        zeroing proj. Gradient flow:
            dL/dα    = Σ_M (M · dL/dX) != 0 at init  -> the gate wakes on step 1;
            dL/dW_proj = α·(…) = 0 at init            -> proj stays COLD until α > 0
                          ("gate first, proj second").
        This is the historically-superseded v1 cold-start design, kept here as the
        experimental contrast to A1 (superseded in 2026-08-31 because the zeroed proj
        made A1 strictly better-behaved; the question this arm answers: does a learnable
        from-zero gate that delays proj training match / harm the A1 curve?).

Why NOT α-gate stacked on A1's zero-init proj: with proj≡0 the branch M≡0 AND
dL/dα = Σ M·(…) = 0 forever -> the whole cond branch is permanently dead (a silent
re-train of the baseline). Identity MUST come from α=0 with proj at DEFAULT init, or
the branch never wakes. (v1 semantics, user-confirmed.)

Interface parity with AdaLN (train.py _capture_adaln_diag / layer_grad_norm rely on
the members `ln_g`/`proj`/`ln_local` and the stashed `_diag`):
  * self.ln_g      LayerNorm(cond_dim)          (LN over the WeMM global embedding)
  * self.proj      Linear(cond_dim, 2*n_embd)   -> gamma,beta split (DEFAULT init)
  * self.ln_local  LayerNorm(n_embd, affine=ln_local_affine)
  * self._diag     detached last-forward gamma/beta/delta_norm/x_norm (+ alpha)
  * diag_summary() same-key dict as AdaLN.diag_summary() plus "alpha".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from patch_policy.adaln import _stats


class AdaLNGate(nn.Module):
    """Alpha-gated residual AdaLN: X_fused = X + alpha * (gamma*LN_local(X) + beta).

    alpha init 0 -> X_fused == X EXACTLY at init (bitwise: 0.0 * M == 0), irrespective
    of proj. proj is default-initialized (NOT zeroed) so M != 0 and the gate gets
    gradient on the first real backward (post-VQ-fit epoch 2).
    """

    def __init__(
        self,
        cond_dim: int = 2048,
        n_embd: int = 512,
        hidden_dim: int | None = None,
        zero_init: bool = False,   # NOTE: default False = proj DEFAULT init (identity via alpha)
        alpha_init: float = 0.0,
        ln_local_affine: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 2 * n_embd
        assert hidden_dim == 2 * n_embd, (
            f"spec: Linear(cond_dim, hidden_dim) must split into gamma,beta of n_embd each; "
            f"got hidden_dim={hidden_dim}, 2*n_embd={2*n_embd}"
        )
        assert alpha_init == 0.0, (
            "WeMM-AdaLN-zero: alpha_init must be 0 (identity start comes from alpha==0); "
            "a nonzero alpha would corrupt the input stream at init"
        )
        self.cond_dim = cond_dim
        self.n_embd = n_embd
        self.ln_g = nn.LayerNorm(cond_dim)              # LN_global(g)
        self.proj = nn.Linear(cond_dim, hidden_dim)     # -> gamma,beta split
        if zero_init:
            # Opt-in to the DEAD branch (proj≡0 -> M≡0 -> alpha never learns). Kept only
            # for config parity; build_model asserts against this combo for the gate run.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        # else: DEFAULT nn.Linear init (weight ~ U(-1/sqrt(in),1/sqrt(in)), bias ~ U(0)-same)
        # so M = gamma*LN(X)+beta is NONZERO at init (proj 'live'), identity held by alpha.
        self.ln_local = nn.LayerNorm(n_embd, elementwise_affine=ln_local_affine)
        # the learnable scalar gate, init 0 -> exact identity at init.
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32) + alpha_init)
        self._diag = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x [B,T,P,n_embd]; cond [B,T,cond_dim] -> fused [B,T,P,n_embd]."""
        g = self.ln_g(cond)                       # [B,T,cond_dim]
        h = self.proj(g)                          # [B,T,2*n_embd]
        gamma, beta = h.chunk(2, dim=-1)          # [B,T,n_embd]
        gamma = gamma.unsqueeze(-2)               # [B,T,1,n_embd]
        beta = beta.unsqueeze(-2)
        m = gamma * self.ln_local(x) + beta       # [B,T,P,n_embd] un-gated branch (M)
        fused = x + self.alpha * m                # alpha=0 -> fused == x exactly
        with torch.no_grad():
            a = self.alpha.detach()
            self._diag = {
                "gamma": gamma.detach(),
                "beta": beta.detach(),
                # delta_norm = |alpha*M| (actual written modulation; mod_ratio comparable to A1)
                "delta_norm": float((a * m).detach().float().norm()),
                # raw un-gated branch magnitude (evidence M != 0 even while alpha == 0)
                "m_norm": float(m.detach().float().norm()),
                "x_norm": float(x.detach().float().norm()),
                "alpha": float(a),
            }
        return fused

    def diag_summary(self) -> dict:
        """Compact scalar summary of the last forward (gamma/beta stats, mod ratio, alpha)."""
        d = self._diag or {}
        mod = (d.get("delta_norm", 0.0) / (d.get("x_norm", 0.0) + 1e-6))
        return {
            "mod_ratio": mod,
            "delta_norm": d.get("delta_norm"),
            "m_norm": d.get("m_norm"),
            "x_norm": d.get("x_norm"),
            "alpha": d.get("alpha"),
            "gamma": _stats(d["gamma"]) if "gamma" in d else None,
            "beta": _stats(d["beta"]) if "beta" in d else None,
        }
