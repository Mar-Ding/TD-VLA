"""Truncated diffusion schedule + DDIM step + inference timestep sequences.

SELF-IMPLEMENTED on purpose. `wemm-p1` — the env that `rollout_locked_gpu2.sh`
activates, i.e. the env every arm actually runs in — has neither `diffusers` nor
`sklearn`, and installing into it would perturb the existing VQ-BeT/AdaLN arms.
So the beta schedule and the DDIM step are reproduced here in ~40 lines of torch,
and P1b cross-checks them against `diffusers` in the `patch-policy` env (which
does have it) rather than taking fidelity on faith.

Why not just call DDIMScheduler.step:
    prev_timestep = timestep - num_train_timesteps // num_inference_steps
With Diffusers' own `set_timesteps(1000)` that delta is ALWAYS 1, so a truncated
loop of `[10, 0]` actually does: eval@10 -> step 10->9 (moves ~6% of the way),
then eval@0 -> step 0->final. The second call returns x0_hat outright, so the
"2 step" inference in the reference is one live step plus one no-op. That is API
semantics, not a bug we can patch around — so `ddim_step` here takes an EXPLICIT
(t_cur, t_prev) pair with arbitrary span, and the sequence always ends at t=0.
The reference's behaviour is reproducible via `make_eval_timesteps_diffusiondrive`
for attribution in P5, not as the production path.

DELIBERATE DEVIATION from the reference's inference path (documented, not silent):
`transfuser_model_v2.py:519` additionally clamps the *latent* before encoding it as
the denoiser query, `x_boxes = torch.clamp(img, min=-1, max=1)`. We do NOT. That clamp
exists only in the reference's `forward_test`, never in training, so it puts a
train/inference distribution shift into the reference's own pipeline; our training path
is unclamped, and adding the clamp at inference only would import that mismatch for no
measured benefit. It is one line to add behind a flag if P5 ever shows the denoiser
extrapolating badly. P1b reports it as a known difference.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

__all__ = [
    "TruncatedDiffusionSchedule",
    "make_inference_timesteps",
    "inference_chain",
    "make_diffusers_timesteps",
    "make_eval_timesteps_diffusiondrive",
    "tradeoff_table",
]


def make_beta_schedule(
    num_train_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: str = "scaled_linear",
) -> torch.Tensor:
    """Bit-for-bit match of diffusers' beta schedules.

    scaled_linear: linspace(sqrt(b0), sqrt(b1), N) ** 2
    linear:        linspace(b0, b1, N)

    The linspace and the square are done in FLOAT32 because that is what diffusers
    does (`torch.linspace(..., dtype=torch.float32) ** 2`). Computing in float64 and
    casting afterwards is mathematically identical but NOT bit-identical: measured
    max|diff| 3.7e-9 against `DDIMScheduler.betas`. That is numerically irrelevant and
    still worth removing, because the audit's claim is parity with the reference and
    "agrees to within float32 rounding" invites a reader to wonder what else differs.
    """
    if schedule == "scaled_linear":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
                              dtype=torch.float32) ** 2
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps,
                              dtype=torch.float32)
    raise ValueError(f"unsupported beta_schedule {schedule!r}")


class TruncatedDiffusionSchedule(nn.Module):
    """alphas_cumprod table + add_noise + DDIM step, restricted to t < train_t_max.

    Buffers, not Parameters (same reason as normalizer.py — the optimiser param
    classifier in train.py would otherwise reject them).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "scaled_linear",
        train_t_max: int = 50,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super().__init__()
        assert 1 <= train_t_max <= num_train_timesteps, (
            f"train_t_max {train_t_max} must be in [1, {num_train_timesteps}]"
        )
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        self.train_t_max = int(train_t_max)
        # `clip_sample=True, clip_sample_range=1.0` are diffusers' DEFAULTS for
        # DDIMScheduler, and the reference constructs its scheduler as
        # `DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
        #  prediction_type="sample")` (transfuser_model_v2.py:400-404) with no other
        # kwargs -- so it inherits them and DOES clamp x0_hat to [-1, 1] every step.
        # Omitting it here would have been a silent behavioural difference from the
        # reference, which is exactly the class of thing an audit is supposed to catch.
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)

        betas = make_beta_schedule(num_train_timesteps, beta_start, beta_end, beta_schedule)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)
        # diffusers' DDIMScheduler.final_alpha_cumprod — the "prev" value used when
        # prev_timestep < 0, which is what makes the final step collapse to x0_hat.
        # 1.0 is the `set_alpha_to_one=True` branch, which is diffusers' DEFAULT and
        # therefore what the reference gets by omission. (The other branch would give
        # alphas_cumprod[0] = 0.9999, which would make the last step a ~1% mixture
        # instead of an identity -- a difference the audit pins numerically.)
        self.register_buffer("final_alpha_cumprod", torch.tensor(1.0), persistent=True)

    # ---- table lookups -------------------------------------------------
    def abar(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar_t. `t` may be a scalar, a tensor, or -1 (meaning "final")."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.alphas_cumprod.device, dtype=torch.long)
        t = t.to(self.alphas_cumprod.device).long()
        out = self.alphas_cumprod[t.clamp(min=0)]
        return torch.where(t < 0, self.final_alpha_cumprod.to(out.dtype), out)

    def b(self, t) -> torch.Tensor:
        """b_t = sqrt(1 - alpha_bar_t): the noise scale actually present at t."""
        return (1.0 - self.abar(t)).clamp_min(0.0).sqrt()

    # ---- training ------------------------------------------------------
    def sample_train_timesteps(self, shape, device, generator=None) -> torch.Tensor:
        """t ~ Uniform{0, ..., train_t_max - 1}. NOT 1..50 — the lower bound is inclusive
        of 0 so the model sees the near-clean end of the band it will start from."""
        return torch.randint(0, self.train_t_max, shape, device=device, generator=generator)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * noise.

        `t` broadcasts against x0's leading dims (unsqueeze-trailing, exactly as
        diffusers does), so t of shape [B] works for x0 of [B, K, W, A], and t of
        shape [B, T] works for x0 of [B, T, K, W, A].
        """
        abar = self.abar(t)
        while abar.ndim < x0.ndim:
            abar = abar.unsqueeze(-1)
        return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise

    # ---- inference -----------------------------------------------------
    def ddim_step(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t_cur: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """One DDIM step, prediction_type="sample" (model output IS x0_hat).

        `t_prev < 0` means "past the end of the chain": alpha_bar_prev falls back to
        final_alpha_cumprod == 1.0, so sqrt(a_prev) == 1 and sqrt(1 - a_prev) == 0 and
        the step returns the model's output EXACTLY, carrying NO residual noise. That
        identity is what guarantees the loop terminates. The phrase to be precise about
        is which tensor comes back: it is `clamp(x0_hat, ±clip_sample_range)` when
        `clip_sample` is on (the reference's setting), not `x0_hat` unconditionally. For
        an in-range x0_hat the two coincide; for an out-of-range one they do not, and
        P1b checks both cases against diffusers rather than asserting the loose version.

        In the PRODUCTION chain this branch is never taken: the last transition is
        `(25, 0)`, and `t_prev == 0` reads alphas_cumprod[0] = 0.99989, so the final step
        is a genuine 0.83-travel transition. The -1 fallback belongs to the reference's
        `[10, 0]` attribution mode, where `t=0` is the last model call.

        ORDER MATTERS for `clip_sample`. diffusers computes the implied eps from the
        UNCLIPPED prediction, then clamps x0_hat, then combines:

            eps_hat = (x_t - sqrt(a_cur) * x0_hat_raw) / b_cur      <- unclipped
            x0_use  = clamp(x0_hat_raw, ±clip_sample_range)         <- clipped
            x_prev  = sqrt(a_prev) * x0_use + sqrt(1 - a_prev) * eps_hat

        Clipping x0_hat *before* forming eps_hat would instead treat the clamped value
        as the model's actual output and move the step somewhere else. Replicated in
        that order so `clip_sample` is genuinely the reference's operation.
        """
        a_cur = self.abar(t_cur)
        a_prev = self.abar(t_prev)
        while a_cur.ndim < x_t.ndim:
            a_cur = a_cur.unsqueeze(-1)
        while a_prev.ndim < x_t.ndim:
            a_prev = a_prev.unsqueeze(-1)
        eps_hat = (
            (x_t - a_cur.sqrt() * x0_hat) / (1.0 - a_cur).clamp_min(1e-12).sqrt()
        )
        if self.clip_sample:
            x0_use = x0_hat.clamp(-self.clip_sample_range, self.clip_sample_range)
        else:
            x0_use = x0_hat
        return a_prev.sqrt() * x0_use + (1.0 - a_prev).clamp_min(0.0).sqrt() * eps_hat

    def step_weights(self, t_cur, t_prev) -> dict:
        """Exact convex weights of one DDIM step, for the P1b audit.

        With eps_hat = (x_t - sqrt(a_cur) x0_hat) / b_cur, the step is exactly
            x_prev = w0 * x0_hat + w1 * x_t
            w1 = b_prev / b_cur
            w0 = sqrt(a_prev) - (b_prev / b_cur) * sqrt(a_cur)
        At small t, sqrt(a) ~ 1 so w0 ~ 1 - b_prev/b_cur, which is the "each step only
        mixes in x0_hat by a fraction" intuition. We report the EXACT w0/w1 and only
        use the approximation to describe them; travel = w0 / (w0 + w1) is how much of
        the way from x_t to x0_hat the step actually travels.

        NOTE: these weights describe the step with `clip_sample` INACTIVE. Clipping bounds
        x0_hat before it is mixed in, which is a projection onto the legal action range
        (-1..1 == the normaliser's limits, so it is exactly "clamp to legal actions") and
        does not change how much of x_t is retained. Every travel/`cumulative_travel`
        number in the audit is therefore the unclipped mixing ratio, which is what those
        numbers are meant to describe.
        """
        a_cur, a_prev = float(self.abar(t_cur)), float(self.abar(t_prev))
        b_cur, b_prev = math.sqrt(max(1e-12, 1.0 - a_cur)), math.sqrt(max(0.0, 1.0 - a_prev))
        w1 = b_prev / b_cur
        w0 = math.sqrt(a_prev) - w1 * math.sqrt(a_cur)
        tot = w0 + w1
        return {
            "t_cur": int(t_cur), "t_prev": int(t_prev),
            "alpha_bar_cur": round(a_cur, 10), "alpha_bar_prev": round(a_prev, 10),
            "b_cur": round(b_cur, 8), "b_prev": round(b_prev, 8),
            "w0_x0hat": round(w0, 8), "w1_xt": round(w1, 8),
            "approx_travel_1_minus_bprev_over_bcur": round(1.0 - w1, 8),
            "exact_travel_w0_over_sum": round(w0 / tot, 8) if tot else 0.0,
        }


def make_inference_timesteps(t_start: int, n_steps: int) -> List[int]:
    """THE PRODUCTION CHAIN. `n_steps` MODEL-INPUT timesteps, all in (0, t_start].

    t_start=49, n_steps=1 -> [49]              transitions: 49 -> 0
    t_start=49, n_steps=2 -> [49, 25]          transitions: 49 -> 25 -> 0
    t_start=49, n_steps=4 -> [49, 37, 25, 13]  transitions: ... -> 13 -> 0

    Every transition is a REAL DDIM step: `n_steps` model calls, `n_steps` non-trivial
    transitions, and the chain's endpoint 0 is reached by the LAST transition rather
    than by a model call AT t=0.

    Why this exists instead of the diffusers layout (`make_diffusers_timesteps`):
    diffusers' `set_timesteps(N)` on T=1000 emits a sequence whose LAST ENTRY IS 0, so
    the final model call happens on a (nearly clean) input and its `step()` has
    `prev_timestep < 0`, which returns `x0_hat` verbatim -- one no-op. Paired with
    `prev_timestep = t - T//N` spacing, a "2-step" run is therefore closer to one live
    step plus one identity. Measured on this schedule: 49->0 travels 89.4% while 10->9
    travels 5.25%. Calling that a 2-step truncated diffusion is not defensible, so the
    canonical chain spends its step budget on ACTUAL transitions and the t=0 terminal
    is exercised separately, as a parity assertion (`test_termination`), not as a step.

    Integer floor arithmetic, not float rounding: `round(24.5)` is 24 under banker's
    rounding and 25 under half-up, and this number ends up in a published table.
    """
    assert n_steps >= 1, f"n_steps must be >= 1, got {n_steps}"
    assert t_start >= 0, f"t_start must be >= 0, got {t_start}"
    assert t_start >= n_steps, (
        f"t_start {t_start} < n_steps {n_steps}: the chain would contain duplicate or "
        "non-positive timesteps. Lower infer_steps or raise infer_t_start."
    )
    out = [t_start - (i * t_start) // n_steps for i in range(n_steps)]
    for i in range(len(out) - 1):
        assert out[i] > out[i + 1], f"timestep chain not strictly descending: {out}"
    assert out[-1] > 0, f"last model input must be > 0 (0 is the endpoint): {out}"
    return out


def inference_chain(t_start: int, n_steps: int) -> List[tuple]:
    """The (t_current, t_next) pairs actually handed to `ddim_step`, in order.

    This is the object every audit prints, because it is the thing a reader needs to
    check: with n_steps=2 the two model calls are (49->25) and (25->0), NOT (49->0)
    followed by an identity.
    """
    ts = make_inference_timesteps(t_start, n_steps)
    return [(t, ts[i + 1] if i + 1 < len(ts) else 0) for i, t in enumerate(ts)]


def make_diffusers_timesteps(t_start: int, n_steps: int) -> List[int]:
    """diffusers' own `set_timesteps` construction, restricted to the truncated band.

    `step_ratio = (t_start + 1) // n_steps` then `(arange(0, n) * step_ratio)[::-1]`,
    i.e. the reference's `step_ratio = 20 / step_num` formula (:509) generalised. The
    LAST ENTRY IS 0, so the last model call runs on a near-clean input and its step is
    the terminal identity. NOT used in production -- kept so the scheduler audit can put
    diffusers' layout side by side with `make_inference_timesteps` and show exactly
    where they differ.
    """
    assert n_steps >= 1 and t_start >= 0
    ratio = (t_start + 1) // n_steps
    raw = [(i * ratio) for i in range(n_steps)][::-1]
    return raw


def make_eval_timesteps_diffusiondrive(step_num: int, magic: int = 20) -> List[int]:
    """Reproduces the reference's roll_timesteps for ATTRIBUTION ONLY (P5).

    transfuser_model_v2.py:509-511 does `step_ratio = 20 / step_num` then
    `(arange(0, step_num) * step_ratio).round()[::-1]`, giving [10, 0] for step_num=2.
    Paired with diffusers' unit prev_timestep spacing this yields one live step and
    one no-op. Kept here so P5 can measure that claim instead of arguing it.
    """
    import numpy as np

    step_ratio = magic / step_num
    return (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(int).tolist()


def tradeoff_table(sched: TruncatedDiffusionSchedule, t_start: int, n_steps: int) -> List[dict]:
    """Per-transition exact weights for the PRODUCTION chain -- P1b's evidence table."""
    return [sched.step_weights(t_cur, t_next)
            for t_cur, t_next in inference_chain(t_start, n_steps)]


def cumulative_travel(sched: TruncatedDiffusionSchedule, t_start: int,
                      n_steps: int) -> List[dict]:
    """How far toward x0_hat the chain has travelled after each step, cumulatively.

    Each DDIM step is exactly `x <- w0*x0_hat + w1*x` (see `step_weights`), so per step
    the distance to x0_hat multiplies by w1. The chain therefore reaches
    `1 - prod(w1)` of the way after k steps. This is the number that makes "a step from
    t to t/2 travels about half the remaining way" checkable rather than hand-waved.
    """
    rows, remain = [], 1.0
    for t_cur, t_next in inference_chain(t_start, n_steps):
        w = sched.step_weights(t_cur, t_next)
        remain *= w["w1_xt"]
        rows.append({**w, "cumulative_travel": round(1.0 - remain, 8)})
    return rows


def recommend_t_start(sched: TruncatedDiffusionSchedule, anchor_err_norm: float,
                      lo: int = 0, hi: Optional[int] = None) -> dict:
    """Pick the inference start timestep whose noise scale best matches the anchor error.

    The reasoning (see doc/2026-09-16_P1b_diffusion_scheduler_audit.md): the denoiser sees
    noised-anchor as input. If the noise we add is FAR SMALLER than the anchor's own
    distance to the target, the input "looks cleaner than an anchor is", and there is
    no denoising work left for the later DDIM steps to do. So t_start should be where
    b_t = sqrt(1 - abar_t) is comparable to the typical anchor error.
    """
    hi = sched.train_t_max - 1 if hi is None else hi
    best_t, best_gap = lo, float("inf")
    table = []
    for t in range(lo, hi + 1):
        b_t = float(sched.b(t))
        gap = abs(b_t - anchor_err_norm)
        table.append({"t": t, "b_t": round(b_t, 8), "gap_to_anchor_err": round(gap, 8)})
        if gap < best_gap:
            best_t, best_gap = t, gap
    return {
        "anchor_err_norm": round(float(anchor_err_norm), 8),
        "recommended_t_start": int(best_t),
        "b_at_recommended": round(float(sched.b(best_t)), 8),
        "table": table,
    }
