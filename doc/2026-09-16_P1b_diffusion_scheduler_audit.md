# P1b — Diffusion scheduler audit

**Scope.** Validate the self-implemented truncated-diffusion schedule and DDIM step in
`patch_policy/diffusion.py` against `diffusers` field by field, and settle the
`num_inference_steps=2` question that the reference implementation leaves ambiguous.

**Why self-implemented at all.** The environment every arm actually runs in (`wemm-p1`,
activated by `rollout_locked_gpu2.sh`) has neither `diffusers` nor `sklearn`, and
installing into it would perturb the existing VQ-BeT/AdaLN arms. So the beta schedule and
the DDIM step are ~40 lines of torch here, and this audit is what makes that acceptable:
the re-implementation is checked against the library, not against a reading of it.

**Reproduce.**

```bash
# server, patch-policy env (has diffusers). NOT wemm-p1 -- it does not.
cd ~/td_pusht
CUDA_VISIBLE_DEVICES='' ~/miniconda3/envs/patch-policy/bin/python scripts/audit_scheduler.py
```

Result: `ALL CHECKS PASSED` over 52 checks; JSON written to
`outputs/diffusion_scheduler_audit.json`. `diffusers==0.36.0`. Pure CPU.

Contract tests that do **not** need diffusers and run in `wemm-p1`:
`python tests/test_td_contract.py` → `ALL 30 CHECKS PASSED`.

---

## 1. What the reference actually does

Read from `base/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`.
Every claim below is a line cite, not an inference.

| # | Fact | Source |
|---|---|---|
| 1 | `DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample")` — **no other kwargs** | `:400-404` |
| 2 | `self.diffusion_scheduler.set_timesteps(1000, device)` | `:508` |
| 3 | `step_ratio = 20 / step_num`; `roll_timesteps = (arange(0, step_num) * step_ratio).round()[::-1]` → **`[10, 0]`** for `step_num=2` | `:509-511` |
| 4 | noise level `trunc_timesteps = ones(bs) * 8`, i.e. `add_noise(t=8)` | `:518` |
| 5 | first denoiser call is labelled `t=10`, not 8 | `:510` loop vs `:518` |
| 6 | `x_boxes = torch.clamp(img, min=-1, max=1)` before encoding the denoiser query | `:519` |
| 7 | the loop body's `for k in roll_timesteps` calls `scheduler.step(..., timesteps=k)` | `:530-540` |

Two consequences worth stating plainly:

- **Fact 1 with no overrides means the reference inherits diffusers' defaults**, and two
  of those defaults are load-bearing: `clip_sample=True` (see §6) and
  `set_alpha_to_one=True` (see §7). This audit's first version passed
  `clip_sample=False, set_alpha_to_one=False` by hand and produced three false FAILs. A
  parity audit has to *inherit* the reference's defaults, not restate its own guesses.
- **Fact 3 means `roll_timesteps` is not diffusers' `set_timesteps` output.** It is
  hand-rolled and has nothing to do with `num_inference_steps`. The `[10, 0]` values go
  into `step()` as *timesteps*, and `step()` then computes its own `prev_timestep`.

### 1.1 `prev_timestep` is always `t - 1`

`DDIMScheduler.step` sets

```python
prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
```

With fact 2 (`num_inference_steps = 1000`) that is `1000 // 1000 == 1`, **always**. So the
reference's `[10, 0]` loop performs:

| model call | input label | `prev_timestep` | `alpha_bar_prev` | travel |
|---|---|---|---|---|
| #1 | `t = 10` | `9` | `0.99887723` | **0.0525** |
| #2 | `t = 0` | `-1` → `final_alpha_cumprod = 1.0` | `1.0` | **1.0000** |

Call #1 moves 5.25% of the way. Call #2 has `sqrt(1 - a_prev) == 0`, so it returns the
model's output verbatim — it is a terminal identity, not a transition. **The reference's
"2-step" inference contains exactly one live DDIM transition.**

That is API semantics, not a bug that can be patched: `step()` derives its own spacing
from `num_inference_steps`, so no choice of `roll_timesteps` can widen it. A genuine
2-step chain from the same start would be `[(10,5), (5,0)]` with travels `0.2845`,
`0.6048` — which the reference cannot express.

---

## 2. Item-by-item parity table

Ours vs `diffusers`, reference constructed exactly as fact 1.

| Item | Ours | diffusers | max abs diff | Verdict |
|---|---|---|---|---|
| `betas` | `linspace(√b₀,√b₁,1000,float32)²` | same | `0.000e+00` | bit-identical |
| `alphas_cumprod` | `cumprod(1-β)` | same | `0.000e+00` | bit-identical |
| `add_noise` t∈{0,1,8,10,25,49} | `√ᾱ·x₀+√(1-ᾱ)·ε` | same | `0.000e+00` | bit-identical |
| `final_alpha_cumprod` | `1.0` | `1.0` (`set_alpha_to_one=True` default) | — | identical |
| `clip_sample` / `_range` | `True` / `1.0` | `True` / `1.0` (defaults) | — | identical |
| `steps_offset` | `0` | `0` (default) | — | identical |
| `step(t→t-1)` t∈{49,25,10,1,0} | explicit-span step | `step()` | `0.000e+00` | bit-identical |
| `step` with wild `x₀` (t=49…0) | clamp then mix | `step()` | `0.000e+00` | bit-identical |
| wide-span step vs closed form | `w₀x̂₀+w₁x_t` | (no analogue) | `≤3.6e-07` | fp32 exact |
| termination at `t_prev<0` | returns `x̂₀` | `step()` | `0.000e+00` | bit-identical |

The float32 detail: computing betas in float64 and casting later is mathematically the
same but **not** bit-identical (`max|diff| 3.7e-9`). That is numerically irrelevant and
still worth removing — "agrees to within float32 rounding" invites the question of what
*else* differs. The linspace and the square are now both float32, as diffusers does.

The wide-span step has no diffusers analogue because `step()` cannot produce it (§1.1);
it is verified against the closed form instead, which is the stronger check anyway.

---

## 3. The mandated 2-step chain

`infer_t_start = 49`, `infer_steps = 2`. Printed by `describe_chain()` in every P2/P3 log:

```
  inference chain: infer_steps=2 infer_t_start=49
    model call #1: t_current =  49  t_next =  25  | alpha_bar 0.99111623 -> 0.99645531  | travel 0.3690
    model call #2: t_current =  25  t_next =   0  | alpha_bar 0.99645531 -> 0.99989998  | travel 0.8321
```

| call | `t_current` | `t_next` | `ᾱ_cur` | `ᾱ_next` | `w₀` | `w₁` | travel | cumulative |
|---|---|---|---|---|---|---|---|---|
| #1 | 49 | 25 | 0.99111623 | 0.99645531 | 0.369368 | 0.631670 | 0.3690 | 0.3683 |
| #2 | 25 | 0 | 0.99645531 | 0.99989998 | 0.832272 | 0.167976 | 0.8321 | 0.8939 |

**Both calls are real transitions**, both are non-trivial (>2% travel — actually ≥36%),
and **the chain never contains `t = 0`**: 0 is the endpoint reached *by* call #2, not an
input. This is the property the reference lacks and the reason the production chain is
built this way rather than by copying `[10, 0]`.

The construction is integer floor arithmetic (`t_start - (i*t_start)//n_steps`), not float
rounding — `round(24.5)` is 24 under banker's rounding and 25 under half-up, and this
number ends up in a published table.

For comparison, in the same tree:

| layout | n=2 | contains `t=0`? |
|---|---|---|
| production (`make_inference_timesteps`) | `[49, 25]` | no |
| diffusers' `set_timesteps` layout | `[25, 0]` | **yes — that call is the no-op** |
| reference `roll_timesteps` | `[10, 0]` | **yes — and its two calls are 5.25% + identity** |

---

## 4. `alpha_cumprod` and `b_t` across the truncation band

`b_t = sqrt(1 - ᾱ_t)` is the noise scale actually present at `t`. Training band is
`t ~ U{0..49}` — **inclusive of 0**, i.e. `torch.randint(0, 50, ...)`
(`transfuser_model_v2.py:465`), not `1..50`.

| `t` | `ᾱ_t` | `b_t` |
|---|---|---|
| 0 | 0.99989998 | 0.010001 |
| 1 | 0.99979734 | 0.014236 |
| 8 | 0.99900216 | 0.031589 |
| 10 | 0.99874932 | 0.035365 |
| 20 | 0.99730128 | 0.051949 |
| 25 | 0.99645531 | 0.059537 |
| 30 | 0.99552268 | 0.066913 |
| 40 | 0.99338090 | 0.081358 |
| 49 | 0.99111623 | 0.094254 |

Even at the top of the band the noise scale is ~0.094, i.e. under 10% of the signal —
this is a *truncated* diffusion, and §8 shows it matters.

---

## 5. `||x_before − x_after||` and the oracle ceiling

Measured on the **real P1 anchors** (`anchors/pusht_kmeans_K20.pt`, sha256
`44762e64a540ccf5…`) with `x̂₀ := the anchor itself`, i.e. an **oracle denoiser**. This
isolates the chain's geometry from any trained network: it is the best case.

| call | `t → t_next` | `‖x_before − x_after‖` | `‖x_after − x₀‖` | ratio | `w₁` |
|---|---|---|---|---|---|
| #1 | 49 → 25 | 0.475105 | 0.813392 | 1.584013 | 0.631670 |
| #2 | 25 → 0 | 0.676869 | 0.136535 | 5.957409 | 0.167976 |

`‖x_t0 − x₀‖ = 1.288424` initially → **`0.136535` remaining (10.6%)**.

The step is `x ← w₀x̂₀ + w₁x`, so `x_after − x₀ = w₁(x_before − x₀) + (w₀+w₁−1)x₀`. It is
**affine, not exactly convex**: `w₀ + w₁ = 1.00103815` at (49,25) and `1.00024796` at
(25,0), so the distance ratio lands within ~0.1% of `1/w₁` rather than on it. The audit
prints those deviations instead of tuning a tolerance until it passes; the exact affine
identity is checked separately against the closed form (max diff `3.6e-07`, §2).

**The ceiling this implies, stated as a limitation:** because the last step retains
`w₁ = 0.168` of the noisy `x_t`, **even a perfect denoiser cannot return the target
exactly.** The oracle residual is `L1 = 0.007510` in normalised units ≈ **1.9 px**
(the normaliser's scale is ≈0.004008 x-units/px). So a 2-step chain contributes roughly
2 px of irreducible error, against the K=20 anchor floor of 22.6 px (§8). The chain
length is **not** the binding constraint; the anchor error and the network are.

---

## 6. `clip_sample` — a real operation, found missing on the first pass

diffusers defaults to `clip_sample=True, clip_sample_range=1.0`, and the reference passes
neither, so **it clamps `x̂₀` to [-1,1] on every DDIM step.** The first version of
`ddim_step` did not. Order matters and is now replicated exactly:

```python
eps_hat = (x_t - sqrt(a_cur) * x0_hat) / b_cur   # from the UNCLIPPED prediction
x0_use  = clamp(x0_hat, ±clip_sample_range)      # then clamp
x_prev  = sqrt(a_prev) * x0_use + sqrt(1-a_prev) * eps_hat
```

Clamping *before* forming `eps_hat` would instead treat the clamped value as the model's
output and land elsewhere. Parity, with `x̂₀` deliberately scaled ×3 (range ±2.70):

| `t` | ours vs `diffusers` | divergence if `clip_sample=False` |
|---|---|---|
| 49 | `0.000e+00` | `1.693e+00` |
| 25 | `0.000e+00` | `1.697e+00` |
| 10 | `0.000e+00` | `1.699e+00` |
| 1 | `0.000e+00` | `1.700e+00` |
| 0 | `0.000e+00` | `1.700e+00` |

So the flag is not cosmetic: omitting it changes the step by up to ~1.7 in normalised
units — roughly 425 px. It is exposed as `td.clip_sample` / `td.clip_sample_range`,
defaulting to the reference's values.

Semantically this is well-behaved here: the normaliser's `limits` mode maps the train
split to *exactly* [-1,1], so clamping to [-1,1] is precisely "clamp to the legal action
range". Note it does **not** guarantee the returned sample stays in range — lowering
`x̂₀` *raises* `eps_hat`, and the sum can exceed 1. That is diffusers' behaviour too, and
the contract test asserts the property that actually holds:

```
step(clip on) - step(clip off) == sqrt(a_prev) * (clamp(x0) - x0)     max|residual| = 9.5e-07
```

---

## 7. Termination at `t = 0`

`t_prev < 0` (past the chain's end) → `ᾱ_prev = final_alpha_cumprod = 1.0` → `√ᾱ_prev = 1`
and `√(1-ᾱ_prev) = 0`, so the step returns the model's output with **no residual noise**.

Two distinct claims, both checked:

| claim | evidence |
|---|---|
| `t=0 → prev<0` matches diffusers exactly | `max|diff| = 0.000e+00` |
| both return the model's output with zero residual noise | `max|theirs − x̂₀| = 0.0` and `max|ours − x̂₀| = 0.0` |
| `prev<0` from arbitrary `t` also returns the output exactly | `t ∈ {1,9,10,25,49}` → `0.0` each |
| which tensor comes back is `clamp(x̂₀)`, not `x̂₀` unconditionally | when `x̂₀` is out of range the two differ; both forms are asserted |

**The production chain never takes this branch.** Its last transition is `(25, 0)`, and
`t_next = 0` reads `alphas_cumprod[0] = 0.99989998`, giving a genuine 0.8321-travel step.
The `-1` fallback belongs to the reference's `[10, 0]` attribution mode, where `t=0` is
the last *model call*. This distinction is the whole point of §1.1 and §3.

An earlier assertion in this audit said the terminal step "returns `x̂₀` exactly" without
qualification. That is true only for in-range `x̂₀`; with `clip_sample` on, an out-of-range
`x̂₀` comes back clamped. The docstring and the tests now state the precise version.

---

## 8. Why `infer_t_start = 49`, and the finding that must not be papered over

`infer_t_start` is **derived from P1's measured anchor error**, not copied from the
reference's 8:

```
anchor error (normalised, median)            0.14510        # from P1, K=20, full train split
recommended infer_t_start                    49  ->  b_t = 0.09425
    t= 0  b_t 0.01000   gap 0.13510
    t= 8  b_t 0.03159   gap 0.11351
    t=10  b_t 0.03537   gap 0.10973
    t=20  b_t 0.05195   gap 0.09315
    t=30  b_t 0.06691   gap 0.07818
    t=40  b_t 0.08136   gap 0.06374
    t=49  b_t 0.09425   gap 0.05084
SATURATED AT TOP OF BAND: True
```

**The recommendation saturates: `t = 49` is the largest timestep in the training band,
and its noise scale `b_49 = 0.09425` is still smaller than the anchor's own median
distance to the target (`0.14510`).**

Read that carefully, because it changes what this head is:

> The denoiser's input is `add_noise(anchor, ε, t)`. Even at maximum `t`, the noise added
> is *smaller than the error the anchor already has*. So the network is not primarily
> being asked to denoise — it is being asked to **correct**. The anchor supplies the
> mode and most of the trajectory; the diffusion steps refine it.

Consequences, stated rather than hidden:

1. **The reference's `t = 8` would have been much worse here.** `b_8 = 0.0316` is 4.6×
   below the anchor error — the input would look cleaner than an anchor already is.
2. **Raising `infer_steps` buys little on the mechanism it is named for.** The leverage
   is in the correction `x̂₀`, and call #2 already travels 83% (§3).
3. **This is partly a K=20 artifact.** K=32 lowers the median anchor error to `0.11999`,
   which is still above `b_49`. The saturation is not resolved by a bigger K alone.
4. It is **not** an implementation defect — the weights, the chain and the termination are
   all verified against diffusers above. It is a statement about where the learning
   signal lives, and it should temper any expectation that this head behaves like
   full-band diffusion.

---

## 9. Deliberate deviations from the reference

| deviation | status |
|---|---|
| `roll_timesteps = [10, 0]`, i.e. one live step + an identity | **Not used in production.** Reproducible via `make_eval_timesteps_diffusiondrive` for attribution in P5. |
| `add_noise(t=8)` while labelling the first call `t=10` | **Not reproduced.** Our noise level and denoiser label are always the same `t`. The reference contradicts itself here (`:518` vs `:510`). |
| latent clamped to [-1,1] before the denoiser query (`:519`) | **Deliberately omitted.** That clamp exists only in `forward_test`, never in training, so it puts a train/inference shift into the reference's own pipeline. Our training path is unclamped; adding the clamp at inference only would import that mismatch for no measured benefit. One line behind a flag if P5 ever shows the denoiser extrapolating. |
| `clip_sample`, `set_alpha_to_one` | **Now reproduced** (they were silently missing before this audit). |

---

## 10. What this audit got wrong on its first pass

Recorded because it is the evidence that the checks are independent rather than tuned:

1. Constructed the reference scheduler with `clip_sample=False, set_alpha_to_one=False`
   → 3 false FAILs, including the false alarm that our `final_alpha_cumprod` was wrong.
   **The code was right; the audit's defaults were invented.**
2. Asserted the terminal step returns `x̂₀` unconditionally → caught the real
   `clip_sample` omission, but from a mis-stated premise (§7).
3. Compared termination at `t=10/49` by calling `diffusers.step(t_cur - 1)`, which makes
   diffusers' `prev = t_cur - 2` — an ordinary step, not a terminal one. diffusers can
   only reach `prev < 0` from `t=0` (§1.1). The comparison has to respect that asymmetry.
4. Asserted the clamped step "stays in range" — false for *both* implementations, since
   lowering `x̂₀` raises `eps_hat` (§6).
5. Asserted `‖x_after − x₀‖ = w₁‖x_before − x₀‖` exactly — the step is affine, not
   exactly convex; `w₀+w₁ = 1.00104` (§5).

Every one of these was a defect in the test, and each is now either fixed or replaced by
a property that actually holds.

---

## 11. Verdict

| requirement | status |
|---|---|
| beta schedule / `alphas_cumprod` == diffusers | ✅ bit-identical |
| `add_noise` == diffusers | ✅ bit-identical at t ∈ {0,1,8,10,25,49} |
| DDIM step == `diffusers.step` | ✅ bit-identical, incl. `clip_sample` and wide spans |
| real `num_inference_steps=2` with two live transitions | ✅ `[(49,25), (25,0)]`, travels 0.369 / 0.832 |
| no hard-coded `[10, 0]`; `[10, 0]` demoted to attribution | ✅ |
| `t=0` termination parity | ✅ exact, both against diffusers and as a property |
| `infer_t_start` derived from P1, not copied | ✅ 49 — **saturated; see §8** |
| deviations from the reference documented | ✅ §9, 4 entries |

**One-line summary:** the scheduler is a faithful port, including two defaults
(`clip_sample`, `set_alpha_to_one`) that were initially missing and are now verified
bit-identical against `diffusers`; the production chain is a genuine two-transition
truncated diffusion rather than the reference's one transition plus an identity; and the
P1-derived `infer_t_start` saturates, meaning this head is doing **correction on top of a
mode anchor**, not full-band denoising.
