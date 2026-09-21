# P1 — PushT action corpus, normaliser, and k-means anchors

**Scope.** Characterise the PushT action distribution, fit the action normaliser the TD
head will use, and produce the K=20 anchor artifact — then report the number that decides
what the head can possibly achieve: the **anchor floor**.

**Reproduce.**

```bash
# server, CPU only (0 GPU memory)
cd ~/td_pusht
CUDA_VISIBLE_DEVICES='' ~/miniconda3/envs/wemm-p1/bin/python -m patch_policy.anchors --k 20
CUDA_VISIBLE_DEVICES='' ~/miniconda3/envs/wemm-p1/bin/python scripts/report_p1.py
```

Outputs `anchors/pusht_action_anchor_audit.json` and `anchors/pusht_kmeans_K20.pt`.
`wemm-p1` has no `sklearn`, so k-means is a pure-torch implementation (`kmeans_torch`,
k-means++ init + Lloyd, `torch.Generator` seeded for cross-version reproducibility).

---

## 1. Corpus and split

| item | value |
|---|---|
| dataset | `~/patch_policy_datasets/pusht_dataset` |
| split | `split_episodes`, `randperm(seed=42)`, `train_fraction=0.9` |
| episodes | **185 train / 21 val** (asserted, matching the harness) |
| windows | 23 136 train / 2 514 val (obs W=5, act W=5) |
| chunks | **115 680 train / 12 570 val** (T=5 per window) |
| `train_idx` sha256 | `01d11cb29fb82ff33d67c3fb47b30967…` |

The chunk corpus follows the VQ-BeT arm's `--export-shared-vq` definition
(`train.py:736-742`): every **train** window is expanded by `_unpack_actions` into T
chunks. Window overlap is deliberately **not** de-duplicated, so each chunk's weight
equals the frequency with which it appears as a training target. Val windows never enter
the corpus.

Raw actions — **reported, not redefined** (the spec forbids redefining PushT actions):

```
pixel range    x [12.0, 511.0]   y [25.0, 511.0]
```

---

## 2. Normaliser (`LinearNormalizer`, mode `limits`)

Ported from `base/patch_policy/utils/normalizer.py`. Fit on the **train split only**.
`limits` mode maps `[min, max] → [-1, 1]` exactly.

| field | value |
|---|---|
| `scale` | `[0.004008, 0.004115]` repeating over `(W,A)` |
| `offset` | `[-1.048096, -1.102881]` repeating over `(W,A)` |
| `input_min` | `[12.0, 25.0, …]` |
| `input_max` | `[511.0, 511.0, …]` |
| `input_mean` | `[228.69, 292.46, …]` (drifts slightly along W, as expected) |
| `input_std` | `[102.33, 97.60, …]` |
| norm train range | `[-1.0, 1.0]` |
| norm val range | `[-0.99599195, 0.90380764]` |
| **`frac outside [-1,1]`** | **`0`** ← required to be 0 |

Two things this establishes:

- The train split lands in `[-1,1]` **exactly**, so `clip_sample=True` (§P1b 6) is exactly
  "clamp to the legal action range" rather than an arbitrary box.
- Val never exceeds the train range, so the normaliser does not extrapolate on held-out
  data. Val's max is `0.9038`, i.e. val stays inside the box with margin.

The scale is stored over `W*A = 10` as a **1-D** tensor: `normalizer._normalize` does
`x.reshape(-1, scale.shape[0])`, which requires a 1-D scale. (A 2-D scale would reshape
to the wrong width — this is a real constraint from the ported code, not a preference.)

---

## 3. k-means anchors, K=20

Fitted on the normalised **train** chunks, flattened to `115680 × (W·A)`. Anchors are
stored in **normalised** space as a `[20, 5, 2]` buffer with `requires_grad=False` — the
same space the diffusion and `mode_distance` operate in.

| item | value |
|---|---|
| artifact | `anchors/pusht_kmeans_K20.pt` |
| sha256 | `44762e64a540ccf5fb4ebff271f4ad7195d1f5e3675f5b62266a95ae711866d7` |
| inertia | 25 256.8515625 |
| min pairwise anchor distance | 0.34809 |

### 3.1 Nearest-anchor error (normalised)

Reference metric from DiffusionDrive's `LossComputer`: L2 over `A`, mean over `W`.

| K | split | mean | median | P90 | P95 | max |
|---|---|---|---|---|---|---|
| 8 | train | 0.23060 | 0.22396 | 0.34873 | 0.39596 | 0.76453 |
| 8 | val | 0.23302 | 0.22715 | 0.35151 | 0.38725 | 0.63963 |
| 16 | train | 0.16619 | 0.16172 | 0.25893 | 0.28837 | 0.60969 |
| 16 | val | 0.16528 | 0.16329 | 0.26204 | 0.28299 | 0.44873 |
| **20** | **train** | **0.15008** | **0.14510** | **0.23337** | **0.26710** | **0.60969** |
| **20** | val | 0.14757 | 0.14379 | 0.23563 | 0.26408 | 0.44873 |
| 32 | train | 0.12506 | 0.11999 | 0.19481 | 0.22373 | 0.47369 |
| 32 | val | 0.12664 | 0.12315 | 0.19500 | 0.22590 | 0.42334 |

Train and val agree closely at every K, so the anchors are not overfit to the train
split. K=8/16 are audit-only and must not be used for training.

### 3.2 Anchor floor — the ceiling on this method

**If the head did nothing but output the nearest anchor**, the raw-pixel L1 error would be:

| K=20 | mean | P50 | P90 | max |
|---|---|---|---|---|
| train | 23.492 px | **22.588 px** | 36.801 px | 102.878 px |
| val | 23.158 px | 22.320 px | 37.424 px | — |

This is the number P2's pass criterion has to be read against. The plan's gate was
`P50 ≤ 0.15` in normalised units; measured **0.14510**, so K=20 **passes, but only just**
— by 3%.

**The floor is the real constraint on this project.** A K=20 anchor is on average 22.6 px
away from the target, while PushT success requires `final_coverage ≥ 0.95`. Every py
improvement the head makes must come from *refinement on top of* the anchor, because the
anchor alone cannot be accurate enough. Two further observations:

- **Raising K has sharply diminishing returns.** K=16 → 20 buys 0.16172 → 0.14510 (10%);
  20 → 32 buys 0.14510 → 0.11999 (17%) at 60% more modes. K=32 does not clear the problem.
- **Low coverage at small τ.** Fraction of train chunks whose nearest anchor is within τ:
  `τ=0.02 → 0.0027`, `τ=0.05 → 0.0478`, `τ=0.10 → 0.2470`, `τ=0.20 → 0.7950`. Only 25% of
  chunks have an anchor within 0.10. Train/val agree throughout (`0.2542` at τ=0.10), so
  this is a property of the distribution, not of the fit.

### 3.3 Cluster occupancy and mode ambiguity

| item | value |
|---|---|
| smallest cluster | 2 363 chunks |
| largest cluster | 11 143 chunks |
| mean | 5 784.0 chunks |
| **dead modes** | **0** |
| share range | 0.0204 – 0.0963 |

Per-anchor counts (`[4538, 5045, 7474, 2363, 8725, 5267, 5241, 6304, 3444, 7271, 2585,
5768, 3255, 6994, 7973, 8337, 8386, 11143, 2964, 2603]`): a 4.7× spread between smallest
and largest, but **no dead mode**, so all K=20 candidates receive gradient. That matters
for the mode-classification loss — a dead mode would be an anchor the head can never
select.

**Mode ambiguity.** The first version of this metric (`second / nearest > 0.9`) was
vacuous: `second ≥ nearest` by definition, so it reported 1.0000 unconditionally. The
correct phrasing asks how *often the nearest is not clearly nearest*:

| item | value |
|---|---|
| `second/nearest` median | 1.662 |
| `second/nearest` P10 | 1.083 |
| ambiguous (within 1.1×) | **0.1178** |

Median 1.662 means the nearest anchor is typically well separated. But **11.8% of chunks
have a second anchor within 1.1× of the nearest**, and min pairwise anchor distance is
0.34809 — so for roughly one chunk in eight the "correct" mode is genuinely ambiguous.
That sets a realistic ceiling on mode accuracy: expecting 100% is not reasonable, and a
mode-accuracy figure in the 0.85–0.95 band is consistent with the geometry rather than
evidence of a defect.

---

## 4. Inferred inference start timestep

`infer_t_start` is **derived from the measured anchor error**, not copied from the
reference's 8. The rule: the denoiser's input is `add_noise(anchor, ε, t)`, so `t` should
be where `b_t = √(1-ᾱ_t)` is comparable to the anchor's own error. If the added noise is
far smaller than the anchor's existing error, the input "looks cleaner than an anchor is"
and the later DDIM steps have no denoising work to do.

```
anchor error (normalised, median)   0.14510
recommended infer_t_start           49  ->  b_t = 0.09425
    t= 0  b_t 0.01000   gap 0.13510
    t= 8  b_t 0.03159   gap 0.11351
    t=10  b_t 0.03537   gap 0.10973
    t=20  b_t 0.05195   gap 0.09315
    t=30  b_t 0.06691   gap 0.07818
    t=40  b_t 0.08136   gap 0.06374
    t=49  b_t 0.09425   gap 0.05084
SATURATED AT TOP OF BAND: True
```

**This saturates and the finding must not be papered over.** `t=49` is the largest
timestep in the training band, and `b_49 = 0.09425` is still **below** the anchor's median
error of `0.14510`. The consequences are analysed in
[`2026-09-16_P1b_diffusion_scheduler_audit.md`](2026-09-16_P1b_diffusion_scheduler_audit.md) §8; in one line: the head
is doing **correction on top of a mode anchor, not full-band denoising.** Also note the
reference's `t = 8` (`b_8 = 0.0316`) would have been 4.6× below the anchor error — a
direct argument for deriving this value rather than inheriting it.

Two unit bugs found and fixed while getting here, both worth recording because either
would have produced a plausible-looking wrong answer:

1. **k-means flattening.** `norm_tr.reshape(N, -1)` folded the window axis `T` into the
   feature axis (giving 50 features instead of 10) and then failed to reshape back to
   `[K, W, A]`. Fixed to `reshape(N*T, W*A)`.
2. **Unit mismatch in `recommend_t_start`.** It was passed `anchor_only_l1_p50`
   (**pixels**, 22.6) to compare against `b_t` (**normalised**, 0.094) — a ratio of ~233,
   which saturates the search at `t=49` for a completely spurious reason. Now passed
   `nearest_dist_median` (normalised). The saturation reported above is from the corrected
   version.

---

## 5. Verdict

| gate | required | measured | status |
|---|---|---|---|
| split matches harness | 185 / 21 episodes | 185 / 21 | ✅ |
| `frac outside [-1,1]` | `0` | `0` | ✅ |
| norm train range | exactly `[-1,1]` | `[-1.0, 1.0]` | ✅ |
| anchor floor P50 (normalised) | `≤ 0.15` | **0.14510** | ⚠️ **passes by 3%** |
| dead modes at K=20 | `0` | `0` | ✅ |
| val never fitted on | anchors + normaliser from train only | ✅ | ✅ |

**One-line summary:** the normaliser is exact and the K=20 anchor set is healthy (no dead
modes, train/val agreement), but its **floor is 22.6 px against a 0.95-coverage success
criterion** — so the anchor cannot solve PushT by itself, ~12% of chunks have an
ambiguous nearest anchor, and the P1-derived `infer_t_start` saturates at the top of the
training band.
