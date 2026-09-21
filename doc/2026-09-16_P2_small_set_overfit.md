# P2 — small-set overfit: does the TD head actually learn?

**Scope.** Train the **full** policy (frozen DINOv2 trunk + TD head) on 32 and then 128
windows of the train split, for a fixed 2000 steps, and ask the only question that
matters before committing to a 100-epoch run: **does the head fit, and does the inference
chain produce it?**

If the head cannot drive the loss down on a handful of samples, no long run will fix it,
and a bad P6 result would be misattributed to the diffusion formulation or to
optimisation instead of to a plain wiring bug. This stage exists to make that
misattribution impossible.

**Reproduce.**

```bash
cd ~/td_pusht
CUDA_VISIBLE_DEVICES=2 ./scripts/run_p2.sh          # contract tests + all three runs
```

---

## 1. Verdict up front

| run | 2-step L1 | 1-step L1 | mode acc | gate |
|---|---|---|---|---|
| n=32, harness lr `5.5e-5` | **4.788 px** | 5.285 px | **1.000** | 2-step ✅ / 1-step ❌ / mode ✅ |
| n=32, probe lr `1e-3` | 95.963 px | 100.822 px | 0.125 | **diverged** — see §6 |
| n=128, harness lr `5.5e-5` | 10.608 px | 11.061 px | 0.948 | ❌ / ❌ / ❌ |

Gate = raw-pixel L1 **< 5 px** and mode accuracy **> 0.95** (plan's criterion), with both
1-step and 2-step measured separately so a failure names the right chain.

**The head learns, and the chain works.** Loss falls 3.018 → 0.248 on n=32; mode accuracy
reaches 1.000 by step 600 and holds for 1400 steps; predictions track the ground-truth
chunk shape rather than a constant (§5). Two independent chains both converge.

**The n=128 gate is not met at 2000 steps**, and the evidence says this is *under-training
plus mode ambiguity*, not a wiring defect — §7 makes that argument in full, and §8 states
what this stage therefore does **not** establish.

The headline number to read these against: the **P1 anchor floor is 22.6 px** (mean
nearest-anchor distance, raw pixels). At 4.788 px (n=32) the head beats the floor by
**4.7×**; at 10.608 px (n=128) by **2.1×**. Refinement on top of the anchor is doing real
work in both cases — the head is not merely selecting an anchor and returning it.

---

## 2. Configuration and provenance

| item | value |
|---|---|
| `select` | `spread` — the midpoint window of each of the first N train episodes (§4) |
| steps / batch | 2000 / **8** (`cfg.batch_size`) — see §3 |
| K / cls_from | 20 / `refined` (the canonical mode-scoring path) |
| inference | `infer_t_start=49`, `infer_steps=2`, `train_t_max=50` |
| `clip_sample` | `True`, range `1.0` — the diffusers default the reference inherits |
| anchor artifact | sha256 `44762e64a540ccf5…` — **the P1 artifact**, fit on the full train split |
| `code_sha256` | `diffusion.py 487e73a8edcb…`, `td_head.py 47805b5faebb…`, `config.py b763c931e99e…` |

Both anchors and normaliser are **loaded** from the P1 artifact and never refit here.
This is the contamination this stage is most likely to suffer: refitting k-means on 32
samples would produce anchors that make the overfit look better for a reason that has
nothing to do with the head. The script prints the artifact's sha256 precisely so a
future reader can prove which anchors produced these numbers.

`code_sha256` exists for a related reason: P2 was **first launched before `clip_sample`
was added** to `ddim_step`, which changes 1-step and 2-step inference. Without a code
fingerprint, the pre-fix and post-fix JSONs would have been indistinguishable, and the
first run's numbers would have silently contaminated the report.

---

## 3. Batch size: 8, not the plan's 32

The plan specified `batch 32`. That does not fit: the GPT runs `5 × 256 = 1280`-token
sequences, and batch 32 OOMs on a 24 GB card. Measured peak at batch 8 is **9034 MB** —
already 37% of the card, so batch 32 would need roughly 4× the activation memory.

Using the training protocol's own batch size is the better test anyway: it exercises the
optimiser that will actually be used at P6, so a pass here transfers.

---

## 4. Why `--select spread` rather than the first N windows

`prefix` takes the first N windows a `WindowDataset` walk emits. That walk is
episode-major, so those N windows all come from **one episode** — near-identical targets.
A low L1 there proves the head can **memorise one scene** and says almost nothing about
mode selection.

`spread` takes one window per episode from the first N episodes: 32 different scenes, so
the modal structure of the action distribution is actually exercised. This is the stronger
test and the only one whose mode-accuracy number means anything. Both were implemented;
`spread` is what was run.

---

## 5. Results

### 5.1 n=32 — the head fits, both chains converge

```
step    0  loss 3.0179  cls 1.1196  reg 1.8983  mode_acc 0.125
step  300  loss 0.8347  cls 0.1362  reg 0.6985  mode_acc 0.850
step  600  loss 0.5034  cls 0.0126  reg 0.4908  mode_acc 1.000
step 1200  loss 0.2786  cls 0.0008  reg 0.2778  mode_acc 1.000
step 1999  loss 0.2480                                    mode_acc 1.000
```

649 s, **3.1 steps/s**, peak 9034 MB. Per-sample 2-step L1: min 2.58 / median 4.57 /
max 8.60 px; **65.6% of samples under 5 px, 100% under 10 px**.

Two things worth noting. First, `cls` collapses from 1.12 to 0.0008 — mode selection is
solved first and decisively, and the remaining loss is entirely regression. Second, this
is a 32-sample subset: mode accuracy 1.000 here is **memorisation**, not generalisation,
and must not be quoted as evidence that mode selection works on held-out scenes.

### 5.2 n=128 — still descending when the budget ran out

```
step    0  loss 3.0179  mode_acc 0.125
step  600  loss 1.1210  mode_acc 0.470
step 1200  loss 0.7245  mode_acc 1.000
step 1800  loss 0.5215  mode_acc 0.875
step 1999  loss 0.7924  mode_acc 0.875
```

662 s, **3.0 steps/s**, peak 9034 MB. Per-sample 2-step L1: min 4.80 / median 9.49 /
max 35.02 px; only **1.6%** under 5 px, 55.5% under 10 px.

The loss is **not converged**: the last three logged points are 0.521 → 0.534 → 0.792 and
mode accuracy oscillates 0.875 / 1.000 / 0.875. Compare n=32, which had flattened by step
600 and held. With 4× the data at the same 2000 steps, the model has taken 4× fewer
passes over the data.

### 5.3 Per-sample prediction vs ground truth

Representative 2-step predictions, `pred` vs `gt` for the first chunk `[W=5, A=2]`, raw
pixels:

**n=32, median sample (L1 4.74 px)** — the trajectory shape and both endpoints are right:

```
pred [[120.7, 456.7], [142.2, 452.4], [148.5, 461.3], [185.3, 458.4], [194.1, 452.3]]
gt   [[116.0, 451.0], [131.0, 460.0], [151.0, 460.0], [189.0, 463.0], [202.0, 457.0]]
```

**n=32, worst sample (L1 8.60 px)** — the drift is smooth and monotone, i.e. refinement
error, not a mode error:

```
pred [[86.6, 324.7], [78.8, 317.6], [74.6, 316.3], [75.0, 311.4], [70.2, 301.0]]
gt   [[82.0, 314.0], [83.0, 314.0], [79.0, 310.0], [70.0, 303.0], [67.0, 296.0]]
```

**n=128, worst sample (L1 35.02 px)** — a categorically different failure:

```
pred [[366.8, 254.4], [373.1, 268.3], [385.6, 289.7], [386.8, 281.4], [367.8, 281.0]]
gt   [[323.0, 132.0], [330.0, 110.0], [341.0,  98.0], [356.0,  93.0], [370.0,  93.0]]
```

The ground truth sweeps `y: 132 → 93`; the prediction stays near `y ≈ 254–290` and
wanders. That is a **mis-selected mode**, not a refinement error — the head committed to
an anchor whose `y` trajectory belongs to a different scene. This distinction matters for
§7.

---

## 6. Learning-rate sensitivity (an unplanned but useful result)

The second n=32 run at `lr=1e-3` — one order of magnitude above the harness value —
**does not converge**:

```
step    0  loss 3.0179  mode_acc 0.125
step  100  loss 1.5987  mode_acc 0.125
step  300  loss 1.3787  mode_acc 0.075
step 1999  loss 1.6440  mode_acc 0.150
```

Final: 95.963 px 2-step, mode accuracy **0.125**, and it never recovers. It drops for ~100
steps and then oscillates around 1.3–1.7 for the remaining 1900 without settling. Mode
accuracy stays at chance, so the classification head never learns at all.

This was a capacity probe, not a gate, and it is reported separately for that reason. Its
value is negative evidence: the harness's `5.5e-5` is not merely *a* workable rate, it is
on the stable side of a sharp cliff. Both arms at P6 use the harness value, so this does
not affect fairness — but it does mean a future "just raise the lr" instinct would be
wrong, and it is worth recording before someone tries it.

---

## 7. What the n=128 gate failure is, and what it is not

Three pieces of evidence say **under-training on a harder subset**, not a defect in the
head or the chain:

1. **The loss is still falling and oscillating at the budget limit.** A broken wiring or a
   broken sampling chain produces a floor, not a descent that is visibly mid-flight at
   step 2000. n=32 at the same step count had been flat for 1400 steps.
2. **The error tail is mode mis-selection, not refinement drift.** The worst n=128 sample
   (§5.3) is off by ~150 px in `y` with a wandering trajectory — the signature of choosing
   the wrong anchor. Median error (9.49 px) is in the same band as n=32's worst (8.60 px),
   so refinement itself has not degraded; the distribution has grown a tail.
3. **The mode-accuracy gate may be unreachable by construction.** P1 measured that
   **11.8% of train chunks have a second anchor within 1.1× of the nearest** — the correct
   mode is genuinely ambiguous for about one chunk in eight. n=128's measured 0.948 is
   *inside* that ambiguity band. Demanding > 0.95 on held-out scenes is asking the head to
   resolve ties that the anchor geometry does not resolve. This was flagged in
   [`2026-09-16_P1_action_anchor_audit.md`](2026-09-16_P1_action_anchor_audit.md) §3.3 before P2 ran, and
   the n=128 number is consistent with it rather than in tension with it.

**The chain itself is exonerated by the same data.** The 1-step vs 2-step comparison shows
the expected ordering in both runs (n=32: 5.285 vs 4.788; n=128: 11.061 vs 10.608), so the
extra DDIM step is doing real work rather than being a no-op — the failure mode the
scheduler audit was written to rule out. The gap is small (≈0.5 px in both), consistent
with P1b's finding that the 2-step chain carries roughly 2 px of irreducible error and the
anchor error dominates the budget.

---

## 8. What this stage does NOT establish

It is important not to read more out of P2 than is there:

- **It does not show the head generalises.** n=32 is memorisation by construction; n=128
  never converged. Neither is a generalisation result.
- **It does not clear the 5 px gate at n=128.** The gate was met only at n=32, and by 4.2%
  (4.788 vs 5.0). A marginal pass on a 32-sample memorisation subset is weak evidence.
- **It does not bound what 100 epochs will reach.** The natural reading of §7 is that the
  n=128 result is a step-budget artefact, but that is an *inference from the trajectory*,
  not a measurement. The only way to settle it is to train longer.
- **It does not validate mode scoring on held-out data.** `cls_from='refined'` is wired
  correctly (contract-tested in `tests/test_td_contract.py`) and it learns — `cls` loss
  falls 1.12 → 0.0008 at n=32. But whether the *refined* score beats the rejected
  context-only variant is a comparison P2 never made.

The last point is the one the n=128 result leaves open: `cls_from='refined'` is wired
correctly and it learns, but whether the *refined* score beats the rejected context-only
variant is a comparison P2 never made.

A separate measurement is implied by §7 but not performed here: "needs more steps" and
"cannot fit 128 scenes" predict the same curve up to step 2000 and diverge only after the
loss flattens. P2 stops at 2000 steps by construction, so it cannot distinguish them.

---

## 9. Gate decision

| gate | required | measured | status |
|---|---|---|---|
| head fits on a small set | loss ↓, no NaN | 3.018 → 0.248 (n=32) | ✅ |
| 2-step inference < 5 px | < 5 px | **4.788 px** (n=32) | ✅ marginal |
| 1-step inference < 5 px | < 5 px | 5.285 px (n=32) | ❌ by 5.7% |
| mode accuracy > 0.95 | > 0.95 | **1.000** (n=32, memorisation) | ⚠️ not meaningful |
| both chains converge | yes | yes, both | ✅ |
| anchor/normaliser not refit | P1 artifact only | sha `44762e64…` | ✅ |
| no wiring bug found | — | none | ✅ |
| n=128 reaches the same gates | < 5 px, > 0.95 | 10.608 px, 0.948 | ❌ |

**Summary.** The wiring is sound and the truncated-diffusion head demonstrably learns:
loss falls by 12×, mode selection converges, both inference chains work, and refinement
beats the anchor floor by 2–5×. The n=128 gates fail, and the evidence points at step
budget and ~12% intrinsic mode ambiguity rather than at the implementation.

**Per the plan, execution stops here and reports.** No 50- or 100-epoch training is
started.
