# P3 — smoke test: does the TD arm survive the real train→ckpt→rollout path?

**Scope.** Run the TD arm end to end on GPU2 through the **real** harness — the real
training loop, the real checkpoint writer, the real rollout evaluator — at a size small
enough that a failure is cheap to find. P3 is not a performance measurement; it is the
first time the arm is driven by the same code the P6 comparison will use.

**Reproduce.**

```bash
cd ~/td_pusht && ./scripts/run_p3.sh      # ~5 min total, GPU2 only
```

---

## 1. Verdict

| gate | required | measured | status |
|---|---|---|---|
| no NaN in training | none | none (100/100 steps finite) | ✅ |
| `train.py --smoke` verdict | `SMOKE PASS` | `SMOKE PASS` | ✅ |
| no NaN/Inf gradient | none | none | ✅ |
| ckpt round-trip | writes + loads | 2 ckpts written, loaded by the evaluator | ✅ |
| rollout produces `final_coverage` | a number | `0.1305` | ✅ |
| records `policy_ms_per_step` | recorded | `29.99 ms` | ✅ |
| records `peak_gpu_mem_mb` | recorded | `1012.7 MB` | ✅ |
| GPU2 lock mutual exclusion (P0 item J) | exclusive | exclusive, falsified-tested | ✅ |
| GPU0 / GPU1 untouched | 0 change | 439 MiB / 21 MiB before **and** after | ✅ |

**The arm runs end to end.** Every stage of the real pipeline accepted it, including the
two places that would have failed loudly if the duck-typing contract were incomplete:
the evaluator's `model.vqvae_is_fit` read and its `model.vqvae.parameters()` walk (the TD
stub returns True and an empty list respectively, printed in the rollout log as
`vqvae_is_fit=True, vqvae_requires_grad=False`).

**`final_coverage = 0.1305` is not a performance result.** It is 2 episodes from a
checkpoint trained for 93 steps on 6 episodes. It proves the evaluator can consume the TD
arm's output, and nothing about the policy's quality.

---

## 2. Two places the plan and the harness disagreed

Both were resolved in favour of the harness, without adding a deviation to `train.py`.

**2.1 `--smoke` writes no checkpoint, so P3 cannot have a smoke-only rollout.**

`train.py` sets `cfg.save_ckpt = False` under `--smoke` in **all three** data branches
(`:639`, `:665`, `:696`), and the only pre-training `save_ckpt(0)` call sits behind
`if not args.smoke and not args.resume:` (`:1083`). So the plan's "`--smoke` training +
`--smoke` rollout" cannot be a single checkpoint's journey.

Rather than add a fifth deviation to `train.py` for a smoke-only save, step 5 runs a
**tiny but real** 1-epoch / 6-episode training (`--max-episodes 6 --epochs 1
--ckpt-epochs 0,1`) through the ordinary checkpoint path. That is the better test anyway:
it exercises the checkpoint code P6 will actually use, not a special case. Cost: 31 s.

**2.2 The harness's smoke checklist cannot see the module this project adds.**

`train.py:1319-1355` verifies gradients on seven named parameters: `gpt.wte`, `gpt.wpe`,
`gpt.h[0].attn.c_attn`, `gpt.h[0].mlp.c_fc`, `gpt.ln_f`, `bin_head.net[0]`,
`offset_head.net[0]`. That list predates the TD arm and **does not include `denoiser`** —
so a TD run could print `SMOKE PASS` with the entire 46-parameter denoiser disconnected
from the loss, and its own output would never say so.

`scripts/audit_td_grads.py` closes that from outside: it walks **every** parameter, groups
by module, and gates the four modules that must be on the loss path.

| module | params | with grad | non-finite |
|---|---|---|---|
| `gpt` | 102 | 102 | 0 |
| `denoiser` | 46 | 46 | 0 |
| `bin_head` | 6 | 6 | 0 |
| `offset_head` | 6 | 6 | 0 |

Denoiser breakdown: `anchor_encoder` 6/6, `blocks` 36/36, `time_mlp` 4/4.

That audit's first version **passed while measuring nothing**: it grouped submodules by
`n.split(".")[1]`, which on `anchor_encoder.0.weight` yields `"0"`, so it printed four
vacuous `denoiser.0 … denoiser.3  0/0 PASS` lines. Fixed to split on `[0]`, and the check
now fails on an empty group rather than passing. Recorded because a green check that
measures nothing is more dangerous than a red one.

---

## 3. The GPU2 lock: the plan's path was wrong, and it mattered

P0 deferred its item 9 (lock mutual exclusion) to P3 because the script did not exist yet.
Writing it surfaced that **the path the plan specified is not the lock anyone else takes.**

| | path | exists? |
|---|---|---|
| plan's requirement | `~/audits/gpu2_rollout.lock` | **no** — `~/audits/` does not exist |
| the existing driver's actual lock | `~/sibling_pkg/audits/gpu2_rollout.lock` | yes |
| TD's implementation | `~/sibling_pkg/audits/gpu2_rollout.lock` | yes — the same file |

The existing `~/sibling_pkg/scripts/rollout_locked_gpu2.sh` declares `LOCK=audits/gpu2_rollout.lock`
— a **relative** path — so it resolves against the caller's cwd, which in practice is
`~/sibling_pkg`. `find ~ -maxdepth 4 -name gpu2_rollout.lock` returns **exactly one**
file, and it is that one. Had TD used the plan's absolute path, it would have locked a
**different inode**, and both drivers would have acquired "their own" lock and rolled out on
GPU2 simultaneously while each believed it was serialized — the precise failure the absolute
path was introduced to prevent.

`scripts/test_gpu2_lock.sh` guards this with four checks, and the first one is the
load-bearing one:

```text
[PASS] the lock we take IS the file the existing sibling_pkg driver takes
[PASS] second holder is REFUSED while the lock is held (flock -n fails)
[PASS] lock is acquirable again after release (the file is real and usable)
J RESULT: PASS
```

**The test was falsified to prove it has teeth.** Re-run with the plan's bogus path:

```
GPU2_LOCK=~/audits/gpu2_rollout.lock bash scripts/test_gpu2_lock.sh
  [FAIL] lock mismatch: theirs resolves to '~/sibling_pkg/audits/gpu2_rollout.lock',
         ours is '~/audits/gpu2_rollout.lock' <- mutual exclusion would be a fiction
  [PASS] second holder is REFUSED while the lock is held (flock -n fails)
  [PASS] lock is acquirable again after release (the file is real and usable)
  J RESULT: FAIL
```

Note what this shows: with the wrong path, the flock mechanics still pass (checks 2 and 3)
— the lock is real and it does exclude *itself*. Only check 1 catches that it excludes
nothing that matters. A test that only asked "does flock work" would have certified the
broken configuration.

**A second, unfixable hazard, recorded not solved.** `~/peer-project/scripts/auto_rollout.sh`
also targets `GPU=2` and takes **no lock at all** — it calls `eval_rollout` directly. No
lock in this project can exclude it. It cannot currently fire: its last log entry is
`2026-09-05 22:41:24`, and all 54 of its targets (28 + 26) either already have their
`rollout_seed0.json` marker or have no checkpoint on disk, so it is a read-only poller
waking every 60 s to find nothing. That is a property of its inputs, not of any lock — if
anyone drops a `ckpt_epoch*.pt` into those output dirs it fires on GPU2 unannounced.

---

## 4. Measured numbers

### 4.1 Smoke training — 100 steps, 6 episodes, batch 2

```
step   1  loss=2.8726  cls=1.138  off=1.735  grads all ok  mem alloc=0.73G reserved=2.65G
step 100  loss=2.5666  cls=0.356  off=2.211  grads all ok  mem alloc=0.73G reserved=2.88G
```

| item | value |
|---|---|
| verdict | **`SMOKE PASS`** |
| benchmark | 100 steps in **9.3 s** → **93 ms/iter** |
| peak VRAM | **2787 MB** |
| NaN/Inf gradients | none |
| `vqvae_is_fit` seen by training | `True` (no silent no-op) |

`cls` falls 1.138 → 0.356 across 100 steps, so the classification head is visibly moving.
The total loss is noisy because 100 steps over 6 episodes is a pipeline check, not a
convergence measurement.

### 4.2 Tiny real training — 1 epoch, 6 episodes

```
[ckpt] epoch 0 step 0  -> ckpt_epoch0_step0.pt
[epoch 0/1] val_loss=2.8948 (3s)
[epoch 1/1] val_loss=1.9235 (31s)
[ckpt] epoch 1 step 93 -> ckpt_epoch1_step93.pt
```

| ckpt | size |
|---|---|
| `ckpt_epoch0_step0.pt` | **201 MB** |
| `ckpt_epoch1_step93.pt` | **500 MB** |

Both were written and the evaluator loaded `ckpt_epoch1_step93.pt` successfully, so the
round-trip is exercised in the direction that matters (write by `train.py`, read by
`eval_rollout.py`).

**Disk note.** The plan budgeted ~417 MB per checkpoint. Measured: **500 MB** with
optimizer state, 201 MB without. With the keep-only-best policy implying ~3 ckpts per arm,
that is ~1.5 GB per arm rather than ~1.3 GB — still inside the 24 GB free, but the P6 disk
gate should be re-checked against 500 MB, not 417 MB.

### 4.3 Smoke rollout — 2 episodes, 2 envs, under the lock

| metric | value |
|---|---|
| `final_coverage_mean` | 0.1305 |
| `max_coverage_mean` | 0.2093 |
| `success_rate_095` | 0.0 |
| `episode_steps_mean` | 302.0 |
| `policy_ms_per_step_mean` | **29.99 ms** |
| `encoder_ms_per_step_mean` | 10.06 ms |
| `env_step_ms_per_step_mean` | 12.05 ms |
| `total_ms_per_step_mean` | **39.08 ms** |
| `peak_gpu_mem_mb` | **1012.7 MB** |
| protocol | `max_steps=302`, temporal ensemble W=5, `action_rng_seed=0` |

**These timings and the memory figure are NOT comparable to the n100 protocol.** They are
2 envs; the P6 protocol uses 10. `peak_gpu_mem_mb` in particular is env-count dependent, so
1012.7 MB here says nothing against the 9997 MB sibling reference. P4/P5 produce the
comparable numbers.

The rollout used the real 2-step chain: `_sample` walks `self.chain()` =
`[(49, 25), (25, 0)]`, so both model calls are genuine DDIM transitions, as established in
`2026-09-16_P1b_diffusion_scheduler_audit.md`.

---

## 5. Found during P3, to carry into P4/P5

**`eval_rollout.py` does not pin the TD noise generator.** The plan requires P5's 1-step
vs 2-step comparison to share an eps source, via `set_noise_generator(seed = 31337 + train_seed)`
following the `cond_gen` convention at `eval_rollout.py:324`. The evaluator never calls it.
Inference therefore falls back to `_gen()`'s lazy default: a single
`torch.Generator(device).manual_seed(31337)` created on first use.

Consequences, stated precisely:

- **P3 is unaffected.** The rollout is deterministic and reproducible; the seed is simply
  fixed at 31337 rather than offset by `train_seed`.
- **Across processes** the fallback happens to give both a 1-step and a 2-step run the same
  *initial* eps, because each process starts a fresh generator at 31337.
- **Within a process** the generator advances, and a 1-step chain draws eps once per policy
  call while a 2-step chain draws once per chain — so after the first call the two arms
  consume different eps. That divergence is inherent to comparing chains of different
  length, not a bug, but it means "same noise source" is only exactly true for the first
  call and the P5 write-up must say so.

This is a P5 requirement that is currently unmet, not a P3 failure. It needs an explicit
decision (pin it in the evaluator, or pin it per-run from the driver) before P5's numbers
mean what the plan wants them to mean.

**The harness's smoke checklist is blind to new modules by construction.** It enumerates
parameter names rather than walking the model. Any future head added to this repo will hit
the same gap; `scripts/audit_td_grads.py` is the reusable answer.
