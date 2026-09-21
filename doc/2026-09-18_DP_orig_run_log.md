# ORIGINAL Patch-Policy DP arm — run log (2026-09-18)

Third arm of the three-way comparison: **upstream's own Diffusion Policy**, run
as-is, *not* ported into our harness. This log records what it is, what was
deviated from upstream and why, and every measured number, in the same shape as
`2026-09-16_P4_run_log.md`.

Status: **training launched 2026-09-18 15:29:29 at `epochs=40`; the plan was cut
to 10 epochs at 16:15 (§8a). The live job is stopped by
[scripts/stop_dp_at_ep10.sh](../scripts/stop_dp_at_ep10.sh) the moment
`model_9.pt` lands (~17:00), then one n100 at ~46.6 min → ~17:55.**
Sections 6–9 below are pre-training measurements and are complete; §10 onward is
filled in when the run finishes.

---

## 0. Identity

| | |
|---|---|
| Arm | `patch_policy_dp_orig` — upstream `train_policy.py` + `train_pusht_diffusion.yaml` |
| Code | `~/dp_orig` — byte-faithful copy of `~/patch_policy` + **2 documented deviations** |
| Upstream reference | `~/patch_policy` — **read-only, verified untouched** (guard B) |
| Conda env | `patch-policy` (python 3.9) — this arm needs `diffusers`; `wemm-p1` does not have it |
| GPU | 2 (`CUDA_VISIBLE_DEVICES=2`) |
| Run dir | `~/dp_orig/outputs/pusht/dp_orig_40ep` |

The other two arms (TD, VQ-BeT) live in our own harness (`patch_policy/`,
`~/td_pusht`). This one deliberately does not.

---

## 1. What this arm is, and why it is not ours

The user's instruction was 「我需要的是 patch policy 的 diffusion，不要私自改结构」
and 「原版 Patch policy DP，单独跑一个」. The resolved design forks were:

| Fork | Decision |
|---|---|
| Which code path | **上游原样跑** — not ported into our harness, not TD-aligned |
| Where it lands | **复制到独立目录** `~/dp_orig`; `~/patch_policy` never written |
| ckpt cadence | **加最小改动每 10ep 存** |

**Consequence that must be stated in any comparison table: DP-vs-TD is NOT a
single-variable comparison.** Only TD-vs-VQ-BeT is (same file, one line of
`action_head` differs). This arm differs in *harness*:

| | TD / VQ-BeT | DP (this arm) |
|---|---|---|
| obs window | 5 | **2** |
| pred_horizon | 5 | **6** (5+2−1) |
| n_embd / n_head | 512 / 8 | **256 / 4** |
| batch_size | — | **256** |
| lr | — | **1e-4** |
| epochs (config default) | 100 | **1200** |
| scheduler | TD's audited DDIM/`scaled_linear`/`sample` | **`DDPMScheduler(100, squaredcos_cap_v2, clip_sample=True, epsilon)`** |
| inference | 1–2 steps | **full DDPM-100 + EMA weights** |
| action consumption | temporal ensemble | upstream eval: **open-loop chunk**; our rollout: temporal ensemble (§7) |

It also runs a *different class* (`models.diffusion_policy.diffusion_policy.DiffusionPolicy`,
a `TransformerForDiffusion` that consumes patch tokens directly) — there is no
GPT trunk in this arm at all.

---

## 2. The two source deviations (and nothing else)

Applied by [scripts/deploy_dp_orig.sh](../scripts/deploy_dp_orig.sh).

**1. `configs/env_vars/env_vars.yaml`: `save_path: ~/patch_policy` → `~/dp_orig`.**
Forced: `hydra.run.dir` is built from `${env_vars.save_path}`, so without this the
copy would write into the read-only reference. Changes **where**, not **what**.

**2. `train_policy.py`: save a ckpt at every `eval_on_env`, not only on a new best.**
Upstream's best-gate means a 40-epoch run keeps `model_final.pt` plus at most a
couple of mid-run bests — not enough to roll out ep10/20/30/40 separately, which
is what the standing protocol requires. `best_eval_metric` is still tracked and
still printed; only the `save_model(epoch)` call moves out of the `if`. The
recipe, optimizer, scheduler and EMA are untouched.

Generated diff of deviation 2 (the whole of it — 8 lines of context, CRLF preserved):

```
@@ -545,12 +545,15 @@
                 best_eval_metric = current_eval_metric
-                save_model(epoch)
             else:
                 print(
                     f"Eval metric {current_eval_metric} <= prev best {best_eval_metric}, "
-                    f"NOT saving ckpt"
+                    f"saving ckpt anyway"
                 )
+            # [dp_orig] deviation 2/2: store on EVERY eval_on_env, not only on a
+            # new best, so ep10/20/30/40 can each be rolled out afterwards.
+            # See scripts/deploy_dp_orig.sh for the rationale.
+            save_model(epoch)
```

`train_policy.py` bytes 31694 → 31916; `wc -l` 691 → 694; line ending preserved
as CRLF. The upstream tree is mixed-ending (`train_policy.py` 691/691 CRLF,
`train_pusht_diffusion.yaml` 94/94 CRLF, `diffusion_policy.py` 912/912 CRLF,
`env_vars.yaml` 0/18 LF), so the patch helper opens with `newline=""` and rebuilds
the anchor using the file's own terminator.

> A first version of the deploy script used `pathlib.read_text()/write_text()`,
> whose universal-newline translation silently rewrote `train_policy.py` CRLF→LF —
> an unlisted whole-file deviation (3 lines longer, 454 bytes *smaller*). Caught
> because the generated `diff -u` showed all 691 lines changed. Fixed; re-run
> with `FORCE=1`.

### Deployment integrity (the guards that make the copy self-proving)

* **Guard A** — `diff -r -q` over the whole tree (excluding `outputs/`,
  `checkpoints/`, `wandb/`, `__pycache__`) must report **exactly 2** differing
  files. It does. Every other byte is identical to upstream.
* **Guard B** — `find $SRC -type f -newer $MARKER` must be empty. It is:
  `~/patch_policy` was not written.

---

## 3. The three CLI deviations (not source changes)

From [scripts/run_dp_orig.sh](../scripts/run_dp_orig.sh), all passed as hydra
overrides:

| Override | Upstream | Ours | Why |
|---|---|---|---|
| `num_workers` | 23 | **0** | measured OOM, below |
| `num_env_evals` | 100 | **10** | inline evals are a progress signal only (§7); they exist to trigger the per-eval ckpt save |
| `num_final_evals` | 100 | **10** | same |
| `epochs` | 1200 | **40** | the user's request |

### `num_workers=0` — the first smoke died, this is why

With the config's `num_workers: 23` the run died on the **first batch**:

```
RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed
```

`ps --sort=-rss` showed three `pt_data_worker` at **20.5 / 20.5 / 20.0 GB** RSS.
`PushTDataset` materialises the whole dataset (~20 GB) inside **every** worker, so
23 workers demand ~460 GB on a 251 GB box and the OOM killer takes them out.
`/dev/shm` (126 G, 90 G free), RAM and ulimits were all fine — the 61 GB "used" in
`free(1)` was our own workers. **Upstream's DP config had never been run, so its
23 was never validated.** `num_workers=0` keeps exactly one copy, in the main
process (RSS 18.5 GB, observed). The failed log is preserved as
`smoke_dp_ep1/train.log.failed_oom_dataloader`.

---

## 4. Smoke (gate before the 40-epoch run)

`MODE=smoke` → `epochs=1 num_env_evals=10 num_final_evals=10 num_workers=0`,
GPU2, 2026-09-18 14:49:50 → 15:00:21.

Validated end to end: dataset loads → DINO frozen → `DiffusionPolicy` builds →
optimizer steps → train step advances → `model_init.pt` written → final eval runs
→ `model_final.pt` written. No `Killed` / `RuntimeError` / `Traceback` / NaN.

```
100%|██████████| 91/91 [08:23<00:00,  5.53s/it]
100%|██████████| 1/1 [09:09<00:00, 549.26s/it]
Train loss: 0.35081166143600756
Eval on goal 0/1, 10 episodes
```

Note `epochs=1` means `(0+1) % eval_on_env_freq == 1 ≠ 0`, so **the smoke exercised
no inline eval** — only training plus the final eval. The inline path was checked
statically instead: `train_policy.py` mentions `vqvae` exactly twice, both under
`if hasattr(saved_model, "vqvae_is_fit")`, so a `DiffusionPolicy` checkpoint (which
has no such attribute) skips it.

Artifacts: `model_init.pt` 73,683,490 B, `model_final.pt` 73,683,833 B,
`snapshot.pt` 147,309,357 B (model + optimizer state, **fixed filename, atomic
tmp+rename — overwritten each epoch, does not accumulate**),
`completion_idx_final.json` = 10 empty lists (0 successes at 1 epoch),
`eval_1_0.mp4` 36,821 B.

---

## 5. Per-step cost — the measurement that sets the budget

Rolled out the smoke checkpoint at n=10 (one batch of 302 steps) on GPU2 with
[scripts/eval_dp_orig.py](../scripts/eval_dp_orig.py):

```
[eval] batch 1/1: seeds [0..9] 302 steps, 279.6s, final_cov mean 0.0015,
       policy 879ms/step, enc 12ms, env 30ms
```

| term | ms/step | share of wall |
|---|---|---|
| **policy (DDPM-100 + EMA)** | **879.4 ± 38.5** | **95.4 %** |
| encoder (DINO) | 11.9 | 1.3 % |
| `env.step` (10 subproc envs) | 30.1 | 3.3 % |
| **total** | **921.3** | |

`final_coverage_mean 0.00154`, `max_coverage_mean 0.0869`, `success_rate_095 0.0`,
`episode_steps_mean 302.0`, `peak_gpu_mem_mb 244.2`.

The 879 ms is **not** arithmetic. A single denoiser forward here is ~1 GFLOP
(batch 10, 6 query tokens, 512 memory tokens, 8 layers × 256 dim), which on a 3090
is ~0.03 ms of real work; 8.79 ms per forward is ~300× off, i.e. the 100-step
Python loop is **kernel-launch bound** (~40 launches/step × 100 steps). So this is
a property of how upstream's `_predict` is written, not a tuning problem, and it
is not ours to change (「不要私自改结构」).

### Budget consequence (measured, not estimated)

| stage | unit | ×N | total |
|---|---|---|---|
| training, 1 epoch = 91 steps @ ~5.6 s/it | ~510 s | 10 | **1.42 h** |
| inline eval (10 ep, 302 steps, open-loop) | ~72 s | 1 (at ep 10) | 1 min |
| **n100 rollout (temporal ensemble)** | **46.6 min** | 1 ckpt | **0.78 h** |
| | | | **≈ 2.2 h** |

> The `×N` column was 40 / 4 / 4 before the 2026-09-18 amendment (§8a). At 10
> epochs only one inline eval ever fires — the one at epoch 10 that also triggers
> the save — and only one ckpt exists to roll out. The two `unit` figures are
> measurements and did not change; only their multiplicity did.

`episode_steps_mean 302.0` is exactly the cap, confirmed in source:
`PushWrapper.step` does `if self.step_idx > 300: done = True` *before*
`self.step_idx += 1`, so the first done is on call 302. (The success-`done` in
`PushTKeypointsEnv.step` is commented out upstream: `done = False  # coverage >
self.success_threshold`.)

---

## 6. ckpt naming, and the ep40/model_final identity

`save_model(epoch)` is gated on `(epoch+1) % 10 == 0`, and **filenames are
0-based**. A 40-epoch run therefore writes:

```
model_9.pt   model_19.pt   model_29.pt   model_39.pt     (= after ep 10/20/30/40)
model_init.pt                                          (= before ep 1)
model_final.pt
```

`model_39.pt` and `model_final.pt` are the **same weights**: the only thing between
them is the final eval and two saves; no training happens. So a 40-epoch run has
**four distinct rollout points, not five**. (That identity is now unobservable
rather than merely unchecked — see the amendment in §8a: the run stops at epoch
10, so neither `model_39.pt` nor `model_final.pt` is ever written.)

### What the 10-epoch cut leaves (§8a)

| file | written | note |
|---|---|---|
| `model_init.pt` | before epoch 1 | 73,683,490 B |
| **`model_9.pt`** | after epoch 10 | **the whole deliverable** — the only rollout point |
| `snapshot.pt` | every epoch, overwritten | model + optimizer state, 147,309,357 B, fixed name |
| `model_final.pt` | never | the process is stopped, not allowed to finish |
| `completion_idx_final.json`, `eval_*.mp4` | never | same |

**`save_model` is not atomic.** `train_policy.py:485/487` is a plain
`torch.save(cbet_model, "{}/model_{}.pt".format(...))` — no tmp+rename, unlike the
`snapshot.pt` path. A signal landing inside that write leaves a truncated file
carrying the correct name. This is why the 10-epoch stop is done by a watcher
that waits for a size-stable `model_9.pt` rather than by a timer: a partial file
cannot pass a two-sample stability test.

---

## 7. Protocol declarations (two, both load-bearing)

**(a) The reported rollout protocol is temporal ensembling, matching the other two
arms.** Upstream's own `eval_on_env` takes the `if use_diffusion:` branch, which
executes the whole 5-step chunk open-loop before re-planning — one policy forward
per 5 env steps. Our rollout takes one forward per env step and ensembles the last
5 chunks. Both are legitimate DP protocols, but they are **different** protocols,
and TD/VQ-BeT use temporal ensembling, so comparing DP-open-loop against
TD-ensemble would compare action-consumption schemes rather than action heads.
Cost of the aligned choice: **4.2×** (n100 46.6 min vs ~11 min open-loop).

**(b) The inline `final coverage mean` printed during training is a progress signal
only.** It is measured under upstream's open-loop protocol (a), so it must not be
compared with the numbers in §5/§8 or with the other arms. This is why
`num_env_evals` was shrunk rather than trusted: the cheap inline eval's only jobs
are to trigger the per-eval ckpt save (the save lives *inside* the eval branch, so
the eval cannot be switched off) and to show that training is not diverging. It is
not a substitute for the n100.

**(c) DP-vs-TD is not single-variable** — see §1.

---

## 8. Launch

```
=== run_dp_orig mode=full 2026-09-18 15:29:29 ===
  gpu=2 epochs=40 num_env_evals=10 num_final_evals=10 num_workers=0
  run_dir=~/dp_orig/outputs/pusht/dp_orig_40ep
  pid=2586672
  alive
```

Preflight at launch: disk 184 G free (80 % used, plenty — 6 ckpts ≈ 0.44 GB plus
one 147 MB snapshot); no compute apps on any GPU; no rollout of ours running; run
dir fresh. `+3:31` health check: `25/91 @ 5.60 s/it`, RSS 18.5 GB, no
`Killed`/`Error`/`Traceback`, `model_init.pt` written, GPU2 1695 MHz / 72 °C.

---

## 8a. Amendment 2026-09-18 16:15 — the plan is cut from 40 to 10 epochs

**Decision (user, verbatim):** 「修改一下训练计划，我们这个DP，只到10个epoch就行，就看
看速度和收敛效果」. This arm is a **cost-and-convergence probe**, not a final result,
so 40 epochs buy nothing that 10 do not already show: the per-epoch wall and the
policy-forward latency are epoch-count-independent, and the loss trajectory is
well resolved by 10 points.

**Why the live job was not killed on the spot.** At 16:12 the run was at epoch 5/40,
43:04 in. Killing there would have discarded all of it and produced no ckpt at all
(the first save point is epoch 10). Letting it reach the epoch-10 save costs ~42
min and yields exactly the ckpt the amended plan asks for. So: continue, and stop
at the ckpt.

**How it is stopped** — [scripts/stop_dp_at_ep10.sh](../scripts/stop_dp_at_ep10.sh),
detached on the server. It polls `model_9.pt` every 20 s and sends SIGTERM only
after the size has been **>70 MB and unchanged across two consecutive samples**,
then SIGKILLs if that is not enough after 10 s. A timer would not be safe here:
`save_model` is a bare `torch.save` with no tmp+rename (§6), so a signal inside
the write leaves a truncated file with a valid-looking name. A stability test
cannot be fooled by that. The watcher has a 2 h ceiling and, if it expires, kills
nothing and says so.

Two operational traps hit while launching it, both recorded because both cost a
round trip and neither is specific to this script:

* **`pgrep -f` self-matches the invoking `bash -c`.** The first attempt guarded the
  launch with `pgrep -f '[s]top_dp_at_ep10.sh'` — but the same command line also
  contained `chmod +x ~/td_pusht/scripts/stop_dp_at_ep10.sh`, i.e. the
  **unbracketed** path, so pgrep matched the wrapper itself and reported "already
  running". The watcher never started (the log file's absence was the tell, not
  the message). The bracket trick only defeats the match when the *only* mention
  of the name on that command line is the bracketed one.
* **A trailing foreground command after a background job hangs the ssh channel.**
  `... setsid nohup ./script > /dev/null 2>&1 < /dev/null & sleep 3; echo launched`
  left the `bash -c` wrapper alive indefinitely (it was still there two minutes
  later, holding a dead channel), and `ssh_run.py` blocked on it until its 120 s
  timeout. The recipe that works — and the one `run_dp_orig.sh` already used — puts
  the whole launch inside a subshell with nothing after it:
  `( ... setsid nohup ... > LOG 2>&1 < /dev/null & )`. The stray wrapper was
  killed by pid after confirming its command line, and the watcher (a session
  leader of its own, fds on `/dev/null`) was unaffected — as intended.

**State at the amendment** (measured, not projected):

| ep | train loss | epoch wall |
|---|---|---|
| 1 | 0.350812 | 564.5 s |
| 2 | 0.218037 | 530.7 s |
| 3 | 0.186358 | 517.5 s |
| 4 | 0.178590 | 512.8 s |
| 5 | 0.172354 | 510.3 s |

Monotone down; deltas −0.133 → −0.032 → −0.008 → −0.006. Five points is an
early-convergence reading, **not** a plateau verdict — the only verdict this arm
accepts is the n100 (§7(b)). Health: 0 hits for `Traceback|RuntimeError|Killed|nan`
in the whole log, GPU2 100 % util / 6904 MiB / 81 °C, GPU0 and GPU1 idle, disk 184 G
free. Epoch 1's loss `0.35081166176350564` matches the smoke's
`0.35081166143600756` to 8 significant figures — the recipe did not drift between
smoke and full run.

**Consequential changes:** `run_dp_orig.sh` `MODE=full` now defaults to `EPS=10`
(overridable); `run_dp_rollout.sh` is untouched and its ep20/30/40 entries now
print `[skip]` instead of running — expected output, not failure. Revised budget
is in §5; revised ETA is in §9.

---

## 9. Rollout plan (post-training, exclusive)

[scripts/run_dp_rollout.sh](../scripts/run_dp_rollout.sh) `MODE=full` — after the
§8a amendment that is **`model_9.pt` alone**, n=100, `seed=0`, **serial**, GPU2,
under `flock ~/sibling_pkg/audits/gpu2_rollout.lock` (absolute path, shared
with the sibling driver; a relative lock would let two drivers each hold "the" lock).
The driver still lists ep20/30/40; those ckpts will not exist and print `[skip]`.

`seed=0` matches the other two arms' `--rng-seed 0`; both feed the same
`env.seed([seed + 10b + j])` scheme, so all three arms see the same 100 initial
states. Noise is pinned (torch/np/random all seeded before the loop) — which
matters more here than for the other arms, since DP inference is 100 denoiser
steps and an unpinned generator would put real variance on every coverage number.

**Verification state of the rollout path — both halves now validated.**

`eval_dp_orig.py` was validated first: it produced `/tmp/dp_diag.json` (§5), which
is where every number in that table came from. The shell driver
`run_dp_rollout.sh` was **not** validated at the time this section was written, so
it was exercised in `test` mode before `MODE=full` was allowed near it. Result,
2026-09-18:

```
=== run_dp_rollout mode=test gpu=2 2026-09-18 17:40:31 ===
--- ep_smoke: model_final.pt 73683833 bytes, n=20 ---
[ok] ep_smoke -> .../smoke_dp_ep1/rollout/eval_dp_ep_smoke.json
  "final_coverage_mean": 0.0007698964466329856,
  "success_rate_095": 0.0,
  "latency_ms_per_action_mean": 911.5808702795612,
  "peak_gpu_mem_mb": 244.60498046875,
=== done 2026-09-18 17:50:28 ===
```

9.97 min wall, no error — ckpt load, the 3-D/4-D shape branch, env seeding, JSON
emit and the `[ok]` grep path all run. The cross-project lock passed separately:

```
holder: got lock
second: refused (GOOD)          <- flock -n 9 fails => driver exits 3, no GPU2 double-run
third:  got lock (GOOD)         <- released cleanly
```

**A latency spread worth carrying forward:** this n=20 smoke pass reports
`911.58 ms` where the n=10 diagnostic (§5) on **the same checkpoint** reported
`879.38 ms` — **+3.7 %**. Same weights, same script, same card. So this arm's
per-step latency is not reproducible to better than ~4 %, which is the resolution
limit on any DP latency claim of that size.

**Bugs found and fixed while building this path** (all before any result existed):

1. `action[:, -1]` on DP's **3-D** `[N, W, A]` output. `_predict` returns
   `action_pred[:, obs_horizon-1 : obs_horizon-1+action_horizon, :]` = `[N, 5, 2]` —
   there is no `T` axis because there is no GPT readout horizon. Copied verbatim
   from the other arms, where the tensor is `[N, T, W, A]`. **This one does not
   crash**: `[:, -1]` yields `[N, 2]` and `[:, 0]` then yields `[N]`, a 1-D action
   `env.step` would broadcast into nonsense. Fixed with an explicit dim branch.
2. Missing `@torch.no_grad()` on the rollout loop. Upstream's `eval_on_env` carries
   it; without it the 100-step DDPM loop builds an autograd graph per env step.
3. `from envs.venv import SubprocVectorEnv` failed with `ModuleNotFoundError:
   No module named 'envs'` — `python /path/to/script.py` puts the **script's**
   directory on `sys.path[0]`, not the cwd, so running it from `~/dp_orig` still
   does not make `envs`/`models`/`datasets` importable. `train_policy.py` never hits
   this because it is run as `python train_policy.py` from inside the tree.
4. `UnsupportedInterpolationType: Unsupported interpolation type eval` — the DP
   config computes `pred_horizon` via `${eval:'${a} + ${b} - 1'}`, a custom resolver
   registered by `train_policy.py:34`, not by Hydra. A bare `@hydra.main` on the same
   config dies on it.
5. `UnboundLocalError: local variable 'final_covs' referenced before assignment` —
   the accumulators were `+=`-ed inside the nested `rollout()`. Cost: one full
   302-step batch ran to completion and then threw the result away.

### Revised ETA (after §8a), with actuals

| stage | projected | actual |
|---|---|---|
| epochs 6–10 | ~16:55 | 16:48 (each epoch ~510 s, no drift) |
| inline eval at ep10 + the save it triggers | ~16:57 | eval 16:48:04; `model_9.pt` 16:48 |
| watcher detects + two stability samples + SIGTERM | ~16:58 | **16:49:05** |
| `run_dp_rollout.sh test` (validates the driver, smoke ckpt) | ~17:05 | started 17:41 |
| `MODE=full` → one n100 on `model_9.pt` | **~17:52** | projected ~18:37 from 17:41 |

The `MODE=full` figure moved from the pre-amendment ~01:00 to ~18:37 for two
reasons: the ckpt count went from four to one (3.11 h → 0.78 h of rollout), and
the actual launch happened at 17:41 rather than 17:05.

`run_dp_rollout.sh`'s `test` budget was corrected from ~3 min to **~9.3 min** (and
§9's earlier ~5 min with it): n=20 is 2 batches of 302 steps at the measured
921 ms/step = 557 s, plus startup. Episodes never end early — the success-`done`
is commented out upstream — so 302 steps is exact, not a ceiling.

---

## 10. Training results

*(to be filled when the run finishes)*

## 11. Rollout results

*(to be filled)*

## 12. Artifacts on disk

*(to be filled)*

## 13. Not recorded

* **Everything downstream of epoch 10 does not exist, by design (§8a).** The run
  is stopped rather than allowed to finish, so there is no `model_final.pt`, no
  final eval, no `completion_idx_final.json`, no `eval_*.mp4` for the ep10 ckpt.
  The ep10 inline eval *does* run (it is what triggers the save) but its output
  is a progress signal only (§7(b)) and is not carried forward. This is not a
  gap to close later — a 10-epoch arm has no end-of-run artifacts to have.
* **The `model_39.pt` ≡ `model_final.pt` sha256 identity is unobservable**, not
  merely unmeasured: neither file is written. Stated in §6 so that a later reader
  does not go looking for a check that has no subject.
* The inline-eval coverage values — they went to the offline wandb runs, which
  did not write `wandb-summary.json`/`wandb-history.jsonl` (only `requirements.txt`
  and the binary `.wandb` log are present for the smoke's
  `offline-run-20260918_144950-hfg9l283` and the full run's
  `offline-run-20260918_152934-gmxfamb8`). Not worth reconstructing: per §7(b)
  they are a progress signal, not a result. The full run's wandb is in any case
  terminated by SIGTERM rather than exiting, so it is the less likely of the two
  to have flushed anything.
