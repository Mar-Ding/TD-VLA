# P0 — Patch Policy ↔ TD head interface audit

**Scope.** The TD arm must differ from the VQ-BeT arm in **exactly one thing**: the action
head. This document is the evidence for that claim. It audits the *contract* the shared
harness imposes on a policy object, and it audits the *file-level* diff against the
sibling tree the fork came from — because "only the action head differs" is a statement
about both.

**Reproduce.**

```bash
# local work tree (checks A–H: file identity + the duck-typing contract)
python scripts/audit_interface.py

# server, CPU/GPU-agnostic (adds check I: package resolution)
cd ~/td_pusht && CUDA_VISIBLE_DEVICES='' \
    ~/miniconda3/envs/wemm-p1/bin/python -u scripts/audit_interface.py --remote
```

Measured: **43/43 PASS** locally, **27/27 PASS** on the server. (The count differs
because check A compares against a sibling tree that only exists in the local work
tree; see §2.)

Every check below *calls* the thing it asserts. Nothing here is a reading of the source.

---

## 1. Why this stage exists at all

The harness (`train.py`, `eval_rollout.py`) was written against `BehaviorTransformer`. It
reaches into the policy object through duck typing, not through an interface:

| harness site | what it does | what breaks if absent |
|---|---|---|
| `train.py:747` | `assert model.vqvae_is_fit` | **silently trains nothing** — the loss is zeroed at `vqbet.py:447-448` |
| `train.py:1193-1196` | indexes `loss_dict["total_loss" / "classification_loss" / "offset_loss"]` | `KeyError`, mid-epoch |
| `train.py:1221-1222`, `:1328-1329` | prints `model.bin_head.net[0].weight.grad` | `AttributeError`; and the *shape* of `bin_head` is therefore load-bearing |
| `train.py:931-939` | groups params by the prefixes `"bin_head."` / `"offset_head."` | an empty group is **silent** |
| `train.py` ckpt/resume | `model.vqvae.state_dict()` / `.training` / `.parameters()` | `AttributeError` |
| `eval_rollout.py:653` | consumes `predicted_action[N,5,5,2]`, takes `[:, -1]` | wrong rollout, no error |

Two of these fail **silently**. That is the reason this stage is a script and not a
paragraph: a passing P2 with a broken `vqvae_is_fit` would look like a converging TD head
and would actually be a head receiving no gradient.

---

## 2. The deviation ledger — the honest core of P0

The plan's phrasing was "17 files identical to the sibling tree modulo the package
rename, with 2 enumerated deviations." **That was wrong, and the corrected ledger is
below.** `normalizer.py` is not a copy of anything in the sibling (which has no normaliser
at all), `config.py` is not identical (it gained `TDConfig`), and there are four new
modules — not two.

### 2.1 Identical to the sibling, modulo `s/sibling_pkg/patch_policy/g` — 12 files

`gpt.py`, `encoder.py`, `dataset.py`, `vqvae.py`, `vqbet.py`, `coord_decoder.py`,
`adaln.py`, `adaln_gate.py`, `global_policy.py`, `precompute.py`, `precompute_cond.py`,
`qwen3vl_official.py`.

Compared as **text**, after substituting the package name, not as a bare sha256 — the
package was renamed, so a hash comparison would be meaningless. The comparison this
buys is stronger than a hash anyway: it proves the difference is *only* the package
name.

### 2.2 Enumerated deviations, each with a budget in changed lines

Budget = added + removed lines from `difflib`, and it is enforced. The budget is the
point: the deviation is allowed, but it must not grow without someone noticing.

| file | changed | budget | what it is |
|---|---|---|---|
| `__init__.py` | 13 | 20 | docstring rewritten to describe **this** project rather than the fork |
| `eval_rollout.py` | 9 | 25 | (a) usage header `~/sibling_pkg` → `~/td_pusht`; (b) `BASE` readable from `$PATCH_POLICY_BASE`, **default unchanged** |
| `train.py` | 17 | 40 | the `build_model` dispatch (5 lines) + 2 return-annotation widenings + its docstring. **The only substantive harness change in this project** |
| `config.py` | 93 | 130 | `TDConfig` dataclass + the `Config.td` field. **No pre-existing field was altered** |

`config.py` was **not** in the plan's deviation list, and `train.py`'s annotation
widenings (`BehaviorTransformer` → `nn.Module` in two signatures) were not either. Both
are recorded here because an unlisted deviation is indistinguishable from a mistake.

### 2.3 New modules — 4, plus one ported from the base repo

| module | provenance | what it is |
|---|---|---|
| `normalizer.py` | ported from `base/patch_policy/utils/normalizer.py` | **not** a copy: 351 → 162 lines, dropping `zarr`, `numpy` and the dict-of-fields machinery, keeping `limits` mode. The sibling has no normaliser, so comparing it there would invent a failure |
| `anchors.py` | no upstream | pure-torch k-means over PushT action chunks |
| `diffusion.py` | no upstream | self-contained truncated DDIM — `wemm-p1` has no `diffusers`, and the plan forbids installing into it |
| `td_head.py` | no upstream | `TruncatedDiffusionHead` + `TruncatedDiffusionPolicy` |

The check asserts that each of these has **no** sibling counterpart, so a future reader
cannot assume any of them was copied from a tree that is supposed to be frozen.

---

## 3. Bugs found in this audit's own first version

Recorded because both would have produced a confident wrong conclusion.

**3.1 The diff metric was positional.** The first version counted differing lines by
zipping the two files line-by-line. A single inserted line shifts every subsequent line,
so it reported **`train.py`: 1105 differing lines of 1390** and **`eval_rollout.py`: 851
of 905** — i.e. "the fork rewrote 80% of the harness," which would have been a
catastrophic finding. The real numbers, from `difflib`, are **17** and **9**. Fixed by
diffing properly and enforcing a per-file budget. The lesson is general: a line-zipped
comparison answers "how many lines are at a different index," not "how many lines
changed."

**3.2 `inspect.signature` on a bound method omits `self`.** The first version compared
`inspect.signature(model.forward)` (bound → no `self`) against
`inspect.signature(BehaviorTransformer.forward)` (a plain function → has `self`), and
reported a spurious parameter mismatch. Fixed by stripping `self`, and extended to compare
**defaults** as well, since the harness calls the policy both positionally and by keyword
depending on the path.

**3.3 Not a bug, but a false-alarm class avoided.** Check A can only run where the
sibling tree exists. On the server it does not, and the first version recorded that as a
FAIL. An alarm about the absence of a *comparison baseline* is not an alarm about this
code, so `--remote` now downgrades it to a note (and `--sibling <path>` can point at one).

---

## 4. The contract checks

### B. Trunk readout — 3 checks

`gpt(dummy[2,5,256,384]) → [2,5,256]`, `readout_dim == 256`.

This is the `clean` line's consequence and the one place the plan's dimensions moved.
`native_parity: true` gives `lm_head: Linear(512→256)`, `gelu='tanh'` and a head MLP with
no final activation, so **the head's input width is 256, not 512**. The audit asserts
`readout_dim == cfg.policy.n_embd // 2` rather than a literal, so no head module can
hard-code either width — hard-coding 512 breaks the clean line, and hard-coding 256
breaks the base line.

### C. `forward` contract parity — 6 checks

Parameter names **and defaults** match `BehaviorTransformer.forward`; the call returns a
3-tuple; `predicted_action` is `[N,5,5,2]`; loss is a scalar with grad; and the inference
path (`action_seq=None`) returns the same 3-tuple shape.

### D. `loss_dict` — 3 checks

`total_loss`, `classification_loss`, `offset_loss` all present and finite. These are the
three keys `train.py:1193-1196` indexes unconditionally; `mode_acc` is a TD-only addition
and does not disturb them.

### E. VQ surface — 4 checks

`vqvae_is_fit` is `True`; **assigning to it is absorbed** rather than raising (the
historical `--resume` bug recorded at `vqbet.py:258`); the `vqvae` stub has
`.state_dict()` / `.training` / `.parameters()`; `_maybe_fit_vq(force=True)` — reachable
only from `--export-shared-vq` — raises `SystemExit` instead of emitting an empty VQ
artifact that a later arm could load and mistake for a trained tokenizer.

### F. Checkpoint round-trip — 2 checks

No `_vqvae_model.*` or `vqvae.*` key ever appears in a TD `state_dict()` — an empty
tokenizer is the honest encoding of "there is no tokenizer." Save → load → identical.

Cross-arm loading is **not** tested and is **not** supported: TD has 3 param groups and
VQ-BeT has 2, so `optim_param_names` is naturally incompatible. This is by design, not a
defect.

### G. Rollout shape — 2 checks

`pred[:, -1] → [N,5,2]`, and the W=5 temporal ensemble stacks to `[W,N,5,2]` whose
`[:, 0]` is the action consumed. `eval_rollout.py:653` reads exactly this.

### H. Encoder frozen — 2 checks

All encoder parameters `requires_grad=False`; `n_patches=256`, `output_dim=384`.

### I. Package resolution on the server — 4 checks

```
BASE         -> ~/patch_policy
envs         -> ~/patch_policy/envs
patch_policy -> ~/td_pusht/patch_policy
```

`envs` must come from BASE (the read-only upstream is the only copy of PushT's vendored
gym env, and nothing here ever writes to it), while `patch_policy` must come from **our**
tree. The no-shadowing claim holds only because BASE has no `patch_policy/` subdir — so
that is asserted directly rather than assumed, and `BASE` is read from
`$PATCH_POLICY_BASE` with the same default the real code uses.

The check imports `patch_policy.eval_rollout` and inspects where things landed, rather
than re-deriving the `sys.path` rule. A reimplementation of the rule would keep passing
after the real one broke.

---

## 5. Check J — resolved at P3, and the plan's path was wrong

The plan's item 9 is **GPU2 rollout lock mutual exclusion**: `rollout_locked_gpu2.sh` must
take the absolute path `~/audits/gpu2_rollout.lock`, the same lock the
existing sibling driver takes, so the two drivers cannot both hold "their own" lock and put
two n100 rollouts (~10 GB each) on a 24 GB card simultaneously.

It could not be tested at P0 because the script did not exist. P0 recorded it as
**DEFERRED** rather than pass — claiming a lock passes when there is no lock would be
fiction, and it is the kind of fiction that costs an OOM at P6.

**Outcome at P3: PASS, but not at the path this document specified.**

The plan's absolute path is **not** the lock anyone else holds. `~/sibling_pkg/scripts/rollout_locked_gpu2.sh`
declares `LOCK=audits/gpu2_rollout.lock` — **relative** — so it resolves against the
caller's cwd, which is `~/sibling_pkg`:

| | path | exists? |
|---|---|---|
| plan required (this doc, item 9) | `~/audits/gpu2_rollout.lock` | **no** — `~/audits/` does not exist |
| the existing driver's real lock | `~/sibling_pkg/audits/gpu2_rollout.lock` | yes |
| TD's implementation | `~/sibling_pkg/audits/gpu2_rollout.lock` | yes — the same file |

`find ~ -maxdepth 4 -name gpu2_rollout.lock` returns exactly **one** hit. Had
TD obeyed this document literally, it would have locked a **different inode**, both
drivers would have acquired "their own" lock, and two rollouts would have landed on GPU2
while each believed it was serialized — precisely the failure the absolute path was
introduced to prevent. **The requirement was right; the specific path in it was wrong.**

`scripts/test_gpu2_lock.sh` now asserts the identity, not just the mechanism:

```text
[PASS] the lock we take IS the file the existing sibling_pkg driver takes
[PASS] second holder is REFUSED while the lock is held (flock -n fails)
[PASS] lock is acquirable again after release (the file is real and usable)
J RESULT: PASS
```

Falsified against the path this document originally specified — check 0 fails, checks 1
and 2 still pass, i.e. the flock mechanics are fine and only the file identity catches it:

```text
GPU2_LOCK=~/audits/gpu2_rollout.lock bash scripts/test_gpu2_lock.sh
  [FAIL] lock mismatch: theirs resolves to '~/sibling_pkg/audits/gpu2_rollout.lock',
         ours is '~/audits/gpu2_rollout.lock' <- mutual exclusion would be a fiction
  J RESULT: FAIL
```

A second hazard is recorded but **not solvable from this project**:
`~/peer-project/scripts/auto_rollout.sh` also targets `GPU=2` and takes no lock at all. No lock
in this repo can exclude it. Details and the evidence that it cannot currently fire are in
`2026-09-16_P3_smoke_test.md` §3.

---

## 6. Disk gate

The plan's gate is `df -h` free space **≥ 15 G** before P6. Last measured: **24 G**, which
passes but is not comfortable (98% full).

The plan's per-checkpoint estimate was **~417 MB**. P3 measured the real thing:

| checkpoint | measured |
|---|---|
| with optimizer state | **500 MB** |
| without | **201 MB** |

So the "keep only the best" peak is ~1.5 GB per arm, not ~1.3 GB. Still inside 24 G, but
**the gate should be re-checked against 500 MB**, and re-confirmed immediately before P6 —
the margin is 9 GB and other projects share the volume.

---

## 7. Verdict

| gate | required | measured | status |
|---|---|---|---|
| shared files identical modulo the rename | 12 / 12 | 12 / 12 | ✅ |
| deviations enumerated **and** within budget | 4 / 4 | 4 / 4 | ✅ |
| new modules have no sibling counterpart | 4 / 4 | 4 / 4 | ✅ |
| `gpt` readout `[2,5,256]`, `readout_dim == 256` | exact | exact | ✅ |
| `forward` parity (names + defaults + triple + shape) | exact | exact | ✅ |
| `loss_dict` carries the 3 mandatory keys | 3 / 3 | 3 / 3 | ✅ |
| no VQ tokenizer keys in a TD checkpoint | 0 keys | 0 keys | ✅ |
| encoder fully frozen, 256 × 384 | exact | exact | ✅ |
| server package resolution, no shadowing | `envs`←BASE, `pkg`←cwd | as required | ✅ |
| GPU2 lock mutual exclusion | the lock others hold | same inode, falsified-tested | ✅ (P3) |
| disk free before P6 | ≥ 15 G | 24 G | ⚠️ re-confirm |

**One-line summary:** the "one variable" claim now has a file-level ledger rather than a
promise — 12 files bit-identical modulo the rename, 4 deviations bounded and justified
(`train.py` 17 lines, `eval_rollout.py` 9, `config.py` 93, `__init__.py` 13), 4 new
modules with stated provenance — and every silent-failure mode the harness can produce
(`vqvae_is_fit`, the three `loss_dict` keys, `bin_head.net[0].weight`, the
`bin_head.`/`offset_head.` prefixes, the `pred[:, -1]` rollout shape) is asserted by
calling it.

Item 9 was **deferred here in good faith and closed at P3 with a correction**: the
mechanism passed, but the absolute path this document demanded was not the lock any other
driver takes (§5). The disk gate remains open, and its per-checkpoint number is now
measured rather than estimated — see §6.

**Note on provenance of this document.** P0 was reported as "passed" before this file
existed — the doc and the script behind it were written afterwards. The ledger in §2 is
therefore the *first* record of the deviation set, and it corrects the plan's count (2
deviations → 4, plus a base-repo port the plan did not name) rather than restating it.
