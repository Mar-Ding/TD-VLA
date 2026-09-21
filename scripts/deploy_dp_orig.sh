#!/bin/bash
# Deploy a byte-faithful copy of the upstream patch_policy tree so the ORIGINAL
# Diffusion-Policy arm can be trained standalone, without ever writing to
# ~/patch_policy (the project's stated read-only reference).
#
# WHY A COPY AT ALL, and not a port into our own harness
# The user chose "上游原样跑" over porting the DP into patch_policy/ with the
# TD-aligned recipe. That means the DP arm deliberately does NOT share our
# harness: upstream has its own train loop, its own window (2, not 5), its own
# 256/4/8 capacity and its own DDPM-100/squaredcos_cap_v2/epsilon+EMA recipe.
# The consequence -- which must be stated in any later comparison -- is that
# DP-vs-TD is NOT a single-variable comparison; only TD-vs-VQ-BeT is.
#
# THE TWO DEVIATIONS (and nothing else)
#   1. configs/env_vars/env_vars.yaml: save_path $HOME/patch_policy
#      -> $HOME/dp_orig.  Forced: hydra.run.dir is built from
#      ${env_vars.save_path}, so without this the copy would write into the
#      read-only reference. This changes WHERE, not WHAT.
#   2. train_policy.py: save a ckpt at EVERY eval_on_env, not only when the eval
#      beats the running best.  Upstream's best-gate (the `if current_eval_metric
#      > best_eval_metric` at ~:504) means a 40-epoch run keeps model_final.pt
#      plus at most a couple of mid-run bests -- not enough to roll out
#      ep10/20/30/40 separately, which is what the standing protocol requires.
#      best_eval_metric is still tracked and still printed; only the store gate
#      moves.  The recipe, the optimizer, the scheduler and the EMA are untouched.
#
# LINE ENDINGS ARE PART OF THE BYTES. The upstream tree is MIXED: train_policy.py
# is 691/691 CRLF, train_pusht_diffusion.yaml 94/94 CRLF, diffusion_policy.py
# 912/912 CRLF, env_vars.yaml 0/18 (LF). A first version of this script used
# pathlib read_text()/write_text(), whose universal-newline translation silently
# rewrote train_policy.py from CRLF to LF -- an unlisted, whole-file deviation
# (the file got 3 lines longer yet 454 bytes smaller). patch() below therefore
# opens with newline="" and rebuilds the anchors using the file's own ending.
#
# ckpt FILENAMES ARE 0-BASED. save_model(epoch) with eval gated on
# (epoch+1) % eval_on_env_freq == 0 means a 40-epoch run writes
#   model_9.pt model_19.pt model_29.pt model_39.pt   (= ep10/20/30/40)
# plus model_init.pt and model_final.pt. The rollout driver must use these names.
#
# GUARD: the script ends with two independent checks -- `diff -r -q` over the
# whole tree must report EXACTLY the two deviated files, and nothing in SRC may
# have an mtime newer than the marker.
set -euo pipefail

SRC=${SRC:-$HOME/patch_policy}
DST=${DST:-$HOME/dp_orig}
MARKER=/tmp/dp_orig_deploy_marker.$$
# Non-interactive shells on this box have no `python` on PATH; the project
# convention is the absolute interpreter path.
PY=${PY:-$HOME/miniconda3/envs/wemm-p1/bin/python}
[ -x "$PY" ] || PY=$(command -v python3 || command -v python)

echo "=== deploy_dp_orig $(date '+%F %T') ==="
echo "  SRC=$SRC  (must stay byte-identical)"
echo "  DST=$DST  (receives the two deviations)"

if [ -e "$DST" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[abort] $DST already exists. Re-run with FORCE=1 to replace it wholesale."
    echo "        (Refusing by default: tar would overlay onto a PATCHED tree, the"
    echo "         anchors would no longer match, and the failure would look like a"
    echo "         changed upstream rather than a re-run.)"
    exit 2
fi
[ "${FORCE:-0}" = "1" ] && rm -rf "$DST"

touch "$MARKER"

# ---- 1. copy code only ---------------------------------------------------------
# outputs/ is 18G and is the OLD tree's results, checkpoints/ is VTP-S (638M) and
# is only reachable via encoder: vtp_s*, which the DP config does not use.
mkdir -p "$DST"
tar -C "$SRC" -cf - \
    --exclude=./outputs --exclude=./checkpoints --exclude=./wandb \
    --exclude=__pycache__ --exclude='*.pyc' . | tar -C "$DST" -xf -
echo "[copy] $(find "$DST" -type f | wc -l) files, $(du -sh "$DST" | cut -f1)"

# ---- 2. the two deviations, applied to the COPY only ---------------------------
"$PY" - "$DST" <<'PY'
import hashlib
import pathlib
import sys

dst = pathlib.Path(sys.argv[1])


def patch(rel, old_lines, new_lines, why):
    """Replace one anchor, preserving the file's own line ending byte-for-byte.

    old_lines/new_lines are lists of lines WITHOUT terminators; they are joined
    with whichever ending the file already uses. Reading and writing with
    newline="" disables Python's universal-newline translation, so an untouched
    line's bytes are unchanged and a CRLF file stays CRLF.
    """
    p = dst / rel
    with open(p, "r", newline="") as fh:
        s = fh.read()
    nl = "\r\n" if "\r\n" in s else "\n"
    old = nl.join(old_lines) + nl
    new = nl.join(new_lines) + nl
    n = s.count(old)
    if n != 1:
        raise SystemExit(
            f"[ABORT] {rel}: expected exactly 1 occurrence of the anchor, found {n}.\n"
            f"        The upstream file is not what this script was written against; "
            f"refusing to guess. Anchor was:\n{old!r}"
        )
    out = s.replace(old, new)
    with open(p, "w", newline="") as fh:
        fh.write(out)
    print(f"[patch] {rel}: {why}")
    print(f"        line ending preserved as {nl!r}; "
          f"bytes {len(s)} -> {len(out)}; sha256 now "
          f"{hashlib.sha256(out.encode()).hexdigest()}")


patch(
    "configs/env_vars/env_vars.yaml",
    ["save_path: $HOME/patch_policy"],
    ["save_path: $HOME/dp_orig"],
    "deviation 1/2: outputs land in the copy, not in the read-only reference",
)

patch(
    "train_policy.py",
    [
        "                best_eval_metric = current_eval_metric",
        "                save_model(epoch)",
        "            else:",
        "                print(",
        '                    f"Eval metric {current_eval_metric} <= prev best {best_eval_metric}, "',
        '                    f"NOT saving ckpt"',
        "                )",
    ],
    [
        "                best_eval_metric = current_eval_metric",
        "            else:",
        "                print(",
        '                    f"Eval metric {current_eval_metric} <= prev best {best_eval_metric}, "',
        '                    f"saving ckpt anyway"',
        "                )",
        "            # [dp_orig] deviation 2/2: store on EVERY eval_on_env, not only on a",
        "            # new best, so ep10/20/30/40 can each be rolled out afterwards.",
        "            # See scripts/deploy_dp_orig.sh for the rationale.",
        "            save_model(epoch)",
    ],
    "deviation 2/2: ckpt at every eval_on_env (best_eval_metric still tracked)",
)
PY

# ---- 3. guard A: the differing-file set must be EXACTLY the two above ----------
echo
echo "=== guard A: diff -r -q over the whole tree (expect exactly 2 lines) ==="
DIFFS=$(diff -r -q \
    -x outputs -x checkpoints -x wandb -x __pycache__ -x '*.pyc' \
    "$SRC" "$DST" || true)
echo "$DIFFS"
NDIFF=$(printf '%s\n' "$DIFFS" | grep -c . || true)
if [ "$NDIFF" != "2" ]; then
    echo "[FAIL] expected exactly 2 differing files, got $NDIFF -- the copy is NOT"
    echo "       byte-faithful-plus-two-deviations. Investigate before training."
    rm -f "$MARKER"; exit 1
fi
echo "  OK: exactly the two deviated files differ; every other byte is identical"

echo
echo "=== the deviation list, generated rather than asserted ==="
for f in configs/env_vars/env_vars.yaml train_policy.py; do
    echo "--- $f ---"
    diff -u "$SRC/$f" "$DST/$f" | sed -n '3,$p' || true
done

# ---- 4. guard B: nothing in SRC was written -----------------------------------
echo
echo "=== guard B: files in SRC modified after the marker ==="
TOUCHED=$(find "$SRC" -type f -newer "$MARKER" 2>/dev/null || true)
if [ -n "$TOUCHED" ]; then
    echo "[FAIL] SRC was written to:"; echo "$TOUCHED"; rm -f "$MARKER"; exit 1
fi
echo "  none -- $SRC is untouched"
rm -f "$MARKER"
echo "=== deploy_dp_orig done $(date '+%F %T') ==="
