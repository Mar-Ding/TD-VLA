#!/bin/bash
# Run ONE eval_rollout on GPU2 under a CROSS-DRIVER flock, so this project's rollouts
# cannot overlap another driver's on the same card (each n100 rollout peaks ~10GB; two
# at once contend or OOM on 24GB).
#
# Usage (from a driver):
#   bash scripts/rollout_locked_gpu2.sh <config> <ckpt> <out> <log> [rng_seed] [extra...]
#
# =====================================================================================
# THE LOCK PATH IS THE WHOLE POINT OF THIS FILE. READ BEFORE EDITING.
# =====================================================================================
# The plan specified the lock as the ABSOLUTE path
#     $HOME/audits/gpu2_rollout.lock
# "the same lock the sibling driver takes". That path was WRONG, and using it would have
# produced exactly the failure the absolute path was meant to prevent:
#
#   * The existing driver `~/sibling_pkg/scripts/rollout_locked_gpu2.sh` writes
#         LOCK=audits/gpu2_rollout.lock          <-- RELATIVE
#     which resolves against the CALLER'S CWD. In practice the driver runs from
#     ~/sibling_pkg, so the real lock file is
#         $HOME/sibling_pkg/audits/gpu2_rollout.lock
#     That is the only gpu2_rollout.lock that exists on disk (verified by
#     `find $HOME -maxdepth 4 -name gpu2_rollout.lock` -> exactly one hit).
#   * $HOME/audits/ DOES NOT EXIST. A lock there is a DIFFERENT inode, so both
#     drivers would each acquire "their own" lock and roll out on GPU2 simultaneously
#     while both believed they were serialized.
#
# So this file takes the absolute path of the lock that actually exists and is actually
# held by the other driver. The path is not a convention, it is a specific file.
#
# SECOND, UNRESOLVED HAZARD (not fixable from here): `~/peer-project/scripts/auto_rollout.sh`
# also targets GPU=2 and takes NO LOCK AT ALL -- it calls eval_rollout directly. No lock
# in this file can exclude it. As of 2026-09-16 it cannot fire (its last log entry is
# 2026-09-05 22:41 and every remaining target either already has its rollout_seed0.json
# marker or has no ckpt on disk, so it is a read-only poller), but that is a property of
# its inputs, not a property of the lock. Recorded here because if anyone ever drops a
# ckpt_epoch*.pt into those output dirs, it fires on GPU2 unannounced.
# =====================================================================================
set -u

# Absolute path to the ONE real lock file. Overridable for testing only.
LOCK=${GPU2_LOCK:-$HOME/sibling_pkg/audits/gpu2_rollout.lock}

CONFIG=$1; CKPT=$2; OUT=$3; LOG=$4; RNGSEED=${5:-0}
shift 5 2>/dev/null || shift $#
EXTRA=("$@")

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")" "$(dirname "$LOCK")"

# Block up to 3h for the other holder to finish, then run. -x = exclusive.
exec 9>"$LOCK"
flock -x -w 10800 9 || {
    echo "[lock] $(date '+%F %T') could not acquire GPU2 lock ($LOCK) within 3h" >> "$LOG"
    exit 2
}
echo "[lock] $(date '+%F %T') GPU2 lock acquired ($LOCK) -> $(basename "$OUT") rng_seed=$RNGSEED" >> "$LOG"

CUDA_VISIBLE_DEVICES=2 $HOME/miniconda3/envs/wemm-p1/bin/python -u \
    -m patch_policy.eval_rollout \
    --config "$CONFIG" --ckpt "$CKPT" --out "$OUT" --rng-seed "$RNGSEED" \
    --attn current "${EXTRA[@]}" >> "$LOG" 2>&1
rc=$?
echo "[lock] $(date '+%F %T') rollout done rc=$rc -> $(basename "$OUT")" >> "$LOG"
exit $rc
