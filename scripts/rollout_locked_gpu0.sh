#!/bin/bash
# Run ONE eval_rollout on GPU0 under a CROSS-DRIVER flock. Twin of
# rollout_locked_gpu2.sh, differing in exactly two things: the card, and the lock path.
#
# Usage (from a driver):
#   bash scripts/rollout_locked_gpu0.sh <config> <ckpt> <out> <log> [rng_seed] [extra...]
#
# =====================================================================================
# WHY THIS RUN'S TIMINGS ARE NOT USABLE AS TIMINGS -- READ BEFORE QUOTING policy_ms
# =====================================================================================
# The user's assignment for the 400-epoch run (2026-09-20) puts rollouts on GPU0 while
# BOTH trainings run on GPU1 and GPU2. That is a deliberate, accepted tradeoff, and its
# cost is recorded here so no later reader averages these milliseconds into a comparison:
#
#   1. GPU0 IS THE THERMALLY THROTTLED CARD. Under load it sits at 84-88 C with SM
#      ~1090 MHz -- 52% of what a cool card holds. The project's standing rule is
#      「计时对比必须避开 GPU0」.
#   2. IT RUNS CONCURRENTLY WITH TWO TRAININGS. Concurrency on this box has been
#      measured to inflate the per-step clock ~2.9x, with DINO encoding as the
#      thermometer (16.6-17.1 ms clean vs 48-58 ms contended).
#
# So: **coverage from this run is valid; policy_ms / encoder_ms / total_ms are not.**
# The timing comparison must come from a separate, exclusive rerun on a cool card.
# `encoder_ms` is still worth reading -- not as a duration, but as the thermometer that
# tells you whether a given rollout was contended while it ran.
#
# =====================================================================================
# THE LOCK IS A DIFFERENT FILE FROM THE GPU2 ONE, ON PURPOSE. DO NOT SHARE IT.
# =====================================================================================
# A lock is meaningful only if every process that touches the card takes the SAME inode.
# GPU0 is a different card from GPU2, and two rollouts may legitimately run at once --
# one per card -- so gpu0 and gpu2 must NOT share a lock or they would serialize each
# other for no reason.
#
# What MUST share this lock: the two arm workers of THIS run (K=20 and K=40), which both
# target GPU0. `scripts/run_400ep_rollout.sh` is launched once per arm, so without a
# shared lock those two processes would each roll out on GPU0 simultaneously -- ~10 GB
# each on a 24 GB card, plus mutual slowdown, while both believed they were serialized.
#
# The lock lives next to the GPU2 one because that directory is the one that actually
# exists: `$HOME/audits/gpu2_rollout.lock` also exists, but `rollout_locked_gpu2.sh`
# documents (verified by find) that the lock the OTHER project's driver holds is
# `$HOME/sibling_pkg/audits/gpu2_rollout.lock`. Same directory, new filename.
# =====================================================================================
set -u

# Absolute path. Overridable for testing only.
LOCK=${GPU0_LOCK:-$HOME/sibling_pkg/audits/gpu0_rollout.lock}

CONFIG=$1; CKPT=$2; OUT=$3; LOG=$4; RNGSEED=${5:-0}
shift 5 2>/dev/null || shift $#
EXTRA=("$@")

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")" "$(dirname "$LOCK")"

# Block up to 1h for the other arm's rollout to finish, then run. -x = exclusive.
# 1h rather than the GPU2 twin's 3h: a throttled n100 here is ~9-20 min, so an hour is
# several rollouts of slack, and failing fast surfaces a stuck peer sooner.
exec 9>"$LOCK"
flock -x -w 3600 9 || {
    echo "[lock] $(date '+%F %T') could not acquire GPU0 lock ($LOCK) within 1h" >> "$LOG"
    exit 2
}
echo "[lock] $(date '+%F %T') GPU0 lock acquired -> $(basename "$OUT") rng_seed=$RNGSEED" >> "$LOG"

# Thermometer sample at the boundary. Not a duration -- see the header.
echo "[lock] $(date '+%F %T') gpu0 pre  $(nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv,noheader)" >> "$LOG"

CUDA_VISIBLE_DEVICES=0 $HOME/miniconda3/envs/wemm-p1/bin/python -u \
    -m patch_policy.eval_rollout \
    --config "$CONFIG" --ckpt "$CKPT" --out "$OUT" --rng-seed "$RNGSEED" \
    --attn current "${EXTRA[@]}" >> "$LOG" 2>&1
rc=$?

echo "[lock] $(date '+%F %T') gpu0 post $(nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv,noheader)" >> "$LOG"
echo "[lock] $(date '+%F %T') rollout done rc=$rc -> $(basename "$OUT")" >> "$LOG"
exit $rc
