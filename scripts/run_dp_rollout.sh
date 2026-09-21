#!/bin/bash
# Roll out the standalone ORIGINAL Diffusion-Policy arm (~/dp_orig) at n=100.
#
# RUN THIS ONLY AFTER TRAINING HAS FINISHED. The standing protocol for this
# project is: ckpts are stored during training, and every rollout is measured
# afterwards, serially, with the cards otherwise idle -- because concurrency and
# card identity both move the clock by more than the arm difference does (see
# the GPU0-thermal note: card spread 531-626 s = 18 % vs concurrency 3.9 %).
#
# CKPT NAMES ARE 0-BASED. A run gated on (epoch+1) % 10 == 0 writes
#   model_9.pt  model_19.pt  model_29.pt  model_39.pt     (after ep 10/20/30/40)
# The list below is kept in full so that extending the arm past 10 epochs needs no
# edit here; a ckpt that does not exist is skipped with a printed [skip], not an
# error. For the 10-epoch run (amended 2026-09-18, see run_dp_orig.sh) exactly
# one of these exists -- model_9.pt -- so the three [skip] lines are the expected
# output, not a failure. (model_39.pt, if it ever exists, is the SAME WEIGHTS as
# model_final.pt -- but the 10-epoch run is STOPPED before either is written, so
# that identity is now unobservable rather than merely unchecked. Recorded in §6
# of the run log.)
#
# PROTOCOL ALIGNMENT. eval_dp_orig.py applies the same temporal ensembling as the
# TD and VQ-BeT arms, so the three arms' numbers are protocol-comparable. It
# does NOT reproduce upstream's inline eval, which executes the chunk open-loop.
# Those inline numbers are a progress signal only -- see the script docstring.
#
# MODE=test : one small n=20 pass over the SMOKE ckpt. Validates the whole
#             rollout path (ckpt load, shape branch, env seeding, JSON) before
#             committing to the n100 that the full run costs. Budget it as
#             ~9.3 min, not the ~3 min originally written here: n=20 is 2 batches
#             of 302 steps and the measured per-step cost is 921 ms
#             (302 x 2 x 0.921 = 557 s), and episodes never end early -- the
#             success-done is commented out upstream.
# MODE=full : the four distinct ckpts of the 40-epoch run, n=100 each.
#
# GPU: default 2. GPU0 is thermally self-limited (SM ~1090 MHz vs GPU1's 1562)
# so it must not be used for anything whose wall clock will be compared. GPU1 and
# GPU2 are equivalent; if the three arms are later timed on different cards the
# card identity must be recorded next to the timing.
set -u

MODE=${1:-test}
GPU=${GPU:-2}
BASE=$HOME/dp_orig
PY=$HOME/miniconda3/envs/patch-policy/bin/python
# The rollout script lives in our deployed tree, NOT in dp_orig: dp_orig must
# stay a byte-faithful copy of upstream plus the two documented deviations, and
# adding a file to it would quietly become deviation three.
SCRIPT=$HOME/td_pusht/scripts/eval_dp_orig.py
# Absolute path, shared with the sibling driver -- a relative lock would let two
# drivers each hold "the" lock and roll out on GPU2 simultaneously.
LOCK=$HOME/sibling_pkg/audits/gpu2_rollout.lock

RUN40="$BASE/outputs/pusht/dp_orig_40ep"
OUTDIR="$RUN40/rollout"
SMOKE="$BASE/outputs/pusht/smoke_dp_ep1/model_final.pt"

case "$MODE" in
  test)
    JOBS="$SMOKE ep_smoke 20"
    OUTDIR="$BASE/outputs/pusht/smoke_dp_ep1/rollout"
    ;;
  full)
    # ep10 is the only one that will exist for the 10-epoch run; the other three
    # are listed so the arm can be extended without editing this file, and are
    # skipped (see the 0-based-names note above).
    JOBS="$RUN40/model_9.pt  ep10 100
$RUN40/model_19.pt ep20 100
$RUN40/model_29.pt ep30 100
$RUN40/model_39.pt ep40 100"
    ;;
  *) echo "usage: $0 [test|full]"; exit 2 ;;
esac

mkdir -p "$OUTDIR"

# Serialise against every other rollout on this card, ours and the sibling one.
if [ -e "$LOCK" ]; then
    exec 9>"$LOCK"
    if ! flock -n 9; then
        echo "[abort] another rollout holds $LOCK; refusing to contend"
        exit 3
    fi
else
    echo "[warn] $LOCK does not exist -- running WITHOUT the cross-project lock"
    echo "       (check the path before trusting any timing from this run)"
fi

echo "=== run_dp_rollout mode=$MODE gpu=$GPU $(date '+%F %T') ==="
# A here-string, not `printf | while`: a piped loop runs in a subshell, so the
# `exit 1` on a failed rollout below would kill only the subshell -- the driver
# would print "=== done ===" and return 0 while a rollout had actually failed.
# This driver produces the deliverable numbers unattended, so a failure that
# reports success is the one bug it must not have.
while read -r CKPT TAG NEV; do
    [ -n "$CKPT" ] || continue
    if [ ! -f "$CKPT" ]; then
        echo "[skip] $TAG: no ckpt at $CKPT"
        continue
    fi
    JSON="$OUTDIR/eval_dp_$TAG.json"
    if [ -f "$JSON" ]; then
        echo "[skip] $TAG: $JSON already exists (delete it to redo)"
        continue
    fi
    echo "--- $TAG: $(basename "$CKPT") $(stat -c%s "$CKPT") bytes, n=$NEV ---"
    cd "$BASE" || exit 1
    # seed=0 matches the other two arms' rollout seed. --rng-seed 0 in their
    # driver and cfg.seed here feed the same env.seed([0..99]) scheme, so all
    # three arms see the same 100 initial states.
    if CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline MUJOCO_GL=egl \
        DP_CKPT="$CKPT" DP_TAG="$TAG" DP_OUT="$JSON" DP_NUM_EVALS="$NEV" \
        "$PY" "$SCRIPT" \
            encoder=dino_patch seed=0 hydra.run.dir="$OUTDIR/hydra_$TAG" \
            > "$OUTDIR/log_$TAG.txt" 2>&1
    then
        echo "[ok] $TAG -> $JSON"
        grep -E 'final_coverage_mean|success_rate_095|latency_ms_per_action_mean|peak_gpu_mem_mb' "$JSON"
    else
        echo "[FAIL] $TAG -- last 30 lines of log:"
        tail -30 "$OUTDIR/log_$TAG.txt"
        exit 1
    fi
done <<< "$JOBS"
echo "=== done $(date '+%F %T') ==="
