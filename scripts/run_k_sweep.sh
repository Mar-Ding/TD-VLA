#!/bin/bash
# Anchor-count sweep: three trainings, one per card, varying ONLY K.
#
# K in {5, 10, 40}; K=20 already exists as P4 (outputs/p4_td, 40 epochs, train-seed 0),
# so this completes the 4-point curve with the SAME epoch setting the user asked for.
#
# NO ROLLOUTS HERE. The user's instruction is to train all three first, keep every
# checkpoint, and rollout only afterwards when nothing else is competing. So this driver
# trains and stops. Each arm leaves 4 checkpoints (ep10/20/30/40) on disk.
#
# WHY ONE CARD PER ARM: the three runs are independent, so running them in parallel costs
# nothing in correctness and saves ~2/3 of the wall clock. The thing it DOES cost is
# comparability of the throughput numbers (steps/s, s/epoch): three concurrent runs share
# CPU, disk and the DINO encoder's host-side work, so their epoch times are not the solo
# number. Everything that matters for the science -- loss, coverage, mode usage -- is
# unaffected. The per-arm epoch times are recorded anyway, next to P4's solo 959.4 s, so
# the inflation is visible rather than assumed.
set -u
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1

EPOCHS=40
CKPT_EPOCHS=10,20,30,40
TRAIN_SEED=0
# K:physical_gpu. K only changes bin_head's output width (K) and the anchor tensor, so
# memory is essentially identical across arms; the assignment is arbitrary and fixed
# only so the run is reproducible.
ARMS="5:0 10:1 40:2"

mkdir -p logs
echo "=== K sweep start $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h $HOME | tail -1

# GUARD: refuse to start on a card that already has a compute process. The point of this
# sweep is a clean measurement; silently sharing a card would fold another job's load into
# our epoch wall clock with no way to tell afterwards which numbers were affected.
for pair in $ARMS; do
    K=${pair%%:*}; GPU=${pair##*:}
    busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$busy" ]; then
        echo "[ABORT] GPU$GPU already has a compute process:" >&2
        echo "$busy" >&2
        echo "[ABORT] refusing to launch K=$K there" >&2
        exit 1
    fi
done
echo "[guard] all target cards free"

PIDS=""
for pair in $ARMS; do
    K=${pair%%:*}; GPU=${pair##*:}
    CFG=configs/pusht_dino_p256_td_k${K}.yaml
    OUT=outputs/ksweep_k${K}
    LOG=logs/ksweep_k${K}_train.log

    # Bind the card to a config we have actually checked, rather than to a filename we
    # hope is right: a K=40 run on the K=5 anchors would be a silent, expensive mix-up.
    # `tr -d '\r'` because the configs are CRLF and a `$`-anchored grep against a CRLF
    # line never matches -- the guard fired on its own config the first time this ran.
    grep -q "^  n_modes: ${K}$" <(tr -d '\r' < "$CFG") || {
        echo "[ABORT] $CFG has no '  n_modes: $K'" >&2; exit 1; }
    grep -q "^  anchor_artifact: anchors/pusht_kmeans_K${K}.pt$" <(tr -d '\r' < "$CFG") || {
        echo "[ABORT] $CFG does not point at the K=$K anchor artifact" >&2; exit 1; }
    [ -f "anchors/pusht_kmeans_K${K}.pt" ] || { echo "[ABORT] K=$K anchors missing" >&2; exit 1; }

    echo "=== K=$K GPU$GPU cfg=$CFG out=$OUT ==="
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PY -u -m patch_policy.train \
        --config "$CFG" --train-seed "$TRAIN_SEED" --epochs "$EPOCHS" \
        --ckpt-epochs "$CKPT_EPOCHS" --save-dir "$OUT" --attn current > "$LOG" 2>&1 &
    echo "  pid=$! log=$LOG"
    PIDS="$PIDS $!:K$K"
    sleep 5
done

echo "=== all three launched, waiting $(date '+%F %T') ==="
rc_all=0
for entry in $PIDS; do
    pid=${entry%%:*}; K=${entry##*:}
    wait "$pid"; rc=$?
    echo "  K=$K pid=$pid rc=$rc at $(date '+%F %T')"
    [ "$rc" -ne 0 ] && rc_all=1
done

echo "=== K sweep done $(date '+%F %T') rc_all=$rc_all ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h $HOME | tail -1
for pair in $ARMS; do
    K=${pair%%:*}
    echo "--- K=$K ---"
    tail -3 "logs/ksweep_k${K}_train.log"
    ls -l "outputs/ksweep_k${K}/seed0/" 2>/dev/null
done
echo "=== NO ROLLOUTS: run them later, alone, per the instruction ==="
