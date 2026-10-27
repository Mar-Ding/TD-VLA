#!/bin/bash
# 400-epoch TD run, FROM SCRATCH, one arm per card:
#   K=20 -> GPU1   outputs/p4_td_400ep
#   K=40 -> GPU2   outputs/ksweep_k40_400ep
#
# USER INSTRUCTION (2026-09-20): 「在GPU1，GPU2上进行400epoch的完整试验，分别跑K=20,和K=40的
# 训练，GPU0用于rollout，每5个epoch的ckpt进行次rollout，这次不是续训是从头开始」
#
# FROM SCRATCH, AND THAT IS ENFORCED. No `--resume`. The preflight refuses to start if the
# save dir already holds any ckpt_epoch*.pt, so an accidental relaunch cannot silently
# continue a half-finished run and then get read as a fresh 400-epoch curve.
#
# WHY 400 AND NOT 100. Both arms already have a 100-epoch curve (K=20: outputs/p4_td;
# K=40: outputs/ksweep_k40), and 2026-09-19's result was that ep40->100 bought +0.014 /
# +0.034 -- both below the ~0.060 an n=100 pair can resolve -- while train_loss kept
# falling. The plateau showed up at ep30-60. What that leaves genuinely unmeasured is the
# regime PAST 100 epochs: "ep30-60 is the ceiling" and "it climbs again after ep100" are
# both still open. This run is what closes it. It is not a rerun of the 100-epoch arms.
#
# CKPT CADENCE. --ckpt-epochs is every 5th epoch (5,10,...,400 = 80 per arm) because the
# rollout cadence is per-ckpt and 5 epochs is the resolution needed to see a plateau move.
# The rollouts are NOT this script's job -- see run_400ep_rollout.sh -- so that a killed
# rollout worker can be relaunched without touching a live training.
#
# DISK. 80 ckpts x ~470 MB = ~37.6 GB per arm, ~75 GB for both. That fits in the 178 GB
# free, but only the keep-best pair is meant to survive: run_400ep_rollout.sh prunes each
# arm down to {current best, newest} after every rollout. If that worker is NOT running,
# this training fills ~75 GB.
#
# CONCURRENCY CAVEAT. Two runs share host CPU / disk / DINO decode, so their epoch wall
# clocks are NOT solo numbers (K=40 measured 1265-1335 s/epoch contended vs 947 s/epoch
# solo). Loss, coverage and mode usage are unaffected. Per the standing instruction every
# rollout happens later, alone on its own card.
#
# Usage (on the server, detached):
#   cd ~/td_pusht && ( setsid nohup bash scripts/run_400ep_train.sh \
#       > logs/run400ep_train_driver.log 2>&1 < /dev/null & )
set -u
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1

EPOCHS=400
TRAIN_SEED=0
CKPT_EPOCHS=$(seq -s, 5 5 400)

# K:physical_gpu:config:save_dir
ARMS="20:1:configs/pusht_dino_p256_td.yaml:outputs/p4_td_400ep
40:2:configs/pusht_dino_p256_td_k40.yaml:outputs/ksweep_k40_400ep"

mkdir -p logs
echo "=== 400ep from-scratch train start $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
df -h $HOME | tail -1
echo "  ckpt grid: ep5..$EPOCHS every 5 -> $(echo "$CKPT_EPOCHS" | tr ',' '\n' | wc -l) ckpts/arm"

# ---------------------------------------------------------------- preflight
echo "--- preflight ---"
for entry in $ARMS; do
    K=$(echo "$entry" | cut -d: -f1)
    GPU=$(echo "$entry" | cut -d: -f2)
    CFG=$(echo "$entry" | cut -d: -f3)
    OUT=$(echo "$entry" | cut -d: -f4)
    SEED_DIR="$OUT/seed$TRAIN_SEED"

    # The card must be free. A foreign job folds into our epoch wall clock and eats the
    # memory headroom.
    busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$busy" ]; then
        echo "[ABORT] GPU$GPU already has a compute process:" >&2
        echo "$busy" >&2
        exit 1
    fi

    # The config must be the one that defines this arm's K. A K=40 run on the K=20 anchors
    # would be a silent, expensive mix-up.
    grep -q "^  n_modes: ${K}$" <(tr -d '\r' < "$CFG") || {
        echo "[ABORT] $CFG has no '  n_modes: $K'" >&2; exit 1; }
    grep -q "^  anchor_artifact: anchors/pusht_kmeans_K${K}.pt$" <(tr -d '\r' < "$CFG") || {
        echo "[ABORT] $CFG does not point at the K=$K anchor artifact" >&2; exit 1; }
    [ -f "anchors/pusht_kmeans_K${K}.pt" ] || {
        echo "[ABORT] missing anchors/pusht_kmeans_K${K}.pt" >&2; exit 1; }

    # FROM-SCRATCH GUARD. Any existing ckpt means this dir is not empty, and launching on
    # top of it would produce a curve that is neither the old run nor a fresh one.
    EXIST=$(ls "$SEED_DIR"/ckpt_epoch*_step*.pt 2>/dev/null | head -5)
    if [ -n "$EXIST" ]; then
        echo "[ABORT] $SEED_DIR is not empty -- this run is FROM SCRATCH:" >&2
        echo "$EXIST" >&2
        echo "[ABORT] choose a new --save-dir, or delete the dir deliberately." >&2
        exit 1
    fi

    # Idempotency: refuse if this exact arm is already training.
    if pgrep -af "patch_policy.train.*--save-dir $OUT" > /dev/null; then
        echo "[ABORT] a patch_policy.train for $OUT is already running:" >&2
        pgrep -af "patch_policy.train.*--save-dir $OUT" >&2
        exit 1
    fi

    echo "  K=$K GPU$GPU cfg=$CFG out=$OUT (empty, from scratch) -- preflight OK"
done
echo "[guard] preflight passed for both arms"

# ---------------------------------------------------------------- launch
PIDS=""
for entry in $ARMS; do
    K=$(echo "$entry" | cut -d: -f1)
    GPU=$(echo "$entry" | cut -d: -f2)
    CFG=$(echo "$entry" | cut -d: -f3)
    OUT=$(echo "$entry" | cut -d: -f4)
    LOG="logs/$(echo "$OUT" | tr '/' '_')_train.log"

    echo "=== K=$K GPU$GPU -> e=1..$EPOCHS from scratch, ckpt every 5, log=$LOG ==="
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PY -u -m patch_policy.train \
        --config "$CFG" --train-seed "$TRAIN_SEED" --epochs "$EPOCHS" \
        --ckpt-epochs "$CKPT_EPOCHS" --save-dir "$OUT" --attn current \
        > "$LOG" 2>&1 &
    echo "  pid=$! $(date '+%T')"
    PIDS="$PIDS $!:K$K:$LOG"
    sleep 8
done

# ------------------------------------------------- post-launch gate verification
# A failed gate is a SystemExit, i.e. an INSTANT exit, not a slow degradation. Waiting
# 150 s distinguishes "training" from "died at load" while relaunching is still cheap.
echo "=== post-launch verification (waiting 150 s) $(date '+%F %T') ==="
sleep 150
rc_all=0
for entry in $PIDS; do
    K=$(echo "$entry" | cut -d: -f2)
    LOG=$(echo "$entry" | cut -d: -f3)
    pid=$(echo "$entry" | cut -d: -f1)
    echo "--- K=$K pid=$pid ---"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  [FAIL] process is GONE after 150 s -- it exited at load. Log tail:"
        tail -25 "$LOG" | sed 's/^/    /'
        rc_all=1
        continue
    fi
    if grep -q "FATAL" "$LOG"; then
        echo "  [FAIL] log contains FATAL:"
        grep -n "FATAL" "$LOG" | sed 's/^/    /'
        rc_all=1
        continue
    fi
    # `--epochs 400` must have been honoured; if the yaml's 100 won, the run is 4x short
    # and every later "ep400" claim would be false.
    if ! grep -qE "epochs[= ]+400|EPOCHS[= ]+400|e=1\.\.400" "$LOG"; then
        echo "  [warn] no explicit '400 epochs' line in the log -- confirm --epochs took effect"
        grep -nE "epoch|EPOCH" "$LOG" | head -5 | sed 's/^/    /'
    fi
    echo "  [ok] alive after 150 s, no FATAL"
done
if [ "$rc_all" -ne 0 ]; then
    echo "=== ABORTING: at least one arm failed to launch. Killing the survivors. ==="
    for entry in $PIDS; do kill "$(echo "$entry" | cut -d: -f1)" 2>/dev/null; done
    exit 1
fi
echo "[guard] both arms are training"

echo "=== NEXT: launch the two rollout workers, one per arm ==="
echo "    cd ~/td_pusht && ( setsid nohup bash scripts/run_400ep_rollout.sh 20 \\"
echo "        > logs/run400ep_rollout_k20.log 2>&1 < /dev/null & )"
echo "    cd ~/td_pusht && ( setsid nohup bash scripts/run_400ep_rollout.sh 40 \\"
echo "        > logs/run400ep_rollout_k40.log 2>&1 < /dev/null & )"

# ---------------------------------------------------------------- wait
echo "=== both arms training, waiting $(date '+%F %T') ==="
for entry in $PIDS; do
    pid=$(echo "$entry" | cut -d: -f1)
    K=$(echo "$entry" | cut -d: -f2)
    wait "$pid"; rc=$?
    echo "  K=$K pid=$pid rc=$rc at $(date '+%F %T')"
    [ "$rc" -ne 0 ] && rc_all=1
done

echo "=== 400ep train done $(date '+%F %T') rc_all=$rc_all ==="
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader
df -h $HOME | tail -1
for entry in $ARMS; do
    OUT=$(echo "$entry" | cut -d: -f4)
    K=$(echo "$entry" | cut -d: -f1)
    echo "--- K=$K ($OUT) ckpts on disk: $(ls "$OUT/seed0"/ckpt_epoch*_step*.pt 2>/dev/null | wc -l) ---"
    ls -l "$OUT/seed0"/*.pt 2>/dev/null | awk '{print "  ", $5, $9}' | tail -5
    echo "    rollouts done: $(ls "$OUT/rollout"/*/rollout_seed0.json 2>/dev/null | wc -l)/80"
done
exit $rc_all
