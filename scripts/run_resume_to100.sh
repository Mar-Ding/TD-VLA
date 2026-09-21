#!/bin/bash
# Resume the K=20 and K=40 arms from ep40 to ep100, one arm per card.
#
# USER INSTRUCTION (2026-09-18): 「对于 K=20，K=40 的两个试验在 GPU1 和 GPU2 上续训原来的 ckpt，
# 这里你要严格注意续训的合理和连续，我们最后训到 100ep」. ckpt grid chosen by the user:
# 50,60,70,80,90,100 (6 new ckpts per arm).
#
# RESUME CORRECTNESS was settled BEFORE this script was written, by
# `scripts/probe_resume.py`: 12/12 gates PASS on both arms (see the log it printed and
# doc/2026-09-17_K_sweep_run_log.md). The four that would silently corrupt rather
# than crash:
#   * Adam moments present, 160 slots == 160 trainable tensors
#   * optimizer param LAYOUT byte-identical (3 sub-optimizers, 108/6/46) -- AdamW restores
#     by POSITION, so a layout drift attaches the wrong momentum to the wrong tensor
#   * recorded lr [5.5e-5 x3] == configured lr          (train.py:270 hard-stops otherwise)
#   * freeze scope 160 == 160, vqvae_is_fit True
#
# CONTINUITY is structural, not luck, and the code lines are cited here so a later reader
# can re-derive it instead of trusting this comment:
#   * LR is CONSTANT -- no scheduler, no warmup, no decay (train.py:920). Resuming does not
#     re-enter a schedule, because there is no schedule to re-enter.
#   * RNG incl. the dataloader generator is restored (train.py:1051-1059), so the shuffle
#     order for e=41..100 continues exactly where an UNINTERRUPTED 100-epoch run would be.
#   * `for epoch in range(resume_epoch, cfg.epochs)` (train.py:1088) -> e=41..100, exactly
#     60 more epochs, no re-training of ep40.
#   * metrics.json in save_dir is MERGED (train.py:1060-1067), so the curve stays continuous
#     rather than restarting at e=1.
#
# CONCURRENCY CAVEAT. Two runs share host CPU / disk / DINO decode, so their epoch wall
# clocks are NOT solo numbers -- same tradeoff the 3-way K sweep documented. Loss, coverage
# and mode usage are unaffected. Per the standing instruction, every rollout happens later,
# alone, on an otherwise idle card; this driver deliberately trains only.
#
# Usage (on the server, detached):
#   cd ~/td_pusht && ( setsid nohup bash scripts/run_resume_to100.sh \
#       > logs/resume_to100_driver.log 2>&1 < /dev/null & )
set -u
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1

EPOCHS=100
FROM_EPOCH=40
CKPT_EPOCHS=50,60,70,80,90,100
TRAIN_SEED=0

# K:physical_gpu:config:save_dir. GPU assignment is the user's: K=20 -> GPU1, K=40 -> GPU2.
ARMS="20:1:configs/pusht_dino_p256_td.yaml:outputs/p4_td
40:2:configs/pusht_dino_p256_td_k40.yaml:outputs/ksweep_k40"

mkdir -p logs
echo "=== resume-to-100 start $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h $HOME | tail -1

# ---------------------------------------------------------------- preflight
echo "--- preflight ---"
for entry in $ARMS; do
    K=$(echo "$entry" | cut -d: -f1)
    GPU=$(echo "$entry" | cut -d: -f2)
    CFG=$(echo "$entry" | cut -d: -f3)
    OUT=$(echo "$entry" | cut -d: -f4)
    SEED_DIR="$OUT/seed$TRAIN_SEED"
    CK="$SEED_DIR/ckpt_epoch${FROM_EPOCH}_step115680.pt"

    # The card must be free. A foreign job on it folds into our epoch wall clock and, worse,
    # into our memory headroom -- TD's peak is 1956 MB but the DINO decode is host-side.
    busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$busy" ]; then
        echo "[ABORT] GPU$GPU already has a compute process:" >&2
        echo "$busy" >&2
        exit 1
    fi

    # The config must be the one that trained this arm. A K=40 run on the K=20 anchors would
    # be a silent, expensive mix-up -- the same guard run_k_sweep.sh used.
    grep -q "^  n_modes: ${K}$" <(tr -d '\r' < "$CFG") || {
        echo "[ABORT] $CFG has no '  n_modes: $K'" >&2; exit 1; }
    grep -q "^  anchor_artifact: anchors/pusht_kmeans_K${K}.pt$" <(tr -d '\r' < "$CFG") || {
        echo "[ABORT] $CFG does not point at the K=$K anchor artifact" >&2; exit 1; }

    [ -f "$CK" ] || { echo "[ABORT] resume ckpt missing: $CK" >&2; exit 1; }
    SZ=$(stat -c%s "$CK")
    [ "$SZ" -gt 400000000 ] || {
        echo "[ABORT] $CK is only $SZ B -- refusing to resume from a stub" >&2; exit 1; }

    # FIRST-resume guard. If a ckpt above ep40 already exists, this arm has been resumed
    # before; resuming again from ep40 would FORK it, and the metrics merge would keep the
    # later numbers, so the two histories would silently interleave.
    ABOVE=$(ls "$SEED_DIR"/ckpt_epoch*_step*.pt 2>/dev/null \
            | sed -n 's/.*ckpt_epoch\([0-9]*\)_.*/\1/p' | awk -v f="$FROM_EPOCH" '$1>f')
    if [ -n "$ABOVE" ]; then
        echo "[ABORT] K=$K already has ckpt(s) above ep$FROM_EPOCH: $(echo $ABOVE | tr '\n' ' ')" >&2
        echo "[ABORT] resume from the HIGHEST one instead, not from ep$FROM_EPOCH" >&2
        exit 1
    fi

    # Idempotency: refuse if this exact arm is already training.
    if pgrep -af "patch_policy.train.*--save-dir $OUT" > /dev/null; then
        echo "[ABORT] a patch_policy.train for $OUT is already running:" >&2
        pgrep -af "patch_policy.train.*--save-dir $OUT" >&2
        exit 1
    fi

    echo "  K=$K GPU$GPU cfg=$CFG out=$OUT resume=$(basename $CK) ($SZ B) -- preflight OK"
done
echo "[guard] preflight passed for both arms"

# ---------------------------------------------------------------- launch
PIDS=""
for entry in $ARMS; do
    K=$(echo "$entry" | cut -d: -f1)
    GPU=$(echo "$entry" | cut -d: -f2)
    CFG=$(echo "$entry" | cut -d: -f3)
    OUT=$(echo "$entry" | cut -d: -f4)
    CK="$OUT/seed$TRAIN_SEED/ckpt_epoch${FROM_EPOCH}_step115680.pt"
    LOG="logs/$(echo "$OUT" | tr '/' '_')_resume_train.log"

    echo "=== K=$K GPU$GPU -> e=41..$EPOCHS, ckpt at $CKPT_EPOCHS, log=$LOG ==="
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PY -u -m patch_policy.train \
        --config "$CFG" --train-seed "$TRAIN_SEED" --epochs "$EPOCHS" \
        --ckpt-epochs "$CKPT_EPOCHS" --save-dir "$OUT" --resume "$CK" \
        > "$LOG" 2>&1 &
    echo "  pid=$! $(date '+%T')"
    PIDS="$PIDS $!:K$K:$LOG"
    sleep 8
done

# ------------------------------------------------- post-launch gate verification
# A failed resume gate is a SystemExit, i.e. an INSTANT exit, not a slow degradation. So
# waiting 150 s and reading the log distinguishes "training" from "died at load" while it
# is still cheap to relaunch. ResNet-free check, but the failure it catches is the whole
# reason probe_resume.py exists.
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
    grep -E "^\[resume\]" "$LOG" | sed 's/^/  /' || echo "  [warn] no [resume] lines yet"
    if grep -q "FATAL" "$LOG"; then
        echo "  [FAIL] log contains FATAL:"
        grep -n "FATAL" "$LOG" | sed 's/^/    /'
        rc_all=1
        continue
    fi
    if grep -q "merged metrics.json history" "$LOG"; then
        echo "  [ok] resuming, metrics merged, no FATAL"
    else
        echo "  [warn] running but no 'merged metrics.json history' line yet -- inspect"
    fi
done
if [ "$rc_all" -ne 0 ]; then
    echo "=== ABORTING: at least one arm failed to resume. Killing the survivors. ==="
    for entry in $PIDS; do kill "$(echo "$entry" | cut -d: -f1)" 2>/dev/null; done
    exit 1
fi
echo "[guard] both arms are training"

# ---------------------------------------------------------------- wait
echo "=== both arms training, waiting $(date '+%F %T') ==="
for entry in $PIDS; do
    pid=$(echo "$entry" | cut -d: -f1)
    K=$(echo "$entry" | cut -d: -f2)
    wait "$pid"; rc=$?
    echo "  K=$K pid=$pid rc=$rc at $(date '+%F %T')"
    [ "$rc" -ne 0 ] && rc_all=1
done

echo "=== resume-to-100 done $(date '+%F %T') rc_all=$rc_all ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h $HOME | tail -1
for entry in $ARMS; do
    K=$(echo "$entry" | cut -d: -f1)
    OUT=$(echo "$entry" | cut -d: -f4)
    echo "--- K=$K ($OUT) ---"
    ls -l "$OUT/seed0"/*.pt 2>/dev/null | awk '{print "  ", $5, $9}'
done
echo "=== NO ROLLOUTS: they run later, alone (standing instruction) ==="
