#!/bin/bash
# THE DELIBERATE OPPOSITE OF run_k_rollout.sh: run the three K arms' ep40 checkpoints
# SIMULTANEOUSLY, one per GPU, to measure what concurrency costs.
#
# This exists because every other rollout in this project is serialized on purpose, and
# serialization removes the very quantity the user now wants. So this driver breaks
# serialization -- but only in the one place that is safe to break it, and only for the
# duration of one measurement.
#
# WHAT MAKES THE RESULT INTERPRETABLE
# The exclusive series (run_k_rollout.sh) ran all 12 rollouts on GPU2. This series puts
# K=5 on GPU0, K=10 on GPU1, K=40 on GPU2 -- the same cards the trainings used. So:
#   * K=40 is the clean control: same ckpt, same protocol, same card (GPU2) as its
#     exclusive run, with the ONLY difference being that two neighbours are now running.
#     Its delta is attributable to concurrency alone.
#   * K=5 and K=10 changed card as well as concurrency, and GPU0 is separately thermally
#     capped (85C vs its own 83C target, 840MHz of 2100 -- see the K-sweep run log S3).
#     Their deltas conflate "different card" with "neighbours". They are recorded, not
#     attributed. Separating them needs a solo re-run on GPU0 and GPU1.
# Nothing else changes: same ckpt, same n=100 / 10 envs / rng_seed 0 / train_seed 0, and
# the outputs go to a NEW directory (ep40_concurrent) so the exclusive JSONs are untouched.
#
# THE LOCK. rollout_locked_gpu2.sh flocks per-rollout, which would serialize the three and
# measure nothing. This driver instead takes the SAME absolute lock file ONCE for the whole
# concurrent window, then launches all three inside it. That keeps the cross-driver
# exclusion against the sibling driver intact while letting our own three run together.
# Its window is short (one n100 per arm), so holding the lock this way cannot starve the
# other driver for long.
set -u
cd "$(dirname "$0")/.."

PY=$HOME/miniconda3/envs/wemm-p1/bin/python
LOCK=${GPU2_LOCK:-$HOME/sibling_pkg/audits/gpu2_rollout.lock}
DLOG=logs/k_ep40_concurrent_driver.log
CSV=logs/k_ep40_concurrent_gpu.csv
ARMS="5:0 10:1 40:2"
CKPT_STEP=115680

mkdir -p logs

if pgrep -f "patch_policy.eval_rollout" >/dev/null; then
    echo "[abort] $(date '+%F %T') an eval_rollout is already running; this test needs the cards to itself" >> "$DLOG"
    exit 3
fi
for pair in $ARMS; do
    G=${pair##*:}
    apps=$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader)
    if [ -n "$apps" ]; then
        echo "[abort] $(date '+%F %T') GPU$G is not free ($apps); a foreign job would be a third confound" >> "$DLOG"
        exit 3
    fi
done

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -x -w 600 9 || {
    echo "[abort] $(date '+%F %T') could not take $LOCK within 10 min; the sibling driver holds it" >> "$DLOG"
    exit 2
}

echo "=== K ep40 CONCURRENT driver start $(date '+%F %T') ===" >> "$DLOG"
echo "    arms $ARMS | n=100 envs=10 rng_seed=0 train_seed=0 | out -> ep40_concurrent" >> "$DLOG"
echo "    lock $LOCK held for the whole window (blocks the sibling driver, not our own three)" >> "$DLOG"

# Sample all three cards while the rollouts run. The point of the run is the *shape* of the
# contention, and that shape is only visible as a time series -- a before/after pair would
# miss the ramp-up, which is where the neighbours start stealing SMs.
: > "$CSV"
( while :; do
    echo "$(date '+%F %T'),$(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits | tr '\n' ';')" >> "$CSV"
    sleep 15
  done ) &
SAMPLER=$!

PIDS=""
for pair in $ARMS; do
    K=${pair%%:*}; G=${pair##*:}
    CFG=configs/pusht_dino_p256_td_k${K}.yaml
    CKPT=outputs/ksweep_k${K}/seed0/ckpt_epoch40_step${CKPT_STEP}.pt
    OUT=outputs/ksweep_k${K}/rollout/ep40_concurrent
    LOG=logs/rollout_ksweep_k${K}_ep40_concurrent.log
    if [ ! -f "$CKPT" ]; then
        echo "[MISS] $(date '+%F %T') K=$K ckpt $CKPT not found" >> "$DLOG"; continue
    fi
    mkdir -p "$OUT"
    (
        s=$(date +%s)
        CUDA_VISIBLE_DEVICES=$G $PY -u -m patch_policy.eval_rollout \
            --config "$CFG" --ckpt "$CKPT" --out "$OUT" --rng-seed 0 \
            --num-evals 100 --num-envs 10 --train-seed 0 --attn current > "$LOG" 2>&1
        rc=$?
        e=$(date +%s)
        echo "[done] $(date '+%F %T') K=$K gpu=$G rc=$rc wall=$((e-s))s" >> "$DLOG"
    ) &
    echo "[launch] $(date '+%F %T') K=$K gpu=$G pid=$! -> $(basename "$OUT")" >> "$DLOG"
    PIDS="$PIDS $!"
done

wait $PIDS
kill $SAMPLER 2>/dev/null
echo "=== K ep40 CONCURRENT driver end $(date '+%F %T') ===" >> "$DLOG"

for pair in $ARMS; do
    K=${pair%%:*}
    echo "--- K=$K ---" >> "$DLOG"
    $PY - "outputs/ksweep_k${K}/rollout/ep40_concurrent/rollout_seed0.json" >> "$DLOG" 2>&1 <<'PYEOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("    (no rollout json)"); raise SystemExit
b = json.load(open(p)).get("base_metrics", {})
for k in ("final_coverage_mean", "final_coverage_ci95", "success_rate_095",
          "encoder_ms_per_step_mean", "policy_ms_per_step_mean",
          "env_step_ms_per_step_mean", "total_ms_per_step_mean", "peak_gpu_mem_mb"):
    v = b.get(k)
    if v is None: continue
    print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")
PYEOF
done
