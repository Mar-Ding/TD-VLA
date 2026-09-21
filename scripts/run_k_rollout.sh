#!/bin/bash
# Roll out every K-sweep ckpt, ONE AT A TIME, on GPU2, in this order:
#     K=5  (ep10,20,30,40)  ->  K=10 (ep10,20,30,40)  ->  K=40 (ep10,20,30,40)
# 12 rollouts, n=100 / 10 envs / rng_seed 0 / train_seed 0.
#
# WHY THIS IS A DRIVER AND NOT 12 HAND-TYPED COMMANDS
# The whole point of the sequence is that the timings are comparable. A skipped
# `--train-seed 0`, a swapped --out, or two rollouts overlapping on GPU2 would each
# silently poison the table, and none of the three would show up as an error. So the
# order, the flags and the serialization live in one file that can be read and diffed.
#
# WHY SERIAL: each n100 rollout peaks ~10GB on a 24GB card; two at once contend or OOM,
# and worse, the two arms' ms/step numbers then measure each other rather than the model.
# Serialization is enforced by the flock inside rollout_locked_gpu2.sh, which takes the
# ABSOLUTE path of the one real lock file that the sibling driver also holds. See that
# file's header -- the relative-path version of the lock would have let two drivers
# each acquire "their own" lock and run concurrently while both believed they were safe.
#
# RESUMABLE: a rollout whose $OUT/rollout_seed0.json already exists is skipped, so the
# driver can be killed and relaunched at any point without redoing finished work.
#
# THE THERMOMETER: GPU0 on this box thermally throttles to ~840 MHz of 2100 (85C vs its
# own 83C target) and ran the K=5 training at 1.79x the wall clock of the same work on a
# cool card. Every rollout here therefore runs on GPU2, and the driver samples GPU2's
# temperature/clocks/power before and after each rollout so a contaminated run can be
# identified after the fact instead of being averaged into the table. The second, sharper
# thermometer is `encoder_ms_per_step_mean` in the JSON: byte-identical work across arms,
# reading ~16.7-17.4 ms clean and 48-58 ms when contended.
set -u
cd "$(dirname "$0")/.."

PY=$HOME/miniconda3/envs/wemm-p1/bin/python
DLOG=logs/k_rollout_driver.log
N_EVALS=100
N_ENVS=10
RNG_SEED=0
TRAIN_SEED=0
mkdir -p logs

stamp() { date '+%F %T'; }

gpu_note() {   # $1 = label
    local apps clk
    apps=$(nvidia-smi -i 2 --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
    clk=$(nvidia-smi -i 2 --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu \
              --format=csv,noheader)
    echo "  [$1] $(stamp) gpu2 {$clk} compute_apps=${apps:-none}" >> "$DLOG"
}

# --- preflight: refuse to start while any sweep training is alive -------------------
if pgrep -f "patch_policy.train --config configs/pusht_dino_p256_td_k" >/dev/null; then
    echo "[abort] $(stamp) a K-sweep training is still running; rollouts must wait for it" >> "$DLOG"
    exit 3
fi
if pgrep -f "patch_policy.eval_rollout" >/dev/null; then
    echo "[abort] $(stamp) another eval_rollout is already running" >> "$DLOG"
    exit 3
fi

echo "=== K-sweep rollout driver start $(stamp) ===" >> "$DLOG"
echo "    order K=5 -> K=10 -> K=40, each ep10/20/30/40 | n=$N_EVALS envs=$N_ENVS rng=$RNG_SEED train=$TRAIN_SEED" >> "$DLOG"

for K in 5 10 40; do
  CFG=configs/pusht_dino_p256_td_k${K}.yaml
  for EP in 10 20 30 40; do
    CKPT=$(ls outputs/ksweep_k${K}/seed0/ckpt_epoch${EP}_step*.pt 2>/dev/null | head -1)
    OUT=outputs/ksweep_k${K}/rollout/ep${EP}
    RLOG=logs/rollout_ksweep_k${K}_ep${EP}.log

    if [ -f "$OUT/rollout_seed0.json" ]; then
        echo "[skip] $(stamp) K=$K ep$EP already has rollout_seed0.json" >> "$DLOG"
        continue
    fi
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "[MISS] $(stamp) K=$K ep$EP: no ckpt matching ckpt_epoch${EP}_step*.pt" >> "$DLOG"
        continue
    fi
    if [ ! -f "$CFG" ]; then
        echo "[MISS] $(stamp) K=$K ep$EP: no config $CFG" >> "$DLOG"
        continue
    fi

    echo "[run ] $(stamp) K=$K ep$EP $(basename "$CKPT") $(( $(stat -c %s "$CKPT") / 1048576 ))MB" >> "$DLOG"
    gpu_note "pre  K=$K ep$EP"
    T0=$(date +%s)

    bash scripts/rollout_locked_gpu2.sh "$CFG" "$CKPT" "$OUT" "$RLOG" "$RNG_SEED" \
        --num-evals "$N_EVALS" --num-envs "$N_ENVS" --train-seed "$TRAIN_SEED"
    rc=$?

    T1=$(date +%s)
    gpu_note "post K=$K ep$EP"
    echo "[done] $(stamp) K=$K ep$EP rc=$rc wall=$((T1-T0))s" >> "$DLOG"

    "$PY" - "$OUT/rollout_seed0.json" >> "$DLOG" 2>&1 <<'PYEOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("       (no rollout json written)"); raise SystemExit
b = json.load(open(p)).get("base_metrics", {})
for k in ("final_coverage_mean", "final_coverage_ci95", "max_coverage_mean",
          "success_rate_095", "success_rate_ci95", "n_episodes_reached_095",
          "encoder_ms_per_step_mean", "policy_ms_per_step_mean",
          "env_step_ms_per_step_mean", "total_ms_per_step_mean",
          "peak_gpu_mem_mb", "episode_steps_mean"):
    if k in b:
        v = b[k]
        print(f"       {k} = {v:.4f}" if isinstance(v, float) else f"       {k} = {v}")
PYEOF
  done
done

echo "=== K-sweep rollout driver end $(stamp) ===" >> "$DLOG"
