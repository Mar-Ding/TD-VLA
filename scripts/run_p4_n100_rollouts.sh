#!/bin/bash
# Bring the K=20 (P4) arm's four rollouts onto the n=100 protocol.
#
# WHY THIS EXISTS
# P4's four rollouts (ep10/20/30/40) are n=20 (P4 log §8). The K-sweep's twelve are
# n=100. That mismatch is why the four-group K table in
# doc/2026-09-19_ablation_record.md §A.4 cannot be read as one curve: K=20 has
# exactly one n=100 point (ep30, run 2026-09-17 08:53 -> 09:02, P4 log §A). This
# driver adds the missing three so all four groups sit on the same protocol.
#
# WHAT IT DOES NOT DO
# It does not touch the n=20 outputs. Each new rollout goes to its own out-dir
# `rollout/ep{EP}_n100`, exactly as the ep30 addendum did, so the §8 numbers stay
# readable side by side with their n=100 counterparts -- and the superset check P4
# §A.4 performed for ep30 (episodes 0-19 bit-identical across the two runs) becomes
# repeatable for the other three.
#
# EXCLUSIVITY
# Every rollout goes through scripts/rollout_locked_gpu2.sh, so it takes the same
# absolute-path cross-driver flock ($HOME/sibling_pkg/audits/gpu2_rollout.lock) as
# the sibling driver and the DP driver. Concurrency has been measured to inflate the clock
# ~2.9x (P4 log §A.2), and the standing rule is that these numbers are only comparable
# when the cards are otherwise idle. Hence the GPU2 preflight below.
#
# BUDGET
# Measured, not estimated: the ep30 n=100 rollout took 543 s of loop (9 m 03 s wall
# including ~90 s of model load + DINO init + env resets). Three ckpts -> ~28 min.
#
# Usage (on the server, detached):
#   cd ~/td_pusht && ( setsid nohup bash scripts/run_p4_n100_rollouts.sh \
#       > logs/p4_n100_driver.log 2>&1 < /dev/null & )
set -u

BASE=${BASE:-$HOME/td_pusht}
CFG=${CFG:-configs/pusht_dino_p256_td.yaml}
S0="$BASE/outputs/p4_td/seed0"
RUN="$BASE/outputs/p4_td/rollout"
LOCKED="$BASE/scripts/rollout_locked_gpu2.sh"

# ep30 is deliberately absent: its n=100 run already exists (P4 log §A).
EPS="10 20 40"

cd "$BASE" || exit 1
echo "=== p4 n100 rollouts start $(date '+%F %T') ==="

# Preflight 1: the three ckpts must be the real thing. P4's own ckpt-arrival guard used
# a 400 MB floor because a pre-training epoch-0 ckpt is ~200 MB; same test here, so a
# truncated or stub file is caught before 9 minutes are spent on it.
for EP in $EPS; do
    CK=$(ls "$S0"/ckpt_epoch${EP}_step*.pt 2>/dev/null | head -1)
    if [ -z "$CK" ]; then echo "[abort] no ckpt for ep$EP in $S0"; exit 1; fi
    SZ=$(stat -c%s "$CK")
    if [ "$SZ" -lt 400000000 ]; then
        echo "[abort] ep$EP ckpt is only $SZ B (< 400 MB) -- refusing to roll out a stub"
        exit 1
    fi
    echo "  ep$EP -> $(basename "$CK") $SZ B"
done

# Preflight 2: a foreign process on GPU2 would be folded into the wall clock with no way
# to separate it afterwards. The lock only excludes other drivers that take the lock.
busy=$(nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$busy" ]; then
    echo "[abort] GPU2 already has a compute process:"
    echo "$busy"
    exit 1
fi
echo "[guard] GPU2 free: $(nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

# A `for` loop, not `printf | while`: a piped loop runs in a subshell, so the `exit 1`
# below would kill only the subshell and this driver would report success after a
# failed rollout. That exact bug was found and fixed in run_dp_rollout.sh; not
# reintroducing it here.
rc_all=0
for EP in $EPS; do
    CK=$(ls "$S0"/ckpt_epoch${EP}_step*.pt 2>/dev/null | head -1)
    OUT="$RUN/ep${EP}_n100"
    JSON="$OUT/rollout_seed0.json"
    if [ -f "$JSON" ]; then
        echo "[skip] ep$EP: $JSON exists (delete it to redo)"
        continue
    fi
    mkdir -p "$OUT"
    echo "--- ep$EP: $(basename "$CK") n=100 $(date '+%F %T') ---"
    if bash "$LOCKED" "$CFG" "$CK" "$OUT" \
            "$BASE/logs/p4_rollout_ep${EP}_n100.log" 0 \
            --num-evals 100 --num-envs 10 --train-seed 0; then
        echo "[ok] ep$EP $(date '+%F %T')"
        grep -E '"final_coverage_mean"|"success_rate_095"|"policy_ms_per_step_mean"|"peak_gpu_mem_mb"' "$JSON"
    else
        echo "[FAIL] ep$EP rc=$? -- see logs/p4_rollout_ep${EP}_n100.log"
        rc_all=1
    fi
done

echo "=== p4 n100 rollouts done $(date '+%F %T') rc_all=$rc_all ==="
nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
exit $rc_all
