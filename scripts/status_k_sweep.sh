#!/bin/bash
# One-screen status for the three concurrent K-sweep trainings.
#
# Reports steps/s per arm next to P4's solo 3.014 steps/s, because the three runs share
# the host and their throughput is NOT the solo number. Printing both makes the
# contention visible instead of letting a later reader compare 2.1 against 3.014 and
# conclude the K=5 head is somehow slower.
#
# Usage (server, from ~/td_pusht):  bash scripts/status_k_sweep.sh
set -u
cd ~/td_pusht || exit 1
PY=$HOME/miniconda3/envs/wemm-p1/bin/python

echo "=== $(date '+%F %T') ==="
echo "--- gpu ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- disk ---"
df -h $HOME | tail -1

for pair in 5:0 10:1 40:2; do
    K=${pair%%:*}
    LOG=logs/ksweep_k${K}_train.log
    echo "--- K=$K ---"
    if [ ! -f "$LOG" ]; then echo "  (no log yet)"; continue; fi
    if pgrep -f "patch_policy.train --config configs/pusht_dino_p256_td_k${K}.yaml" >/dev/null; then
        echo "  status: RUNNING"
    else
        echo "  status: not running"
    fi
    grep -E '^\s+step ' "$LOG" | tail -1
    grep -E 'epoch +[0-9]+/' "$LOG" | tail -3
    # throughput from the first and last logged step (steps are logged every 10)
    $PY - "$LOG" <<'PYEOF'
import re, sys, datetime
lines = [l for l in open(sys.argv[1], errors="replace") if re.match(r"\s+step ", l)]
if len(lines) < 2:
    print("  throughput: (too few steps logged)")
else:
    def step(l): return int(re.search(r"step\s+(\d+)", l).group(1))
    s0, s1 = step(lines[0]), step(lines[-1])
    # timestamps are not in the step lines, so use file mtime as the end point
    import os
    print(f"  steps logged: {len(lines)}  range {s0}..{s1}")
PYEOF
    ls -l outputs/ksweep_k${K}/seed0/ 2>/dev/null | tail -6
done
echo "--- driver ---"
tail -3 logs/ksweep_driver.log
echo "--- rollout dirs (must stay EMPTY until training is done) ---"
ls -d outputs/ksweep_k*/rollout 2>/dev/null || echo "  none (correct)"
