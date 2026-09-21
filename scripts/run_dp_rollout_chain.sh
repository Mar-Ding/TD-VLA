#!/bin/bash
# The DP arm's post-training rollout sequence: validate the driver, then run it.
#
# WHY CHAINED RATHER THAN TWO COMMANDS
# run_dp_rollout.sh had never been exercised end to end before this point (the run
# log §9 records that: only its flock had been tested). Its `test` mode costs
# ~9.3 min and its `full` mode ~46.6 min, so the order matters -- a broken driver
# must be caught in 9 minutes, not 46. Chaining them in one detached process also
# keeps the whole sequence inside a single `setsid` so it survives the ssh
# channel, which two separate launches would not.
#
# EXCLUSIVITY IS THE POINT OF THE MEASUREMENT
# Both modes go through run_dp_rollout.sh, so both take the same absolute-path
# flock ($HOME/sibling_pkg/audits/gpu2_rollout.lock) and neither will start if
# another rollout holds it. That is deliberate: this project's standing rule is
# that rollout wall-clocks are only comparable when measured with the cards
# otherwise idle -- concurrency has been observed to inflate the clock ~2.9x and
# card identity by 18 %. Before launching, confirm every GPU is free:
#     nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
set -u

BASE=${BASE:-$HOME/td_pusht}
RUN=${RUN:-$HOME/dp_orig/outputs/pusht/dp_orig_40ep}
LOG="$RUN/rollout_driver.log"
DRIVER="$BASE/scripts/run_dp_rollout.sh"

{
    echo "=== chain start $(date '+%F %T') ==="
    echo "--- stage 1/2: test (validates the driver on the smoke ckpt, ~9.3 min) ---"

    if bash "$DRIVER" test; then
        echo "--- stage 1 OK; stage 2/2: full (one n100 on model_9.pt, ~46.6 min) ---"
        if bash "$DRIVER" full; then
            echo "=== chain done $(date '+%F %T') -- both stages OK ==="
            exit 0
        fi
        echo "=== chain FAILED $(date '+%F %T') -- the full rollout did not complete ==="
        exit 1
    fi

    echo "=== chain ABORTED $(date '+%F %T') -- test failed, full NOT run ==="
    echo "    The driver is unvalidated; nothing should be read from a full run"
    echo "    until this passes. See the stage-1 output above."
    exit 1
} >> "$LOG" 2>&1
