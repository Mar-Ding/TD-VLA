#!/bin/bash
# Stop the standalone ORIGINAL-DP run once the ep10 checkpoint has landed.
#
# WHY A WATCHER AND NOT JUST A KILL
# The run was launched with epochs=40; the plan was then cut to 10 epochs (user
# decision 2026-09-18: 「只到10个epoch就行，就看看速度和收敛效果」). Killing
# immediately would throw away the ~45 min already spent, and killing at an
# arbitrary moment is not safe: save_model() writes with a plain torch.save
# (train_policy.py:485/487) -- NO tmp+rename, unlike snapshot.pt -- so a signal
# landing inside the write leaves a truncated file that still carries the right
# name. This waits for model_9.pt, confirms its size is unchanged across two
# samples, and only then signals. A partial file cannot pass that test.
#
# WHICH CKPT, AND WHY THERE IS EXACTLY ONE
# save_model(epoch) is gated on (epoch+1) % eval_on_env_freq == 0 and filenames
# are 0-based, so epoch 10 (index 9) is the first and only save point in a
# 10-epoch run: model_9.pt. model_init.pt is the only other ckpt, written before
# epoch 1. Because the process is stopped rather than allowed to finish, the
# end-of-run artifacts -- model_final.pt, completion_idx_final.json, the final
# eval -- are never produced; that is deliberate and recorded in the run log §13.
#
# Usage (on the server, detached):
#   setsid nohup $HOME/td_pusht/scripts/stop_dp_at_ep10.sh \
#       > /dev/null 2>&1 < /dev/null &
set -u

RUN=${RUN:-$HOME/dp_orig/outputs/pusht/dp_orig_40ep}
PIDF="$RUN/pid"
CKPT="$RUN/model_9.pt"
LOG="$RUN/stop_watcher.log"
STABLE=${STABLE:-2}        # consecutive equal-size samples required
INTERVAL=${INTERVAL:-20}   # seconds between samples
CEILING=${CEILING:-7200}   # 2 h; ep10 is ~45 min out at the time of writing
MINSZ=70000000             # a real ckpt is 73,683,490 B; anything less is partial

exec >> "$LOG" 2>&1
echo "=== stop_dp_at_ep10 $(date '+%F %T') waiting for $CKPT (${INTERVAL}s samples, ${STABLE} stable) ==="

PID=$(cat "$PIDF" 2>/dev/null || true)
if [ -z "$PID" ]; then
    echo "[abort] no pid file at $PIDF -- nothing to stop, and nothing to trust"
    exit 2
fi
echo "[init] pid=$PID"

prev=0
same=0
deadline=$(( $(date +%s) + CEILING ))
while :; do
    if ! kill -0 "$PID" 2>/dev/null; then
        # The run can also end by itself (crash, or a 40-epoch completion if the
        # cut were somehow not reached). Either way the ckpt is what matters.
        if [ -f "$CKPT" ]; then
            echo "[note] pid $PID gone on its own; $CKPT present, $(stat -c%s "$CKPT") bytes"
            exit 0
        fi
        echo "[FAIL] pid $PID gone and $CKPT was never written -- the run died, read train.log"
        exit 1
    fi

    if [ -f "$CKPT" ]; then
        sz=$(stat -c%s "$CKPT")
        if [ "$sz" -gt "$MINSZ" ] && [ "$sz" = "$prev" ]; then
            same=$((same + 1))
            if [ "$same" -ge "$STABLE" ]; then
                echo "[stop] $CKPT stable at $sz bytes over $same samples -> SIGTERM $PID"
                kill -TERM "$PID" 2>/dev/null
                for _ in $(seq 10); do
                    kill -0 "$PID" 2>/dev/null || break
                    sleep 1
                done
                if kill -0 "$PID" 2>/dev/null; then
                    echo "[stop] still alive after SIGTERM -> SIGKILL"
                    kill -KILL "$PID" 2>/dev/null
                    sleep 2
                fi
                if kill -0 "$PID" 2>/dev/null; then
                    echo "[FAIL] pid $PID survived SIGKILL -- investigate by hand"
                    exit 1
                fi
                echo "=== done $(date '+%F %T'); run stopped, ep10 ckpt intact ==="
                sha256sum "$CKPT"
                exit 0
            fi
        else
            if [ "$sz" != "$prev" ]; then
                echo "[wait] $CKPT present, $sz bytes -- sampling again in ${INTERVAL}s"
            fi
            same=0
            prev=$sz
        fi
    fi

    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "[FAIL] ${CEILING}s ceiling passed without a stable $CKPT -- run still alive,"
        echo "       nothing was killed. Check train.log before doing anything."
        exit 1
    fi
    sleep "$INTERVAL"
done
