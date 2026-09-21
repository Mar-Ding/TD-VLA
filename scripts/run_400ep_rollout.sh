#!/bin/bash
# Rollout + keep-best-prune worker for ONE arm of the 400-epoch run.
#
#   bash scripts/run_400ep_rollout.sh 20     # K=20, outputs/p4_td_400ep
#   bash scripts/run_400ep_rollout.sh 40     # K=40, outputs/ksweep_k40_400ep
#
# Launched ONCE PER ARM as two independent processes. That is deliberate: K=40 trains
# ~1.4x slower per epoch than K=20, so an epoch-synchronised loop would block K=20's
# remaining rollouts for days waiting on K=40 to catch up. Two workers each advance at
# their own arm's pace; they are serialized against each other on GPU0 by
# rollout_locked_gpu0.sh's flock, which is the ONLY thing they share.
#
# Per epoch: wait for the ckpt to land and stop changing -> rollout n=100 -> read
# final_coverage_mean -> update best -> prune.
#
# WHY n=100 AND NOT LESS. It matches the protocol every existing point used
# (--num-evals 100 --num-envs 10 --rng-seed 0 --train-seed 0), so the 400-epoch curve is
# directly comparable to the 100-epoch one. ci95 half-width is ~0.042, i.e. two rollouts
# need ~0.060 apart to be distinguished -- that resolution is what makes the plateau
# question answerable at all.
#
# TIMINGS FROM THIS RUN ARE NOT USABLE AS TIMINGS. Rollouts run on GPU0, which throttles
# to ~1090 MHz under load, while two trainings are running. Coverage is valid; the
# millisecond columns are not. See the header of rollout_locked_gpu0.sh.
#
# PRUNE RULE -- keep = {best, newest among the remaining, the one just rolled out}:
#   * `best` is the whole point of the run; nothing else is worth re-evaluating.
#   * `newest` must survive because it is the only resume point if a training dies.
#   * the one just rolled out is protected so a rollout in flight can never delete its
#     own input mid-run.
# When training is over, prune collapses to {best} only -- that is the agreed end state
# (~0.5 GB/arm instead of ~37 GB).
#
# RESTARTABLE: an epoch whose rollout_seed0.json already exists is skipped, so this can be
# killed and relaunched at any point. Re-running it is also how you resume after a reboot.
#
# Usage (on the server, detached; ONE PER ARM):
#   cd ~/td_pusht && ( setsid nohup bash scripts/run_400ep_rollout.sh 20 \
#       > logs/run400ep_rollout_k20.log 2>&1 < /dev/null & )
set -u
BASE=${BASE:-$HOME/td_pusht}
cd "$BASE" || exit 1

K=${1:?usage: run_400ep_rollout.sh <20|40>}
case "$K" in
    20) CFG=configs/pusht_dino_p256_td.yaml;      OUT=outputs/p4_td_400ep ;;
    40) CFG=configs/pusht_dino_p256_td_k40.yaml;  OUT=outputs/ksweep_k40_400ep ;;
    *)  echo "usage: run_400ep_rollout.sh <20|40>" >&2; exit 1 ;;
esac
SEED_DIR="$BASE/$OUT/seed0"
LOCKED="$BASE/scripts/rollout_locked_gpu0.sh"
ROLL="$BASE/$OUT/rollout"

EPS=$(seq 5 5 400)
N_EVALS=100; N_ENVS=10; RNG_SEED=0; TRAIN_SEED=0
# 5 epochs at the contended K=40 rate (~1400 s/ep) is ~117 min; 240 gives headroom for a
# slow first epoch without masking a dead training for long.
WAIT_MIN=240
mkdir -p "$BASE/logs"

stamp() { date '+%F %T'; }
train_alive() { pgrep -f "patch_policy.train.*--save-dir $OUT" >/dev/null; }

# Echo the ckpt path for epoch $1 once it has landed and stopped changing, or "" if it
# will never appear. Diagnostics go to stderr so they land in the driver log without
# polluting the captured path.
#
# A TIMEOUT WHILE TRAINING IS ALIVE MUST NOT SKIP THE EPOCH. A skipped epoch is a hole in
# the curve, and holes are what this run exists to avoid. Contended K=40 does ~1400 s per
# epoch, so 5 epochs can exceed WAIT_MIN; when that happens we keep waiting (up to 4
# cycles) rather than moving on. Only an absent ckpt with NO training process behind it
# means the epoch is genuinely gone.
#
# The >400 MB floor: a pre-training epoch-0 ckpt is ~200 MB and a truncated write is
# anything at all. Same test every earlier driver used. Size must be stable across two
# samples so a file mid-write is never handed to eval_rollout.
wait_ckpt() {
    local ep=$1 prev="" cur="" sz tries=0 cycles=0
    while :; do
        cur=$(ls "$SEED_DIR"/ckpt_epoch${ep}_step*.pt 2>/dev/null | head -1)
        if [ -n "$cur" ]; then
            sz=$(stat -c%s "$cur" 2>/dev/null || echo 0)
            if [ "$sz" -gt 400000000 ] && [ "$sz" = "$prev" ]; then echo "$cur"; return 0; fi
            prev="$sz"; cur=""
        fi
        if [ -z "$cur" ] && ! train_alive; then
            echo "[miss] $(stamp) K=$K ep$ep: no ckpt and no training for $OUT -- stopping" >&2
            echo ""; return 1
        fi
        tries=$((tries + 1))
        if [ "$tries" -ge $((WAIT_MIN * 2)) ]; then
            cycles=$((cycles + 1))
            if [ "$cycles" -ge 4 ]; then
                echo "[warn] $(stamp) K=$K ep$ep still absent after $((4 * WAIT_MIN)) min but training is alive -- giving up on this epoch" >&2
                echo ""; return 1
            fi
            echo "[wait] $(stamp) K=$K ep$ep past ${WAIT_MIN} min (cycle $cycles/4), training alive -- continuing to wait" >&2
            tries=0
        fi
        sleep 30
    done
}

echo "=== 400ep rollout worker K=$K start $(stamp) ==="
echo "  config=$CFG  seed_dir=$SEED_DIR  n=$N_EVALS envs=$N_ENVS rng=$RNG_SEED"
df -h $HOME | tail -1

[ -f "$CFG" ] || { echo "[abort] missing $CFG"; exit 1; }
grep -q "^  n_modes: ${K}$" <(tr -d '\r' < "$CFG") \
    || { echo "[abort] $CFG is not the K=$K config"; exit 1; }

BEST_CK=""; BEST_COV=""; BEST_EP=""
rc_all=0; ran=0; skipped=0; missing=0

for EP in $EPS; do
    JSON="$ROLL/ep${EP}/rollout_seed0.json"
    LOG="$BASE/logs/run400ep_rollout_k${K}_ep${EP}.log"

    if [ -f "$JSON" ]; then
        skipped=$((skipped + 1))
        # still fold an already-done epoch into the best tracking, so a relaunch after a
        # kill does not forget which ckpt was best (and then delete it)
        cov=$($HOME/miniconda3/envs/wemm-p1/bin/python -c \
              "import json,sys;print(json.load(open(sys.argv[1]))['base_metrics']['final_coverage_mean'])" \
              "$JSON" 2>/dev/null)
        if [ -n "${cov:-}" ] && { [ -z "$BEST_COV" ] || awk -v a="$cov" -v b="$BEST_COV" 'BEGIN{exit !(a>b)}'; }; then
            BEST_CK=$(ls "$SEED_DIR"/ckpt_epoch${EP}_step*.pt 2>/dev/null | head -1)
            BEST_COV="$cov"; BEST_EP="$EP"
        fi
        continue
    fi

    # ---- wait for the ckpt to land and stop changing ------------------------------
    CK=$(wait_ckpt "$EP") || CK=""
    if [ -z "$CK" ]; then
        missing=$((missing + 1))
        # training gone => every later epoch is gone too
        train_alive || break
        continue
    fi

    # ---- rollout ------------------------------------------------------------------
    mkdir -p "$ROLL/ep${EP}"
    echo "[run ] $(stamp) K=$K ep$EP $(basename "$CK") n=$N_EVALS"
    T0=$(date +%s)
    if bash "$LOCKED" "$CFG" "$CK" "$ROLL/ep${EP}" "$LOG" "$RNG_SEED" \
            --num-evals "$N_EVALS" --num-envs "$N_ENVS" --train-seed "$TRAIN_SEED"; then
        rc=0
    else
        rc=$?
    fi
    T1=$(date +%s)
    ran=$((ran + 1))
    if [ "$rc" -ne 0 ] || [ ! -f "$JSON" ]; then
        echo "[fail] $(stamp) K=$K ep$EP rc=$rc wall=$((T1-T0))s -- see $LOG"
        rc_all=1
        # do NOT delete a ckpt we failed to evaluate: it cannot be ranked, and it may be
        # the best one. Leave it; the next pass will retry the rollout.
        continue
    fi

    cov=$($HOME/miniconda3/envs/wemm-p1/bin/python -c \
          "import json,sys;print(json.load(open(sys.argv[1]))['base_metrics']['final_coverage_mean'])" \
          "$JSON")
    echo "[done] $(stamp) K=$K ep$EP cov=$cov wall=$((T1-T0))s best=${BEST_EP:-none}(${BEST_COV:-—})"

    if [ -z "$BEST_COV" ] || awk -v a="$cov" -v b="$BEST_COV" 'BEGIN{exit !(a>b)}'; then
        BEST_CK="$CK"; BEST_COV="$cov"; BEST_EP="$EP"
    fi

    # ---- prune: keep {best, newest}, never the one just evaluated ------------------
    NEWEST=$(ls -t "$SEED_DIR"/ckpt_epoch*_step*.pt 2>/dev/null | head -1)
    for old in "$SEED_DIR"/ckpt_epoch*_step*.pt; do
        [ -e "$old" ] || continue
        [ "$old" = "$BEST_CK" ] && continue
        [ "$old" = "$NEWEST" ]  && continue
        [ "$old" = "$CK" ]      && continue
        echo "[prune] $(stamp) rm $(basename "$old") (best=${BEST_EP:-none})"
        rm -f "$old"
    done
done

# ---- training over? collapse to {best} only ---------------------------------------
if ! train_alive; then
    echo "[final] $(stamp) K=$K training has ended; collapsing to best-only (ep${BEST_EP:-none})"
    for old in "$SEED_DIR"/ckpt_epoch*_step*.pt; do
        [ -e "$old" ] || continue
        [ "$old" = "$BEST_CK" ] && continue
        echo "[prune] $(stamp) rm $(basename "$old") (final, best=${BEST_EP:-none})"
        rm -f "$old"
    done
else
    echo "[final] $(stamp) K=$K training still alive; keeping {best,newest}"
fi

echo "=== 400ep rollout worker K=$K end $(stamp) rc_all=$rc_all ==="
echo "  ran=$ran skipped=$skipped missing=$missing"
echo "  best: ep${BEST_EP:-none} cov=${BEST_COV:-—}  ckpt=$(basename "${BEST_CK:-none}")"
echo "  rollouts on disk: $(ls "$ROLL"/*/rollout_seed0.json 2>/dev/null | wc -l)/80"
df -h $HOME | tail -1
exit $rc_all
