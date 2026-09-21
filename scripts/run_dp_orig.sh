#!/bin/bash
# Launcher for the standalone ORIGINAL Diffusion-Policy arm in ~/dp_orig.
#
# This arm is deliberately NOT part of our harness: it is upstream's own
# train_policy.py + train_pusht_diffusion.yaml, run from a byte-faithful copy
# (see scripts/deploy_dp_orig.sh). Consequences to keep in mind when reading
# any number that comes out of it:
#   * window 2 (ours is 5), pred_horizon 6, n_embd 256 / n_head 4, batch 256,
#     lr 1e-4 -- so DP-vs-TD is NOT a single-variable comparison
#   * it runs in the `patch-policy` conda env, not `wemm-p1`
#   * inference is full DDPM: 100 denoiser forwards per action chunk, using the
#     EMA weights. That makes EVAL, not training, the dominant cost.
#
# MODE=smoke  : epochs=1, 10-episode evals. Validates dataset/encoder/DP
#               model/optimizer/train step/ckpt save/eval path end to end.
# MODE=full   : epochs=10 (amended 2026-09-18, see below), ckpt at every eval
#               (deploy deviation 2), inline evals kept SMALL on purpose -- see
#               the note below.
#
# 2026-09-18 AMENDMENT: the plan was cut from 40 to 10 epochs. The user's reason
# is that this arm exists to show SPEED and CONVERGENCE, not to be a 40-epoch
# final result: 「只到10个epoch就行，就看看速度和收敛效果」. 10 epochs still
# brackets the arm's cost honestly (the per-epoch wall and the policy-forward
# latency do not depend on epoch count) and it produces exactly one rollout
# point, model_9.pt. The already-running 40-epoch job was stopped by
# scripts/stop_dp_at_ep10.sh at the moment that ckpt landed rather than killed
# on the spot -- see that script for why mid-write kills are unsafe. EPS can be
# raised again (EPS=40) if the arm is ever extended; nothing else changes.
#
# WHY THE INLINE EVALS ARE SMALLER THAN UPSTREAM'S
# Upstream's own config would run five 100-episode evals (4 inline + 1 final).
# At ~100 forwards per action chunk that is the single largest item in the
# budget. The reported numbers are supposed to come from the n100 rollouts run
# AFTER training finishes (the standing protocol), so the inline evals only have
# to (a) trigger the per-eval ckpt save -- note the save lives INSIDE the
# eval branch, so the eval cannot simply be switched off -- and (b) give a
# progress signal. Hence num_env_evals=10 / num_final_evals=10. This is a CLI
# override, not a source change.
#
# WHY num_workers IS 0 AND NOT UPSTREAM'S 23
# Measured, not guessed. With the config's num_workers: 23 the run died on the
# FIRST batch: `RuntimeError: DataLoader worker (pid ...) is killed by signal:
# Killed`. ps showed why -- each `pt_data_worker` had grown to ~20 GB RSS
# because PushTDataset materialises the whole dataset inside every worker, so
# 23 workers demand ~460 GB on a 251 GB box. The OOM killer takes workers out.
# Nothing else on the machine was at fault (the 61 GB "used" in free(1) was our
# own workers). num_workers=0 keeps exactly one copy, in the main process.
# Upstream's DP config had never been run, so its 23 was never validated.
set -u

MODE=${1:-smoke}
GPU=${GPU:-2}                      # GPU0 is thermally capped (see the memory note)
NW=${NW:-0}                        # see the num_workers note above
BASE=$HOME/dp_orig
PY=$HOME/miniconda3/envs/patch-policy/bin/python

case "$MODE" in
  smoke) RUN=smoke_dp_ep1; EPS=1;  NEV=10;  NFIN=10  ;;
  # The run directory keeps the name dp_orig_40ep even though the arm now stops
  # at 10: that is where the live job's pid file, ckpts and watcher log are, and
  # renaming it mid-run would break all three plus every path in the run log.
  # The name records the ORIGINAL plan, not the current one -- §8a of the log
  # says so in the same words. EPS stays overridable for a later extension.
  full)  RUN=dp_orig_40ep; EPS=${EPS:-10}; NEV=10; NFIN=10 ;;
  *) echo "usage: $0 [smoke|full]"; exit 2 ;;
esac

RUN_DIR="$BASE/outputs/pusht/$RUN"
LOG="$RUN_DIR/train.log"
mkdir -p "$RUN_DIR"

# Preflight: this is the only DP job; a foreign rollout would confound the
# clock (coverage is contention-invariant, but do not add a variable for free).
if pgrep -f "patch_policy.eval_rollout" >/dev/null; then
    echo "[abort] one of our rollouts is running; the cards should be idle for a fresh run"
    exit 3
fi
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
if [ -n "$apps" ]; then
    echo "[abort] GPU$GPU already has compute apps ($apps)"
    exit 3
fi

cd "$BASE"
echo "=== run_dp_orig mode=$MODE $(date '+%F %T') ==="
echo "  gpu=$GPU epochs=$EPS num_env_evals=$NEV num_final_evals=$NFIN num_workers=$NW"
echo "  run_dir=$RUN_DIR"

# Detached: a launched job must not hold the ssh channel open.
( CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline MUJOCO_GL=egl setsid "$PY" train_policy.py \
      --config-name train_pusht_diffusion encoder=dino_patch \
      epochs="$EPS" num_env_evals="$NEV" num_final_evals="$NFIN" num_workers="$NW" \
      hydra.run.dir="$RUN_DIR" \
      > "$LOG" 2>&1 < /dev/null & echo $! > "$RUN_DIR/pid" )

sleep 3
PID=$(cat "$RUN_DIR/pid" 2>/dev/null)
echo "  pid=$PID log=$LOG"
if kill -0 "$PID" 2>/dev/null; then echo "  alive"; else echo "  DIED IMMEDIATELY -- read the log"; fi
