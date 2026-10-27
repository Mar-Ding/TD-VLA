#!/bin/bash
# P4 -- quick validation. Single seed, 40 epochs, rollout every 10.
#
# Runs on the SERVER, from ~/td_pusht, launched under nohup so it survives the ssh
# session that starts it.
#
# WHY 40 AND NOT 20: the plan allows 20-40. The gate is a coverage number that must
# clear 0.1, and the failure mode worth distinguishing is "not learning yet" vs "not
# going to learn" -- at 20 epochs those look identical. 40 buys a second and third
# rollout point, so a flat curve is evidence rather than a single uninterpretable
# sample. It is still half of P6's 100.
#
# STAGE STRUCTURE, and why rollout is interleaved rather than batched at the end:
# a checkpoint is 500 MB (measured at P3). 4 of them is 2 GB on a volume with 24 GB
# free. Evaluating as we go means the coverage curve exists even if the run is killed
# at epoch 25 -- which matters, because this is the first long run of this arm and
# "it died at hour 6 having produced nothing" is the outcome to design out.
#
# NO keep-best PRUNING HERE (P6 has it). P4 keeps all 4 ckpts deliberately: the whole
# point is to compare epoch 10 against 40, and re-rollouting a deleted ckpt means
# retraining. 2 GB is affordable at this scale; it would not be at P6's 20 ckpts.
set -u
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1
CFG=configs/pusht_dino_p256_td.yaml
OUT=outputs/p4_td
mkdir -p logs "$OUT"

EPOCHS=40
CKPT_EPOCHS=10,20,30,40
ROLL_EVALS=20
ROLL_ENVS=10

echo "=== P4 start $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h $HOME | tail -1

# ---------------------------------------------------------------- stage A: training
echo "=== P4-A training: $EPOCHS epochs, ckpt at $CKPT_EPOCHS ==="
nohup env CUDA_VISIBLE_DEVICES=2 $PY -u -m patch_policy.train --config "$CFG" \
    --train-seed 0 --epochs "$EPOCHS" --ckpt-epochs "$CKPT_EPOCHS" \
    --save-dir "$OUT" --attn current > logs/p4_train.log 2>&1 &
TRAIN_PID=$!
echo "  train pid=$TRAIN_PID log=logs/p4_train.log"

# --------------------------------------------------------------------- stage B: rollout
# Wait for a ckpt to exist AND stop growing. The stability check is not paranoia:
# save is a plain torch.save into the destination path, so a reader that races it gets
# a truncated file. Requiring two equal sizes 20 s apart costs 20 s per epoch and
# removes the whole class of "rollout loaded a half-written checkpoint" failures.
wait_for_ckpt() {
    local ep=$1 t=0 f s prev=""
    while [ $t -lt 14400 ]; do          # 4 h per epoch; a full epoch measured << that
        f=$(ls -1 "$OUT"/seed0/ckpt_epoch"${ep}"_*.pt 2>/dev/null | head -1)
        if [ -n "$f" ]; then
            s=$(stat -c%s "$f" 2>/dev/null || echo 0)
            # 400 MB floor: the pre-training ckpt (no optimizer state) is ~200 MB, so
            # this also filters out a stale epoch-0 file being mistaken for epoch N.
            if [ "$s" -gt 400000000 ] && [ "$s" = "$prev" ]; then
                echo "$f"; return 0
            fi
            prev=$s
        fi
        # If training is gone and the ckpt still is not there, it is never coming.
        if ! kill -0 "$TRAIN_PID" 2>/dev/null && [ -z "$f" ]; then
            echo "  [p4] training exited before epoch $ep ckpt appeared" >&2
            return 1
        fi
        sleep 20; t=$((t + 20))
    done
    echo "  [p4] timed out waiting for epoch $ep ckpt" >&2
    return 1
}

best_cov=-1; best_ep=-1
for EP in 10 20 30 40; do
    echo "=== P4-B waiting for epoch $EP ckpt ==="
    CKPT=$(wait_for_ckpt "$EP") || { echo "  skipped epoch $EP"; continue; }
    echo "  ckpt=$CKPT ($(stat -c%s "$CKPT") B)"

    ROLLOUT_DIR="$OUT/rollout/ep$EP"
    # --num-evals 20 --num-envs 10 => 2 sequential batches of 10. Same env count as the
    # n100 protocol so the ms/step numbers are comparable to P6's; only the eval count
    # is smaller. rng_seed 0 to match the P6 rollout protocol exactly.
    bash scripts/rollout_locked_gpu2.sh "$CFG" "$CKPT" "$ROLLOUT_DIR" \
        "logs/p4_rollout_ep${EP}.log" 0 \
        --num-evals "$ROLL_EVALS" --num-envs "$ROLL_ENVS" --train-seed 0
    rc=$?
    echo "  rollout ep$EP rc=$rc"

    J="$ROLLOUT_DIR/rollout_seed0.json"
    if [ -f "$J" ]; then
        cov=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['base_metrics']['final_coverage_mean'])" "$J")
        echo "  ep$EP final_coverage_mean=$cov"
        awk -v a="$cov" -v b="$best_cov" 'BEGIN{exit !(a>b)}' && { best_cov=$cov; best_ep=$EP; }
    else
        echo "  ep$EP: no rollout JSON produced"
    fi
done

# ----------------------------------------------------------------------- stage C: summary
echo "=== P4-C summary $(date '+%F %T') ==="
echo "  best: ep$best_ep final_coverage_mean=$best_cov"
for EP in 10 20 30 40; do
    J="$OUT/rollout/ep$EP/rollout_seed0.json"
    if [ -f "$J" ]; then
        $PY - "$J" "$EP" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("base_metrics", {})
print(f"  ep{sys.argv[2]:>3}  final_cov={m.get('final_coverage_mean', float('nan')):.4f}"
      f"  max_cov={m.get('max_coverage_mean', float('nan')):.4f}"
      f"  success={m.get('success_rate_095', float('nan')):.3f}"
      f"  policy_ms={m.get('policy_ms_per_step_mean', float('nan')):.2f}"
      f"  total_ms={m.get('total_ms_per_step_mean', float('nan')):.2f}"
      f"  peak_mem={m.get('peak_gpu_mem_mb', float('nan')):.0f}")
PYEOF
    fi
done
echo "=== P4 DONE $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h $HOME | tail -1
