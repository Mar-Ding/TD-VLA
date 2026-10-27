#!/bin/bash
# P3 -- smoke test. Six short steps, all on GPU2, none of them a long training run.
#
# Two things here are not what the plan literally said, both because the harness
# disagreed with the plan and the harness is the authority:
#
#  1. `train.py --smoke` sets `cfg.save_ckpt = False` in ALL THREE data branches
#     (:639, :665, :696) and the only `save_ckpt(0)` call sits behind
#     `if not args.smoke and not args.resume:` (:1083). So a smoke run writes NO
#     checkpoint -- and P3 needs one for the rollout. Rather than add a fifth deviation
#     to `train.py`, step 5 runs a TINY but REAL 1-epoch / 6-episode training through the
#     ordinary checkpoint path. That is a better test than a smoke-only save would be,
#     because it exercises the checkpoint code the P6 run will actually use.
#
#  2. The harness's own smoke checklist names seven parameters and does not include
#     `denoiser` -- the module this project adds. Step 2 covers it from outside instead.
set -x
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1
CFG=configs/pusht_dino_p256_td.yaml
mkdir -p logs

echo "=== GPU state before (GPU0/GPU1 must stay untouched) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== P3-1 contract tests (CPU only) ==="
CUDA_VISIBLE_DEVICES='' $PY -u tests/test_td_contract.py 2>&1 | tail -4

echo "=== P3-2 gradient reachability, incl. denoiser (CPU only) ==="
CUDA_VISIBLE_DEVICES='' $PY -u scripts/audit_td_grads.py 2>&1 | tail -6

echo "=== P3-3 GPU2 lock mutual exclusion (no GPU) ==="
bash scripts/test_gpu2_lock.sh

echo "=== P3-4 smoke training: 100 steps, 6 episodes, batch 2 ==="
CUDA_VISIBLE_DEVICES=2 $PY -u -m patch_policy.train --config "$CFG" --smoke \
    --train-seed 0 --save-dir outputs/p3_smoke_train --attn current \
    > logs/p3_smoke_train.log 2>&1
echo "  smoke_train rc=$?"

echo "=== P3-5 tiny REAL training to produce a checkpoint (1 epoch, 6 episodes) ==="
CUDA_VISIBLE_DEVICES=2 $PY -u -m patch_policy.train --config "$CFG" \
    --train-seed 0 --max-episodes 6 --epochs 1 --ckpt-epochs 0,1 \
    --save-dir outputs/p3_ckpt --attn current > logs/p3_ckpt_train.log 2>&1
echo "  ckpt_train rc=$?"
ls -la outputs/p3_ckpt/seed0/ 2>&1

echo "=== P3-6 smoke rollout: 2 episodes, under the GPU2 lock ==="
CKPT=$(ls -1 outputs/p3_ckpt/seed0/ckpt_epoch1_*.pt 2>/dev/null | head -1)
echo "  ckpt=$CKPT"
bash scripts/rollout_locked_gpu2.sh "$CFG" "$CKPT" outputs/p3_smoke_rollout \
    logs/p3_rollout.log 0 --smoke
echo "  rollout rc=$?"

echo "P3 DONE"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
