#!/bin/bash
# P2 small-set overfit: the full policy, N windows of the TRAIN split, GPU2 ONLY.
#
# `-u` is not decoration: without it Python block-buffers stdout when redirected, and a
# 1-2 hour run produced a log containing nothing but DINOv2 warnings -- i.e. the run was
# unmonitorable and a hang would have been indistinguishable from progress.
set -x
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1

nvidia-smi --query-gpu=index,memory.used --format=csv
$PY -u tests/test_td_contract.py
$PY -u scripts/overfit_small.py --n-samples 32  --select spread --steps 2000 --lrs 5.5e-5,1e-3 --out outputs/p2_overfit
$PY -u scripts/overfit_small.py --n-samples 128 --select spread --steps 2000 --lrs 5.5e-5      --out outputs/p2_overfit
echo "P2 DONE"
nvidia-smi --query-gpu=index,memory.used --format=csv
