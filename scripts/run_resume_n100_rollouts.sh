#!/bin/bash
# Roll out the six new ckpts of each resumed arm -- K=20 and K=40 -- at ep50/60/70/80/90/100,
# n=100, ONE AT A TIME on GPU2. 12 rollouts (2 arms x 6 epochs).
#
# WHY THIS EXISTS
# The 2026-09-18/19 resume (scripts/run_resume_to100.sh) carried both arms from ep40 to
# ep100 and, per the standing instruction 「对于产生的 ckpt 先存储，最后当所有训练结束后在不受
# 干扰竞争时进行逐个 rollout」, wrote ckpts only. This driver is that instruction's second
# half: the rollouts run afterwards, alone, with nothing else on the cards.
#
# PROTOCOL -- byte-for-byte the one the existing 16 rollouts used, so these 12 points append
# to the same table instead of starting a new one:
#   --num-evals 100 --num-envs 10 --rng-seed 0 --train-seed 0, GPU2, serial
#
# WHY SERIAL ON GPU2 AND NOT PARALLEL ON THREE IDLE CARDS
# All three cards are idle right now, so parallelism is available -- and is still wrong.
# Concurrency on this box has been measured to inflate the step clock ~2.9x, with DINO
# encoding as the thermometer (16.6-17.1 ms clean vs 48-58 ms contended). That clock is the
# quantity the comparison table is *about*, so buying wall clock here would destroy the
# measurement it was meant to produce. GPU2 specifically, because that is the card the
# existing 16 points were measured on; mixing cards would import GPU0's thermal throttle
# (it holds ~1090 MHz of 2100 under load, 52% of a cool card) as if it were a property of
# the model.
#
# OUTPUT NAMING -- per arm, following that arm's OWN existing convention
#   K=20 -> outputs/p4_td/rollout/ep{EP}_n100   (its bare ep{EP} dirs hold the n=20 legacy runs)
#   K=40 -> outputs/ksweep_k40/rollout/ep{EP}   (no legacy; there the bare name IS the n=100 one)
# At ep50-100 neither arm has a legacy run, so the suffix is only for within-arm uniformity
# of the directory listing. Do not "fix" one of the two to match the other: each name means
# what that arm's earlier dirs mean.
#
# RESUMABLE: a rollout whose rollout_seed0.json already exists is skipped, so this driver can
# be killed and relaunched at any point without redoing finished work.
#
# BUDGET: measured 537-560 s per n=100 rollout on GPU2 -> 12 x ~9.5 min ~= 1.9 h.
#
# Usage (on the server, detached):
#   cd ~/td_pusht && ( setsid nohup bash scripts/run_resume_n100_rollouts.sh \
#       > logs/resume_n100_driver.log 2>&1 < /dev/null & )
set -u
BASE=${BASE:-$HOME/td_pusht}
cd "$BASE" || exit 1

LOCKED="$BASE/scripts/rollout_locked_gpu2.sh"
EPS="50 60 70 80 90 100"
N_EVALS=100
N_ENVS=10
RNG_SEED=0
TRAIN_SEED=0
mkdir -p "$BASE/logs"

stamp() { date '+%F %T'; }

gpu_note() {   # $1 = label. Records the card state so a contaminated rollout can be
               # identified after the fact instead of being silently averaged into the table.
    local apps clk
    apps=$(nvidia-smi -i 2 --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
    clk=$(nvidia-smi -i 2 --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu \
              --format=csv,noheader)
    echo "  [$1] $(stamp) gpu2 {$clk} compute_apps=${apps:-none}"
}

# arm spec: K:config:outputs-root
ARMS="20:configs/pusht_dino_p256_td.yaml:outputs/p4_td
40:configs/pusht_dino_p256_td_k40.yaml:outputs/ksweep_k40"

echo "=== resume n100 rollouts start $(stamp) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
df -h $HOME | tail -1

# ---------------------------------------------------------------- global preflight
# Both arms' training must be over. A training process on either card folds into every
# rollout's clock below, and nothing downstream can undo it.
if pgrep -f "patch_policy.train" >/dev/null; then
    echo "[abort] a patch_policy.train is still running:"
    pgrep -af "patch_policy.train"
    exit 3
fi
if pgrep -f "patch_policy.eval_rollout" >/dev/null; then
    echo "[abort] another eval_rollout is already running:"
    pgrep -af "patch_policy.eval_rollout"
    exit 3
fi
busy=$(nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$busy" ]; then
    echo "[abort] GPU2 already has a compute process:"
    echo "$busy"
    exit 3
fi
echo "[guard] no training, no eval, GPU2 free: $(nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

# ---------------------------------------------------------------- per-artifact preflight
# Every ckpt and config is checked BEFORE the first rollout starts, so a missing ep100 or a
# K-mismatched config surfaces in seconds rather than 90 minutes in.
PLAN=""
for entry in $ARMS; do
    K=$(echo "$entry" | cut -d: -f1)
    CFG=$(echo "$entry" | cut -d: -f2)
    ROOT=$(echo "$entry" | cut -d: -f3)

    [ -f "$CFG" ] || { echo "[abort] missing config $CFG"; exit 1; }
    # The config must be the one that trained this arm. Rolling K=40's ckpt out under the
    # K=20 config would rebuild the model with 20 anchors and either crash on shape or, if
    # the shapes happened to line up, silently evaluate a different model.
    grep -q "^  n_modes: ${K}$" <(tr -d '\r' < "$CFG") \
        || { echo "[abort] $CFG has no '  n_modes: $K'"; exit 1; }
    grep -q "^  anchor_artifact: anchors/pusht_kmeans_K${K}.pt$" <(tr -d '\r' < "$CFG") \
        || { echo "[abort] $CFG does not point at the K=$K anchor artifact"; exit 1; }

    for EP in $EPS; do
        CK=$(ls "$ROOT"/seed0/ckpt_epoch${EP}_step*.pt 2>/dev/null | head -1)
        [ -n "$CK" ] || { echo "[abort] no ckpt for K=$K ep$EP in $ROOT/seed0"; exit 1; }
        SZ=$(stat -c%s "$CK")
        # 400 MB floor: a pre-training epoch-0 ckpt is ~200 MB and a truncated write is
        # anything at all. Same test the P4 n100 driver used.
        [ "$SZ" -gt 400000000 ] \
            || { echo "[abort] K=$K ep$EP ckpt is only $SZ B -- refusing to roll out a stub"; exit 1; }
        if [ "$K" = "20" ]; then OUTDIR="$ROOT/rollout/ep${EP}_n100"; else OUTDIR="$ROOT/rollout/ep${EP}"; fi
        PLAN="$PLAN
$K|$CFG|$CK|$OUTDIR|$EP"
        printf '  K=%-2s ep%-3s %-34s %10s B -> %s\n' "$K" "$EP" "$(basename "$CK")" "$SZ" "$OUTDIR"
    done
done
echo "[guard] all 12 ckpts and both configs present"

# ---------------------------------------------------------------- serial rollouts
# A `for` loop over a here-string, NOT `printf | while`: a piped loop runs in a subshell, so
# the exit status below would be lost and this driver would report success after a failed
# rollout. That exact bug was found and fixed once already in run_dp_rollout.sh.
rc_all=0
done_n=0
while IFS='|' read -r K CFG CK OUT EP; do
    [ -n "${K:-}" ] || continue
    JSON="$OUT/rollout_seed0.json"
    RLOG="$BASE/logs/resume_rollout_k${K}_ep${EP}.log"

    if [ -f "$JSON" ]; then
        echo "[skip] $(stamp) K=$K ep$EP already has rollout_seed0.json"
        continue
    fi
    mkdir -p "$OUT"
    echo "[run ] $(stamp) K=$K ep$EP $(basename "$CK") n=$N_EVALS envs=$N_ENVS rng=$RNG_SEED train=$TRAIN_SEED"
    gpu_note "pre  K=$K ep$EP"
    T0=$(date +%s)

    if bash "$LOCKED" "$CFG" "$CK" "$OUT" "$RLOG" "$RNG_SEED" \
            --num-evals "$N_EVALS" --num-envs "$N_ENVS" --train-seed "$TRAIN_SEED"; then
        rc=0
    else
        rc=$?
    fi

    T1=$(date +%s)
    gpu_note "post K=$K ep$EP"
    echo "[done] $(stamp) K=$K ep$EP rc=$rc wall=$((T1-T0))s"
    [ "$rc" -ne 0 ] && { rc_all=1; echo "       ^ see $RLOG"; }
    done_n=$((done_n + 1))
done <<< "$PLAN"

echo "=== resume n100 rollouts end $(stamp) rc_all=$rc_all (ran $done_n) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h $HOME | tail -1

# ---------------------------------------------------------------- summary table
$HOME/miniconda3/envs/wemm-p1/bin/python - "$BASE" <<'PYEOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
rows = []
for K, root in ((20, "outputs/p4_td"), (40, "outputs/ksweep_k40")):
    for ep in (50, 60, 70, 80, 90, 100):
        d = "ep%d_n100" % ep if K == 20 else "ep%d" % ep
        p = base / root / "rollout" / d / "rollout_seed0.json"
        if not p.exists():
            rows.append((K, ep, None)); continue
        j = json.load(open(p))
        rows.append((K, ep, j["base_metrics"]))
print()
print("  K  | ep  | final_cov |  ci95  | succ@0.95 | policy_ms | encoder_ms | env_ms | total_ms | peak_mem")
print("-----+-----+-----------+--------+-----------+-----------+------------+--------+----------+---------")
for K, ep, b in rows:
    if b is None:
        print("  %-2d | %-3d | (no rollout json)" % (K, ep)); continue
    # final_coverage_ci95 is a [lo, hi] LIST, not a scalar -- %f on it raises TypeError.
    ci = b["final_coverage_ci95"]
    cis = "[%.4f,%.4f]" % tuple(ci) if isinstance(ci, (list, tuple)) else "%.4f" % ci
    print("  %-2d | %-3d |   %7.4f | %-12s |  %6.4f  | %9.2f | %10.2f | %6.2f | %8.2f | %7d" % (
        K, ep, b["final_coverage_mean"], cis, b["success_rate_095"],
        b["policy_ms_per_step_mean"], b["encoder_ms_per_step_mean"],
        b["env_step_ms_per_step_mean"], b["total_ms_per_step_mean"], b["peak_gpu_mem_mb"]))
PYEOF

echo "=== next: fold these 12 points into doc/2026-09-19_ablation_record.md A.4 ==="
exit $rc_all
