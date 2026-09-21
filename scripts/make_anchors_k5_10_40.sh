#!/bin/bash
# Build the K in {5, 10, 40} anchor artifacts for the anchor-count sweep.
#
# WHY THIS IS A SEPARATE STEP AND NOT PART OF THE TRAINING DRIVER: the anchors are an
# INPUT to the TD head, not something training fits. If they were regenerated inside
# each run, a K=5 run and a K=40 run could disagree about the normaliser (they must not)
# and there would be no artifact to diff afterwards. Generating them once, up front,
# makes them checkable files with sha256s.
#
# THE NORMALISER MUST NOT DEPEND ON K. It is fit on the train split before k-means runs
# and no K enters `ActionNormalizer.fit`, so all three artifacts must carry byte-identical
# `normalizer` buffers -- which also means they match the K=20 artifact P4 trained with.
# This script does not assert that (it cannot see the K=20 file); `--audit-ks` is set to
# just the production K to halve the k-means work, and the check happens in the summary
# below by printing the normaliser stats of each K side by side.
#
# CPU only. No GPU is touched -- it can run alongside anything.
set -u
PY=$HOME/miniconda3/envs/wemm-p1/bin/python
cd ~/td_pusht || exit 1
CFG=configs/pusht_dino_p256_td.yaml

echo "=== anchors k{5,10,40} start $(date '+%F %T') ==="
for K in 5 10 40; do
    echo "=== K=$K start $(date '+%F %T') ==="
    # --report goes to a per-K file: the default path is the K=20 audit report and
    # overwriting it would destroy P1's record.
    $PY -u -m patch_policy.anchors \
        --config "$CFG" --k "$K" --audit-ks "$K" --seed 0 \
        --out anchors --report "anchors/audit_K${K}.json"
    echo "=== K=$K rc=$? $(date '+%F %T') ==="
done

echo "=== artifacts ==="
ls -l anchors/pusht_kmeans_K5.pt anchors/pusht_kmeans_K10.pt anchors/pusht_kmeans_K40.pt
sha256sum anchors/pusht_kmeans_K*.pt
echo "=== normaliser must be IDENTICAL across K ==="
$PY - <<'PYEOF'
import json
for k in (5, 10, 40, 20):
    try:
        r = json.load(open(f"anchors/audit_K{k}.json"))
    except FileNotFoundError:
        # K=20's report predates this script and has the default name.
        r = json.load(open("anchors/pusht_action_anchor_audit.json"))
    ns = r["normalizer_stats"]
    flat = json.dumps({kk: vv for kk, vv in ns.items()}, sort_keys=True)
    import hashlib
    print(f"  K={k:>2}  normalizer_stats sha256 {hashlib.sha256(flat.encode()).hexdigest()[:24]}")
PYEOF
echo "=== done $(date '+%F %T') ==="
