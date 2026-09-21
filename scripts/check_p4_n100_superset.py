"""Are the K=20 n=100 rollouts a strict superset of the n=20 ones?

P4 §A.4 performed this check once, for ep30 only, by hand:
"两者在前 20 个场景上逐 episode 完全一致". That single fact is what licenses
reading the two runs as the same measurement protocol at different n -- if the
first 20 episodes drifted, the n=20 and n=100 numbers would be two different
samples and the 0.8210 -> 0.6149 drop would not be attributable to n alone.

This script re-runs that check for all four epochs (ep10/20/30/40) instead of
just ep30, so the claim in doc/2026-09-19_ablation_record.md §A.4 rests on
a repeatable command rather than on one manual diff.

It does NOT re-run any rollout. Inputs only.

  n=20  : outputs/p4_td/rollout/ep{EP}/rollout_seed0.json              (P4 log §8)
  n=100 : outputs/p4_td/rollout/ep{EP}_n100/rollout_seed0.json          (this backfill)

Run on the server:
    cd ~/td_pusht && python scripts/check_p4_n100_superset.py
"""
import json
from pathlib import Path

ROOT = Path.home() / "td_pusht" / "outputs" / "p4_td" / "rollout"
EPS = [10, 20, 30, 40]


def load(p):
    return json.load(open(p)) if p.is_file() else None


def by_seed(j):
    """eval_seed-indexed map. The JSONs carry `episodes` as a flat list whose
    entries hold `eval_seed`; the runner seeds episodes 0..n-1, so the key is
    the same scene index in both runs -- which is exactly what makes the check
    meaningful (same scenes, more of them)."""
    return {int(e["eval_seed"]): e for e in j["episodes"]}


print("=== n=20 vs n=100 superset check (K=20, P4 arm) ===\n")
print(f"{'ep':>3} {'n20':>4} {'n100':>5} {'shared':>6} {'bit-identical':>13} "
      f"{'max|dp|':>9} {'mean20':>9} {'mean_first20_of_n100':>21} {'mean_full_n100':>15}")
print("-" * 100)

rows = []
for EP in EPS:
    j20 = load(ROOT / f"ep{EP}" / "rollout_seed0.json")
    j100 = load(ROOT / f"ep{EP}_n100" / "rollout_seed0.json")
    if j20 is None or j100 is None:
        missing = [t for t, j in (("n20", j20), ("n100", j100)) if j is None]
        print(f"{EP:>3} {'[MISSING ' + ','.join(missing) + ']':>60}")
        continue

    a, b = by_seed(j20), by_seed(j100)
    shared = sorted(set(a) & set(b))
    diffs = [abs(a[s]["final_coverage"] - b[s]["final_coverage"]) for s in shared]
    # TOL, not == 0. The same rollout re-run on the same card reproduces to a few
    # ULP of fp64: measured 2.22e-16 .. 4.44e-16 (2-4 ULP at cov ~0.5-0.8), and one
    # epoch came back exactly 0.0. Those are the same computation with rounding
    # noise, not a substituted scene -- a different scene would land O(0.1) apart,
    # 14 orders of magnitude above this. TOL = 1e-9 is far under any coverage
    # difference that could matter and far over ULP noise.
    TOL = 1e-9
    n_ident = sum(1 for d in diffs if d <= TOL)
    m20 = sum(a[s]["final_coverage"] for s in shared) / len(shared)
    m100_first = sum(b[s]["final_coverage"] for s in shared) / len(shared)
    m100_full = j100["base_metrics"]["final_coverage_mean"]
    rows.append((EP, len(a), len(b), len(shared), n_ident, max(diffs), m20, m100_first, m100_full))
    print(f"{EP:>3} {len(a):>4} {len(b):>5} {len(shared):>6} {n_ident:>13} "
          f"{max(diffs):>9.2e} {m20:>9.4f} {m100_first:>21.4f} {m100_full:>15.4f}")

print()
print("verdict per epoch (agreement tolerance 1e-9 = ULP-scale, see comment above):")
for (EP, n20, n100, shared, n_ident, mx, m20, m100_first, m100_full) in rows:
    ok = (n_ident == shared == n20)
    print(f"  ep{EP}: {'SUPERSET' if ok else 'NOT A SUPERSET'} -- "
          f"{n_ident}/{shared} shared scenes agree within 1e-9, max|d| = {mx:.3e}")
    if ok:
        print(f"        same-scene means: n=20 {m20:.4f} vs first-20-of-n100 {m100_first:.4f} "
              f"(delta {m100_first - m20:+.4f})")
        print(f"        the n=20 -> n=100 headline is therefore an n effect, not a resample: "
              f"{m20:.4f} -> {m100_full:.4f} ({m100_full - m20:+.4f})")

print()
print("If every row says SUPERSET, the four-group K table in §A.4 can put the K=20")
print("column on the n=100 line without an asterisk: the extra 80 scenes are added,")
print("not substituted. If any row does not, that epoch's n=20 and n=100 numbers are")
print("different samples and the table must keep them apart.")
