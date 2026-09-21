"""Concurrent-vs-exclusive report for the ep40 rollouts.

WHY THIS EXISTS AS A FILE: the ad-hoc version of the GPU time-series parse was a single
`python -c` with f-strings nested inside a shell double-quoted string, and the shell ate
the format specs. Same failure mode as the earlier hand-typed progress reports -- the bug
is in the reporting, never in the run. So the parse lives here, diffable.

WHAT IT ANSWERS
1. Did concurrency change the POLICY or only the CLOCK? Compares the 3 concurrent ep40
   JSONs against their exclusive counterparts. These use the same checkpoint and the same
   rng_seed 0, so the coverage numbers must come out identical; that is asserted, not
   eyeballed. K=5 additionally moved GPU2 -> GPU0, so it doubles as a cross-device
   determinism check.
2. What did concurrency cost? Per-arm ms and wall-clock deltas. Only K=40 is a controlled
   comparison: same checkpoint, same card (GPU2), only the two neighbours differ. K=5 and
   K=10 also changed card, so their deltas conflate card with concurrency and are printed
   as measurements, not attributions.
3. Was it a power wall or SM contention? From logs/k_ep40_concurrent_gpu.csv, sampled every
   15 s across all three cards: a power wall shows clocks sagging on all three together
   while temperatures stay moderate; SM contention shows clocks holding while utilisation
   squeezes. The three-card total power against 3x350 W separates the two.
"""
import json
import re
from pathlib import Path

KS = (5, 10, 40)
CARD = {5: "GPU2 -> GPU0", 10: "GPU2 -> GPU1", 40: "GPU2 -> GPU2"}
STEP = 115680
CSV = Path("logs/k_ep40_concurrent_gpu.csv")
POWER_CAP_PER_CARD = 350.0


def _num(s: str):
    """Leading number of an nvidia-smi csv field, ignoring any trailing unit."""
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("base_metrics", {})


def g(b, key):
    if not b or key not in b:
        return None
    v = b[key]
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def pct(a, b):
    return "  n/a" if not a or not b else f"{(b / a - 1) * 100:+.1f}%"


def wall_from_driver(k: int) -> int | None:
    log = Path("logs/k_ep40_concurrent_driver.log")
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("[done]") and f" K={k} " in line:
            for tok in line.split():
                if tok.startswith("wall="):
                    return int(tok[5:].rstrip("s"))
    return None


def main() -> None:
    excl = {k: load(Path(f"outputs/ksweep_k{k}/rollout/ep40/rollout_seed0.json")) for k in KS}
    conc = {k: load(Path(f"outputs/ksweep_k{k}/rollout/ep40_concurrent/rollout_seed0.json"))
            for k in KS}

    print("=== 1. coverage: policy output must be identical (same ckpt, same rng_seed 0) ===")
    allsame = True
    for k in KS:
        a, b = excl[k], conc[k]
        if not a or not b:
            print(f"  K={k:<3} missing (exclusive={bool(a)} concurrent={bool(b)})")
            allsame = False
            continue
        same = abs(a["final_coverage_mean"] - b["final_coverage_mean"]) < 1e-12
        allsame &= same
        print(f"  K={k:<3} excl {a['final_coverage_mean']:.4f}  conc "
              f"{b['final_coverage_mean']:.4f}  {'IDENTICAL' if same else 'DIFFERS <<<'}"
              f"   succ {a['success_rate_095']:.2f} -> {b['success_rate_095']:.2f}")
    if allsame:
        print("  => concurrency changed only the clock; it did not change a single episode "
              "outcome. K=5 also moved cards (GPU2->GPU0), so this is cross-device too.")

    print("\n=== 2. timing: exclusive (all on GPU2) vs concurrent ===")
    print(f"{'K':>3} {'cards':>12} {'enc excl':>9} {'enc conc':>9} {'d':>7} "
          f"{'pol excl':>9} {'pol conc':>9} {'d':>7} {'tot excl':>9} {'tot conc':>9} "
          f"{'d':>7} {'wall excl':>10} {'wall conc':>10} {'d':>7}")
    WALL_EXCL = {5: 541, 10: 543, 40: 560}   # from logs/k_rollout_driver.log, ep40 rows
    for k in KS:
        a, b = excl[k], conc[k]
        if not a or not b:
            continue
        wc = wall_from_driver(k)
        we = WALL_EXCL[k]
        print(f"{k:>3} {CARD[k]:>12} "
              f"{a['encoder_ms_per_step_mean']:>9.2f} {b['encoder_ms_per_step_mean']:>9.2f} "
              f"{pct(a['encoder_ms_per_step_mean'], b['encoder_ms_per_step_mean']):>7} "
              f"{a['policy_ms_per_step_mean']:>9.2f} {b['policy_ms_per_step_mean']:>9.2f} "
              f"{pct(a['policy_ms_per_step_mean'], b['policy_ms_per_step_mean']):>7} "
              f"{a['total_ms_per_step_mean']:>9.2f} {b['total_ms_per_step_mean']:>9.2f} "
              f"{pct(a['total_ms_per_step_mean'], b['total_ms_per_step_mean']):>7} "
              f"{we:>10} {str(wc) if wc else '--':>10} "
              f"{pct(we, wc) if wc else '  n/a':>7}")
    print("  Only K=40 is controlled (same card both times). K=5/K=10 changed card as well,")
    print("  so their deltas mix card identity with concurrency -- measured, not attributed.")
    print("  NOTE: the exclusive series' per-arm `wall` above is the row from "
          "logs/k_rollout_driver.log, same ckpt, same protocol.")

    # ---- 3. GPU time series ---------------------------------------------------------
    if not CSV.exists():
        print("\n=== 3. GPU time series: logs/k_ep40_concurrent_gpu.csv not found ===")
        return
    rows = []
    for line in CSV.read_text(errors="replace").splitlines():
        parts = line.strip().split(",", 1)
        if len(parts) < 2:
            continue
        for seg in parts[1].split(";"):
            # nvidia-smi csv appends units ("0 MHz", "70 C", "309.37 W") unless `nounits`
            # is passed. Strip them here rather than requiring it: the sampler that wrote
            # this file did not, and a parser that only accepts one of the two spellings
            # is exactly the kind of reporting bug this file exists to avoid.
            f = [_num(x) for x in seg.split(",")]
            if len(f) >= 6 and None not in f[:6]:
                rows.append({"ts": parts[0], "gpu": int(f[0]), "temp": int(f[1]),
                             "clk": int(f[2]), "util": int(f[3]),
                             "w": float(f[4]), "mem": int(f[5])})
    print(f"\n=== 3. GPU time series while all three rollouts ran "
          f"({len(rows) // 3} samples @15 s) ===")
    print(f"{'GPU':>4} {'clk min':>8} {'clk max':>8} {'clk mean':>9} {'temp max':>9} "
          f"{'util mean':>10} {'W mean':>7} {'W max':>7}")
    for gi in (0, 1, 2):
        r = [x for x in rows if x["gpu"] == gi]
        if not r:
            continue
        print(f"{gi:>4} {min(x['clk'] for x in r):>8} {max(x['clk'] for x in r):>8} "
              f"{sum(x['clk'] for x in r) / len(r):>9.0f} {max(x['temp'] for x in r):>9} "
              f"{sum(x['util'] for x in r) / len(r):>10.0f} "
              f"{sum(x['w'] for x in r) / len(r):>7.0f} {max(x['w'] for x in r):>7.0f}")

    tot = [sum(x["w"] for x in rows[i:i + 3]) for i in range(0, len(rows) - 2, 3)
           if rows[i]["gpu"] == 0]
    if tot:
        print(f"\n  three-card total power: mean {sum(tot) / len(tot):.0f} W, max {max(tot):.0f} W"
              f"   (3 x {POWER_CAP_PER_CARD:.0f} W = {3 * POWER_CAP_PER_CARD:.0f} W of headroom)")
        print("  a power wall would show all three clocks sagging together up here; a mean "
              "well under the cap points at SM/host contention instead.")

    print("\n  time series (one line per minute):")
    for i in range(0, len(rows), 12):
        grp = rows[i:i + 3]
        if len(grp) < 3:
            break
        cells = "  ".join(f"g{x['gpu']}:{x['clk']:>4}MHz {x['temp']:>2}C "
                          f"{x['util']:>3}% {x['w']:>3.0f}W" for x in grp)
        print(f"    {grp[0]['ts']}  {cells}  total {sum(x['w'] for x in grp):>4.0f}W")


if __name__ == "__main__":
    main()
