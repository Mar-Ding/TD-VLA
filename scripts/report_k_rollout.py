"""Summarise the K-sweep rollouts: one row per ckpt, plus the cross-K tables.

Reads every outputs/ksweep_k*/rollout/ep*/rollout_seed0.json the driver produced.
Missing cells are printed as `--`, never skipped silently: a table that silently drops
the rollout still running would read as a complete table.

THE THERMOMETER COLUMN IS NOT DECORATION. GPU0 on this box thermally throttles to
~840 MHz of 2100 and its training ran 1.79x slower than the same work on a cool card,
so every timing here was taken on GPU2 under a cross-driver flock. `encoder_ms` times
byte-identical work in all 12 rollouts, so its spread measures the *measurement
conditions*, not the model: ~17 ms means clean, ~50 ms means something else was on the
card. Rows above the clean band are flagged `!` and their ms columns must not be quoted
as the arm's inference cost -- their coverage numbers remain usable, since contention
slows the clock but does not change what the policy outputs.
"""
import json
from pathlib import Path

KS = (5, 10, 40)
EPS = (10, 20, 30, 40)
CLEAN_ENCODER_MS = 25.0   # clean readings are 16.7-17.4; contended are 48-58


def load(k: int, ep: int):
    p = Path(f"outputs/ksweep_k{k}/rollout/ep{ep}/rollout_seed0.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("base_metrics", {})


def cell(b, key, fmt="{:.4f}"):
    v = b.get(key) if b else None
    return "--" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))


def main() -> None:
    rows = {(k, ep): load(k, ep) for k in KS for ep in EPS}
    done = sum(1 for b in rows.values() if b)

    print(f"=== K-sweep rollouts: {done}/{len(rows)} done ===")
    print(f"{'K':>3} {'ep':>3} {'final_cov':>10} {'ci95':>8} {'max_cov':>8} "
          f"{'succ@.95':>9} {'n_reach':>8} {'enc_ms':>7} {'pol_ms':>7} "
          f"{'env_ms':>7} {'tot_ms':>7} {'peak_mb':>8}")
    for k in KS:
        for ep in EPS:
            b = rows[(k, ep)]
            if not b:
                print(f"{k:>3} {ep:>3} {'--':>10}")
                continue
            enc = b.get("encoder_ms_per_step_mean")
            flag = "!" if enc is not None and enc > CLEAN_ENCODER_MS else " "
            print(f"{k:>3} {ep:>3} {cell(b,'final_coverage_mean'):>10} "
                  f"{cell(b,'final_coverage_ci95','{:.3f}'):>8} "
                  f"{cell(b,'max_coverage_mean','{:.3f}'):>8} "
                  f"{cell(b,'success_rate_095','{:.3f}'):>9} "
                  f"{cell(b,'n_episodes_reached_095'):>8} "
                  f"{cell(b,'encoder_ms_per_step_mean','{:.2f}'):>7} "
                  f"{cell(b,'policy_ms_per_step_mean','{:.2f}'):>7} "
                  f"{cell(b,'env_step_ms_per_step_mean','{:.2f}'):>7} "
                  f"{cell(b,'total_ms_per_step_mean','{:.2f}'):>7} "
                  f"{cell(b,'peak_gpu_mem_mb','{:.0f}'):>8}{flag}")
    print(f"  (! = encoder_ms > {CLEAN_ENCODER_MS:.0f} ms, i.e. this rollout was not "
          f"running alone; its ms columns are not the arm's inference cost)")

    print("\nfinal_coverage by matched epoch (only cells where all three K have it):")
    for ep in EPS:
        vals = {k: rows[(k, ep)] for k in KS}
        if not all(vals.values()):
            continue
        line = "  ".join(f"K={k} {vals[k]['final_coverage_mean']:.4f}" for k in KS)
        best = max(KS, key=lambda k: vals[k]["final_coverage_mean"])
        print(f"  ep{ep:<3} {line}   -> max K={best}")

    # Best epoch per K. This is the number the comparison table actually quotes, and it
    # is only meaningful if the epoch was chosen by coverage rather than by val_loss --
    # which is why the sweep kept every ckpt instead of pruning to the val-best one.
    print("\nbest epoch per K (by final_coverage_mean):")
    for k in KS:
        got = [(ep, rows[(k, ep)]["final_coverage_mean"])
               for ep in EPS if rows[(k, ep)] and "final_coverage_mean" in rows[(k, ep)]]
        if not got:
            print(f"  K={k:<3} no rollouts yet")
            continue
        ep, cov = max(got, key=lambda t: t[1])
        print(f"  K={k:<3} ep{ep} {cov:.4f}   (all: "
              + ", ".join(f"ep{e} {c:.4f}" for e, c in got) + ")")

    # Wall clock from the driver log, so a rollout that took 3x longer than its peers is
    # visible even when its own JSON looks normal.
    dlog = Path("logs/k_rollout_driver.log")
    if dlog.exists():
        print("\ndriver log (run/done lines):")
        for line in dlog.read_text(errors="replace").splitlines():
            if line.startswith(("[run ]", "[done]", "[skip]", "[MISS]", "[abort]")):
                print("  " + line)


if __name__ == "__main__":
    main()
