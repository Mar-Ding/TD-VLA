"""One-shot progress report for the three concurrent K-sweep trainings.

WHY THIS IS A FILE AND NOT AN SSH ONE-LINER: the same parse was hand-typed four
times and broke three of them (a 2-tuple unpacked as 3, `%W` birth time returning
0 on this filesystem, and twice a printf with one slot too few). Every one of
those was a bug in the *reporting*, never in the training, but a progress report
that can silently mis-print is worse than no report. The parsing, the interval
bookkeeping and the matched-step table live here now, testable and diffable.

TWO THINGS THIS GETS RIGHT THAT THE AD-HOC VERSIONS GOT WRONG:

1. `[epoch N/40] val_loss=... (Xs)` -- X is `time.time()-start`, i.e. wall clock
   since the training process started, NOT the validation duration and NOT the
   epoch duration. The per-epoch wall clock is the DIFFERENCE between consecutive
   markers. That is exactly how P4's run log §5.1 table was built, so the numbers
   here are directly comparable to P4's 959.4 s/epoch solo figure.

2. The step line is printed every 10 steps, and the arms run at different speeds,
   so the "latest step" of two arms is NOT at the same point in training.
   Comparing their losses directly is meaningless. The matched-step table below is
   the only cross-arm loss comparison that means anything.

STATE: the previous sample (timestamp + per-arm step) is stored in
`outputs/ksweep_progress_state.json`, so consecutive runs compute a real interval
instead of guessing. `--baseline` overrides it with a fixed point and does not
read or write the state file.

Usage (server, from ~/td_pusht):
    python scripts/report_k_progress.py
    python scripts/report_k_progress.py --baseline "2026-09-17 09:37:06" 2850 4970 3890
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

KS = (5, 10, 40)
LOGDIR = Path("logs")
STATE = Path("outputs/ksweep_progress_state.json")
P4_SOLO_STEPS_PER_S = 3.014   # outputs/p4_td solo run, doc §5.1
P4_SOLO_S_PER_EPOCH = 959.4

STEP_RE = re.compile(r"\s+step\s+(\d+)\s+loss=([\d.]+) \(cls=([\d.]+) off=([\d.]+)\)")
EPOCH_RE = re.compile(r"\[epoch\s+(\d+)/(\d+)\]\s+val_loss=([\d.]+)\s+\((\d+)s\)")
DATA_RE = re.compile(r"\[data\]\s+train windows\s+(\d+),\s+val windows\s+(\d+)")
GRAD_RE = re.compile(r"grads=\{([^}]*)\}")
MEM_RE = re.compile(r"mem=alloc=([\d.]+)G reserved=([\d.]+)G")


def log_path(k: int) -> Path:
    return LOGDIR / f"ksweep_k{k}_train.log"


def scan(k: int) -> dict:
    """Parse one arm's training log into everything the report needs."""
    steps, epochs = {}, []
    train_windows = None
    grads, mem, last_line = None, None, ""
    for line in log_path(k).read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if m:
            steps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            last_line = line
        m = EPOCH_RE.search(line)
        if m:
            epochs.append((int(m.group(1)), float(m.group(3)), int(m.group(4))))
        m = DATA_RE.search(line)
        if m:
            train_windows = int(m.group(1))
        m = GRAD_RE.search(line)
        if m:
            grads = m.group(1)
        m = MEM_RE.search(line)
        if m:
            mem = (float(m.group(1)), float(m.group(2)))
    return {"steps": steps, "epochs": epochs, "train_windows": train_windows,
            "grads": grads, "mem": mem, "last_line": last_line}


def batch_size() -> int:
    """Read batch_size out of the shared config rather than hardcoding it."""
    for cfg in ("configs/pusht_dino_p256_td_k10.yaml", "configs/pusht_dino_p256_td.yaml"):
        p = Path(cfg)
        if p.exists():
            m = re.search(r"^batch_size:\s*(\d+)\s*$", p.read_text(), re.M)
            if m:
                return int(m.group(1))
    raise SystemExit("[progress] could not find batch_size in configs/")


def running(k: int) -> bool:
    return subprocess.run(
        ["pgrep", "-f", f"configs/pusht_dino_p256_td_k{k}.yaml"],
        capture_output=True).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs=4, metavar=("TS", "K5", "K10", "K40"),
                    help='fixed start point, e.g. --baseline "2026-09-17 09:37:06" 2850 4970 3890')
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    data = {k: scan(k) for k in KS}
    bs = batch_size()
    train_windows = next(d["train_windows"] for d in data.values() if d["train_windows"])
    steps_per_epoch = -(-train_windows // bs)          # ceil
    total_steps = args.epochs * steps_per_epoch

    latest = {k: (max(data[k]["steps"]) if data[k]["steps"] else 0) for k in KS}
    now = time.strftime("%F %T")

    if args.baseline:
        ts = time.mktime(time.strptime(args.baseline[0], "%Y-%m-%d %H:%M:%S"))
        base = {k: int(v) for k, v in zip(KS, args.baseline[1:])}
        src = "baseline (fixed)"
    elif STATE.exists():
        st = json.loads(STATE.read_text())
        ts, base, src = st["ts"], {int(k): v for k, v in st["steps"].items()}, "previous run"
    else:
        ts, base, src = None, None, None
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"ts": time.time(), "at": now, "steps": latest}, indent=2))

    print(f"=== {now} ===  (interval source: {src or 'none -- first run'})")
    for k in KS:
        print(f"  K={k:<3} {'RUNNING' if running(k) else 'NOT RUNNING':<12} "
              f"grads={'ok' if data[k]['grads'] else 'n/a'}  mem={data[k]['mem']}")
    print(f"\n  steps/epoch = {train_windows}/{bs} = {steps_per_epoch}   "
          f"total {args.epochs}ep = {total_steps}")

    print(f"\n{'K':>3} {'step':>7} {'epoch':>6} {'d_step':>7} {'steps/s':>8} {'loss':>9} "
          f"{'cls':>7} {'off':>7} {'ETA_h':>7} {'s/epoch':>8}")
    for k in KS:
        s = latest[k]
        row = data[k]["steps"].get(s)
        if not row:
            continue
        rate = None
        if base and ts:
            el = time.time() - ts
            d = s - base[k]
            rate = d / el if el > 0 else None
        r = f"{rate:8.3f}" if rate else f"{'n/a':>8}"
        eta = f"{(total_steps - s) / rate / 3600:7.2f}" if rate else f"{'n/a':>7}"
        spe = f"{steps_per_epoch / rate:8.0f}" if rate else f"{'n/a':>8}"
        print(f"{k:>3} {s:>7} {s / steps_per_epoch:>6.2f} "
              f"{(s - base[k]) if base else 0:>7} {r} "
              f"{row[0]:>9.4f} {row[1]:>7.4f} {row[2]:>7.4f} {eta} {spe}")
    if base and ts:
        print(f"  (interval = {time.time() - ts:.0f} s;  P4 solo reference: "
              f"{P4_SOLO_STEPS_PER_S} steps/s, {P4_SOLO_S_PER_EPOCH} s/epoch)")

    # Epoch markers: the seconds in parens are cumulative since process start, so
    # consecutive differences ARE the per-epoch wall clock (P4 §5.1's method).
    print("\n  epoch markers (val_loss @ cumulative-elapsed -> epoch wall clock):")
    for k in KS:
        eps = data[k]["epochs"]
        if not eps:
            print(f"    K={k:<3} none yet")
            continue
        parts = []
        for i, (e, vl, cum) in enumerate(eps):
            wall = f"{cum - eps[i - 1][2]}s" if i else f"setup {cum}s"
            parts.append(f"ep{e} vl={vl:.4f} @{cum}s ({wall})")
        print(f"    K={k:<3} " + "  ".join(parts))

    # Matched-step table. This is the ONLY valid cross-arm loss comparison: the
    # arms are at different steps, so their latest losses are not comparable.
    marks = sorted({s for d in data.values() for s in d["steps"] if s % 2000 == 0})
    marks = marks[-8:]
    if marks:
        print("\nmatched-step loss / cls / off  (only rows where an arm has data):")
        print(f"{'step':>6} | {'K=5':<22} | {'K=10':<22} | {'K=40':<22}")
        for s in marks:
            cells = []
            for k in KS:
                v = data[k]["steps"].get(s)
                cells.append("--" if v is None else f"{v[0]:.4f} {v[1]:.3f} {v[2]:.3f}")
            print(f"{s:>6} | {cells[0]:<22} | {cells[1]:<22} | {cells[2]:<22}")
        last = marks[-1]
        if data[10]["steps"].get(last) and data[40]["steps"].get(last):
            g = data[10]["steps"][last][0] - data[40]["steps"][last][0]
            print(f"  (K=10 minus K=40 total loss at step {last}: {g:+.4f})")

    # Epoch-mean curves from metrics.json. The step lines above are single-batch
    # readings -- K=10 swings between 0.39 and 0.72 across matched steps -- so no
    # cross-arm claim can rest on them. These epoch means are what the comparison
    # actually rests on. `val_loss` exists only every eval_freq_epochs=10.
    print("\n  epoch-mean curves (metrics.json):  train_loss cls off [val_loss]")
    curves = {}
    for k in KS:
        p = Path(f"outputs/ksweep_k{k}/seed0/metrics.json")
        c = None
        if p.exists():
            raw = json.loads(p.read_text()).get("metrics", {})
            c = {name: {int(e): v for e, v in d.items()}
                 for name, d in raw.items() if isinstance(d, dict)}
        curves[k] = c
    epos = sorted({e for c in curves.values() if c for e in c.get("train_loss", {})})
    if not epos:
        print("    (no metrics.json yet)")
    else:
        print(f"    {'ep':>4} | {'K=5':<32} | {'K=10':<32} | {'K=40':<32}")
        for e in epos:
            cells = []
            for k in KS:
                c = curves[k]
                if not c or e not in c.get("train_loss", {}):
                    cells.append("--")
                    continue
                vl = c.get("val_loss", {}).get(e)
                cells.append(f"{c['train_loss'][e]:.4f} {c['classification_loss'][e]:.3f} "
                             f"{c['offset_loss'][e]:.3f} "
                             + (f"[{vl:.4f}]" if vl is not None else "[  -   ]"))
            print(f"    {e:>4} | {cells[0]:<32} | {cells[1]:<32} | {cells[2]:<32}")
        # Matched-epoch val is the only fair cross-arm comparison: the arms are at
        # different epochs, so their latest val numbers are not comparable either.
        common = [e for e in epos
                  if all(curves[k] and e in curves[k].get("val_loss", {}) for k in KS)]
        if common:
            last = common[-1]
            print(f"    matched-epoch val @ep{last}: " + "  ".join(
                f"K={k} {curves[k]['val_loss'][last]:.4f}" for k in KS))
        else:
            at = {k: sorted(curves[k].get("val_loss", {})) for k in KS if curves[k]}
            print(f"    no epoch with val for all three yet: {at}")

    print("\n  ckpts:", end=" ")
    ck = sorted(Path("outputs").glob("ksweep_k*/seed0/*.pt"))
    print("none yet (first at ep10)" if not ck else ", ".join(str(p) for p in ck))
    ro = sorted(Path("outputs").glob("ksweep_k*/rollout"))
    print("  rollout dirs:", "none (correct)" if not ro else f"PRESENT -- {ro}")


if __name__ == "__main__":
    main()
