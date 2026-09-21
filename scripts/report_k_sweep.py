"""Summarise the K in {5, 10, 20, 40} anchor audits side by side.

Reads `anchors/audit_K{k}.json` (written by `make_anchors_k5_10_40.sh`) plus the
original K=20 report at its default path, and prints the numbers that decide whether
the sweep is a fair single-variable comparison.

Run from ~/td_pusht:  python scripts/report_k_sweep.py
"""
import json
from pathlib import Path

import torch

KS = (5, 10, 20, 40)
TAUS = ("0.02", "0.05", "0.1", "0.2")

# The K=20 report predates the sweep script and lives at the default path.
K20_REPORT = "anchors/pusht_action_anchor_audit.json"


def load(k: int) -> dict:
    p = Path(K20_REPORT) if k == 20 else Path(f"anchors/audit_K{k}.json")
    return json.loads(p.read_text())


def main() -> None:
    print("=== anchor geometry ===")
    hdr = ("K", "inertia", "dead", "minpair", "l1_p50(px)", "l1_p90(px)",
           "nd_med_tr", "nd_med_va", "rec_t")
    print("%4s %12s %5s %8s %11s %10s %10s %10s %6s" % hdr)
    for k in KS:
        r = load(k)
        a = r["coverage_audit"]["K%d" % k]["train"]
        v = r["coverage_audit"]["K%d" % k]["val"]
        km = r["kmeans"]
        rec = r["recommend_t_start"]["recommended_t_start"]
        print("%4d %12.2f %5d %8.4f %11.4f %10.4f %10.5f %10.5f %6d" % (
            k, km["inertia"], a["dead_modes"], km["min_pairwise_anchor_dist"],
            a["anchor_only_l1_p50"], a["anchor_only_l1_p90"],
            a["nearest_dist_median"], v["nearest_dist_median"], rec))

    print()
    print("=== coverage@tau, nearest-anchor distance (train/val) ===")
    print("%4s | %s" % ("K", " ".join("%21s" % ("tau=" + t) for t in TAUS)))
    for k in KS:
        r = load(k)
        a = r["coverage_audit"]["K%d" % k]["train"]
        v = r["coverage_audit"]["K%d" % k]["val"]
        cells = " ".join("%10.3f/%-10.3f" % (a["coverage"]["tau_" + t],
                                             v["coverage"]["tau_" + t]) for t in TAUS)
        print("%4d | %s" % (k, cells))

    print()
    print("=== mode usage (train chunks per anchor) ===")
    for k in KS:
        r = load(k)
        a = r["coverage_audit"]["K%d" % k]["train"]
        hist = a["cluster_size_hist"]
        print("  K=%2d  min %6d  max %6d  mean %9.1f  dead %d" % (
            k, a["cluster_occupancy_min"], a["cluster_occupancy_max"],
            a["cluster_occupancy_mean"], a["dead_modes"]))
        if k <= 10:
            print("        histogram: %s" % hist)

    print()
    print("=== artifacts ===")
    for k in (5, 10, 40):
        p = torch.load("anchors/pusht_kmeans_K%d.pt" % k, map_location="cpu",
                       weights_only=False)
        an = p["anchors_norm"]
        print("  K=%2d norm %s raw %s meta.K=%s rec_t_start=%s" % (
            k, tuple(an.shape), tuple(p["anchors_raw"].shape),
            p["meta"]["K"], p["meta"]["recommended_t_start"]["recommended_t_start"]))

    # The claim the whole sweep rests on: the normaliser must NOT vary with K.
    print()
    print("=== normaliser invariance (must be one hash) ===")
    import hashlib
    for k in KS:
        r = load(k)
        flat = json.dumps(r["normalizer_stats"], sort_keys=True)
        print("  K=%2d  %s" % (k, hashlib.sha256(flat.encode()).hexdigest()[:32]))


if __name__ == "__main__":
    main()
