"""Print the P1 anchor audit in the form the report needs (runs ON THE SERVER).

Reads `anchors/pusht_action_anchor_audit.json` written by `python -m
patch_policy.anchors` and prints, per K, the nearest-anchor error quantiles, the
cluster occupancy, and the anchor floor -- the four numbers P2/P6 depend on.

Usage (server):
    ~/miniconda3/envs/wemm-p1/bin/python scripts/report_p1.py [--json PATH] [--k 20]
"""
import argparse
import json
from pathlib import Path


def _f(v, nd=5):
    return f"{float(v):.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="anchors/pusht_action_anchor_audit.json")
    ap.add_argument("--k", type=int, default=20, help="the production K to headline")
    args = ap.parse_args()

    r = json.loads(Path(args.json).read_text())
    print("=" * 78)
    print("P1 -- PushT action corpus, normaliser, k-means anchors")
    print("=" * 78)
    m = r["meta"]
    print(f"dataset        {m['dataset_root']}")
    print(f"split          seed {m['split_seed']} fraction {m['train_fraction']} -> "
          f"{m['n_train_episodes']} train / {m['n_val_episodes']} val episodes")
    print(f"windows        {m['n_train_windows']} train / {m['n_val_windows']} val "
          f"(obs W={m['obs_window_size']}, act W={m['action_window_size']})")
    print(f"chunks         {r['coverage_audit'][f'K{args.k}']['n_chunks_train']} train / "
          f"{r['coverage_audit'][f'K{args.k}']['n_chunks_val']} val  (T per window)")
    print(f"train_idx sha  {m['train_episode_indices_sha256'][:32]}...")

    print("\n-- raw actions (this REPORTS, it does not redefine) --")
    print(f"pixel range    x [{_f(r['raw_action_min'][0], 1)}, {_f(r['raw_action_max'][0], 1)}]"
          f"   y [{_f(r['raw_action_min'][1], 1)}, {_f(r['raw_action_max'][1], 1)}]")

    print("\n-- normaliser (LinearNormalizer limits, FIT on train split) --")
    for k, v in r["normalizer_stats"].items():
        if isinstance(v, list):
            print(f"{k:14s} {[round(float(x), 6) for x in v]}")
        else:
            print(f"{k:14s} {v}")
    print(f"norm train     {r['norm_train_range']}")
    print(f"norm val       {r['norm_val_range']}")
    print(f"outside [-1,1] {r['n_outside_unit_interval']}   <- MUST be 0")

    print("\n" + "=" * 78)
    print("NEAREST-ANCHOR ERROR  (reference metric: L2 over A, mean over W; NORMALISED)")
    print("=" * 78)
    hdr = f"{'K':>4} {'split':>6} {'mean':>9} {'median':>9} {'P90':>9} {'P95':>9} {'max':>9}"
    print(hdr)
    for kk, d in sorted(r["coverage_audit"].items(), key=lambda kv: kv[1]["K"]):
        for split in ("train", "val"):
            s = d[split]
            print(f"{d['K']:>4} {split:>6} {_f(s['nearest_dist_mean']):>9} "
                  f"{_f(s['nearest_dist_median']):>9} {_f(s['nearest_dist_p90']):>9} "
                  f"{_f(s['nearest_dist_p95']):>9} {_f(s['nearest_dist_max']):>9}")

    print("\n" + "=" * 78)
    print(f"ANCHOR FLOOR -- the best this K can do without any refinement, K={args.k}")
    print("=" * 78)
    s = r["coverage_audit"][f"K{args.k}"]["train"]
    print(f"anchor_only L1 train  mean {_f(s['anchor_only_l1_mean'], 3)} px   "
          f"P50 {_f(s['anchor_only_l1_p50'], 3)} px   P90 {_f(s['anchor_only_l1_p90'], 3)} px   "
          f"max {_f(s['anchor_only_l1_max'], 3)} px")
    sv = r["coverage_audit"][f"K{args.k}"]["val"]
    print(f"anchor_only L1 val    mean {_f(sv['anchor_only_l1_mean'], 3)} px   "
          f"P50 {_f(sv['anchor_only_l1_p50'], 3)} px   P90 {_f(sv['anchor_only_l1_p90'], 3)} px")
    print("  coverage@tau (fraction of train chunks whose nearest anchor is within tau):")
    for tau, v in s["coverage"].items():
        print(f"    {tau:>8}  train {v:.4f}   val {r['coverage_audit'][f'K{args.k}']['val']['coverage'][tau]:.4f}")

    print("\n" + "=" * 78)
    print(f"CLUSTER OCCUPANCY, K={args.k} (train chunks per anchor)")
    print("=" * 78)
    print(f"smallest {s['cluster_occupancy_min']}   largest {s['cluster_occupancy_max']}   "
          f"mean {s['cluster_occupancy_mean']}   dead {s['dead_modes']}")
    print(f"share     min {_f(s['cluster_share_min'], 4)}   max {_f(s['cluster_share_max'], 4)}")
    print(f"per-anchor chunk counts: {s['cluster_size_hist']}")
    print(f"2nd/1st nearest ratio: median {_f(s['second_over_nearest_median'], 3)} "
          f"P10 {_f(s['second_over_nearest_p10'], 3)}   "
          f"ambiguous (within 1.1x): {_f(s['ambiguous_ratio_within_1.1'], 4)}")
    print(f"min pairwise anchor distance: {_f(r['kmeans']['min_pairwise_anchor_dist'])}")
    print(f"inertia: {r['kmeans']['inertia']}")

    print("\n" + "=" * 78)
    print("INFERENCE START TIMESTEP -- derived from the anchor error, not copied from 8")
    print("=" * 78)
    rec = r["recommend_t_start"]
    print(f"anchor error (normalised, median) {_f(rec['anchor_err_norm'])}")
    print(f"recommended infer_t_start {rec['recommended_t_start']} "
          f"-> b_t = {_f(rec['b_at_recommended'])}")
    tbl = {int(x["t"]): x for x in rec["table"]}
    for t in (0, 8, 10, 20, 30, 40, 49):
        if t in tbl:
            print(f"    t={t:>3}  b_t {_f(tbl[t]['b_t'])}   "
                  f"gap to anchor error {_f(tbl[t]['gap_to_anchor_err'])}")
    sat = rec["recommended_t_start"] >= max(tbl)
    print(f"SATURATED AT TOP OF BAND: {sat}"
          + ("  <- even the largest t in the training band noises LESS than the anchor"
             " is already off by; the head is therefore doing mostly CORRECTION, not"
             " denoising. Report this, do not paper over it." if sat else ""))

    print(f"\nartifact {r['artifacts']['anchors']}\nsha256   "
          f"{r['artifacts']['anchors_sha256']}")


if __name__ == "__main__":
    main()
