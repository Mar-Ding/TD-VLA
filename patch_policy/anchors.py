"""K-means action-chunk anchors for the truncated-diffusion head (P1).

The anchors are the multi-modal prior the diffusion head is centred on: K
canonical action chunks in NORMALISED space, one per mode. They are built from
the TRAIN split only, never from val/test, and the split is reproduced through
`dataset.split_episodes` itself rather than re-derived, so it cannot drift.

Self-implemented k-means (k-means++ init + Lloyd) because `wemm-p1` has no
sklearn and we are not going to modify that env — the existing VQ-BeT/AdaLN arms
share it. A seeded torch.Generator on CPU makes the result reproducible across
torch versions, which matters because the anchor file's sha256 is recorded in the
audit and every checkpoint carries it.

The corpus is built by instantiating the REAL `WindowDataset` with placeholder
embeddings (we only consume its `act` half). That way the sample list and the
repeat-padding at window edges are the training pipeline's own, not a
reimplementation that could silently disagree with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from patch_policy.config import Config
from patch_policy.dataset import WindowDataset, load_actions, load_seq_lengths, split_episodes

__all__ = [
    "unpack_actions_like_vqbet",
    "collect_train_chunks",
    "kmeans_torch",
    "coverage_audit",
    "save_anchor_artifact",
    "load_anchor_artifact",
    "sha256_file",
]


def unpack_actions_like_vqbet(action_seq: torch.Tensor, act_window_size: int) -> torch.Tensor:
    """Verbatim copy of BehaviorTransformer._unpack_actions (vqbet.py:222-229).

    [N, T+W-1, A] -> [N, T, W, A]. P0 asserts this agrees numerically with the real
    method so the two cannot silently diverge.
    """
    n, total_w, act_dim = action_seq.shape
    obs_w = total_w + 1 - act_window_size
    result = action_seq.new_empty((n, obs_w, act_window_size, act_dim))
    for i in range(obs_w):
        result[:, i, :, :] = action_seq[:, i : i + act_window_size, :]
    return result


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_train_chunks(
    cfg: Config, max_episodes: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """[N, W, A] RAW chunks from the TRAIN split, plus provenance metadata.

    One chunk per (window, timestep): window i of an episode contributes T chunks
    (the T action offsets the policy is asked to predict). Overlapping chunks are
    deliberately NOT de-duplicated, so a chunk's weight in the k-means objective
    equals how often it appears as a training target.
    """
    data_dir = Path(cfg.dataset_root)
    seq_lengths = load_seq_lengths(data_dir)
    n_episodes = len(seq_lengths)
    actions = load_actions(data_dir, n_episodes, relative=False)

    train_idx, val_idx = split_episodes(n_episodes, cfg.train_fraction, seed=cfg.split_seed)
    if max_episodes is not None:
        train_idx = train_idx[:max_episodes]

    # Placeholder embeddings: WindowDataset only reads .shape[0] from them for its
    # sample list, and we never touch the obs half. Zero-size trailing dims keep
    # the repeat-padding in __getitem__ cheap.
    emb = [torch.zeros(seq_lengths[i], 0, 0) for i in range(n_episodes)]
    ds = WindowDataset(emb, actions, cfg.obs_window_size, cfg.action_window_size,
                       episode_indices=train_idx)

    chunks: List[torch.Tensor] = []
    for i in range(len(ds)):
        act = ds[i][1]                                   # [T+W-1, A], padded
        chunks.append(unpack_actions_like_vqbet(act.unsqueeze(0),
                                                cfg.action_window_size)[0])
    train_chunks = torch.stack(chunks, 0).float()         # [n_win, T, W, A]

    # val chunks: REPORTED ONLY, never fitted on.
    ds_val = WindowDataset(emb, actions, cfg.obs_window_size, cfg.action_window_size,
                           episode_indices=val_idx)
    val_chunks = torch.stack(
        [unpack_actions_like_vqbet(ds_val[i][1].unsqueeze(0),
                                   cfg.action_window_size)[0] for i in range(len(ds_val))],
        0,
    ).float()

    meta = {
        "dataset_root": str(data_dir),
        "n_episodes": n_episodes,
        "n_train_episodes": len(train_idx),
        "n_val_episodes": len(val_idx),
        "split_seed": cfg.split_seed,
        "train_fraction": cfg.train_fraction,
        "obs_window_size": cfg.obs_window_size,
        "action_window_size": cfg.action_window_size,
        "n_train_windows": len(ds),
        "n_val_windows": len(ds_val),
        "n_train_chunks": int(train_chunks.shape[0]),
        "n_val_chunks": int(val_chunks.shape[0]),
        "train_episode_indices_sha256": hashlib.sha256(
            ",".join(str(i) for i in train_idx).encode()
        ).hexdigest(),
        "val_episode_indices": list(val_idx),
    }
    return train_chunks, val_chunks, meta


def kmeans_torch(
    x: torch.Tensor,
    k: int,
    seed: int = 0,
    iters: int = 100,
    tol: float = 1e-7,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """k-means++ init + Lloyd on CPU. Returns (centroids [K,D], assign [N], inertia).

    Deterministic for a given (x, k, seed): the generator is explicit and CPU-only,
    so this does not depend on the ambient RNG or on CUDA nondeterminism.
    """
    assert x.ndim == 2, f"expected [N,D], got {tuple(x.shape)}"
    n, d = x.shape
    assert k <= n, f"k={k} exceeds n={n}"
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = x.detach().to(torch.float32).cpu()

    # ---- k-means++ seeding ----
    centers = torch.empty(k, d, dtype=torch.float32)
    first = int(torch.randint(0, n, (1,), generator=g).item())
    centers[0] = x[first]
    closest = torch.cdist(x, centers[0:1]).squeeze(1) ** 2
    for c in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centers[c] = x[int(torch.randint(0, n, (1,), generator=g).item())]
        else:
            probs = closest / total
            pick = int(torch.multinomial(probs, 1, generator=g).item())
            centers[c] = x[pick]
        closest = torch.minimum(closest, torch.cdist(x, centers[c : c + 1]).squeeze(1) ** 2)

    # ---- Lloyd ----
    prev_inertia = float("inf")
    assign = torch.zeros(n, dtype=torch.long)
    for it in range(iters):
        d2 = torch.cdist(x, centers) ** 2          # [N,K]
        assign = d2.argmin(dim=1)
        inertia = float(d2.gather(1, assign[:, None]).sum())
        # Recompute centres; empty clusters keep their previous position (a dead mode
        # is information the audit reports, not something to silently reseed away).
        new_centers = centers.clone()
        for c in range(k):
            m = assign == c
            if m.any():
                new_centers[c] = x[m].mean(dim=0)
        centers = new_centers
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"    [kmeans K={k}] iter {it:3d} inertia {inertia:.6f}")
        if abs(prev_inertia - inertia) <= tol * max(1.0, abs(prev_inertia)):
            prev_inertia = inertia
            break
        prev_inertia = inertia
    return centers, assign, prev_inertia


def mode_distance(gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """The reference's mode-selection distance (multimodal_loss.py:131-134).

    L2 over the coordinate dim, then MEAN over the time dim -> [..., K].
    `gt` is [..., W, A], `anchors` is [K, W, A].
    """
    a = anchors.to(gt.dtype).to(gt.device)
    diff = gt.unsqueeze(-3) - a                    # [..., K, W, A]
    return torch.linalg.norm(diff, dim=-1).mean(dim=-1)


def coverage_audit(
    train_chunks: torch.Tensor,
    val_chunks: torch.Tensor,
    ks: Tuple[int, ...] = (8, 16, 20, 32),
    seed: int = 0,
    normalizer=None,
    taus: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
) -> Dict:
    """For each K: how well K anchors can cover the action-chunk distribution.

    The number that matters is `anchor_only_l1_px` — the error floor of the WHOLE
    METHOD. No amount of training can beat it, because the best the head can do is
    pick an anchor and refine it; if K=20 cannot place an anchor near the typical
    GT chunk, the refinement starts from a systematically wrong place. P1 must
    publish this before P2 so a bad result is not misread as an optimisation failure.
    """
    out: Dict[str, dict] = {}
    # Same flattening rule as main(): T belongs to the SAMPLE axis, because an anchor
    # is a single [W,A] chunk. `*_flat` feeds k-means; `*_ch` keeps the chunk structure
    # and is what `_stats` measures against.
    h, w = train_chunks.shape[-2], train_chunks.shape[-1]
    tr_flat, tr_ch = train_chunks.reshape(-1, h * w), train_chunks.reshape(-1, h, w)
    va_ch = val_chunks.reshape(-1, h, w)

    for k in ks:
        centers, assign, inertia = kmeans_torch(tr_flat, k, seed=seed)
        centers_ch = centers.reshape(k, h, w)

        def _stats(ch: torch.Tensor) -> dict:
            d_norm = mode_distance(ch, centers_ch)              # [N,K] reference metric
            nearest, idx = d_norm.min(dim=1)
            # second-nearest, for the ambiguity ratio: how often is the mode target
            # genuinely a coin flip between two anchors?
            d2 = d_norm.clone()
            d2.scatter_(1, idx[:, None], float("inf"))
            second = d2.min(dim=1).values
            ratio = second / nearest.clamp_min(1e-12)           # >= 1 by construction
            chosen = centers_ch[idx]                            # [N,W,A] normalised
            # Exact pixel-space error. unnormalize() is (x - offset)/scale, so for two
            # points |raw_a - raw_b| == |norm_a - norm_b| / |scale| ELEMENTWISE — it is
            # not a single scalar rescale, the per-dim scale differs across the chunk.
            err = (chosen - ch).abs()
            if normalizer is not None:
                err = err / normalizer.scale.reshape(1, *ch.shape[1:]).abs()
            l1_px = err.flatten(1).mean(dim=1)                  # per-chunk mean |err| px
            q = torch.quantile(nearest, torch.tensor([0.5, 0.9, 0.95, 1.0]))
            counts = torch.bincount(idx, minlength=k)
            share = counts.float() / counts.sum()
            return {
                # The P1 report's headline numbers: nearest-anchor error, i.e. how far
                # the typical GT chunk is from the closest of the K anchors (normalised
                # space, reference metric `mode_distance`).
                "nearest_dist_mean": round(float(nearest.mean()), 8),
                "nearest_dist_median": round(float(q[0]), 8),
                "nearest_dist_p90": round(float(q[1]), 8),
                "nearest_dist_p95": round(float(q[2]), 8),
                "nearest_dist_max": round(float(q[3]), 8),
                "anchor_only_l1_p50": round(float(torch.quantile(l1_px, 0.5)), 8),
                "anchor_only_l1_p90": round(float(torch.quantile(l1_px, 0.9)), 8),
                "anchor_only_l1_max": round(float(l1_px.max()), 8),
                "anchor_only_l1_mean": round(float(l1_px.mean()), 8),
                "coverage": {f"tau_{t}": round(float((nearest < t).float().mean()), 6)
                             for t in taus},
                # Cluster occupancy: how the train corpus distributes over the K modes.
                "cluster_occupancy_min": int(counts.min()),
                "cluster_occupancy_max": int(counts.max()),
                "cluster_occupancy_mean": round(float(counts.float().mean()), 2),
                "cluster_share_min": round(float(share.min()), 6),
                "cluster_share_max": round(float(share.max()), 6),
                "cluster_size_hist": counts.tolist(),
                "dead_modes": int((counts == 0).sum()),
                "mode_usage_min": int(counts.min()),
                "mode_usage_max": int(counts.max()),
                # AMBIGUITY. `second >= nearest` holds by construction, so the raw ratio
                # is always >= 1 and a "> 0.9" test would be vacuously true -- it has to
                # be an UPPER bound to mean anything. Reported two ways: the ratio
                # quantiles, and the fraction within 10% (the mode target is then close
                # to a coin flip, and the cls head is being asked an ill-posed question).
                "second_over_nearest_median": round(float(ratio.median()), 6),
                "second_over_nearest_p10": round(float(torch.quantile(ratio, 0.10)), 6),
                "ambiguous_ratio_within_1.1": round(float((ratio <= 1.1).float().mean()), 6),
            }

        out[f"K{k}"] = {
            "K": k,
            "inertia": round(float(inertia), 6),
            "n_chunks_train": int(tr_ch.shape[0]),
            "n_chunks_val": int(va_ch.shape[0]),
            # Units note, because getting this wrong silently inverts a conclusion:
            # `nearest_dist_*` are NORMALISED (comparable to b_t, used for t_start);
            # `anchor_only_l1_*` are PIXELS (the method's error floor on the task's own
            # scale). cluster_occupancy_* are chunk counts.
            "units": {"nearest_dist": "normalised", "anchor_only_l1": "pixels",
                      "cluster_occupancy": "chunks"},
            "train": _stats(tr_ch),
            "val": _stats(va_ch),
        }
    return out


def save_anchor_artifact(
    path: str | Path,
    anchors_raw: torch.Tensor,
    anchors_norm: torch.Tensor,
    normalizer,
    meta: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "anchors_raw": anchors_raw,                 # [K,W,A] pixel space
            "anchors_norm": anchors_norm,               # [K,W,A] normalised space
            "normalizer": normalizer.state_dict(),      # buffers: scale/offset/stats
            "normalizer_stats": normalizer.stats(),
            "meta": meta,
        },
        path,
    )


def load_anchor_artifact(path: str | Path) -> Dict:
    """NOTE: weights_only=False — the payload carries a module state_dict and plain
    python meta, not just tensors (torch >= 2.6 defaults weights_only=True)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="P1: action normaliser + k-means anchors")
    ap.add_argument("--config", default="configs/pusht_dino_p256_td.yaml")
    ap.add_argument("--k", type=int, default=20, help="production K (spec: prefer 20)")
    ap.add_argument("--audit-ks", default="8,16,20,32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="anchors")
    ap.add_argument("--report", default="anchors/pusht_action_anchor_audit.json")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="debug only; MUST stay None for the real artifact")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    from patch_policy.normalizer import ActionNormalizer

    train_chunks, val_chunks, meta = collect_train_chunks(cfg, args.max_episodes)
    print(f"[P1] train chunks {tuple(train_chunks.shape)} | val chunks {tuple(val_chunks.shape)}")
    print(f"[P1] train windows {meta['n_train_windows']} "
          f"({meta['n_train_episodes']} eps) | val windows {meta['n_val_windows']} "
          f"({meta['n_val_episodes']} eps)")
    raw_lo = train_chunks.reshape(-1, train_chunks.shape[-1]).min(0).values
    raw_hi = train_chunks.reshape(-1, train_chunks.shape[-1]).max(0).values
    print(f"[P1] RAW pixel action range: min {raw_lo.tolist()} max {raw_hi.tolist()}")

    normalizer = ActionNormalizer(cfg.action_window_size, 2)
    normalizer.fit(train_chunks)
    norm_tr = normalizer.normalize(train_chunks)
    norm_va = normalizer.normalize(val_chunks)
    outside = int((norm_tr.abs() > 1.0 + 1e-6).sum())
    print(f"[P1] normalised train range [{norm_tr.min():.6f}, {norm_tr.max():.6f}] "
          f"| outside [-1,1]: {outside} (must be 0)")
    print(f"[P1] normalised VAL range  [{norm_va.min():.6f}, {norm_va.max():.6f}]")
    assert outside == 0, "train split did not normalise into [-1,1] exactly"

    # FLATTEN T INTO THE SAMPLE AXIS, not into the feature axis. `train_chunks` is
    # [n_win, T, W, A] -- each window contributes T chunks, and an ANCHOR is one chunk
    # [W,A]. So the k-means corpus is [(n_win*T), W*A]; reshaping to [n_win, T*W*A]
    # would cluster whole windows and hand the head a [K, T*W*A] tensor that cannot be
    # reshaped to [K,W,A] at all.
    n_chunks = norm_tr.shape[0] * norm_tr.shape[1]
    flat_tr = norm_tr.reshape(n_chunks, norm_tr.shape[-2] * norm_tr.shape[-1])
    centers, assign, inertia = kmeans_torch(flat_tr, args.k, seed=args.seed, verbose=True)
    anchors_norm = centers.reshape(args.k, norm_tr.shape[-2], norm_tr.shape[-1])
    print(f"[P1] k-means corpus: {tuple(norm_tr.shape)} -> [{n_chunks}, "
          f"{flat_tr.shape[1]}] chunks, anchors {tuple(anchors_norm.shape)}")
    anchors_raw = normalizer.unnormalize(anchors_norm)

    pairwise = torch.cdist(centers, centers)
    pairwise.fill_diagonal_(float("inf"))
    print(f"[P1] K={args.k} inertia {inertia:.6f} | min pairwise anchor dist "
          f"{float(pairwise.min()):.6f} | dead modes {int((torch.bincount(assign, minlength=args.k) == 0).sum())}")

    ks = tuple(int(x) for x in args.audit_ks.split(",") if x.strip())
    audit = coverage_audit(norm_tr, norm_va, ks=ks, seed=args.seed, normalizer=normalizer)
    for key, r in audit.items():
        t, v = r["train"], r["val"]
        print(f"  {key}: anchor_only_l1_p50 train {t['anchor_only_l1_p50']:.5f} "
              f"val {v['anchor_only_l1_p50']:.5f} | dead {t['dead_modes']} | "
              f"cov@0.10 train {t['coverage']['tau_0.1']:.3f}")

    from patch_policy.diffusion import TruncatedDiffusionSchedule, recommend_t_start
    sched = TruncatedDiffusionSchedule(train_t_max=cfg.td.train_t_max)
    # UNITS: `b_t` is the noise scale of the NORMALISED state, so the anchor error it is
    # matched against must be in normalised units too. `nearest_dist_median` is exactly
    # that (reference `mode_distance`: L2 over A, mean over W). `anchor_only_l1_*` is in
    # PIXELS -- comparing those against b_t would be off by a factor of ~233 and would
    # always saturate at the top of the band.
    anchor_err_norm = audit[f"K{args.k}"]["train"]["nearest_dist_median"]
    rec = recommend_t_start(sched, anchor_err_norm)
    print(f"[P1] recommended infer_t_start = {rec['recommended_t_start']} "
          f"(anchor err {rec['anchor_err_norm']:.5f} vs b_t {rec['b_at_recommended']:.5f})")

    out = Path(args.out)
    save_anchor_artifact(out / f"pusht_kmeans_K{args.k}.pt", anchors_raw, anchors_norm,
                         normalizer, {**meta, "K": args.k, "seed": args.seed,
                                      "inertia": inertia,
                                      "min_pairwise_anchor_dist": float(pairwise.min()),
                                      "recommended_t_start": rec})
    report = {
        "config": {"obs_window_size": cfg.obs_window_size,
                   "action_window_size": cfg.action_window_size, "K": args.k},
        "meta": meta,
        "normalizer_stats": normalizer.stats(),
        "raw_action_min": [round(float(v), 6) for v in raw_lo],
        "raw_action_max": [round(float(v), 6) for v in raw_hi],
        "norm_train_range": [round(float(norm_tr.min()), 8), round(float(norm_tr.max()), 8)],
        "norm_val_range": [round(float(norm_va.min()), 8), round(float(norm_va.max()), 8)],
        "n_outside_unit_interval": outside,
        "kmeans": {"K": args.k, "seed": args.seed, "inertia": inertia,
                   "min_pairwise_anchor_dist": float(pairwise.min())},
        "coverage_audit": audit,
        "recommend_t_start": rec,
        "artifacts": {"anchors": str(out / f"pusht_kmeans_K{args.k}.pt"),
                      "anchors_sha256": sha256_file(out / f"pusht_kmeans_K{args.k}.pt")},
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[P1] wrote {out / f'pusht_kmeans_K{args.k}.pt'} (sha256 "
          f"{report['artifacts']['anchors_sha256'][:16]}...) and {rp}")


if __name__ == "__main__":
    main()
