"""Stage-1b: encode the whole pusht dataset once with frozen WeMM -> embedding cache.

Run once per (model, dataset). Training reads the cache; WeMM is never in the
training graph, so per-batch cost is a 512-dim projection + policy forward.
"""

from __future__ import annotations

import argparse

import torch

from patch_policy.config import Config
from patch_policy.dataset import (
    load_episode_frames,
    load_seq_lengths,
    save_embedding_cache,
)
from patch_policy.encoder import build_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pusht.yaml")
    ap.add_argument("--episode-limit", type=int, default=None, help="smoke: only first N episodes")
    ap.add_argument("--out", type=str, default=None, help="override cache path")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default=None,
                    help="cache storage dtype (default fp32; WeMM-Local-P16 uses fp16)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cache_dtype = args.dtype or "fp32"

    print(f"[precompute] loading {cfg.encoder_name} encoder ({cfg.enc_model_name()}) ...")
    encoder, meta = build_encoder(cfg)
    encoder.to(device)

    seq_lengths = load_seq_lengths(cfg.dataset_root)
    n_episodes = args.episode_limit or len(seq_lengths)
    print(f"[precompute] episodes: {n_episodes} / {len(seq_lengths)}")

    enc = cfg.enc()
    episodes = []
    for ei in range(n_episodes):
        frames = load_episode_frames(cfg.dataset_root, ei)  # [L,3,H,W]
        L = frames.shape[0]
        emb = torch.zeros(L, enc.n_patches, enc.output_dim)
        bs = enc.encode_batch_size
        for s in range(0, L, bs):
            chunk = frames[s : s + bs].to(device)
            emb[s : s + bs] = encoder(chunk).cpu()  # [bs, P, D]
        assert torch.isfinite(emb).all()
        if cache_dtype == "fp16":
            emb = emb.half()  # quantize after fp32 isfinite check
        episodes.append(emb)
        if (ei + 1) % 10 == 0 or ei == n_episodes - 1:
            print(f"  encoded {ei + 1}/{n_episodes} episodes (last {L} frames, "
                  f"per-episode emb {tuple(emb.shape)} {emb.dtype})")

    meta.update(
        {
            "dataset_root": cfg.dataset_root,
            "n_episodes": n_episodes,
            "seq_lengths": seq_lengths[:n_episodes],
            "cache_dtype": cache_dtype,
            "local_mode": getattr(enc, "local_mode", False),
        }
    )
    out = args.out or cfg.precompute_cache
    save_embedding_cache(out, meta, episodes)
    total_frames = sum(e.shape[0] for e in episodes)
    print(f"[precompute] saved {out}: {n_episodes} episodes, {total_frames} frames, "
          f"episode emb shape [L, {enc.n_patches}, {enc.output_dim}] dtype={cache_dtype}")


if __name__ == "__main__":
    main()
