"""Build the fusion conditioning-stream embedding cache (cond_encoder over all frames).

Mirror of precompute.py's frame loop + cache format, but the encoded stream is the
FUSION COND encoder selected by cfg.fusion.cond_encoder (wemm -> WeMM-Embedding-2B
or qwen3vl -> Qwen3-VL-Embedding-2B), NOT the main DINO-P256 dense stream (which
fusion training computes in-memory). This is how WeMM's wemm_cache/pusht_wemm.pt
was produced; qwen3vl writes to its own namespace
qwen3_vl_embedding_2b_image_only/ (never overwrites wemm_cache/).

The same obses/episode_XXX.pth frames are encoded with the same chunk loop, so the
cond cache is per-frame aligned with the DINO-P256 in-memory tokens by construction.
meta keys match what train.py's fusion branch checks (model_name / n_patches /
output_dim) via load_embedding_cache(check_meta=...).

Guards: requires cfg.fusion.enabled, and the cond encoder's output_dim audit-filled
(>0) — for qwen3vl this stays 0 until Phase-2 fills it, and build_cond_encoder
raises NotImplementedError until the Qwen3-VL-Embedding encoder module lands (Phase-1).

Run once per (cond_encoder, dataset):  python -m patch_policy.precompute_cond --config <cfg>
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
from patch_policy.encoder import build_cond_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pusht_dino_p256_qwen3vl_adaln.yaml")
    ap.add_argument("--episode-limit", type=int, default=None, help="smoke: only first N episodes")
    ap.add_argument("--out", type=str, default=None, help="override cache path")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default=None,
                    help="cache storage dtype (default fp32, same as wemm_cache)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    assert cfg.fusion.enabled, "precompute_cond is for fusion runs (fusion.enabled=true)"
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cache_dtype = args.dtype or "fp32"

    cc = cfg.cond_config()
    assert cc.output_dim > 0, (
        f"{cfg.fusion.cond_encoder}.output_dim={cc.output_dim} — must be audit-filled "
        f"(>0) before building the cond cache (no 2048 prior for qwen3vl)"
    )

    print(f"[precompute_cond] building cond stream '{cfg.fusion.cond_encoder}' "
          f"({cfg.cond_model_name()}) P={cc.n_patches} D={cc.output_dim} ...")
    cond_encoder = build_cond_encoder(cfg)  # qwen3vl: NotImplementedError until Phase-1 module lands
    cond_encoder.to(device)

    seq_lengths = load_seq_lengths(cfg.dataset_root)
    n_episodes = args.episode_limit or len(seq_lengths)
    print(f"[precompute_cond] episodes: {n_episodes} / {len(seq_lengths)}")

    episodes = []
    for ei in range(n_episodes):
        frames = load_episode_frames(cfg.dataset_root, ei)  # [L,3,H,W] fp32 0..1
        L = frames.shape[0]
        emb = torch.zeros(L, cc.n_patches, cc.output_dim)
        bs = cc.encode_batch_size
        for s in range(0, L, bs):
            chunk = frames[s : s + bs].to(device)
            emb[s : s + bs] = cond_encoder(chunk).cpu()  # [bs, P, D]
        assert torch.isfinite(emb).all()
        if cache_dtype == "fp16":
            emb = emb.half()  # quantize after fp32 isfinite check
        episodes.append(emb)
        if (ei + 1) % 10 == 0 or ei == n_episodes - 1:
            print(f"  encoded {ei + 1}/{n_episodes} episodes (last {L} frames, "
                  f"per-episode emb {tuple(emb.shape)} {emb.dtype})")

    meta = {
        "encoder": cfg.fusion.cond_encoder,
        "model_name": cc.model_name,
        "dtype": cc.dtype,
        "n_patches": cc.n_patches,
        "output_dim": cc.output_dim,
        "local_mode": getattr(cc, "local_mode", False),
        "image_only": getattr(cc, "image_only", False),
        "dataset_root": cfg.dataset_root,
        "n_episodes": n_episodes,
        "seq_lengths": seq_lengths[:n_episodes],
        "cache_dtype": cache_dtype,
    }
    out = args.out or cfg.fusion.cond_cache
    save_embedding_cache(out, meta, episodes)
    total_frames = sum(e.shape[0] for e in episodes)
    print(f"[precompute_cond] saved {out}: {n_episodes} episodes, {total_frames} frames, "
          f"episode emb shape [L, {cc.n_patches}, {cc.output_dim}] dtype={cache_dtype}")


if __name__ == "__main__":
    main()
