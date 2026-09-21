"""P2 -- small-set overfit: can the TD head actually fit, before any long run?

Trains the FULL policy (frozen DINO trunk + TD head) on the first N windows of the
TRAIN split only, for a fixed number of steps, with no augmentation and no shuffle.
If the head cannot drive the loss down on a handful of samples, no 100-epoch run will
fix it, and a bad P6 result would be misattributed to optimisation or to the diffusion
formulation instead of to a plain wiring bug.

Pass criteria (plan): raw-pixel L1 < 5 px AND mode accuracy > 0.95 on that subset,
with BOTH 1-step and 2-step inference converging.

ANCHOR CONSISTENCY: anchors and the normaliser are LOADED from the P1 artifact, which
was fit on the FULL train split. They are never refit here -- the script prints the
artifact's sha256 so a future reader can prove which anchors produced these numbers.
Refitting on N=32 samples is exactly the contamination this guards against.

Usage (server, GPU2 only):
    CUDA_VISIBLE_DEVICES=2 ~/miniconda3/envs/wemm-p1/bin/python scripts/overfit_small.py \
        --n-samples 32 --steps 2000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Run as `python scripts/overfit_small.py` from the repo root: sys.path[0] is then
# `scripts/`, not the root, so the package would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from patch_policy.anchors import sha256_file  # noqa: E402
from patch_policy.config import Config  # noqa: E402
from patch_policy.dataset import (  # noqa: E402
    WindowDataset,
    load_actions,
    load_episode_frames,
    load_seq_lengths,
    split_episodes,
)
from patch_policy.train import build_model  # noqa: E402


def code_fingerprint() -> dict:
    """sha256 of the three modules that determine the numbers in this report.

    Recorded in the JSON so a reader can tell whether a result came from the code as it
    stands. This is not hypothetical: P2 was first launched before `clip_sample` was
    added to `ddim_step`, which changes 1-step and 2-step inference, and without this
    field the resulting JSON would have looked indistinguishable from a post-fix one.
    """
    root = Path(__file__).resolve().parent.parent / "patch_policy"
    return {name: sha256_file(root / name)
            for name in ("diffusion.py", "td_head.py", "config.py")}


def windows_in_episode(L: int, T: int) -> int:
    """WindowDataset's own count: (T-1) start-virtual windows + max(0, L-T+1) full ones."""
    return (T - 1) + max(0, L - T + 1)


def select_samples(seq_lengths, train_idx, n_samples, T, mode):
    """Pick `n_samples` windows from the train split. Returns (episodes, local_indices).

    `mode="prefix"` -- the first N windows, exactly as a `WindowDataset` walk emits them.
    Because that walk is episode-major, those N windows come from ONE episode, which
    makes this the easiest possible test: near-identical targets, so a low L1 there says
    the head can memorise but says little about mode selection.

    `mode="spread"` -- one window per episode, taken at the episode's midpoint, from the
    first N train episodes. Different scenes entirely, so the modal structure of the
    action distribution is actually exercised. This is the stronger test and the one to
    read the mode-accuracy number from.

    Both return indices into a `WindowDataset` built over exactly the returned episodes,
    so the slicing/edge-padding is the real dataset's, never a reimplementation.
    """
    if mode == "prefix":
        eps, total = [], 0
        for ei in train_idx:
            if total >= n_samples:
                break
            eps.append(ei)
            total += windows_in_episode(seq_lengths[ei], T)
        assert total >= n_samples, f"train split holds only {total} windows"
        return eps, list(range(n_samples))

    assert mode == "spread", mode
    assert n_samples <= len(train_idx), f"need {n_samples} episodes, have {len(train_idx)}"
    eps = list(train_idx[:n_samples])
    # Global window index of episode k's midpoint window, given the episode-major order.
    idxs, base = [], 0
    for ei in eps:
        n = windows_in_episode(seq_lengths[ei], T)
        idxs.append(base + n // 2)
        base += n
    return eps, idxs


def encode_episodes(encoder, data_dir, ep_indices, device, bs=32):
    """DINO-P256 tokens for exactly these episodes -> {ei: [L,256,384]} fp32 on CPU."""
    out = {}
    for ei in ep_indices:
        frames = load_episode_frames(data_dir, ei)
        L = frames.shape[0]
        emb = torch.zeros(L, encoder.n_patches, encoder.output_dim)
        for s in range(0, L, bs):
            emb[s:s + bs] = encoder(frames[s:s + bs].to(device)).cpu()
        assert torch.isfinite(emb).all(), f"non-finite DINO tokens for episode {ei}"
        out[ei] = emb
    return out


@torch.no_grad()
def evaluate(model, ds, n, device, batch=32):
    """Raw-pixel L1 (inference path) + mode accuracy on the overfit subset.

    The HEADLINE number is the 2-step INFERENCE L1, because that is what rollout
    consumes -- a training-time `pred` can look fine while the sampling chain is broken,
    and that is precisely the failure this stage is meant to catch. The train-time L1 is
    reported alongside so the two can be told apart.
    """
    model.eval()
    per_step, per_train, accs = {}, [], []
    preds, gts = [], []
    for s in range(0, n, batch):
        idx = list(range(s, min(s + batch, n)))
        obs = torch.stack([ds[i][0] for i in idx]).to(device)
        act = torch.stack([ds[i][1] for i in idx]).to(device)
        gt = model._unpack_actions(act)                        # [B,T,W,A] raw px

        pred_tr, _, ld = model(obs, None, act)
        per_train.append((pred_tr - gt).abs().flatten(1).mean(1))
        accs.append(ld["mode_acc"])
        # Both chains evaluated separately, so a failure names the right one.
        for steps in (1, 2):
            model.set_inference(steps=steps)
            p_inf, _, _ = model(obs, None, None)
            per_step.setdefault(steps, []).append(
                (p_inf - gt).abs().flatten(1).mean(1))
            if steps == 2:
                preds.append(p_inf.cpu())
        gts.append(gt.cpu())
    model.train()
    l2 = torch.cat(per_step[2])
    out = {
        "raw_l1_infer2step_mean": float(l2.mean()),
        "raw_l1_infer2step_p90": float(torch.quantile(l2, 0.9)),
        "raw_l1_infer2step_max": float(l2.max()),
        "raw_l1_infer1step_mean": float(torch.cat(per_step[1]).mean()),
        "raw_l1_train_pred_mean": float(torch.cat(per_train).mean()),
        "mode_acc": float(sum(accs) / len(accs)),
    }
    return out, torch.cat(preds), torch.cat(gts), l2


def run_once(cfg, args, ds, n, device, lr, steps, tag):
    torch.manual_seed(cfg.seed)
    model = build_model(cfg).to(device)
    model.set_noise_generator(
        torch.Generator(device=device).manual_seed(cfg.td.noise_seed))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg.weight_decay,
                            betas=tuple(cfg.betas))
    print(f"\n[{tag}] lr={lr} steps={steps} n={n} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # Printed per run so the log itself carries the inference chain the numbers came from
    # (the mandated "model call #1: t_current = ... t_next = ..." evidence).
    print(model.describe_chain())
    t0, hist = time.time(), []
    for step in range(steps):
        batch = [i % n for i in range(step * args.batch_size,
                                       (step + 1) * args.batch_size)]
        obs = torch.stack([ds[i][0] for i in batch]).to(device)
        act = torch.stack([ds[i][1] for i in batch]).to(device)
        _, loss, ld = model(obs, None, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % args.log_every == 0 or step == steps - 1:
            hist.append({"step": step, "total_loss": float(loss),
                         "cls": ld["classification_loss"],
                         "reg": ld["offset_loss"],
                         "mode_acc": ld["mode_acc"],
                         "grad_norm": float(gnorm)})
            print(f"  [{tag}] step {step:5d} loss {float(loss):.4f} "
                  f"cls {ld['classification_loss']:.4f} reg {ld['offset_loss']:.4f} "
                  f"mode_acc {ld['mode_acc']:.3f} gnorm {float(gnorm):.3f}")
    dt = time.time() - t0
    metrics, preds, gts, l2 = evaluate(model, ds, n, device, batch=args.batch_size)
    metrics["steps_per_s"] = steps / dt
    metrics["wall_s"] = dt
    metrics["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(f"  [{tag}] FINAL infer2 {metrics['raw_l1_infer2step_mean']:.3f} px  "
          f"infer1 {metrics['raw_l1_infer1step_mean']:.3f}  "
          f"trainpred {metrics['raw_l1_train_pred_mean']:.3f}  "
          f"mode_acc {metrics['mode_acc']:.3f}  |  {dt:.0f}s "
          f"({metrics['steps_per_s']:.1f} steps/s, "
          f"peak {metrics['peak_gpu_mem_mb']:.0f} MB)")
    return {"lr": lr, "steps": steps, "history": hist, "final": metrics,
            "pass_infer2_lt_5px": metrics["raw_l1_infer2step_mean"] < 5.0,
            "pass_infer1_lt_5px": metrics["raw_l1_infer1step_mean"] < 5.0,
            "pass_mode_acc_gt_95": metrics["mode_acc"] > 0.95}, preds, gts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pusht_dino_p256_td.yaml")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default = cfg.batch_size (the training protocol's 8). The plan "
                         "said 32, but the GPT runs 5*256=1280-token sequences, so 32 "
                         "does not fit in 24 GB -- see the OOM note in the P2 report. "
                         "Using the training batch size also makes the overfit a fair "
                         "test of the optimiser that will actually be used.")
    ap.add_argument("--lrs", default="5.5e-5,1e-3",
                    help="first entry is the HARNESS lr (the honest test); the rest are "
                         "capacity probes, reported separately")
    ap.add_argument("--select", choices=("prefix", "spread"), default="spread",
                    help="prefix = the first N windows (all from ONE episode, easiest); "
                         "spread = the midpoint window of each of the first N episodes "
                         "(stronger, actually exercises mode selection)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default="outputs/p2_overfit")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # NOTE: under CUDA_VISIBLE_DEVICES=2 this prints `cuda:None` and reports the ONE
        # visible device, so it does NOT on its own prove the physical card was #2 --
        # the masking makes that unobservable from inside the process. The evidence that
        # GPU0/GPU1 were untouched is the `nvidia-smi --query-gpu=index,memory.used` line
        # that `run_p2.sh` prints immediately before and after the run, in the same log.
        print(f"[P2] visible cuda device {device.index} = {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB); "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    cfg = Config.from_yaml(args.config)
    assert cfg.action_head == "td", f"this is the TD arm's P2, got {cfg.action_head}"
    if args.batch_size is None:
        args.batch_size = cfg.batch_size
    T, W = cfg.obs_window_size, cfg.action_window_size

    data_dir = Path(cfg.dataset_root)
    seq_lengths = load_seq_lengths(data_dir)
    actions = load_actions(data_dir, len(seq_lengths), relative=cfg.use_relative_actions)
    train_idx, val_idx = split_episodes(len(seq_lengths), cfg.train_fraction,
                                        seed=cfg.split_seed)
    print(f"[P2] split: {len(train_idx)} train / {len(val_idx)} val episodes "
          f"(split_seed={cfg.split_seed})")
    assert len(train_idx) == 185, f"expected 185 train episodes, got {len(train_idx)}"

    eps, local_idx = select_samples(seq_lengths, train_idx, args.n_samples, T,
                                    args.select)
    print(f"[P2] select={args.select}: {args.n_samples} windows from {len(eps)} train "
          f"episode(s) {eps if len(eps) <= 12 else str(eps[:12]) + '...'}")

    anchor_path = Path(cfg.td.anchor_artifact)
    if not anchor_path.is_file():
        raise SystemExit(f"[P2] anchor artifact missing: {anchor_path} -- run P1 first")
    sha = sha256_file(anchor_path)
    print(f"[P2] anchors {anchor_path} sha256 {sha}")
    print(f"[P2] K={cfg.td.n_modes} cls_from={cfg.td.cls_from} "
          f"infer {cfg.td.infer_t_start}/{cfg.td.infer_steps} "
          f"train_t_max={cfg.td.train_t_max}")

    from patch_policy.encoder import build_encoder
    enc, _ = build_encoder(cfg)
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    t0 = time.time()
    emb_by_ep = encode_episodes(enc, data_dir, eps, device,
                                bs=cfg.dino.encode_batch_size)
    print(f"[P2] DINO encode: {sum(e.shape[0] for e in emb_by_ep.values())} frames "
          f"in {time.time()-t0:.0f}s")
    del enc
    torch.cuda.empty_cache()

    # Full-length list so `episodes_emb[ei]` resolves; only the involved episodes carry
    # real tokens. Placeholders use zero-size trailing dims (the same trick anchors.py
    # uses) so nothing but shape[0] is ever read from the untouched slots.
    episodes = [torch.zeros(seq_lengths[i], 0, 0) for i in range(len(seq_lengths))]
    for ei, e in emb_by_ep.items():
        episodes[ei] = e
    ds = WindowDataset(episodes, actions, T, W, episode_indices=eps)
    assert len(ds) > max(local_idx), (
        f"subset dataset built {len(ds)} windows, need index {max(local_idx)}")
    # Subset, not a bespoke sampler: the windows are the REAL dataset's, including its
    # repeat-start padding at episode heads.
    ds = torch.utils.data.Subset(ds, local_idx)
    n = len(ds)
    assert n == args.n_samples, f"selected {n} windows, wanted {args.n_samples}"
    print(f"[P2] overfit subset: {n} windows, {len(eps)} episode(s) "
          f"(val split NEVER touched; anchors from the full-train P1 artifact)")

    results = []
    for lr in (float(x) for x in args.lrs.split(",") if x.strip()):
        tag = "harness-lr" if abs(lr - cfg.lr) < 1e-12 else "probe"
        r, preds, gts = run_once(cfg, args, ds, n, device, lr, args.steps, tag)
        results.append(r)
        if tag == "harness-lr":
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"pred": preds, "gt": gts,
                        "anchors_raw": None, "subset": torch.tensor(eps)},
                       out / f"p2_pred_vs_gt_{args.select}_n{n}.npz")

    r0 = results[0]
    best = min(results, key=lambda r: r["final"]["raw_l1_infer2step_mean"])
    print("\n" + "=" * 74)
    print(f"P2 verdict @ harness lr ({r0['lr']}):")
    print(f"  2-step inference L1 < 5 px : {r0['pass_infer2_lt_5px']}  "
          f"({r0['final']['raw_l1_infer2step_mean']:.3f} px)")
    print(f"  1-step inference L1 < 5 px : {r0['pass_infer1_lt_5px']}  "
          f"({r0['final']['raw_l1_infer1step_mean']:.3f} px)")
    print(f"  mode accuracy > 0.95       : {r0['pass_mode_acc_gt_95']}  "
          f"({r0['final']['mode_acc']:.3f})")
    print(f"  (best across all runs: {best['final']['raw_l1_infer2step_mean']:.3f} px "
          f"at lr {best['lr']})")
    print("=" * 74)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "config": args.config, "n_samples": n, "steps": args.steps,
        "n_episodes_in_subset": len(eps),
        "batch_size": args.batch_size, "train_seed": cfg.seed,
        "split_seed": cfg.split_seed, "train_episodes": len(train_idx),
        "val_episodes": len(val_idx),
        "select": args.select, "subset_episodes": eps,
        "anchor_artifact": str(anchor_path), "anchor_sha256": sha,
        "code_sha256": code_fingerprint(),
        "clip_sample": bool(cfg.td.clip_sample),
        "clip_sample_range": float(cfg.td.clip_sample_range),
        "td": {"K": cfg.td.n_modes, "cls_from": cfg.td.cls_from,
               "infer_t_start": cfg.td.infer_t_start, "infer_steps": cfg.td.infer_steps,
               "train_t_max": cfg.td.train_t_max, "num_layers": cfg.td.num_layers},
        "results": results,
        "verdict": {k: v for k, v in r0.items()
                    if k.startswith("pass_")},
    }
    jname = f"p2_overfit_{args.select}_n{n}.json"
    (out / jname).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[P2] wrote {out}/{jname}")
    sys.exit(0)


if __name__ == "__main__":
    main()
