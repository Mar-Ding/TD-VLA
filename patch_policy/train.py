"""Train the visual-policy (VQ-BeT) on precomputed encoder embeddings.

    python -m patch_policy.train --config configs/pusht.yaml            # full training
    python -m patch_policy.train --config configs/dino_pusht.yaml       # DINO-P1
    python -m patch_policy.train --config configs/pusht.yaml --smoke    # 100-step smoke

The vision encoder (WeMM-P1 or DINO-P1) is NOT in the training graph:
observations are precomputed embeddings (patch_policy.precompute). The policy's
input projection (GPT wte) is the visual projection (enc.output_dim -> n_embd;
WeMM 2048 / DINO 384).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_policy.config import Config, seed_everything
from patch_policy.dataset import (
    FusionWindowDataset,
    WindowDataset,
    expand_repeat_patches,
    load_actions,
    load_embedding_cache,
    precompute_dino_in_memory,
    split_episodes,
)
from patch_policy.adaln import _stats
from patch_policy.vqbet import BehaviorTransformer


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    return f"alloc={a:.2f}G reserved={r:.2f}G"


def _atomic_save(obj: dict, path: Path) -> None:
    """Write a ckpt/json atomically: torch.save to `<name>.tmp` then os.replace.

    A machine-level kill mid-`torch.save` previously left a truncated .pt at the
    final path (09-03 ep20 = 71MB / 436MB) that resume + rollout both crashed on
    (`failed finding central directory`). Temp+rename guarantees the final path
    only ever holds a complete file; an interrupted save leaves only a .tmp
    orphan that ready_ckpt/rollout ignore.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    """Atomic metrics.json / STOPPED.json write (same rationale as _atomic_save)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _saved_optim_aligns(saved: dict, optim) -> bool:
    """Whether saved GroupedOptimizer state aligns with the current optimizer's params.

    AdamW state_dict is restored by POSITION (index within each param_group), not by
    param name. The optimizer's param order comes from iterating `decay`/`no_decay`
    Python sets (see vqbet.configure_optimizers), whose order depends on string hashing
    and varies across processes (PYTHONHASHSEED) — and changed when film->adaln renamed
    the AdaLN params. If the model's param order differs from when the ckpt was saved,
    the positional zip misaligns and exp_avg shapes clash (2048 vs 512 crash at
    optim.step). This shape check is a FALLBACK for legacy ckpts that carry no param
    names; any failure is now a hard stop in resume_optimizer (never a silent fresh
    AdamW).
    """
    try:
        if set(saved.keys()) != set(range(len(optim.optimizers))):
            return False
        for i, opt in enumerate(optim.optimizers):
            opt_sd = saved[i]
            saved_groups = opt_sd["param_groups"]
            if len(saved_groups) != len(opt.param_groups):
                return False
            saved_ids = [pid for g in saved_groups for pid in g["params"]]
            cur_params = [p for g in opt.param_groups for p in g["params"]]
            if len(saved_ids) != len(cur_params):
                return False
            state = opt_sd["state"]
            for pid, p in zip(saved_ids, cur_params):
                s = state.get(pid)
                if s is None or "exp_avg" not in s:
                    continue  # param never stepped -> no state to restore
                if tuple(s["exp_avg"].shape) != tuple(p.shape):
                    return False
        return True
    except Exception:
        return False


# ---- resume completeness (bug 3) -------------------------------------------
# A checkpoint must carry everything needed to reproduce the exact continuation.
# AdamW restores by POSITION, so the ordered param NAMES make a misalignment
# detectable instead of silent; RNG state makes the run reproducible; the actual
# LR / epoch / step / VQ mode make the resume auditable.


def _optim_param_names(model: nn.Module, optim) -> dict:
    """{optimizer_index: [[param name per group slot], ...]} — the positional map.

    Saved in every ckpt so the next resume can verify, by NAME, that each AdamW
    state slot still points at the same parameter it was saved for.
    """
    lookup = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for i, opt in enumerate(optim.optimizers):
        out[i] = [[lookup.get(id(p), f"<unnamed:{id(p)}>") for p in g["params"]]
                  for g in opt.param_groups]
    return out


def _optim_params(optim) -> list:
    """Every parameter the policy optimizer actually steps (VQ is in none of them)."""
    return [p for opt in optim.optimizers for g in opt.param_groups for p in g["params"]]


def _optim_lrs(optim) -> list:
    return [float(opt.param_groups[0]["lr"]) for opt in optim.optimizers]


def _trainable_param_names(model: nn.Module) -> list:
    """Names of the parameters that can actually receive a gradient (freeze provenance).

    Saved in every ckpt so `--resume` can detect that the freeze scope changed and refuse,
    instead of silently unfreezing (`--train-adaln-only` forgotten) or silently freezing
    (flag added) parameters that the optimizer would then step. Legacy ckpts lack the key.
    """
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def _rng_state(gen) -> dict:
    """Full RNG snapshot: python / numpy / torch CPU / torch CUDA + the DataLoader
    sampler generator (the actual data-sampling stream)."""
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    if gen is not None:
        st["dataloader"] = gen.get_state()
    return st


def _as_cpu_byte(x):
    """RNG states are CPU uint8 ByteTensors, but `torch.load(..., map_location=device)`
    moves every saved tensor to the device before we ever see it, and
    `torch.set_rng_state` / `Generator.set_state` reject a CUDA tensor with
    'RNG state must be a torch.ByteTensor'. Normalize on the way in."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.uint8)
    if isinstance(x, (list, tuple)):
        return [_as_cpu_byte(i) for i in x]
    return x


def _restore_rng_state(st: dict, gen) -> list:
    """Restore every stream present in `st`; returns the names actually restored."""
    done = []
    if "python" in st:
        random.setstate(st["python"])
        done.append("python")
    if "numpy" in st:
        np.random.set_state(st["numpy"])
        done.append("numpy")
    if "torch" in st:
        torch.set_rng_state(_as_cpu_byte(st["torch"]))
        done.append("torch")
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(_as_cpu_byte(st["torch_cuda"]))
        done.append("torch_cuda")
    if "dataloader" in st and gen is not None:
        gen.set_state(_as_cpu_byte(st["dataloader"]))
        done.append("dataloader")
    return done


def _state_hash(sd: dict, keys=None) -> str:
    """sha256 over (key, dtype, shape, raw bytes) of a state_dict — bitwise identity proof."""
    h = hashlib.sha256()
    for k in (sorted(sd) if keys is None else sorted(keys)):
        v = sd[k]
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def resume_optimizer(ckpt: dict, model: nn.Module, optim, path: str) -> dict:
    """Restore the policy optimizer from a ckpt — or STOP. Never silently rebuild AdamW.

    Order of checks (each failure raises SystemExit, leaving no partially-restored run):
      1. the ckpt must contain optimizer state at all;
      2. if it carries ordered param names (all ckpts saved after this fix do), those
         must match the current optimizer's param-name layout EXACTLY — a mismatch is
         reported with the first differing slot;
      3. otherwise (legacy ckpt) the positional shape check must pass, and the fact
         that verification was shape-only is recorded in the returned meta;
      4. the recorded LR must match the configured LR.
    """
    saved = ckpt.get("optim")
    if saved is None:
        raise SystemExit(
            f"[resume] FATAL: {path} has no optimizer state. Refusing to continue with a "
            "fresh AdamW (that would silently reset all Adam momentum). Re-train or supply "
            "a ckpt that carries 'optim'."
        )
    saved_names = ckpt.get("optim_param_names")
    if saved_names is not None:
        saved_names = {int(k): v for k, v in saved_names.items()}
        cur_names = _optim_param_names(model, optim)
        if saved_names != cur_names:
            detail = "group-count mismatch"
            for i in sorted(set(saved_names) | set(cur_names)):
                a, b = saved_names.get(i, []), cur_names.get(i, [])
                if a != b:
                    for gi, (ga, gb) in enumerate(zip(a, b)):
                        if ga != gb:
                            for si, (sa, sb) in enumerate(zip(ga, gb)):
                                if sa != sb:
                                    detail = (f"opt{i} group{gi} slot{si}: "
                                              f"saved={sa!r} current={sb!r}")
                                    break
                            break
                    break
            raise SystemExit(
                f"[resume] FATAL: optimizer param layout in {path} does not match this "
                f"process ({detail}). AdamW state is restored by POSITION, so continuing "
                "would attach the wrong momentum to the wrong tensors. Refusing to resume "
                "with a fresh AdamW."
            )
        verified = "param_names_exact"
    else:
        if not _saved_optim_aligns(saved, optim):
            raise SystemExit(
                f"[resume] FATAL: {path} has no optim_param_names and its positional "
                "shape check FAILS — the saved AdamW slots cannot be reliably aligned "
                "with this process's parameters. Refusing to resume with a fresh AdamW."
            )
        verified = "positional_shape_only(legacy_ckpt)"
    saved_lr = ckpt.get("lr")
    cur_lr = _optim_lrs(optim)
    if saved_lr is not None:
        saved_lr = [float(x) for x in (saved_lr if isinstance(saved_lr, list) else [saved_lr])]
        if saved_lr != cur_lr:
            raise SystemExit(
                f"[resume] FATAL: ckpt LR {saved_lr} != configured LR {cur_lr}; the LR "
                "schedule/protocol changed since this ckpt was written. Refusing to resume."
            )
    optim.load_state_dict(saved)
    return {"optim_restored": True, "optim_verified_by": verified,
            "lr": cur_lr, "saved_lr": saved_lr}


def build_model(cfg: Config) -> nn.Module:
    """Build the policy. Trunk construction is SHARED; only the action head branches.

    `cfg.action_head` selects the head, and the dispatch is placed at the very END --
    after obs_dim / coord_decoder / adaln / fork_kwargs are computed -- so both arms are
    built from the same values by the same code. "Only the action head differs" is then
    a structural fact rather than a promise. This is deviation #3 (the only substantive
    change to this file); see doc/2026-09-16_P0_interface_audit.md.
    """
    if cfg.policy_mode == "global":
        from patch_policy.global_policy import GlobalActionPolicy
        model = GlobalActionPolicy(cfg)
        return model
    p = cfg.policy
    enc = cfg.enc()
    coord_decoder = None
    obs_dim = enc.output_dim  # GPT wte input: 2048 (direct/repeat) | 512 (coord)
    if cfg.interface_mode() == "coord":
        from patch_policy.coord_decoder import CoordSpatialDecoder
        w = cfg.wemm
        assert w.decoder_out_dim == p.n_embd, (
            f"coord decoder out_dim {w.decoder_out_dim} must equal policy.n_embd {p.n_embd} "
            "(decoder slots are fed to GPT, wte input == n_embd)"
        )
        coord_decoder = CoordSpatialDecoder(
            n_embed_in=enc.output_dim,
            pe_dim=w.decoder_pe_dim,
            hidden=w.decoder_hidden,
            out_dim=w.decoder_out_dim,
            n_patches=cfg.policy_n_patches(),
        )
        obs_dim = w.decoder_out_dim
        print(f"[model] Global-to-Spatial Coord decoder: z[{enc.output_dim}] + FourierPE[{w.decoder_pe_dim}] "
              f"-> {cfg.policy_n_patches()} slots x {w.decoder_out_dim} | params "
              f"{sum(x.numel() for x in coord_decoder.parameters())/1e6:.3f}M | "
              f"PE {coord_decoder.pe.config_dict()}")
    adaln = None
    if cfg.fusion.enabled:
        # WeMM-DINO residual-AdaLN fusion (spec §3-§4): injected between gpt.wte and wpe.
        assert p.n_embd * 2 == cfg.fusion.hidden_dim, (
            f"fusion.hidden_dim {cfg.fusion.hidden_dim} must equal 2*policy.n_embd {2*p.n_embd}"
        )
        if cfg.fusion.alpha_gate:
            # AdaLN-zero variant: X_fused = X + alpha*(gamma*LN(X)+beta), scalar alpha init 0,
            # proj DEFAULT init -> identity from alpha==0 (NOT zeroed proj). Assert the config
            # never pairs the gate with a zeroed proj (branch M=0 AND alpha grad would both be
            # permanently 0 -> dead cond branch = silent baseline re-train).
            from patch_policy.adaln_gate import AdaLNGate

            assert not cfg.fusion.zero_init, (
                "AdaLN-zero: identity comes from alpha=0 — zero_init must be False (proj keeps "
                "DEFAULT init) or the cond branch (M=0) AND the alpha gradient are permanently dead"
            )
            adaln = AdaLNGate(
                cond_dim=cfg.fusion.cond_dim,
                n_embd=p.n_embd,
                hidden_dim=cfg.fusion.hidden_dim,
                zero_init=cfg.fusion.zero_init,
                alpha_init=cfg.fusion.alpha_init,
                ln_local_affine=cfg.fusion.ln_local_affine,
            )
            print(f"[model] WeMM-DINO AdaLN-ZERO fusion (scalar alpha gate from 0): "
                  f"g[{cfg.fusion.cond_dim}] -> LN -> Linear({cfg.fusion.cond_dim},"
                  f"{cfg.fusion.hidden_dim}) -> gamma,beta[{p.n_embd}] "
                  f"| alpha_init={cfg.fusion.alpha_init} (X_fused==X via alpha==0, proj DEFAULT "
                  f"init => M!=0, alpha wakes post-VQ-fit) | X_fused = X + alpha*(gamma*LN(X)+beta)")
        else:
            # WeMM-DINO residual-AdaLN fusion (spec §3-§4): X_fused = X + gamma*LN(X) + beta,
            # Linear zero-init -> identity start.
            from patch_policy.adaln import AdaLN

            adaln = AdaLN(
                cond_dim=cfg.fusion.cond_dim,
                n_embd=p.n_embd,
                hidden_dim=cfg.fusion.hidden_dim,
                zero_init=cfg.fusion.zero_init,
                ln_local_affine=cfg.fusion.ln_local_affine,
            )
            print(f"[model] WeMM-DINO AdaLN fusion: g[{cfg.fusion.cond_dim}] -> LN -> "
                  f"Linear({cfg.fusion.cond_dim},{cfg.fusion.hidden_dim}) -> gamma,beta[{p.n_embd}] "
                  f"| zero_init={cfg.fusion.zero_init} (identity X_fused==X) | "
                  f"X_fused = X + gamma*LN_local(X) + beta")
    # ---- native-parity faithful fork knobs ----
    fork_kwargs = {}
    if p.native_parity:
        enc_ = cfg.enc()
        assert cfg.encoder_name == "dino" and enc_.output_dim == 384 and enc_.n_patches == 256, (
            f"native_parity fork requires DINO-P256 (out=384, P=256); got encoder={cfg.encoder_name} "
            f"out={enc_.output_dim} P={enc_.n_patches}"
        )
        fork_kwargs = dict(
            gpt_output_dim=256,     # lm_head Linear(512->256,bias=False) all-positions -> last-patch
            gelu="tanh",            # GPT-2 new_gelu (native models/.../gpt.py MLP)
            head_final_relu=False,  # torchvision utils.MLP: no activation on final head Linear
        )
        print("[model] native_parity fork: lm_head 512->256 + tanh-GELU + native head MLP "
              "(patch_policy phase9 dino parity)")
    if cfg.action_head == "td":
        from patch_policy.td_head import build_td_policy

        return build_td_policy(cfg, obs_dim=obs_dim, coord_decoder=coord_decoder,
                               adaln=adaln, fork_kwargs=fork_kwargs)
    return BehaviorTransformer(
        obs_dim=obs_dim,
        act_dim=2,
        goal_dim=0,
        views=1,
        obs_window_size=cfg.obs_window_size,
        act_window_size=cfg.action_window_size,
        n_patches=cfg.policy_n_patches(),  # GPT block mask / wpe follow policy P (repeat/coord)
        coord_decoder=coord_decoder,
        adaln=adaln,
        n_layer=p.n_layer,
        n_head=p.n_head,
        n_embd=p.n_embd,
        dropout=p.dropout,
        vqvae_latent_dim=p.vqvae_latent_dim,
        vqvae_n_embed=p.vqvae_n_embed,
        vqvae_groups=p.vqvae_groups,
        vqvae_iters=p.vqvae_iters,
        vqvae_batch_size=p.vqvae_batch_size,
        vqvae_fit_steps=p.vqvae_fit_steps,
        act_scale=p.act_scale,
        offset_loss_multiplier=p.offset_loss_multiplier,
        secondary_code_multiplier=p.secondary_code_multiplier,
        gamma=p.gamma,
        **fork_kwargs,  # native_parity: gpt_output_dim=256 / gelu='tanh' / head_final_relu=False
    )


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, fusion: bool = False
) -> float:
    model.eval()
    losses, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if fusion:
                obs, cond, act = batch
                obs, cond, act = obs.to(device), cond.to(device), act.to(device)
                _, loss, _ = model(obs, None, act, cond)
            else:
                obs, act = batch
                obs, act = obs.to(device), act.to(device)
                _, loss, _ = model(obs, None, act)
            if loss is None:
                continue
            losses += loss.item() * obs.shape[0]
            n += obs.shape[0]
    model.train()
    return (losses / n) if n else float("nan")


def _dump_adaln_rows(out_dir: Path, rows: list) -> None:
    """Persist AdaLN diagnostics (gamma/beta/mod/grad per logged step) as JSONL."""
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "adaln_diag.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _adaln_variance(gb: torch.Tensor) -> dict:
    """gb [B,T,1,D] -> variance decomposition (spec AdaLN Diagnostics #1-4).

      timestep: var over T for FIXED image, mean over (B,D) — "same image, different
                timestep" (temporal conditioning sensitivity)
      image   : var over B for FIXED timestep, mean over (T,D) — "different image,
                same timestep" (sample-wise WeMM conditioning sensitivity)
    Zero for a pure static affine (constant across images/timesteps); >0 => the
    WeMM cond actually shapes gamma/beta per sample/time.
    """
    t = gb[:, :, 0].float()  # [B,T,D]
    return {
        "timestep": float(t.var(dim=1).mean().item()),
        "image": float(t.var(dim=0).mean().item()),
    }


def _capture_adaln_diag(model, epoch: int, step: int, rows: list) -> dict | None:
    """Capture AdaLN gamma/beta/mod/grad stats from the last forward (diagnostics).

    gamma/beta are read from the AdaLN module's _diag (stashed detached, per spec §9);
    grad_adaln is the AdaLN Linear weight grad (after backward, before zero_grad).
    gamma_var / beta_var add the cross-image / cross-timestep variance decomposition
    (spec AdaLN Diagnostics) so the "static affine vs sample-dependent conditioning"
    question is answerable from the per-epoch adaln_diag.jsonl.
    """
    adaln = getattr(getattr(model, "gpt", None), "adaln", None)
    if adaln is None or getattr(adaln, "_diag", None) is None:
        return None
    g = adaln._diag
    gs, bs = _stats(g["gamma"]), _stats(g["beta"])
    mod = g["delta_norm"] / (g["x_norm"] + 1e-6)
    pw = adaln.proj.weight.grad
    row = {
        "step": int(step), "epoch": int(epoch),
        "mod_ratio": mod,
        "delta_norm": g["delta_norm"], "x_norm": g["x_norm"],
        "grad_adaln": float(pw.norm()) if pw is not None else None,
        "gamma": gs, "beta": bs,
        "gamma_var": _adaln_variance(g["gamma"]),
        "beta_var": _adaln_variance(g["beta"]),
    }
    if hasattr(adaln, "alpha"):   # AdaLN-zero scalar gate: track alpha + its grad per step
        row["alpha"] = float(adaln.alpha.detach())
        ag = adaln.alpha.grad
        row["alpha_grad"] = float(ag.abs()) if ag is not None else None
    rows.append(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pusht.yaml")
    ap.add_argument("--smoke", action="store_true", help="100-step pipeline smoke test")
    ap.add_argument("--train-seed", type=int, default=None, dest="train_seed",
                    help="training randomness seed (torch/np/python/model init/dataloader/VQ-BeT)")
    ap.add_argument("--seed", type=int, default=None,
                    help="alias of --train-seed (kept for back-compat)")
    ap.add_argument("--split-seed", type=int, default=None, dest="split_seed",
                    help="dataset split seed (fixed 42; NEVER changes train/val split)")
    ap.add_argument("--cache", default=None,
                    help="override precompute_cache path (e.g. smoke cache)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override cfg.epochs (Go/No-Go short run: 20)")
    ap.add_argument("--max-episodes", type=int, default=None, dest="max_episodes",
                    help="only the first N episodes (tiny overfit / audit / quick runs)")
    ap.add_argument("--save-dir", type=str, default=None,
                    help="override cfg.save_dir (isolate Go/No-Go run)")
    ap.add_argument("--ckpt-epochs", type=str, default="",
                    help="comma list of epoch boundaries to save ckpt at "
                         "('' = final only; 0 = init before training; e.g. 0,5,10,20)")
    ap.add_argument("--resume", default=None,
                    help="resume from a saved ckpt (epoch20 -> 400 continuation). Restores "
                         "model + optimizer + vq_is_fit + step + RNG + dataloader sampler "
                         "state; continues the epoch loop from the ckpt epoch. metrics.json in "
                         "save_dir is merged to keep curves continuous. A ckpt whose optimizer "
                         "state cannot be aligned is a HARD error (never a silent fresh AdamW).")
    # ---- controlled-pair initialization (two arms share backbone + action head + VQ) ----
    ap.add_argument("--init-ckpt", dest="init_ckpt", default=None,
                    help="load SHARED policy-backbone + action-head init weights (everything "
                         "except gpt.adaln.* and vqvae.*) from a --export-shared-init artifact, "
                         "so the two arms start from bit-identical shared weights regardless of "
                         "how much RNG the AdaLN module consumed during its own build")
    ap.add_argument("--vq-ckpt", dest="vq_ckpt", default=None,
                    help="pre-fitted SHARED VQ tokenizer (--export-shared-vq artifact). Held as "
                         "a preset and installed at the SAME trigger point as a normal fit "
                         "(end of epoch 1), so reusing it grants no extra policy-update budget")
    ap.add_argument("--train-adaln-only", dest="train_adaln_only", action="store_true",
                    help="warm-start demo: freeze EVERY parameter except gpt.adaln.* (the 5 "
                         "AdaLN-Zero gate tensors, 2,102,273 params) and train only that branch. "
                         "The optimizer's param set is deliberately left UNCHANGED — AdamW only "
                         "touches params whose .grad is not None, so the gate is the only thing "
                         "that steps and the saved optim_param_names layout stays resume-compatible. "
                         "Requires --vq-ckpt (vqvae.* is frozen too).")
    ap.add_argument("--export-shared-init", dest="export_shared_init", default=None,
                    help="build the model once and write the shared (non-AdaLN, non-VQ) init "
                         "state_dict + hash manifest to this path, then exit")
    ap.add_argument("--export-shared-vq", dest="export_shared_vq", default=None,
                    help="collect the TRAIN-split action corpus and run the standard VQ fit, "
                         "write {vqvae, meta} to this path, then exit (validation windows are "
                         "never collected — evaluate() runs under model.eval())")
    ap.add_argument("--export-meta", dest="export_meta", default=None,
                    help="optional JSON path for export metadata (hashes / counts / config)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    # 种子解耦：split_seed 固定切分，train_seed 只控制训练随机性。
    # split_episodes 用独立 Generator().manual_seed(split_seed)，不受全局 torch seed 影响。
    cfg.split_seed = args.split_seed if args.split_seed is not None else cfg.split_seed
    cfg.seed = args.train_seed if args.train_seed is not None else (
        args.seed if args.seed is not None else cfg.seed
    )
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    ckpt_epochs = {int(x) for x in args.ckpt_epochs.split(",") if x.strip()}
    seed_everything(cfg.seed)  # train randomness only
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- shared policy-backbone/action-head init export -----------------------------
    # Placed BEFORE data loading: it needs only cfg + build_model, so exporting costs
    # seconds instead of an in-memory DINO precompute. Nothing consumes RNG between
    # seed_everything and this build, so the exported weights are exactly what the
    # training path would have produced for this config/seed.
    if args.export_shared_init:
        ref = build_model(cfg)
        sd = ref.state_dict()
        shared_keys = [k for k in sd
                       if not k.startswith("gpt.adaln.") and not k.startswith("vqvae.")]
        blob = {
            "shared": {k: sd[k] for k in shared_keys},
            "meta": {
                "config": args.config, "train_seed": cfg.seed, "split_seed": cfg.split_seed,
                "excluded_prefixes": ["gpt.adaln.", "vqvae."],
                "alpha_gate": cfg.fusion.alpha_gate, "zero_init": cfg.fusion.zero_init,
                "n_keys": len(shared_keys),
                "shared_sha256": _state_hash(sd, shared_keys),
            },
        }
        _atomic_save(blob, Path(args.export_shared_init))
        if args.export_meta:
            _atomic_write_json(blob["meta"], Path(args.export_meta))
        print(f"[shared-init] {len(shared_keys)} tensors of {len(sd)} "
              f"(sha256 {blob['meta']['shared_sha256'][:16]}...) -> {args.export_shared_init}")
        return

    # ---- data --------------------------------------------------------------
    fusion = cfg.fusion.enabled
    # DINO-P256 (native P256 patch tokens) has no disk cache (disk 100% full; base
    # baseline also used in-memory DINO) -> the non-fusion DINO-P256 baseline must
    # take the same in-memory path (minus the WeMM cond).
    dino_inmem = (cfg.encoder_name == "dino" and cfg.enc().postprocess is None)
    n_target = args.max_episodes  # None = all; else slice episodes (audit / quick runs)
    if args.smoke:
        n_target = min(n_target or 10**9, 6)
    if fusion:
        # WeMM-DINO Fusion: primary = DINO-P256 dense tokens (in-memory, no disk
        # cache — disk 100% full; the base DINO-P256 baseline used in-memory DINO
        # too); conditioning = WeMM global embedding (existing disk cache). Both
        # built over the SAME obses/episode_XXX.pth frames -> per-frame aligned
        # by construction (FusionWindowDataset reuses WindowDataset's (ei,start,end)).
        assert cfg.encoder_name == "dino" and cfg.enc().postprocess is None, (
            "fusion requires encoder_name=dino, dino.postprocess=None (native P256 dense tokens)"
        )
        assert cfg.enc().n_patches == 256 and cfg.enc().output_dim == 384, (
            f"fusion expects DINO-P256 (n_patches=256, output_dim=384), got "
            f"n_patches={cfg.enc().n_patches}, output_dim={cfg.enc().output_dim}"
        )
        # cond_dim must equal the cond-encoder's AUDITED output dim AND be positive.
        # qwen3vl starts at output_dim=cond_dim=0 (sentinel) -> this blocks training
        # until Phase-2 fills the measured dim, preventing a mismatched/un-audited
        # cond stream (or a phantom 0-dim AdaLN Linear).
        assert cfg.cond_config().output_dim > 0 and cfg.fusion.cond_dim > 0, (
            f"{cfg.fusion.cond_encoder}.output_dim={cfg.cond_config().output_dim}, "
            f"fusion.cond_dim={cfg.fusion.cond_dim} — cond encoder dim must be "
            f"audit-filled (>0) before fusion training"
        )
        assert cfg.fusion.cond_dim == cfg.cond_config().output_dim, (
            f"fusion.cond_dim {cfg.fusion.cond_dim} != {cfg.fusion.cond_encoder}."
            f"output_dim {cfg.cond_config().output_dim} — cond_dim must equal the "
            f"audited cond-encoder output dim"
        )
        _, cond_episodes = load_embedding_cache(
            cfg.fusion.cond_cache,
            check_meta={
                "model_name": cfg.cond_model_name(),  # wemm | qwen3vl (cond_config)
                "n_patches": 1,
                "output_dim": cfg.fusion.cond_dim,
            },
        )
        n_episodes = len(cond_episodes)
        if n_target is not None:
            n_episodes = min(n_episodes, n_target)
            cond_episodes = cond_episodes[:n_episodes]
        if args.smoke:
            cfg.batch_size = 2
            cfg.policy.vqvae_fit_steps = 2
            cfg.save_ckpt = False
        from patch_policy.encoder import build_encoder

        dino_enc, _ = build_encoder(cfg)
        dino_enc.to(device)
        episodes = precompute_dino_in_memory(dino_enc, cfg.dataset_root, n_episodes, device)
        meta = {"encoder": "dino", "model_name": cfg.enc().name,
                "n_patches": 256, "output_dim": 384}
        print(f"[data] fusion: DINO-P256 in-memory ({sum(e.shape[0] for e in episodes)} frames) "
              f"+ WeMM cond {n_episodes} episodes (cond_cache={cfg.fusion.cond_cache})")
    elif dino_inmem:
        # DINO-P256 baseline: same in-memory DINO-P256 precompute as the fusion path,
        # NO WeMM cond (WindowDataset / cond_seq=None). Data pipeline parity with the
        # fusion run (identical DINO tokens, identical (ei,start,end) windows).
        assert cfg.enc().n_patches == 256 and cfg.enc().output_dim == 384, (
            f"in-memory DINO-P256 requires n_patches=256 output_dim=384, got "
            f"{cfg.enc().n_patches}, {cfg.enc().output_dim}"
        )
        from patch_policy.dataset import load_seq_lengths

        n_episodes = len(load_seq_lengths(cfg.dataset_root))
        if n_target is not None:
            n_episodes = min(n_episodes, n_target)
        if args.smoke:
            cfg.batch_size = 2
            cfg.policy.vqvae_fit_steps = 2
            cfg.save_ckpt = False
        from patch_policy.encoder import build_encoder

        dino_enc, _ = build_encoder(cfg)
        dino_enc.to(device)
        episodes = precompute_dino_in_memory(dino_enc, cfg.dataset_root, n_episodes, device)
        meta = {"encoder": "dino", "model_name": cfg.enc().name,
                "n_patches": 256, "output_dim": 384}
        print(f"[data] DINO-P256 in-memory baseline: {sum(e.shape[0] for e in episodes)} frames, "
              f"{n_episodes} episodes (no disk cache)")
    else:
        meta, episodes = load_embedding_cache(
            args.cache or cfg.precompute_cache,
            check_meta={
                "model_name": cfg.enc_model_name(),
                "n_patches": cfg.enc().n_patches,
                "output_dim": cfg.enc().output_dim,
            },
        )
        # fp16 cache (WeMM-Local-P16) is storage-only; policy wte is fp32 Linear, so
        # upcast to float32 in memory. fp16->fp32 is exact, no precision loss.
        if meta.get("cache_dtype") == "fp16":
            episodes = [e.float() for e in episodes]
            print(f"[data] cache fp16 -> float32 in memory (storage {meta.get('cache_dtype')})")
        n_episodes = len(episodes)
        if n_target is not None:
            n_episodes = min(n_episodes, n_target)
            episodes = episodes[:n_episodes]
        if args.smoke:
            cfg.batch_size = 2
            cfg.policy.vqvae_fit_steps = 2
            cfg.save_ckpt = False
    if args.smoke:
        print(f"[smoke] using {n_episodes} episodes, batch_size={cfg.batch_size}")

    actions = load_actions(cfg.dataset_root, n_episodes, relative=cfg.use_relative_actions)
    train_idx, val_idx = split_episodes(n_episodes, cfg.train_fraction, seed=cfg.split_seed)
    print(f"[data] {n_episodes} episodes -> train {len(train_idx)}, val {len(val_idx)} "
          f"(split_seed={cfg.split_seed} train_seed={cfg.seed})")

    if fusion:
        train_ds = FusionWindowDataset(
            episodes, cond_episodes, actions, cfg.obs_window_size, cfg.action_window_size, train_idx
        )
        val_ds = FusionWindowDataset(
            episodes, cond_episodes, actions, cfg.obs_window_size, cfg.action_window_size, val_idx
        )
    else:
        train_ds = WindowDataset(
            episodes, actions, cfg.obs_window_size, cfg.action_window_size, train_idx
        )
        val_ds = WindowDataset(
            episodes, actions, cfg.obs_window_size, cfg.action_window_size, val_idx
        )
    # Explicit sampler generator (resume completeness, bug 3): with shuffle=True and no
    # generator the RandomSampler draws its per-epoch seed from the GLOBAL torch RNG, so
    # the shuffle stream could not be saved/restored as such. num_workers=0 -> no worker
    # seeding is involved and this single generator fully determines the order.
    train_dl_gen = torch.Generator()
    train_dl_gen.manual_seed(cfg.seed)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        generator=train_dl_gen,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"[data] train windows {len(train_ds)}, val windows {len(val_ds)}")

    # ---- model -------------------------------------------------------------
    model = build_model(cfg).to(device)
    optim = model.configure_optimizers(cfg.weight_decay, cfg.lr, tuple(cfg.betas))
    # grad_norm / layer_grad_norm are computed over the POLICY optimizer's parameters
    # ONLY. The VQ tokenizer is in no optimizer, so any grad it carries is not an
    # applied update and must not enter the statistic (bug 2). The fitted VQ is also
    # grad-free (vqbet._freeze_vqvae), so this scoping is now redundant-but-explicit.
    opt_params = _optim_params(optim)
    print(f"[model] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"policy-optimized params {len(opt_params)} tensors "
          f"({sum(p.numel() for p in opt_params)/1e6:.2f}M)")

    # ---- shared-init / shared-VQ exports (controlled pair; exit before training) ----
    if args.export_shared_vq:
        # Corpus = the action chunks the training forward would collect during epoch 1:
        # every TRAIN window (vqbet._unpack_actions with a batch of 1). Validation windows
        # are never collected — evaluate() runs under model.eval(), where forward()'s
        # `self.training` collection guard is False.
        n_chunks = 0
        for i in range(len(train_ds)):
            act = train_ds[i][-1]
            chunk = model._unpack_actions(act.unsqueeze(0)).to(device)  # VQ lives on `device`
            model._collected_actions.append(chunk)
            n_chunks += chunk.shape[1]
        print(f"[shared-vq] {len(train_ds)} TRAIN windows -> {n_chunks} action chunks "
              f"(val windows: 0)")
        seed_everything(cfg.seed)   # deterministic fit for the shared artifact
        model._maybe_fit_vq(force=True)
        assert model.vqvae_is_fit
        blob = {
            "vqvae": model.vqvae.state_dict(),
            "meta": {
                "config": args.config, "fit_seed": cfg.seed, "split_seed": cfg.split_seed,
                "n_train_windows": len(train_ds), "n_action_chunks": n_chunks,
                "vqvae_iters": cfg.policy.vqvae_iters,
                "vqvae_batch_size": cfg.policy.vqvae_batch_size,
                "vqvae_n_embed": cfg.policy.vqvae_n_embed,
                "vqvae_groups": cfg.policy.vqvae_groups,
                "train_episodes": len(train_idx), "val_episodes": len(val_idx),
                "source": "train split only (val never collected)",
                "vqvae_sha256": _state_hash(model.vqvae.state_dict()),
            },
        }
        _atomic_save(blob, Path(args.export_shared_vq))
        if args.export_meta:
            _atomic_write_json(blob["meta"], Path(args.export_meta))
        print(f"[shared-vq] frozen tokenizer (sha256 "
              f"{blob['meta']['vqvae_sha256'][:16]}...) -> {args.export_shared_vq}")
        return

    # ---- shared policy-backbone/action-head init -----------------------------------
    init_meta = None
    if args.init_ckpt:
        blob = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        shared = blob["shared"] if "shared" in blob else blob
        cur = model.state_dict()
        unexpected = sorted(k for k in shared if k not in cur)
        shape_bad = sorted(k for k in shared if k in cur and tuple(cur[k].shape) != tuple(shared[k].shape))
        # only gpt.adaln.* / vqvae.* may be absent from the shared artifact
        missing_bad = sorted(k for k in cur if k not in shared
                             and not k.startswith("gpt.adaln.") and not k.startswith("vqvae."))
        if unexpected or shape_bad or missing_bad:
            raise SystemExit(
                f"[init-ckpt] FATAL: shared init {args.init_ckpt} does not fit this model "
                f"(unexpected={unexpected[:5]}, shape_mismatch={shape_bad[:5]}, "
                f"missing_non_adaln={missing_bad[:5]})"
            )
        model.load_state_dict(shared, strict=False)
        init_meta = {"path": args.init_ckpt, "n_loaded": len(shared),
                     "sha256_after_load": _state_hash(model.state_dict(), sorted(shared)),
                     "sha256_artifact": blob.get("meta", {}).get("shared_sha256")}
        ok = None
        if init_meta["sha256_artifact"]:
            ok = init_meta["sha256_after_load"] == init_meta["sha256_artifact"]
        print(f"[init-ckpt] loaded {len(shared)} shared tensors from {args.init_ckpt} "
              f"(AdaLN + VQ left at their own init) | bitwise-identical to artifact: {ok}")
        if ok is False:
            raise SystemExit("[init-ckpt] FATAL: loaded shared weights differ from the artifact hash")
    if args.vq_ckpt:
        blob = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        model.load_preset_vqvae(blob["vqvae"])
        meta = blob.get("meta", {})
        print(f"[vq-ckpt] shared frozen tokenizer registered from {args.vq_ckpt} "
              f"(sha256 {meta.get('vqvae_sha256', 'n/a')[:16]}...) — installed at the SAME "
              f"trigger point as a normal fit (end of epoch 1), so no extra update budget")

    # ---- warm-start freeze: train ONLY the AdaLN-Zero branch -------------------------
    # The trunk (DINO-P256 patch policy + action heads + VQ) comes from an already-trained
    # base ckpt and is held fixed; only gpt.adaln.* learns. Because alpha == 0 makes the
    # forward `x + 0.0*m` == x bitwise, step 0 IS the base policy exactly — this is a warm
    # start, not the cold start the two prior AdaLN arms used.
    #
    # The optimizer's param set is deliberately NOT filtered. AdamW only ever touches a param
    # whose `.grad is not None` (the same list `_init_group` builds, and the only place lazy
    # Adam state is created), so a frozen param takes no step, no Adam state and no decoupled
    # weight decay. Leaving the groups intact keeps `_optim_param_names` writing the standard
    # layout, so ckpts stay resume-compatible and no new failure mode is introduced.
    if args.train_adaln_only:
        assert fusion, "--train-adaln-only requires fusion.enabled (there is no gpt.adaln to train)"
        assert getattr(model.gpt, "adaln", None) is not None, \
            "--train-adaln-only requires the alpha-gate AdaLN branch"
        assert args.vq_ckpt, (
            "--train-adaln-only freezes vqvae.* as well; without --vq-ckpt the end-of-epoch-1 "
            "trigger would run a fresh VQ fit whose parameters are all frozen, silently "
            "installing an UNFITTED tokenizer"
        )
        for n, p in model.named_parameters():
            if not n.startswith("gpt.adaln."):
                p.requires_grad_(False)
                # AdamW skips only `grad is None`; a stale grad would re-enable the decoupled
                # weight decay on a frozen parameter.
                p.grad = None
        trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expect = sorted(f"gpt.adaln.{x}" for x in
                        ("alpha", "ln_g.weight", "ln_g.bias", "proj.weight", "proj.bias"))
        assert trainable == expect, f"unexpected trainable set: {trainable}"
        assert n_trainable == 2_102_273, f"trainable param count {n_trainable} != 2102273"
        n_frozen = sum(1 for _ in model.parameters()) - len(trainable)
        print(f"[freeze] train-adaln-only: {len(trainable)} trainable tensors "
              f"({n_trainable:,} params) {trainable}")
        print(f"[freeze] {n_frozen} tensors frozen (requires_grad=False, grad=None); optimizer "
              f"param set UNCHANGED ({len(opt_params)} tensors) -> only the gate can step")

    # ---- graceful-stop handler (infrastructure ONLY — does not change training) ----
    # On SIGTERM/SIGINT: persist the latest model+optimizer+metrics as a No-Go
    # checkpoint (scientific termination, not a training failure). Registered
    # after model/optim are built; holder is updated every step.
    stop = {"step": 0, "epoch": 0, "loss": None, "done": False}

    def _stop_handler(signum, _frame):
        if stop["done"]:
            return
        stop["done"] = True
        e, s = stop["epoch"], stop["step"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"ckpt_abort_epoch{e}_step{s}.pt"
        _atomic_save(
            {"model": model.state_dict(), "config": cfg.to_dict(),
             "step": s, "epoch": e, "optim": optim.state_dict(),
             "vqvae_is_fit": model.vqvae_is_fit,
             # resume completeness (bug 3): an abort ckpt must be as resumable as a
             # scheduled one — optimizer param map, LR, VQ mode, full RNG state.
             "optim_param_names": _optim_param_names(model, optim),
             "lr": _optim_lrs(optim),
             "vqvae_mode": "eval" if not model.vqvae.training else "train",
             "rng": _rng_state(train_dl_gen),
             "resume": dict(resume_meta),
             "trainable_param_names": _trainable_param_names(model),
             "abort_signal": signum, "last_loss": stop["loss"]}, path,
        )
        _atomic_write_json(
            {"config": dict(run_cfg),
             "ckpt_epochs": sorted(ckpt_epochs), "metrics": metrics,
             "abort": {"signal": signum, "epoch": e, "step": s,
                       "last_loss": stop["loss"]}}, out_dir / "metrics.json",
        )
        _atomic_write_json(
            {"termination": "scientific_no_go", "signal": signum,
             "epoch": e, "step": s, "last_loss": stop["loss"],
             "reason": "user-requested graceful stop; NOT a training failure"}, out_dir / "STOPPED.json",
        )
        print(f"[signal {signum}] saved abort ckpt {path} + STOPPED.json (No-Go termination)")
        os._exit(0)

    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGINT, _stop_handler)

    # ---- training ----------------------------------------------------------
    smoke_iters = 100
    step = 0
    log_first = True
    start = time.time()
    # ---- Go/No-Go：per-epoch metrics + ckpt-at-boundaries ----
    # 3-seed 并行：ckpt/metrics 都写在 seed{train_seed}/ 子目录，多卡互不覆盖。
    # ckpt_epochs 语义：0 = 训练前 init；N = 完成 N 个 epoch 后保存（如 0,5,10,20）。
    out_dir = Path(cfg.save_dir) / f"seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)   # eval/ckpt 都写这里；ckpt_epochs 无 0 时首次 eval 前必须已存在
    metrics = {
        "train_loss": {}, "classification_loss": {}, "offset_loss": {},
        "grad_norm": {}, "layer_grad_norm": {}, "val_loss": {},
        # actual policy-update step accounting: `steps` per epoch and the cumulative
        # count of steps that applied a real policy update (VQ already fitted). Epoch 1
        # is zeroed by protocol in every run, so effective < steps by exactly the
        # epoch-1 step count — the shared-VQ preset must not change this.
        "steps_per_epoch": {}, "policy_steps_effective": {},
    }
    # Actual run configuration (LR / wd / betas / seed / epochs / shared artifacts) as
    # written to disk. No scheduler, no warmup, no auto LR decay — these are the exact
    # values every parameter update used.
    run_cfg = {
        "epochs": cfg.epochs, "seed": cfg.seed, "split_seed": cfg.split_seed,
        "batch_size": cfg.batch_size, "policy_P": cfg.policy_n_patches(),
        "encoder": cfg.encoder_name, "policy_mode": cfg.policy_mode,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, "betas": list(cfg.betas),
        "eval_freq_epochs": cfg.eval_freq_epochs,
        "init_ckpt": args.init_ckpt, "vq_ckpt": args.vq_ckpt, "resume": args.resume,
    }
    # 代表层 grad norm（与全局 RMS 同口径：per-step 平方和 -> epoch 末 RMS）。
    # 用户要求：wte / wpe / block0 attention / middle block / last block / bin head / offset head。
    named_params = list(model.named_parameters())
    if cfg.policy_mode == "global":
        layer_groups = {
            "proj": "proj.",
            "time_pe": "time_pe.",
            "block0": "backbone.h.0.",
            "mid_block": "backbone.h.1.",
            "last_block": f"backbone.h.{cfg.global_policy.n_layer - 1}.",
            "bin_head": "bin_head.",
            "offset_head": "offset_head.",
        }
    else:
        layer_groups = {
            "wte": "gpt.wte.",
            "wpe": "gpt.wpe.",
            "block0": "gpt.h.0.",
            "mid_block": "gpt.h.4.",
            "last_block": "gpt.h.7.",
            "bin_head": "bin_head.",
            "offset_head": "offset_head.",
        }
    if getattr(model, "coord_decoder", None) is not None:
        layer_groups["coord_decoder"] = "coord_decoder."
    if fusion and getattr(model.gpt, "adaln", None) is not None:
        layer_groups["adaln"] = "gpt.adaln."
    layer_params = {k: [p for n, p in named_params if n.startswith(prefix)]
                    for k, prefix in layer_groups.items()}
    adaln_rows: list = []  # per-step AdaLN diagnostics (gamma/beta/mod/grad), dumped to adaln_diag.jsonl

    # Resume audit trail: what the previous ckpt was, whether its optimizer/RNG were
    # actually restored, and the VQ mode at the moment training resumes. Written into
    # every ckpt and into metrics.json.
    resume_meta = {"resumed": False, "from": None, "epoch": None, "step": None,
                   "vqvae_is_fit": None, "vqvae_mode": None}

    def save_ckpt(epoch_label: int) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"ckpt_epoch{epoch_label}_step{step}.pt"
        _atomic_save(
            {"model": model.state_dict(), "config": cfg.to_dict(),
             "step": step, "epoch": epoch_label,
             "optim": optim.state_dict(),           # 续训必须（Adam 动量）
             "vqvae_is_fit": model.vqvae_is_fit,    # VQ 已拟合状态
             # ---- resume completeness (bug 3) ----
             "optim_param_names": _optim_param_names(model, optim),  # positional identity
             "lr": _optim_lrs(optim),               # actual LR in the optimizers
             "vqvae_mode": "eval" if not model.vqvae.training else "train",
             "rng": _rng_state(train_dl_gen),       # python/numpy/torch cpu+cuda/sampler
             "policy_steps_effective": eff_steps,   # steps that applied a real update
             # freeze provenance: which params were trainable when this ckpt was written.
             # train-adaln-only ckpts carry exactly the 5 gpt.adaln.* names.
             "trainable_param_names": _trainable_param_names(model),
             "resume": dict(resume_meta)},
            path,
        )
        print(f"[ckpt] epoch {epoch_label} step {step} -> {path}")
        return path

    def eval_and_record(epoch_label: int) -> None:
        v = evaluate(model, val_dl, device, fusion=cfg.fusion.enabled)
        metrics["val_loss"][epoch_label] = v
        print(f"  [epoch {epoch_label}/{cfg.epochs}] val_loss={v:.4f} ({time.time()-start:.0f}s)")

    # ---- resume（epoch20 -> 400 续训，不重启，不改变任何训练语义）----
    eff_steps = 0   # steps that applied a real policy update (VQ already fitted)
    resume_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        # 兼容：2026-09-01 前 ckpt 用旧类名存键 `gpt.film.*`；改名 film->adaln 后
        # 键名为 `gpt.adaln.*`。旧 ckpt 加载时做键名映射，不动架构/不重训。
        sd = ck["model"]
        # 兼容旧 ckpt 键名 gpt.film.* -> gpt.adaln.*
        if any(k.startswith("gpt.film.") for k in sd):
            sd = {
                ("gpt.adaln." + k[len("gpt.film."):] if k.startswith("gpt.film.") else k): v
                for k, v in sd.items()
            }
        model.load_state_dict(sd)
        # Freeze provenance: a ckpt written by --train-adaln-only must not be resumed without
        # the flag (that would silently unfreeze ~34M params), and a full-scope ckpt must not
        # be resumed with it (that would silently freeze them). The optimizer restores by
        # POSITION, so a silent scope change would attach Adam momentum to the wrong tensors.
        saved_trainable = ck.get("trainable_param_names")
        if saved_trainable is not None:
            cur_trainable = _trainable_param_names(model)
            if sorted(saved_trainable) != cur_trainable:
                raise SystemExit(
                    f"[resume] FATAL: {args.resume} was written with {len(saved_trainable)} "
                    f"trainable tensors but this process has {len(cur_trainable)}. Re-run with "
                    "the matching --train-adaln-only setting; resuming across a freeze change "
                    "would silently change which parameters the optimizer updates."
                )
            print(f"[resume] freeze scope matches: {len(cur_trainable)} trainable tensors")
        else:
            print("[resume] WARNING: ckpt predates trainable_param_names -> freeze scope "
                  "cannot be verified (it is whatever this process was launched with)")
        step = int(ck.get("step", 0))
        resume_epoch = int(ck.get("epoch", 0))
        # 协议：VQ 在 epoch1 末 force fit 后冻结。epoch>=1 的 ckpt 一定已拟合。
        # 旧 ckpt 未存该 flag -> 用 epoch>=1 推断；新 ckpt 以存值为准。
        # MUST go through set_vqvae_fit (bug 1): a bare flag assignment leaves the
        # restored VQ in the freshly built model's TRAIN mode, so every forward before
        # the first validation ran get_code() against a live EMA codebook and drifted
        # the codebook/EMA buffers (A1 ep100->105 changed, 105->110 not).
        fitting = bool(ck.get("vqvae_is_fit", ck["epoch"] >= 1))
        if not fitting and resume_epoch >= 1:
            raise SystemExit(
                f"[resume] FATAL: {args.resume} says vqvae_is_fit=False at epoch "
                f"{resume_epoch}; the protocol fits the tokenizer at the end of epoch 1, so "
                "this ckpt is inconsistent. Refusing to resume."
            )
        model.set_vqvae_fit(fitting)
        print(f"[resume] VQ restored: is_fit={model.vqvae_is_fit} "
              f"mode={'eval' if not model.vqvae.training else 'train'} "
              f"requires_grad={any(p.requires_grad for p in model.vqvae.parameters())}")
        # Optimizer: restored, or HARD STOP. No silent fresh AdamW (bug 3).
        optim_info = resume_optimizer(ck, model, optim, args.resume)
        print(f"[resume] optimizer state restored (verified by {optim_info['optim_verified_by']})"
              f" | lr={optim_info['lr']}")
        # RNG: python / numpy / torch cpu+cuda / dataloader sampler generator.
        rng_restored = []
        if ck.get("rng") is not None:
            rng_restored = _restore_rng_state(ck["rng"], train_dl_gen)
            print(f"[resume] RNG restored: {rng_restored}")
        else:
            print("[resume] WARNING: legacy ckpt without RNG state -> RNG starts from the "
                  "launch seed (one-time stream discontinuity; no ckpt from this build "
                  "lacks RNG state)")
        mpath = out_dir / "metrics.json"
        if mpath.exists():
            prev = json.load(open(mpath))
            for k, d in prev.get("metrics", {}).items():
                if k in metrics and isinstance(d, dict) and d:
                    metrics[k] = {int(kk): vv for kk, vv in d.items()}
            print(f"[resume] merged metrics.json history "
                  f"({len(metrics['train_loss'])} epochs of train_loss)")
        resume_meta.update({
            "resumed": True, "from": args.resume, "epoch": resume_epoch, "step": step,
            "vqvae_is_fit": model.vqvae_is_fit,
            "vqvae_mode": "eval" if not model.vqvae.training else "train",
            "optim_restored": True, "optim_verified_by": optim_info["optim_verified_by"],
            "lr": optim_info["lr"], "rng_restored": rng_restored,
        })
        # effective policy steps: taken from the ckpt when present, else derived from the
        # protocol invariant (exactly the epoch-1 step count is zeroed).
        eff_steps = int(ck.get("policy_steps_effective", max(0, step - len(train_dl))))
        print(f"[resume] effective policy-update steps so far: {eff_steps} / {step}")
        metrics["resume"] = dict(resume_meta)
        print(f"[resume] {args.resume}: epoch {resume_epoch} step {step} "
              f"vq_fit={model.vqvae_is_fit} -> continue e={resume_epoch+1}..{cfg.epochs}")

    if not args.smoke and not args.resume:
        if 0 in ckpt_epochs:      # init ckpt = 训练前状态（Phase 2 的 epoch-0 冗余基线）
            save_ckpt(0)
        eval_and_record(0)        # init val baseline（VQ 未拟合，loss≈0，仅参考）

    for epoch in range(1 if args.smoke else resume_epoch, cfg.epochs):
        ep_tl = ep_cls = ep_off = 0.0
        ep_gn2 = torch.zeros((), device=device)
        layer_gn2 = {k: torch.zeros((), device=device) for k in layer_groups}
        ep_n = 0
        for batch in train_dl:
            if fusion:
                obs, cond, act = batch
                obs, cond, act = obs.to(device), cond.to(device), act.to(device)
            else:
                obs, act = batch
                obs, act = obs.to(device), act.to(device)
            # WeMM-Repeat-P256: cache is P=1; expand to policy P=repeat_patches here
            # (pure copy, per-batch — never 256× on disk). Coord decoder: NO expansion —
            # the model's CoordSpatialDecoder turns the P=1 token into spatial slots in
            # forward. Must match eval_rollout exactly.
            if cfg.interface_mode() == "repeat" and cfg.policy_n_patches() != obs.shape[2]:
                obs = expand_repeat_patches(obs, cfg.policy_n_patches())

            if log_first:  # build.md §18: print tensor shapes once
                mode = cfg.interface_mode()
                print(f"    cached {cfg.encoder_name} embedding obs {tuple(obs.shape)} [B,T,P,D] | act {tuple(act.shape)} | interface={mode}")
                # P-generic: 原硬编码 P==1 guard 已放宽为 P==policy_n_patches（P16 用 P=16；Repeat-P256 用 P=256）
                assert obs.shape[-1] == cfg.enc().output_dim, f"D!={cfg.enc().output_dim}: {obs.shape}"
                assert not obs.requires_grad, "cached embedding must not require grad (encoder frozen by construction)"
                if cfg.policy_mode == "global":
                    # Global Policy: P=1 cache -> one temporal state token per frame.
                    assert obs.shape[2] == 1, f"global policy expects the P=1 cache token, got P={obs.shape[2]}"
                    d_pol = model.n_embd
                    proj = model.proj(obs)   # [B,T,1,512]
                    print(f"    global projected state {tuple(proj.shape)} (expect [B,T,1,{d_pol}])")
                    assert proj.shape == (*obs.shape[:3], d_pol), "projection shape != [B,T,1,D_policy]"
                    assert torch.isfinite(proj).all(), "projection NaN/Inf"
                    # temporal PE is the ONLY position signal: state variance across T must be > 0
                    s = model.ln_in(proj.squeeze(-2))
                    pos = torch.arange(obs.shape[1], device=obs.device)
                    s_pe = s + model.time_pe(pos)
                    print(f"    [global] post-temporal-PE state variance across T = "
                          f"{s_pe.var(dim=-2).mean().item():.5f} (expect > 0)")
                elif mode == "coord":
                    # Coord decoder: the P=1 cache token becomes P spatial slots in-model.
                    assert obs.shape[2] == 1, f"coord decoder expects the P=1 cache token, got P={obs.shape[2]}"
                    d_pol = model.gpt.wte.out_features
                    slots = model.coord_decoder(obs)   # [B,T,P,512]
                    print(f"    coord-decoded spatial slots {tuple(slots.shape)} (expect [B,T,{cfg.policy_n_patches()},{d_pol}])")
                    assert slots.shape == (*obs.shape[:2], cfg.policy_n_patches(), d_pol), \
                        "slot shape != [B,T,P,D_policy]"
                    assert torch.isfinite(slots).all(), "decoder output NaN/Inf"
                    # ---- image-dependent interaction (spec §5): Delta_i(z) must vary with image ----
                    # 按 spec §5：交互退化为纯 additive bias 时"报告、不修改训练目标"。
                    # 因此这里是测量 + 持久化到 metrics（audit 可与 ep20 对比），不是硬断言——
                    # 硬 abort 会阻止 20ep Go/No-Go 本体的运行（该运行本身就是"训练能否驱动交互"的检验）。
                    za = obs[0, 0].unsqueeze(0)   # [1,1,D] two different frames
                    zb = obs[1, 0].unsqueeze(0)
                    ta = model.coord_decoder(za)[0, 0]   # [P,D]
                    tb = model.coord_decoder(zb)[0, 0]
                    da = ta - ta[0:1]                     # Delta_i(z_a) (relative to slot 0)
                    db = tb - tb[0:1]
                    cross = (da - db).norm(dim=-1).max().item()
                    mean_cross = (da - db).norm(dim=-1).mean().item()
                    metrics.setdefault("coord_interaction", {}).setdefault("init", cross)
                    if cross <= 1e-4:
                        print(f"    [coord][WARNING] max|Delta_i(z_a)-Delta_i(z_b)| = {cross:.2e} "
                              f"(<=1e-4: init 阶段近 additive positional bias —— 记录并继续，"
                              f"ep20 audit 将对比该值是否随训练增长)")
                    else:
                        print(f"    [coord] max|Delta_i(z_a)-Delta_i(z_b)| = {cross:.4f} / "
                              f"mean {mean_cross:.4f} (init 已图像相关)")
                else:
                    assert obs.shape[2] == cfg.policy_n_patches(), f"P mismatch: {obs.shape} vs policy_n_patches={cfg.policy_n_patches()}"
                    d_pol = model.gpt.wte.out_features
                    proj = model.gpt.wte(obs)
                    print(f"    projected visual token {tuple(proj.shape)} (expect [B,T,{cfg.policy_n_patches()},{d_pol}])")
                    assert proj.shape == (*obs.shape[:3], d_pol), "projection shape != [B,T,P,D_policy]"
                    assert torch.isfinite(proj).all(), "projection NaN/Inf"
                    # ---- WeMM-Repeat identity check (Phase 1): pre-position copies identical,
                    #      post-position tokens differ ONLY by wpe (position-conditioned). ----
                    r = cfg.policy_n_patches()
                    if mode == "repeat" and r > 1:
                        z = obs[:, :, 0].unsqueeze(2)  # [B,T,1,D] the single global token
                        md_pre = (obs - z).abs().max().item()
                        pos = torch.arange(0, obs.shape[1] * r, device=obs.device)
                        pos_tokens = proj.flatten(1, 2) + model.gpt.wpe(pos)  # [B,T*P,D] + [T*P,D]
                        var_post = pos_tokens.var(dim=-2).mean().item()
                        print(f"    [repeat] P={r}: max abs diff among copies pre-position = {md_pre:.3e} "
                              f"(expect 0) | token variance AFTER wpe = {var_post:.5f} (expect > 0)")
                        assert md_pre == 0, f"repeat copies not identical pre-position: {md_pre}"
                        assert var_post > 0, "post-position token variance must be > 0"
                print(f"    {cfg.encoder_name} frozen: by construction (precomputed cache; encoder never in train graph)")
                log_first = False

            pred, loss, loss_dict = model(obs, None, act, cond if fusion else None)
            assert loss is not None
            assert torch.isfinite(loss).all(), f"NaN/Inf loss at step {step}"
            assert torch.isfinite(pred).all(), f"NaN/Inf pred at step {step}"

            optim.zero_grad(set_to_none=True)
            loss.backward()
            # grad_norm = squared sum over the POLICY OPTIMIZER's parameters only (bug 2).
            # It previously iterated model.parameters(), which swept in the VQ decoder's
            # nonzero-but-never-applied grads (the decoder ran outside no_grad in
            # _sample_action and is in no optimizer) and inflated the statistic.
            gn2 = torch.stack(
                [(p.grad ** 2).sum() for p in opt_params if p.grad is not None]
            ).sum()
            ep_gn2 += gn2
            # 代表层 grad norm（每步累加平方和，epoch 末 RMS；仅记录，不改训练）
            for lk, plist in layer_params.items():
                tot = torch.zeros((), device=device)
                for p in plist:
                    if p.grad is not None:
                        tot = tot + (p.grad ** 2).sum()
                layer_gn2[lk] += tot
            if model.vqvae_is_fit:
                eff_steps += 1
            optim.step()
            step += 1
            stop["step"] = step
            stop["loss"] = loss_dict["total_loss"]
            ep_tl += loss_dict["total_loss"]
            ep_cls += loss_dict["classification_loss"]
            ep_off += loss_dict["offset_loss"]
            ep_n += 1

            if step % 10 == 0 or args.smoke:
                if cfg.policy_mode == "global":
                    g = {
                        k: ("ok" if v is not None else "None")
                        for k, v in [
                            ("proj(wproj)", model.proj.weight.grad),
                            ("time_pe", model.time_pe.weight.grad),
                            ("action_queries", model.queries.grad),
                            ("gpt_b0_attn", model.backbone.h[0].attn.c_attn.weight.grad),
                            ("gpt_b0_mlp", model.backbone.h[0].mlp[0].weight.grad),
                            ("bin_head", model.bin_head.net[0].weight.grad),
                            ("offset_head", model.offset_head.net[0].weight.grad),
                        ]
                    }
                else:
                    g = {
                        k: ("ok" if v is not None else "None")
                        for k, v in [
                            ("wte(wproj)", model.gpt.wte.weight.grad),
                            ("wpe", model.gpt.wpe.weight.grad),
                            ("gpt_b0_attn", model.gpt.h[0].attn.c_attn.weight.grad),
                            ("gpt_b0_mlp", model.gpt.h[0].mlp.c_fc.weight.grad),
                            ("bin_head", model.bin_head.net[0].weight.grad),
                            ("offset_head", model.offset_head.net[0].weight.grad),
                        ]
                    }
                    if getattr(model, "coord_decoder", None) is not None:
                        g["coord_decoder"] = "ok" if model.coord_decoder.mlp[0].weight.grad is not None else "None"
                    if args.train_adaln_only:
                        # Under the freeze the trunk/heads above print None BY DESIGN. The live
                        # signal is here: alpha must wake, and proj.weight must stay exactly
                        # zero until it does (at alpha==0, dL/dproj = alpha*(...) = 0 — the
                        # "gate first, proj second" ordering this arm exists to test).
                        pw = model.gpt.adaln.proj.weight.grad
                        ag = model.gpt.adaln.alpha.grad
                        g["adaln.alpha"] = "ok" if ag is not None else "None"
                        g["adaln.proj.W"] = (
                            "None" if pw is None
                            else ("ZERO" if float(pw.abs().max()) == 0.0 else "moving")
                        )
                print(f"  step {step:4d} loss={loss_dict['total_loss']:.4f} "
                      f"(cls={loss_dict['classification_loss']:.3f} off={loss_dict['offset_loss']:.3f}) "
                      f"vq_fit={model.vqvae_is_fit} grads={g} mem={gpu_mem()}")

            if fusion and (step % 10 == 0 or args.smoke):
                r = _capture_adaln_diag(model, epoch + 1, step, adaln_rows)
                if r is not None:
                    gs, bs = r["gamma"], r["beta"]
                    gv, bv = r["gamma_var"], r["beta_var"]
                    print(f"    [adaln] mod_ratio={r['mod_ratio']:.5f} grad_adaln={r['grad_adaln']} "
                          f"gamma(m|s|a|max)={gs['mean']:.4f}|{gs['std']:.4f}|"
                          f"{gs['abs_mean']:.4f}|{gs['max_abs']:.4f} "
                          f"beta(...)={bs['mean']:.4f}|{bs['std']:.4f}|"
                          f"{bs['abs_mean']:.4f}|{bs['max_abs']:.4f} "
                          f"gamma_var(ts|img)={gv['timestep']:.2e}|{gv['image']:.2e} "
                          f"beta_var(ts|img)={bv['timestep']:.2e}|{bv['image']:.2e}")
                    if r.get("alpha") is not None:   # AdaLN-zero gate wakeup trajectory
                        print(f"    [adaln-zero] alpha={r['alpha']:.5e} "
                              f"alpha_grad={r.get('alpha_grad')}")

            if args.smoke and step >= smoke_iters:
                break
        if args.smoke:
            break
        model.finish_epoch()      # 首 epoch 末 force VQ fit（vqvae_fit_steps=null）
        # Go/No-Go per-epoch metrics（epoch 1 记录含首 epoch 末 VQ fit；之后为真实 loss 曲线）
        e = epoch + 1
        stop["epoch"] = e
        metrics["train_loss"][e] = ep_tl / ep_n
        metrics["classification_loss"][e] = ep_cls / ep_n
        metrics["offset_loss"][e] = ep_off / ep_n
        metrics["grad_norm"][e] = (ep_gn2 / ep_n).sqrt().item()  # RMS grad norm
        metrics["layer_grad_norm"][e] = {
            k: (v / ep_n).sqrt().item() for k, v in layer_gn2.items()
        }
        metrics["steps_per_epoch"][e] = ep_n
        metrics["policy_steps_effective"][e] = eff_steps
        if e % cfg.eval_freq_epochs == 0 or e in ckpt_epochs:
            eval_and_record(e)
        if e in ckpt_epochs:
            save_ckpt(e)
        # 周期性落盘 metrics（~4.5 天长跑 crash-safe；不改训练语义，仅记录）
        if e % cfg.eval_freq_epochs == 0 or e == cfg.epochs:
            _atomic_write_json(
                {"config": dict(run_cfg),
                 "ckpt_epochs": sorted(ckpt_epochs), "metrics": metrics},
                out_dir / "metrics.json",
            )
        # AdaLN diagnostics: 每 epoch 落盘（spec: "每个 epoch 持续记录"；crash-safe，不改训练语义）
        _dump_adaln_rows(out_dir, adaln_rows)

    # final val（若末 epoch 已在边界记录则跳过）+ 最终 ckpt + metrics.json
    if not args.smoke:
        e = cfg.epochs
        if e not in metrics["val_loss"]:
            eval_and_record(e)
        if cfg.save_ckpt and e not in ckpt_epochs:
            save_ckpt(e)
        _atomic_write_json(
            {"config": dict(run_cfg),
             "ckpt_epochs": sorted(ckpt_epochs), "metrics": metrics},
            out_dir / "metrics.json",
        )
        _dump_adaln_rows(out_dir, adaln_rows)
        print(f"[done] {step} steps in {time.time()-start:.0f}s | "
              f"final val_loss={metrics['val_loss'][e]:.4f} | metrics -> {out_dir / 'metrics.json'}")

    if args.smoke:
        # ---- explicit verification checklist (parity-confirmed pipeline) ----
        print("\n[SMOKE VERIFICATION]")
        bad = []
        if cfg.policy_mode == "global":
            names = [
                ("projector (proj)", model.proj.weight),
                ("temporal PE (time_pe)", model.time_pe.weight),
                ("action queries", model.queries),
                ("backbone.block0.attn", model.backbone.h[0].attn.c_attn.weight),
                ("backbone.block0.mlp", model.backbone.h[0].mlp[0].weight),
                ("backbone.ln_f", model.backbone.ln_f.weight),
                ("VQ-BeT bin_head", model.bin_head.net[0].weight),
                ("VQ-BeT offset_head", model.offset_head.net[0].weight),
            ]
        else:
            names = [
                ("visual projection (gpt.wte)", model.gpt.wte.weight),
                ("positional (gpt.wpe)", model.gpt.wpe.weight),
                ("gpt.block0.attn", model.gpt.h[0].attn.c_attn.weight),
                ("gpt.block0.mlp", model.gpt.h[0].mlp.c_fc.weight),
                ("gpt.ln_f", model.gpt.ln_f.weight),
                ("VQ-BeT bin_head", model.bin_head.net[0].weight),
                ("VQ-BeT offset_head", model.offset_head.net[0].weight),
            ]
        for name, p in names:
            gr = p.grad
            # --train-adaln-only: the frozen trunk/heads are EXPECTED to carry no grad; the
            # live check moves to the 5 AdaLN tensors just below.
            if args.train_adaln_only and gr is None:
                print(f"  {name}: grad=None (frozen by --train-adaln-only) OK")
                continue
            ok = gr is not None and torch.isfinite(gr).all()
            print(f"  {name}: grad={'present' if gr is not None else 'None'} "
                  f"finite={'yes' if ok else 'no'} {'OK' if ok else 'FAIL'}")
            if not ok:
                bad.append(name)
        if args.train_adaln_only:
            # alpha == 0 keeps proj.weight's grad present-but-exactly-zero; that is correct,
            # so `isfinite` (not `!= 0`) is the pass criterion here.
            for n, p in model.named_parameters():
                if not n.startswith("gpt.adaln."):
                    continue
                gr = p.grad
                ok = gr is not None and torch.isfinite(gr).all()
                print(f"  {n}: grad={'present' if gr is not None else 'None'} "
                      f"finite={'yes' if ok else 'no'} {'OK' if ok else 'FAIL'}")
                if not ok:
                    bad.append(n)
        bad_params = [
            n for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        print(f"  any NaN/Inf grad in model: {'YES' if bad_params else 'no'} "
              f"{'FAIL ' + str(bad_params) if bad_params else 'OK'}")
        print(f"  {cfg.encoder_name} grad: n/a (precomputed cache; encoder never in train graph), "
              f"cache requires_grad={obs.requires_grad} -> {'OK' if not obs.requires_grad else 'FAIL'}")
        # ---- benchmark: VRAM peak + iteration time ----
        dt = time.time() - start
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        print(f"  [bench] {step} steps in {dt:.1f}s -> {dt/max(step,1)*1000:.0f} ms/iter | "
              f"peak VRAM {peak_mb:.0f} MB")
        # ---- repeat-specific final verdict ----
        verdict = "SMOKE PASS" if not bad and not bad_params and not obs.requires_grad else f"SMOKE FAIL {bad}"
        print(f"  RESULT: {verdict}")
        if cfg.policy_n_patches() > 1 and cfg.encoder_name == "wemm":
            marker = "GLOBAL_TO_PATCH_EXPANSION_SMOKE_PASS" if verdict == "SMOKE PASS" else "GLOBAL_TO_PATCH_EXPANSION_SMOKE_FAIL"
            print(f"  FINAL: {marker}  (repeat_patches={cfg.policy_n_patches()}, cache P=1 {cfg.precompute_cache})")


if __name__ == "__main__":
    main()
