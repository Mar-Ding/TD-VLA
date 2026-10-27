"""Closed-loop PushT rollout — replicates base `eval_fine.py` protocol verbatim.

Protocol (each item confirmed against base `PatchPolicy-VTP` source):
    env          PushWrapper(id="pusht", env=PushTKeypointsEnv(render_size=224))
                 x NUM_ENVS via SubprocVectorEnv (envs/venv.py, tianshou-style)
    obs          [N,1,3,224,224] float [0,1]  (render rgb -> rearrange /255)
    history      T=5 deque; model pads (repeat first obs) internally (vqbet.forward)
    action       predicted_action [N,5,5,2]; temporal ensemble over last W=5
                 chunks; execute averaged chunk[0] only; shift chunks left (pad 0)
    max steps    302 (PushWrapper.step: done at step_idx>300; 10 Hz; no early stop)
    eval seeds   eval_seed = train_seed + global_episode_idx  (100 UNIQUE)
    metrics      final/max coverage (env info), reward=sum clip(cov/0.95,0,1),
                 success = final_cov>=0.95, time_to_goal = first reward>=0.999999

The ONLY change vs base: the online vision encoder is selected by
cfg.encoder_name — WeMM-Embedding-2B (wemm) or DINOv2-S/14 avg_pool (dino),
producing [N,1,P,D] -> embed [N,P,D] per frame, instead of base DINO/VTP
patch tokens. Policy ckpt is our seed-scoped training ckpt
(outputs/{save_dir}/seed{N}/ckpt_epoch400_step144800.pt). No model changes.

Usage (server, in ~/td_pusht, wemm-p1 env):
    python -m patch_policy.eval_rollout --config configs/dino_pusht.yaml \
        --train-seed 0 --num-evals 100 --num-envs 10
    python -m patch_policy.eval_rollout --config configs/dino_pusht.yaml \
        --train-seed 0 --smoke --video   # 2 ep + gif
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import einops

# ---- base repo (read-only) on the server: vendored env + venv -----------------
# IMPORT-ONLY: added to sys.path so `from envs.venv import ...` (line ~59) resolves.
# The read-only reference repo is where PushT's vendored gym env lives and there is
# no second copy of it; nothing in this project ever writes here. This package
# (`patch_policy/`, below) resolves from the cwd, NOT from BASE — the reference repo
# has no `patch_policy/` subdir, so there is no shadowing in either direction.
BASE = os.environ.get("PATCH_POLICY_BASE", os.path.expanduser("~/patch_policy"))
if os.path.isdir(BASE):
    sys.path.insert(0, BASE)

from patch_policy.config import Config
from patch_policy.dataset import (
    expand_repeat_patches,
    load_embedding_cache,
    split_episodes,
)
from patch_policy.encoder import build_cond_encoder, build_encoder
from patch_policy.vqbet import BehaviorTransformer
from patch_policy.gpt import set_attn_impl
from patch_policy.train import _adaln_variance, build_model

# env imports (needs gym/pymunk/pygame/cv2/shapely/skimage in wemm-p1 env)
from envs.venv import SubprocVectorEnv
from envs import pusht as pusht_mod


def make_env_fn():
    """Base cfg.env.gym == PushWrapper(id='pusht', env=PushTKeypointsEnv(render_size=224))."""
    return pusht_mod.PushWrapper(
        id="pusht", env=pusht_mod.PushTKeypointsEnv(render_size=224)
    )


def eval_seeds(train_seed: int, num_evals: int) -> list[int]:
    """Base fine_metrics.make_eval_seeds: eval_seed = train_seed + global idx."""
    assert num_evals >= 1
    return [int(train_seed) + i for i in range(num_evals)]


def ci95(mean: float, std: float, n: int):
    if n < 1:
        return None
    se = std / np.sqrt(n)
    return [float(mean - 1.96 * se), float(mean + 1.96 * se)]


def ci_prop(p: float, n: int):
    se = np.sqrt(p * (1 - p) / n)
    return [float(p - 1.96 * se), float(p + 1.96 * se)]


def _summ(xs) -> dict | None:
    """Compact summary of a per-step diagnostic series (None-safe)."""
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str | None:
    """Provenance digest for an artifact a condition depends on (best-effort)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


class _HistPolicyAdapter:
    """Duck-typed wrapper so this evaluator can drive a *pickled patch_policy
    BehaviorTransformer* (known-good HIST ckpt) with the same call it makes on the
    patch_policy model — model(obs, goal, None, cond) -> (action[N,T,W,A], loss, aux).

    HIST forward is (obs_seq, goal_seq, action_seq) -> (pred, loss, dict); it has no
    cond stream (AdaLN fusion is a patch_policy-only addition), so cond is dropped. No
    retraining: the pickled weights are used verbatim.
    """

    def __init__(self, hist):
        self.hist = hist

    def eval(self):
        self.hist.eval()
        return self

    def parameters(self):
        return self.hist.parameters()

    @property
    def vqvae_is_fit(self):
        return self.hist.vqvae_is_fit

    @vqvae_is_fit.setter
    def vqvae_is_fit(self, v):
        self.hist.vqvae_is_fit = v

    @property
    def vqvae(self):
        return self.hist._vqvae_model

    def __call__(self, obs_seq, goal_seq, action_seq=None, cond_seq=None):
        return self.hist(obs_seq, goal_seq, action_seq)


def _load_hist_policy(path: str, device) -> torch.nn.Module:
    """Unpickle a patch_policy BehaviorTransformer ckpt into the current process.

    The pickle references patch_policy classes (models.vq_behavior_transformer.bet);
    BASE is already on sys.path at import. wemm-p1 env needs omegaconf installed
    (accelerate dependency pulled in by bet.py's module-level Accelerator()).
    Returns the HIST nn.Module in eval mode with vqvae_is_fit=True (same as the
    patch_policy branch sets below). No weights are touched.
    """
    m = torch.load(path, map_location=device, weights_only=False)
    assert isinstance(m, torch.nn.Module), f"{path} did not unpickle to an nn.Module"
    assert hasattr(m, "_gpt_model") and hasattr(m, "_vqvae_model"), (
        f"{path}: expected a patch_policy BehaviorTransformer (has _gpt_model/_vqvae_model)"
    )
    m.eval()
    m.vqvae_is_fit = True
    m._vqvae_model.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pusht.yaml")
    ap.add_argument("--attn", choices=("current", "sdpa"), default=None,
                    help="attention kernel for the rollout policy. Default = module default "
                         "('sdpa' since 2026-10-27). Pass 'current' to re-measure a rollout "
                         "under the same kernel that produced the archived n=100 numbers -- "
                         "rollout wall-clock is kernel-sensitive, so compare like with like.")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="override ckpt path")
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--num-evals", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="2 episodes, verbose shapes")
    ap.add_argument("--video", action="store_true", help="save per-episode mp4 (first env each batch)")
    ap.add_argument("--out", default=None, help="output dir (default outputs/pusht/seed{N}/rollout)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preproc", choices=["none", "resize224"], default="none",
                    help="pre-encoder preprocessing. none = current protocol; resize224 = force "
                         "matching the training-precompute input condition (224x224; env render is "
                         "already 224, so this is an identity + assertion)")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="FIXED action-sampling RNG seed for the primary protocol (default 0). "
                         "Applied to torch CPU+CUDA AFTER the model/encoders are built and the "
                         "checkpoint is loaded, immediately before the rollout loop — so model "
                         "init can never consume part of the sampling stream. Re-evaluations "
                         "use pre-fixed extra seeds 1 and 2 (separate runs, separately "
                         "reported; never merged into one 300-scene number).")
    ap.add_argument("--cond-shuffle", choices=["none", "batch", "time"], default="none",
                    help="WeMM-DINO fusion conditional-usage diagnostic (spec): "
                         "none = (DINO_i, WeMM_i); batch = (DINO_i, WeMM_j) shuffle WeMM "
                         "across envs; time = (DINO_i,t, WeMM_i,t') shuffle WeMM across the "
                         "T-history within each env. Only valid when fusion.enabled.")
    ap.add_argument("--cond-ablation",
                    choices=["none", "alpha_zero", "zero_cond", "mean_cond", "shuffled_cond"],
                    default="none",
                    help="INFERENCE-TIME condition ablation on a FROZEN ckpt (no retraining; "
                         "the ckpt FILE is never modified). "
                         "none = bit-identical to the normal protocol. "
                         "alpha_zero = force the AdaLN-Zero gate alpha to 0 (in memory only) — "
                         "the ONLY condition that actually REMOVES the modulation. "
                         "zero_cond / mean_cond = replace the cond with a CONSTANT (all-zeros / "
                         "the training-split mean embedding); the branch still runs on that "
                         "constant, so these are constant-condition controls, NOT 'branch off'. "
                         "shuffled_cond = keep the true cond distribution but break the "
                         "cond<->observation pairing with a FIXED derangement over the N env "
                         "dim (episode-level, applied identically to all T history frames). "
                         "Only valid when fusion.enabled.")
    ap.add_argument("--cond-ablation-seed", type=int, default=20260915,
                    help="fixed RNG seed for the shuffled_cond derangement; recorded in the "
                         "result JSON together with the full permutation mapping")
    ap.add_argument("--probe-sampling", action="store_true",
                    help="record action entropy + VQ code usage by wrapping _sample_action, and "
                         "the AdaLN diag (alpha / gamma / beta). The probe reads cbet_logits "
                         "AFTER the real sampling call, so it consumes NO RNG and leaves the "
                         "action stream untouched. Off by default so a plain run stays "
                         "identical in every respect.")
    ap.add_argument("--hist-ckpt", default=None,
                    help="KNOWN-GOOD REPLAY (audit only): instead of building a patch_policy "
                         "model, load a patch_policy pickled BehaviorTransformer ckpt "
                         "(e.g. phase9 dino_s43/model_final.pt) and run it through THIS "
                         "exact evaluator (same env/metric/DINO/temporal-ensemble). "
                         "Forward contract (obs, goal, None) is the same; no retraining.")
    args = ap.parse_args()

    # NOTE: the action-sampling RNG seed is deliberately NOT applied here. It is applied
    # further down, after the policy/encoders are built and the ckpt is loaded, so that
    # module initialization cannot consume part of the sampling stream.
    if args.preproc == "resize224":
        print("[eval] preproc=resize224: forcing 224x224 pre-WeMM (identity vs env render)")

    num_envs = int(os.environ.get("EV_NUM_ENVS", args.num_envs))
    num_evals = int(os.environ.get("EV_EVAL_EPISODES", args.num_evals if not args.smoke else 2))
    if args.smoke:
        num_evals = min(num_evals, 2)
        num_envs = min(num_envs, 2)
    assert num_evals % num_envs == 0, "episodes must be a multiple of envs"

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device)
    train_seed = args.train_seed
    out_dir = Path(args.out or f"{cfg.save_dir}/seed{train_seed}/rollout")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.hist_ckpt:
        assert not args.smoke, "--hist-ckpt replay is a full-rollout audit, not --smoke"
        assert cfg.encoder_name == "dino", "hist replay requires encoder_name=dino"
        assert not cfg.fusion.enabled, "hist replay runs the non-fusion DINO-P256 model"
        ckpt_path = args.hist_ckpt
    else:
        ckpt_path = args.ckpt or sorted(
            glob.glob(f"{cfg.save_dir}/seed{train_seed}/ckpt_*.pt")
        )[-1]

    tag = "KNOWN-GOOD REPLAY" if args.hist_ckpt else "rollout"
    print(f"[eval] {cfg.encoder_name}-P{cfg.policy_n_patches()} {tag} "
          f"(enc P={cfg.enc().n_patches}) | train_seed={train_seed} "
          f"ckpt={ckpt_path} envs={num_envs} evals={num_evals}")

    # ---- policy (frozen training ckpt) ------------------------------------
    if args.hist_ckpt:
        # KNOWN-GOOD REPLAY (Stage-2 DINO baseline audit): unpickle the patch_policy
        # BehaviorTransformer (phase9 dino_s43 et al.) and run it through THIS exact
        # evaluator (same env / DINO extraction / VQ-BeT sampling / temporal ensemble /
        # 302-step rollout / metrics). No retraining, no re-init — only the policy
        # weights are swapped; obs embeddings come from the current cfg encoder.
        model = _load_hist_policy(args.hist_ckpt, device)
        hist_in = model._gpt_model.transformer.wte.weight.shape[1]
        assert cfg.enc().output_dim == hist_in, (
            f"hist policy input dim {hist_in} != cfg encoder output_dim "
            f"{cfg.enc().output_dim} (replay needs the DINO-P256 encoder it was trained on)"
        )
        print(f"[eval] KNOWN-GOOD REPLAY policy: wte.in={hist_in}, "
              f"lm_head={tuple(model._gpt_model.lm_head.weight.shape)}, "
              f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
              f"vqvae_is_fit={model.vqvae_is_fit}")
        model = _HistPolicyAdapter(model)
    else:
        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model"]
        # 兼容旧 ckpt 键名 gpt.film.* -> gpt.adaln.*（2026-09-01 改名后代码用 adaln 键）
        if any(k.startswith("gpt.film.") for k in sd):
            sd = {
                ("gpt.adaln." + k[len("gpt.film."):] if k.startswith("gpt.film.") else k): v
                for k, v in sd.items()
            }
        model.load_state_dict(sd)
        model.eval()
        # Same fitted-VQ contract as training: eval mode immediately, before any forward
        # (a bare flag assignment would leave the tokenizer in train mode -> live EMA).
        model.set_vqvae_fit(True)
        print(f"[eval] policy params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
              f"vqvae_is_fit={model.vqvae_is_fit}, "
              f"vqvae_requires_grad={any(p.requires_grad for p in model.vqvae.parameters())}")

    # ---- attention kernel (applies to both the plain and the hist-adapter policy) ----
    if args.attn is not None:
        n_attn = set_attn_impl(model, args.attn)
        if n_attn == 0:
            raise SystemExit(f"--attn {args.attn}: no module exposes an attn_impl selector")
        print(f"[attn] rollout kernel={args.attn} on {n_attn} attention module(s) (explicit)")
    else:
        attn_mods = [m.attn_impl for m in model.modules() if hasattr(m, "attn_impl")]
        impls = sorted(set(attn_mods)) or ["none"]
        print(f"[attn] rollout kernel={','.join(impls)} on {len(attn_mods)} attention "
              f"module(s) (module default)")

    # ---- online vision encoder (frozen, wemm | dino per cfg.encoder_name) ---
    encoder, _ = build_encoder(cfg)
    encoder.to(device)
    encoder.eval()
    print(f"[eval] {cfg.encoder_name} encoder ready ({cfg.enc_model_name()})")

    # ---- cond encoder (fusion only; online encode of live obs frames) ----
    # Dispatches on cfg.fusion.cond_encoder (wemm -> WeMM-Embedding-2B,
    # qwen3vl -> Qwen3-VL-Embedding-2B) via build_cond_encoder — must produce the
    # SAME representation the training cond_cache was made with.
    fusion = cfg.fusion.enabled
    cond_encoder = None
    if fusion:
        assert cfg.encoder_name == "dino" and cfg.enc().postprocess is None, (
            "fusion rollout requires encoder_name=dino, dino.postprocess=None "
            "(DINO-P256 dense tokens)"
        )
        cond_encoder = build_cond_encoder(cfg).to(device)
        cond_encoder.eval()
        print(f"[eval] cond encoder ready ({cfg.fusion.cond_encoder}: "
              f"{cfg.cond_model_name()}) cond_dim={cfg.fusion.cond_dim}")
        if args.cond_shuffle != "none":
            print(f"[eval] cond-shuffle = {args.cond_shuffle} (spec conditional-usage diagnostic)")
    # deterministic shuffle generator: stable across reruns of the diagnostic
    cond_gen = torch.Generator(device=device).manual_seed(31337 + train_seed)

    # ---- env ---------------------------------------------------------------
    env = SubprocVectorEnv([make_env_fn for _ in range(num_envs)])
    seeds = eval_seeds(train_seed, num_evals)
    assert len(seeds) == num_evals and len(set(seeds)) == num_evals

    def embed(obs: np.ndarray) -> torch.Tensor:
        """Same as base embed(): [N,V,C,H,W] -> enc -> [N,(V P),E]."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if args.preproc == "resize224":
            # match training-precompute input condition explicitly. env render is already
            # 224x224, so this resize is an identity — assert it to prove no-op.
            pre = torch.nn.functional.interpolate(
                obs_t.reshape(-1, *obs_t.shape[-3:]), size=(224, 224),
                mode="bilinear", align_corners=False,
            ).reshape(obs_t.shape)
            md = (pre - obs_t).abs().max().item()
            if md > 1e-3:
                print(f"[warn] preproc=resize224 changed pixels (max {md:.4f}) — env render not 224?")
            obs_t = pre
        return einops.rearrange(encoder(obs_t), "N V P E -> N (V P) E")

    def embed_cond(obs: np.ndarray) -> torch.Tensor:
        """cond stream for one obs frame: [N,V,C,H,W] -> [N,1,cond_dim] (per-frame)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return einops.rearrange(cond_encoder(obs_t), "N V P E -> N (V P) E")

    # ---- FIXED action-sampling RNG (after model+encoders+ckpt are all in place) -----
    # VQ-BeT picks action codes with torch.multinomial, so the rollout is stochastic.
    # Seeding here (not at launch) makes the sampling stream independent of how much
    # RNG model/encoder construction happened to consume, and identical for every
    # milestone of both arms. CPU + CUDA are both seeded because multinomial runs on
    # the GPU tensors.
    if args.rng_seed is not None:
        torch.manual_seed(int(args.rng_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.rng_seed))
        print(f"[eval] action-sampling RNG seeded AFTER init/ckpt load: "
              f"torch.manual_seed({args.rng_seed}) + cuda.manual_seed_all({args.rng_seed})")

    # ================== inference-time condition ablation ====================
    # Science: does the policy actually USE the image-specific Qwen3 semantic content, or is
    # the gain the conditioning branch's structural / regularization role?
    #
    # Inference-time ONLY: no retraining, and the ckpt FILE is never written. What each
    # condition really does (verified: `cond` reaches the network in exactly ONE place —
    # GPT.forward -> self.adaln, gpt.py:195 — and there is exactly ONE gate, adaln_gate.py:81):
    #   alpha_zero    REMOVES the modulation: fused = x + 0*m == x, bitwise. The only
    #                 condition that is truly "fusion off".
    #   zero_cond     keeps the modulation but feeds it a CONSTANT. ln_g is LayerNorm WITH
    #                 affine, so ln_g(0) = bias != 0 -> M is a NONZERO constant, not zero.
    #   mean_cond     same, with the training-split mean embedding as the constant.
    #   shuffled_cond keeps the true cond distribution, breaks the cond<->obs pairing.
    # So A-C is "content at a fixed code" and A-D is "scene content vs a train-mean code" —
    # NOT "with vs without conditioning". That contrast is A-B alone.
    ablation = args.cond_ablation
    abl_meta: dict = {"cond_ablation": ablation}
    abl_alpha_before = None
    mean_cond = None
    abl_disp: dict = {"cos": [], "dnorm": []}
    abl_gate: dict = {k: [] for k in (
        "alpha", "mod_ratio", "delta_norm", "m_norm", "x_norm",
        "gamma_var_image", "gamma_var_timestep", "beta_var_image", "beta_var_timestep",
        "gamma_abs_mean", "beta_abs_mean")}
    probe: dict = {"on": bool(args.probe_sampling), "ent": [], "calls": 0, "codes": None}
    if ablation != "none":
        assert fusion, f"--cond-ablation {ablation} requires fusion.enabled"
        assert not args.hist_ckpt, "--cond-ablation is a patch_policy fusion diagnostic"
        # The default ckpt selection is `sorted(glob(...))[-1]` = LEXICOGRAPHIC max, which
        # picks ckpt_epoch400, not the best ckpt. An ablation must name its checkpoint.
        assert args.ckpt, ("--cond-ablation requires an explicit --ckpt (the default glob "
                           "would silently select ckpt_epoch400 rather than the best ckpt)")
        print(f"[eval] COND-ABLATION = {ablation} (inference-time; ckpt file NOT modified)")

    if ablation == "alpha_zero":
        _gate = getattr(getattr(model, "gpt", None), "adaln", None)
        assert _gate is not None and hasattr(_gate, "alpha"), (
            "alpha_zero needs model.gpt.adaln.alpha (AdaLN-gated fusion)"
        )
        abl_alpha_before = float(_gate.alpha.detach())
        _gate.alpha.data.zero_()
        abl_meta["alpha_before"] = abl_alpha_before
        abl_meta["alpha_after"] = float(_gate.alpha)
        print(f"[eval]   AdaLN gate alpha {abl_alpha_before:.8f} -> {float(_gate.alpha):.8f} "
              f"(in memory only; the .pt is untouched)")

    if ablation == "mean_cond":
        # ONE fixed mean Qwen embedding over the TRAINING split of the cond cache. Episodes
        # are [L,P,D] with VARIABLE L, so concatenate before averaging (frame-weighted: every
        # training frame counts once). fp32 throughout; only the [1,1,1,D] vector reaches the
        # GPU (the full cache is ~400MB and would pollute peak_gpu_mem_mb).
        cache_path = Path(cfg.fusion.cond_cache)
        _meta, _eps = load_embedding_cache(
            cfg.fusion.cond_cache,
            check_meta={"model_name": cfg.cond_model_name(), "n_patches": 1,
                        "output_dim": cfg.fusion.cond_dim},
        )
        abl_train_idx, abl_val_idx = split_episodes(
            len(_eps), cfg.train_fraction, seed=cfg.split_seed
        )
        _cat = torch.cat([_eps[i].float() for i in abl_train_idx], dim=0)   # [frames,P,D]
        mean_cond = _cat.mean(dim=0).reshape(1, 1, 1, -1).contiguous()      # [1,1,1,D]
        assert mean_cond.ndim == 4 and mean_cond.shape[-1] == cfg.fusion.cond_dim
        abl_meta.update({
            "mean_cond_cache": str(cache_path.resolve()),
            "mean_cond_cache_sha256": _file_sha256(cache_path),
            "mean_cond_cache_n_episodes": len(_eps),
            "mean_cond_train_episodes": len(abl_train_idx),
            "mean_cond_val_episodes": len(abl_val_idx),
            "mean_cond_frames": int(_cat.shape[0]),
            "mean_cond_norm": float(mean_cond.norm()),
            "mean_cond_reduction": "frame-weighted mean over ALL frames of the train episodes",
            "mean_cond_train_idx": abl_train_idx,
            "cache_meta_model_name": _meta.get("model_name"),
            "cache_meta_cache_dtype": _meta.get("cache_dtype"),
        })
        print(f"[eval]   mean_cond from {len(abl_train_idx)}/{len(_eps)} TRAIN episodes "
              f"({_cat.shape[0]} frames) | ||mean|| = {float(mean_cond.norm()):.6f} "
              f"(per-frame conds are L2-normalized, so this is < 1)")
        del _cat, _eps

    if ablation == "shuffled_cond":
        abl_meta["cond_ablation_seed"] = int(args.cond_ablation_seed)
        print(f"[eval]   shuffled_cond: fresh derangement per batch over the {num_envs} env "
              f"dim, seed={args.cond_ablation_seed}, NO fixed points (mapping in the JSON)")

    # Dedicated generator for the derangement so that drawing a permutation can never touch
    # the global action-sampling stream fixed immediately above.
    abl_gen = torch.Generator().manual_seed(int(args.cond_ablation_seed))

    def _draw_derangement(n: int) -> torch.Tensor:
        """Permutation of range(n) with NO fixed point. randperm alone is NOT enough: ~65% of
        random permutations of 10 elements have a fixed point, which would silently leave
        some episodes paired with their OWN cond."""
        for _ in range(100000):
            cand = torch.randperm(n, generator=abl_gen)
            if bool((cand != torch.arange(n)).all()):
                return cand
        raise RuntimeError(f"no derangement of {n} elements found")

    if probe["on"]:
        # The REAL sampling call (and its multinomial RNG draw) happens FIRST and unchanged;
        # the diagnostics below are pure reductions on already-materialized tensors and consume
        # no RNG, so the action stream stays bit-identical to a non-probed run.
        assert not args.hist_ckpt, "--probe-sampling is a patch_policy fusion diagnostic"
        probe["codes"] = torch.zeros(int(model.C), dtype=torch.long)
        _orig_sample_action = model._sample_action

        def _sample_action_probed(cbet_logits, cbet_offsets):
            out = _orig_sample_action(cbet_logits, cbet_offsets)
            with torch.no_grad():
                p = torch.softmax(cbet_logits.detach().float(), dim=-1)    # [N,T,G,C]
                ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)     # [N,T,G]
                probe["ent"].append(float(ent.mean()))
                probe["codes"] += torch.bincount(
                    out[2].detach().reshape(-1).cpu(), minlength=int(model.C)
                )
                probe["calls"] += 1
            return out

        model._sample_action = _sample_action_probed
        print(f"[eval] sampling probe ON (action entropy + VQ code usage; C={int(model.C)})")

    # ---- rollout -----------------------------------------------------------
    final_covs, max_covs, ep_rewards, t2g_all, ep_steps = [], [], [], [], []
    enc_ms_all, pol_ms_all, tot_ms_all = [], [], []
    ep_meta = []
    traces = {}  # episode_id -> per-step arrays (npz output, like base eval_fine)

    num_batches = num_evals // num_envs
    obs_window = cfg.obs_window_size      # T = 5
    act_window = cfg.action_window_size   # W = 5
    smoke_shown = not args.smoke
    env_ms_all = []

    def _embed_and_log(obs: np.ndarray) -> tuple:
        """Encode one obs frame (main DINO + WeMM cond if fusion); return (main, cond, ms)."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e = embed(obs)
        c = embed_cond(obs) if fusion else None
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        enc_ms_all.append(ms)
        return e, c, ms

    for b in range(num_batches):
        batch_seeds = seeds[b * num_envs:(b + 1) * num_envs]
        env.seed(batch_seeds)
        this_obs = env.reset(goal_idx=b)

        # shuffled_cond: a FRESH derangement per batch, held FIXED for all 302 steps of that
        # batch, so env i's whole 5-frame rolling history comes from ONE other env,
        # contemporaneously. Per-batch rather than one global perm so the result does not
        # hinge on a single lucky draw. Drawn from a DEDICATED generator => the global
        # action-sampling stream fixed above is untouched.
        batch_perm = _draw_derangement(num_envs) if ablation == "shuffled_cond" else None
        if batch_perm is not None:
            abl_meta.setdefault("shuffle_perms", []).append(batch_perm.tolist())

        # obs history: repeat-first handled inside model forward (vqbet pads T)
        obs_stack, cond_first, _ = _embed_and_log(this_obs)
        obs_stack = [obs_stack]
        cond_stack = [cond_first] if fusion else None

        done = np.array([False] * num_envs)
        action_list = []
        first_sat = [None] * num_envs
        steps_this_batch = 0
        ep_reward = np.zeros(num_envs)
        # per-env per-step recordings (for video + fine trace)
        frames = [[] for _ in range(num_envs)]
        rec = [{"cov": [], "rew": [], "block": [], "agent": [], "act": []}
               for _ in range(num_envs)]
        enc_ms_ep = np.zeros(num_envs)
        pol_ms_ep = np.zeros(num_envs)
        tot_ms_ep = np.zeros(num_envs)

        while not done.all():
            obs = torch.stack(tuple(obs_stack)).float().to(device)
            obs = einops.rearrange(obs, "T N P E -> N T P E")
            # WeMM-Repeat-P256: expand the single global token per frame to policy P
            # (pure copy; encoder still runs once per frame). Coord decoder: NO expansion —
            # the model's CoordSpatialDecoder turns the P=1 token into slots in forward.
            # Must match train exactly.
            if cfg.interface_mode() == "repeat" and cfg.policy_n_patches() != obs.shape[2]:
                obs = expand_repeat_patches(obs, cfg.policy_n_patches())
            goal = torch.as_tensor([0.0], dtype=torch.float32, device=device)
            goal = einops.repeat(goal, "... -> N T ...", N=num_envs, T=obs_window)

            # WeMM-DINO fusion: assemble [N,T,1,cond_dim] cond history, apply the
            # conditional-usage shuffle diagnostic (batch/time), pass to model.
            cond = None
            if fusion:
                cond = torch.stack(tuple(cond_stack)).float().to(device)
                cond = einops.rearrange(cond, "T N C D -> N T C D")  # [N,T,1,D]
                if args.cond_shuffle == "batch":
                    perm = torch.randperm(cond.shape[0], generator=cond_gen, device=device)
                    cond = cond[perm]                                # (DINO_i, WeMM_j)
                elif args.cond_shuffle == "time":
                    perm = torch.stack([
                        torch.randperm(cond.shape[1], generator=cond_gen, device=device)
                        for _ in range(cond.shape[0])
                    ], dim=0)                                        # [N,T]
                    cond = cond[torch.arange(cond.shape[0], device=device)[:, None], perm]

            # ---- inference-time condition ablation (cond-tensor side) --------------
            # Immediately BEFORE the forward. `none` and `alpha_zero` execute nothing here, so
            # their cond tensor is byte-identical to the normal protocol.
            if ablation in ("zero_cond", "mean_cond", "shuffled_cond"):
                cond_true = cond
                if ablation == "zero_cond":
                    cond = torch.zeros_like(cond)
                elif ablation == "mean_cond":
                    # mean_cond is built on CPU (see above) and .expand_as() does NOT move
                    # devices, so the device has to be named explicitly here or the
                    # displacement diagnostic below sees cuda vs cpu.
                    cond = mean_cond.to(device=cond.device, dtype=cond.dtype).expand_as(cond)
                else:
                    cond = cond[batch_perm.to(cond.device)]
                assert cond.ndim == 4 and cond.shape[-1] == cfg.fusion.cond_dim, (
                    f"ablation {ablation} broke the cond shape: {tuple(cond.shape)}"
                )
                # "FED-COND PERTURBATION" — how far the cond we FEED moved. NOT a usage
                # measure: reported next to the gate's own (alpha / mod_ratio / gamma) numbers
                # because ln_g is a LayerNorm and divides the raw scale out, so a magnitude
                # delta can overstate what the gate can actually see.
                with torch.no_grad():
                    _a = cond_true.detach().float()
                    _b = cond.detach().float()
                    abl_disp["cos"].append(
                        float(torch.nn.functional.cosine_similarity(_a, _b, dim=-1).mean()))
                    abl_disp["dnorm"].append(float((_a - _b).norm(dim=-1).mean()))

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                action, _, _ = model(obs, goal, None, cond)
            torch.cuda.synchronize()
            pol_ms = (time.perf_counter() - t0) * 1000
            pol_ms_all.append(pol_ms); pol_ms_ep += pol_ms

            # ---- diagnostics (deliberately OUTSIDE the timed block) -----------------
            # _stats / _adaln_variance sync the host, so timing them would inflate
            # policy_ms_per_step_mean and peak_gpu_mem_mb and make those non-comparable
            # across conditions. gamma/beta come from the gate's own last-forward `_diag`.
            if probe["on"] or ablation != "none":
                _g = getattr(getattr(model, "gpt", None), "adaln", None)
                if _g is not None and getattr(_g, "_diag", None):
                    ds = _g.diag_summary()
                    gv = _adaln_variance(_g._diag["gamma"])   # [B,T,1,n_embd]
                    bv = _adaln_variance(_g._diag["beta"])
                    abl_gate["alpha"].append(ds["alpha"])
                    abl_gate["mod_ratio"].append(ds["mod_ratio"])
                    abl_gate["delta_norm"].append(ds["delta_norm"])
                    abl_gate["m_norm"].append(ds["m_norm"])
                    abl_gate["x_norm"].append(ds["x_norm"])
                    abl_gate["gamma_var_image"].append(gv["image"])
                    abl_gate["gamma_var_timestep"].append(gv["timestep"])
                    abl_gate["beta_var_image"].append(bv["image"])
                    abl_gate["beta_var_timestep"].append(bv["timestep"])
                    if ds["gamma"]:
                        abl_gate["gamma_abs_mean"].append(ds["gamma"]["abs_mean"])
                    if ds["beta"]:
                        abl_gate["beta_abs_mean"].append(ds["beta"]["abs_mean"])
                    if len(abl_gate["alpha"]) == 1:
                        # Pre-flight numbers: if mod_ratio is ~0 the gate is inert and every
                        # ablation is the same run. gamma_var(image) is the most direct
                        # "does the cond shape the modulation per sample" measure.
                        print(f"[eval]   first-forward AdaLN diag: alpha={ds['alpha']:.6f} "
                              f"mod_ratio={ds['mod_ratio']:.6f} |delta|={ds['delta_norm']:.4f} "
                              f"|M|={ds['m_norm']:.4f} |X|={ds['x_norm']:.4f} "
                              f"gamma_var(image)={gv['image']:.6f} "
                              f"gamma_var(timestep)={gv['timestep']:.6f}")
                    if ablation == "alpha_zero" and "alpha_identity_delta_norm" not in abl_meta:
                        # Free exact proof the gate is OFF while the branch is still computed:
                        # |alpha*M| must be 0 while |M| stays > 0.
                        abl_meta["alpha_identity_delta_norm"] = ds["delta_norm"]
                        abl_meta["alpha_identity_m_norm"] = ds["m_norm"]

            if args.smoke and not smoke_shown:
                print("    ---- Milestone-B §8 tensor checklist (first decision) ----")
                print(f"    {cfg.encoder_name} embedding (single frame) : {tuple(obs_stack[0].shape)} (expect [N,1,{cfg.enc().output_dim}])")
                print(f"    history embeddings                : {tuple(obs.shape)} (expect [N,T,1,{cfg.enc().output_dim}] cache) ")
                if cfg.policy_mode == "global":
                    print(f"    global projected states          : {tuple(model.proj(obs).shape)} (expect [N,T,1,{model.n_embd}])")
                elif cfg.interface_mode() == "coord":
                    print(f"    coord-decoded spatial slots     : {tuple(model.coord_decoder(obs).shape)} (expect [N,T,{cfg.policy_n_patches()},512])")
                else:
                    print(f"    projected visual tokens          : {tuple(model.gpt.wte(obs).shape)} (expect [N,T,{cfg.policy_n_patches()},512])")
                print(f"    policy action chunk              : {tuple(action.shape)} (expect [N,T,W,A]=[{num_envs},{obs_window},{act_window},2])")
                smoke_shown = True

            action_list.append(action[:, -1].cpu().detach().numpy())  # [N,W,A]
            if len(action_list) > act_window:
                action_list = action_list[1:]
            curr_action = np.array(action_list).mean(axis=0)[:, 0]     # [N,A]
            new_list = []
            for a_chunk in action_list:
                new_list.append(np.concatenate(
                    (a_chunk[:, 1:], np.zeros((a_chunk.shape[0], 1, a_chunk.shape[-1]))),
                    axis=1))
            action_list = new_list

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            this_obs, reward, done, info = env.step(curr_action)
            torch.cuda.synchronize()
            env_ms = (time.perf_counter() - t0) * 1000
            env_ms_all.append(env_ms)

            e_new, c_new, enc_ms = _embed_and_log(this_obs)   # encode of the new frame
            enc_ms_ep += enc_ms
            tot_ms = pol_ms + enc_ms
            tot_ms_all.append(tot_ms); tot_ms_ep += tot_ms
            obs_stack.append(e_new)
            obs_stack = obs_stack[-obs_window:]
            if fusion:
                cond_stack.append(c_new)
                cond_stack = cond_stack[-obs_window:]

            ep_reward += reward
            steps_this_batch += 1
            for i in range(num_envs):
                d = info[i]
                if args.video and len(frames[i]) < steps_this_batch:
                    frames[i].append(d["image"])
                rec[i]["cov"].append(float(d["final_coverage"]))
                rec[i]["rew"].append(float(reward[i]))
                rec[i]["block"].append([float(v) for v in d["block_pose"]])
                rec[i]["agent"].append([float(v) for v in d["pos_agent"]])
                rec[i]["act"].append([float(curr_action[i, 0]), float(curr_action[i, 1])])
                if first_sat[i] is None and reward[i] >= 0.999999:
                    first_sat[i] = steps_this_batch

        # ---- finalize batch ----
        for i in range(num_envs):
            gid = b * num_envs + i
            seed = batch_seeds[i]
            steps = steps_this_batch
            final_covs.append(float(info[i]["final_coverage"]))
            max_covs.append(float(info[i]["max_coverage"]))
            t2g_all.append(first_sat[i])
            ep_steps.append(steps)
            ep_rewards.append(float(ep_reward[i]))
            cov = np.array(rec[i]["cov"]); blk = np.array(rec[i]["block"])
            agt = np.array(rec[i]["agent"]); act = np.array(rec[i]["act"])
            ep_meta.append({
                "train_seed": train_seed, "eval_seed": seed, "episode_id": gid,
                "final_coverage": float(info[i]["final_coverage"]),
                "max_coverage": float(info[i]["max_coverage"]),
                "reward": float(ep_reward[i]),
                "success": bool(info[i]["final_coverage"] >= 0.95),
                "episode_steps": steps,
                "time_to_goal": first_sat[i],
                "encoder_ms": float(enc_ms_ep[i] / steps),
                "policy_ms": float(pol_ms_ep[i] / steps),
                "total_ms": float(tot_ms_ep[i] / steps),
            })
            # per-step trace (base eval_fine records the same)
            traces[gid] = {"coverage": cov, "block_pose": blk, "agent_pos": agt,
                           "action": act, "reward": np.array(rec[i]["rew"])}
            if args.video:
                img = np.asarray(frames[i])
                path = out_dir / f"ep_{gid:03d}_seed{seed}.mp4"
                _write_video(img, path)
            if args.smoke:
                blk_drift = np.linalg.norm(blk[-1] - blk[0]) if len(blk) else 0.0
                act_min, act_max = act.min(), act.max()
                print(f"    [diag] ep{gid} seed{seed}: final_cov={cov[-1]:.3f} max_cov={cov.max():.3f} "
                      f"| block drift {blk_drift:.0f}px (start {blk[0][:2]} -> end {blk[-1][:2]}) "
                      f"| action range [{act_min:.1f},{act_max:.1f}] "
                      f"| coverage curve [{cov[0]:.2f} ... {cov[-1]:.2f}]")
        if args.smoke and b == 0:
            break  # smoke: 1 batch of 2 envs

    env.close()
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

    # per-step trajectory npz (base eval_fine writes trajectories/<tag>.npz)
    if traces:
        npz_path = out_dir / f"trajectories_seed{train_seed}.npz"
        np.savez_compressed(npz_path, **{
            f"ep{gid}__{k}": v for gid, t in traces.items() for k, v in t.items()
        })
        print(f"  saved {npz_path} ({len(traces)} episodes)")

    # ---- aggregate (base metrics, identical math to eval_fine.py) ----------
    final_covs = np.array(final_covs); max_covs = np.array(max_covs)
    ep_rewards = np.array(ep_rewards)
    p_succ = float((final_covs >= 0.95).mean())
    t2g_vals = [v for v in t2g_all if v is not None]
    base = {
        "n_evals": int(len(final_covs)),
        "final_coverage_mean": float(final_covs.mean()),
        "final_coverage_std": float(final_covs.std()),
        "final_coverage_ci95": ci95(final_covs.mean(), final_covs.std(), len(final_covs)),
        "max_coverage_mean": float(max_covs.mean()),
        "max_coverage_std": float(max_covs.std()),
        "max_coverage_ci95": ci95(max_covs.mean(), max_covs.std(), len(max_covs)),
        "success_rate_095": p_succ,
        "success_rate_ci95": ci_prop(p_succ, len(final_covs)),
        "reward_mean": float(ep_rewards.mean()),
        "reward_ci95": ci95(ep_rewards.mean(), ep_rewards.std(), len(ep_rewards)),
        "time_to_goal_mean": float(np.mean(t2g_vals)) if t2g_vals else None,
        "time_to_goal_std": float(np.std(t2g_vals)) if t2g_vals else None,
        "n_episodes_reached_095": int(len(t2g_vals)),
        "episode_steps_mean": float(np.mean(ep_steps)),
        "encoder_ms_per_step_mean": float(np.mean(enc_ms_all)),
        "policy_ms_per_step_mean": float(np.mean(pol_ms_all)),
        "env_step_ms_per_step_mean": float(np.mean(env_ms_all)) if env_ms_all else None,
        "total_ms_per_step_mean": float(np.mean(tot_ms_all)),
        "peak_gpu_mem_mb": float(peak_mem),
    }

    # ---- inference-time condition-ablation diagnostics ------------------------
    _ablation_block = None
    if ablation != "none" or probe["on"]:
        _samp = None
        if probe["on"] and probe["codes"] is not None:
            _codes = probe["codes"]
            _tot = float(_codes.sum())
            _p = (_codes.float() / _tot) if _tot > 0 else _codes.float()
            _nz = _p[_p > 0]
            _ent = float(-(_nz * _nz.log()).sum()) if _nz.numel() else 0.0
            _samp = {
                "action_entropy_mean": _summ(probe["ent"]),
                "vq_codes_count": _codes.tolist(),
                "vq_n_samples": int(_codes.sum()),
                "vq_codes_used": int((_codes > 0).sum()),
                "vq_code_entropy_nats": _ent,
                "vq_code_perplexity": float(np.exp(_ent)),
                "note": "code usage = histogram of the SAMPLED centers; entropy = mean entropy "
                        "of the softmax code distribution at each sampled step",
            }
        _ablation_block = {
            "meta": abl_meta,
            "fed_cond_perturbation": {
                "cos_used_vs_true": _summ(abl_disp["cos"]),
                "norm_used_minus_true": _summ(abl_disp["dnorm"]),
                "note": "measures how far the FED cond moved, NOT whether the policy uses it. "
                        "ln_g is a LayerNorm and divides the raw scale out, so a magnitude "
                        "delta overstates what the gate can see — read this next to "
                        "adaln_gate.mod_ratio / gamma_var_image.",
            },
            "adaln_gate": {k: _summ(v) for k, v in abl_gate.items()},
            "sampling": _samp,
        }

    # ---- full fixed-protocol metadata (every knob the score depends on) --------
    ck_epoch = ck_step = None
    if not args.hist_ckpt:
        try:
            _ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ck_epoch, ck_step = _ck.get("epoch"), _ck.get("step")
            del _ck
        except Exception as e:  # metadata only — never fail a rollout over this
            print(f"[eval][warn] could not read epoch/step from ckpt: {e}")
    result = {
        "protocol": "WeMM-P1 rollout (base eval_fine.py protocol) — FIXED SCENE/RNG SET",
        "train_seed": train_seed, "ckpt_path": ckpt_path,
        "ckpt_epoch": ck_epoch, "ckpt_step": ck_step,
        "n_evals": num_evals, "n_envs": num_envs,
        "eval_seeds": seeds,
        "eval_params": {
            # scenes are FIXED, not resampled: eval_seed = train_seed + global episode
            # index, and goal_idx = batch index b (0..num_batches-1) for every env in
            # that batch — identical mapping in both arms and at every milestone.
            "scene_rule": "eval_seed = train_seed + i, i in 0..n_evals-1 (never resampled)",
            "env_seeds": seeds,
            "goal_idx_rule": "goal_idx = batch_index b (per env within the batch)",
            "goal_idx_per_batch": list(range(num_evals // num_envs)),
            "action_rng_seed": args.rng_seed,
            "action_rng_applied": "after model/encoder build + ckpt load, before the loop",
            "max_steps": 302,
            "temporal_ensemble_window": act_window,
            "obs_window_size": obs_window,
            "action_window_size": act_window,
            "num_envs": num_envs, "num_batches": num_evals // num_envs,
            "encoder_online": True,
            "cond_shuffle": args.cond_shuffle,
            "cond_ablation": ablation,
            "cond_ablation_seed": (int(args.cond_ablation_seed) if ablation != "none" else None),
            "probe_sampling": bool(probe["on"]),
            "preproc": args.preproc,
        },
        "encoder": f"{cfg.enc_model_name()} (enc P={cfg.enc().n_patches}, policy P={cfg.policy_n_patches()}, D={cfg.enc().output_dim}, online)",
        "base_metrics": base,
        "ablation": _ablation_block,
        "episodes": ep_meta,
    }
    json_path = out_dir / f"rollout_seed{train_seed}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    csv_path = out_dir / f"episodes_seed{train_seed}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_meta[0].keys()))
        w.writeheader()
        w.writerows(ep_meta)

    print("\n" + "=" * 60)
    if cfg.interface_mode() == "coord":
        rep_tag = f" (Coord-P{cfg.policy_n_patches()})"
    elif cfg.policy_n_patches() != cfg.enc().n_patches:
        rep_tag = f" (Repeat-P{cfg.policy_n_patches()})"
    else:
        rep_tag = ""
    print(f"{cfg.encoder_name.upper()}-P{cfg.policy_n_patches()}{rep_tag} ROLLOUT SUMMARY")
    print("=" * 60)
    print(f"  n_evals                 {base['n_evals']}")
    print(f"  final_coverage          {base['final_coverage_mean']:.4f} ± {base['final_coverage_std']:.4f}")
    print(f"  max_coverage            {base['max_coverage_mean']:.4f} ± {base['max_coverage_std']:.4f}")
    print(f"  reward                  {base['reward_mean']:.2f}")
    print(f"  success (final>=0.95)   {p_succ:.3f}")
    print(f"  time_to_goal            {base['time_to_goal_mean'] if base['time_to_goal_mean'] is not None else 'n/a'}")
    print(f"  encoder/step            {base['encoder_ms_per_step_mean']:.0f} ms")
    print(f"  policy/step             {base['policy_ms_per_step_mean']:.1f} ms")
    print(f"  total/step              {base['total_ms_per_step_mean']:.0f} ms")
    print(f"  peak gpu mem            {base['peak_gpu_mem_mb']:.0f} MB")
    print(f"  saved {json_path}")
    print(f"  saved {csv_path}")
    if args.video:
        print(f"  videos -> {out_dir}/ep_*.mp4")


def _write_video(frames: np.ndarray, path: Path, fps: int = 10) -> None:
    """frames: [T,H,W,C] uint8 -> mp4 (imageio)."""
    import imageio.v2 as iio
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8)
    except Exception as e:  # fallback: gif
        alt = path.with_suffix(".gif")
        iio.mimsave(str(alt), frames, fps=fps)
        print(f"    [video] mp4 failed ({e}), saved {alt}")


if __name__ == "__main__":
    main()
