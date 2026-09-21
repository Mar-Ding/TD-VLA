"""Roll one ORIGINAL-Diffusion-Policy checkpoint out on PushT, n=100.

Adapted from the upstream ~/patch_policy/eval_phase9.py (v3) so the numbers land
in the same shape as every other arm's rollout JSON. Three things differ, and
nothing else:

  1. VQ-BeT-only lines removed. eval_phase9.py does
         model.eval(); model.vqvae_is_fit = True; model._vqvae_model.eval()
     A DiffusionPolicy has no `_vqvae_model`, so those two lines would raise
     AttributeError. It also does not need them: it has no tokenizer to freeze.
  2. config_name is train_pusht_diffusion, not train_probe_bc. The config is
     where `eval_window_size` (2) and `action_window_size` (5) come from, and
     they MUST match the checkpoint's obs_horizon / pred_horizon -- a mismatch
     trips `assert T_obs <= self.obs_horizon` inside _predict.
  3. ckpt / output / tag are parameters (DP_CKPT, DP_OUT, DP_TAG) instead of
     being hardcoded to the phase9 run directory, so ep10/20/30/40 can each be
     rolled out from one driver.
  4. The action tensor is 3-D, not 4-D. Our own arms return [N, T, W, A] and the
     harness reads `action[:, -1]` -> [N, W, A]. DiffusionPolicy._predict returns
     ...[:, obs_horizon-1 : obs_horizon-1+action_horizon, :] = [N, 5, 2] -- there
     is no T axis, because there is no GPT readout horizon to index. The one
     line `a = action[:, -1] if action.dim() == 4 else action` below absorbs
     that, and it is the ONLY place the two shapes differ. Getting it wrong does
     not crash: `action[:, -1]` on a 3-D tensor yields [N, 2], and `[:, 0]` then
     yields [N] -- a 1-D action that env.step would broadcast into nonsense.

WHAT THE REPORTED NUMBER IS, AND WHAT IT IS NOT
The post-training n100 (this script) uses the same temporal ensembling as the
TD and VQ-BeT arms -- last 5 chunks kept, re-meaned and shifted one step each
tick -- so the three arms' protocols match. Upstream's own inline eval does
NOT do this: train_policy.py's `if use_diffusion:` branch executes the whole
5-step chunk open-loop before re-planning. Both are legitimate DP protocols,
but they are different protocols, so the inline `final coverage mean` printed
during training is a PROGRESS SIGNAL ONLY and must not be compared against the
numbers here or against the other arms' numbers.

WHY THE ROLLOUT LOOP IS COPIED RATHER THAN REINVENTED
The v3 seed scheme is the part that makes these numbers comparable: batch b gets
env seeds [seed + 10b .. seed + 10b + 9], so all 100 episodes have distinct
initial states. The v1/v2 bug reseeded with seed+0*10 after each batch, so
batch 1 replayed batch 0's seeds (100 episodes = 90 unique + 10 duplicates).
Re-deriving that logic is exactly how the bug would come back.

NOISE IS PINNED. torch/np/random are all seeded before the loop, and the rollout
is a plain sequential loop with no other RNG consumer, so the DDPM sampling
noise for a given checkpoint is reproducible. This matters more here than for
the other arms: DP inference is 100 denoiser steps, so an unpinned generator
would put real variance on every coverage number.

Usage (on the server, from ~/dp_orig):
  CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline MUJOCO_GL=egl \
  DP_CKPT=<...>/model_39.pt DP_TAG=ep40 DP_OUT=<...>/eval_dp_ep40.json \
  ~/miniconda3/envs/patch-policy/bin/python \
  ~/td_pusht/scripts/eval_dp_orig.py encoder=dino_patch seed=42
"""
import json
import math
import os
import sys
import time
from collections import deque

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "offline")

# `python /path/to/this.py` puts the SCRIPT's directory on sys.path[0], not the
# cwd -- so running this from ~/dp_orig still does not make `envs`, `models` and
# `datasets` importable. train_policy.py never hits this because it is run as
# `python train_policy.py` from inside the tree, which puts the tree itself on
# sys.path. Insert the tree explicitly instead of relying on the caller's cwd.
DP_ROOT = os.environ.get("DP_ROOT", os.path.expanduser("~/dp_orig"))
sys.path.insert(0, DP_ROOT)

import einops
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from envs.venv import SubprocVectorEnv

# The DP config computes pred_horizon with ${eval:'${a} + ${b} - 1'}. That is a
# custom resolver, registered by train_policy.py:34 and NOT by Hydra itself, so a
# bare @hydra.main on the same config dies with
# `UnsupportedInterpolationType: Unsupported interpolation type eval`. Registered
# here with the same call, at import time, because the decorator resolves the
# config when main() is invoked -- i.e. after this line has already run.
OmegaConf.register_new_resolver("eval", eval, replace=True)

T_START = time.perf_counter()
CKPT = os.environ.get("DP_CKPT", "")
OUT = os.environ.get("DP_OUT", "")
TAG = os.environ.get("DP_TAG", "unnamed")
ENC = os.environ.get("DP_ENC", "dino_patch")
NUM_EVALS = int(os.environ.get("DP_NUM_EVALS", "100"))
DP_CONFIGS = os.environ.get("DP_CONFIGS", os.path.expanduser("~/dp_orig/configs"))

if not CKPT or not OUT:
    sys.exit("[abort] DP_CKPT and DP_OUT must both be set")


@hydra.main(config_path=DP_CONFIGS, config_name="train_pusht_diffusion", version_base="1.2")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    import random

    # Pin every RNG before a single model forward -- see the module docstring.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"[eval] tag={TAG} enc={ENC} cuda={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    print(f"[eval] ckpt={CKPT}")
    print(f"[eval] obs_horizon={cfg.eval_window_size} action_window_size={cfg.action_window_size} "
          f"pred_horizon={cfg.pred_horizon} num_evals={NUM_EVALS}")

    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # The file holds the whole pickled nn.Module (that is what save_model writes),
    # normalizer included, so nothing has to be rebuilt or refitted here.
    model = torch.load(CKPT, map_location=device, weights_only=False).to(device)
    model.eval()
    assert hasattr(model, "noise_scheduler"), (
        "this does not look like a DiffusionPolicy checkpoint (no noise_scheduler)"
    )

    cfg.num_envs = int(os.environ.get("DP_NUM_ENVS", "10"))
    num_batches = NUM_EVALS // cfg.num_envs
    assert NUM_EVALS % cfg.num_envs == 0, "num_evals must be a multiple of num_envs"

    env = SubprocVectorEnv(
        [lambda: hydra.utils.instantiate(cfg.env.gym) for _ in range(cfg.num_envs)]
    )
    goal_fn = lambda goal_idx: torch.zeros(1)  # goal_dim=0 -> unconditional

    def embed(enc, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return einops.rearrange(enc(obs), "N V P E -> N (V P) E")

    # @torch.no_grad() is not decoration: upstream's eval_on_env carries it, and
    # without it the 100-step DDPM loop builds an autograd graph per env step.
    # Measured cost of getting this wrong is in the run log; the numbers are
    # unaffected either way (.detach() was already applied to the actions).
    @torch.no_grad()
    def rollout():
        # Accumulators live INSIDE rollout(): `acc += [...]` would otherwise make
        # them locals of this function and die with UnboundLocalError -- which is
        # exactly what happened, after a full 302-step batch had already run.
        final_covs, max_covs = [], []
        lat_ms, enc_ms, env_ms = [], [], []
        t2g_all, ep_steps, ep_rewards = [], [], []
        for goal_idx in range(num_batches):
            # v3 scheme: batch b -> seeds [seed+10b .. seed+10b+9]; 100 unique states.
            base_seed = cfg.seed + goal_idx * cfg.num_envs
            env.seed([base_seed + j for j in range(cfg.num_envs)])
            obs_stack = deque(maxlen=cfg.eval_window_size)
            this_obs = env.reset(goal_idx=goal_idx)
            obs_stack.append(embed(encoder, this_obs))
            done = np.array([False])
            action_list = []
            first_sat = [None] * cfg.num_envs
            steps_this_batch = 0
            ep_reward = np.zeros(cfg.num_envs)
            t_batch = time.perf_counter()
            while not done.all():
                obs = torch.stack(tuple(obs_stack)).float().to(device)
                obs = einops.rearrange(obs, "T N P E -> N T P E")
                goal = torch.as_tensor(goal_fn(goal_idx), dtype=torch.float32, device=device)
                goal = einops.repeat(goal, "... -> N T ...", N=cfg.num_envs, T=cfg.eval_window_size)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                action, _, _ = model(obs, goal, None)
                torch.cuda.synchronize()
                lat_ms.append((time.perf_counter() - t0) * 1000)
                # 4-D [N,T,W,A] for our arms -> take the last readout step's chunk.
                # 3-D [N,W,A] for DiffusionPolicy -> already the chunk. See docstring 4.
                a_chunk = action[:, -1] if action.dim() == 4 else action
                action_list.append(a_chunk.cpu().detach().numpy())
                if len(action_list) > cfg.action_window_size:
                    action_list = action_list[1:]
                curr_action = np.array(action_list).mean(axis=0)[:, 0]
                # temporal ensemble: shift each stored chunk one step and re-mean
                new_list = []
                for a_chunk in action_list:
                    new_list.append(np.concatenate(
                        (a_chunk[:, 1:], np.zeros((a_chunk.shape[0], 1, a_chunk.shape[-1]))), axis=1))
                action_list = new_list
                t_env = time.perf_counter()
                this_obs, reward, done, info = env.step(curr_action)
                env_ms.append((time.perf_counter() - t_env) * 1000)
                t_enc = time.perf_counter()
                obs_stack.append(embed(encoder, this_obs))
                enc_ms.append((time.perf_counter() - t_enc) * 1000)
                ep_reward += reward
                steps_this_batch += 1
                for i in range(cfg.num_envs):
                    if first_sat[i] is None and reward[i] >= 0.999999:
                        first_sat[i] = steps_this_batch
            final_covs += [info[i]["final_coverage"] for i in range(len(info))]
            max_covs += [info[i]["max_coverage"] for i in range(len(info))]
            t2g_all += [v for v in first_sat if v is not None]
            ep_steps.append(steps_this_batch)
            ep_rewards += ep_reward.tolist()
            # Per-batch progress: an n100 is 10 batches of up to 302 steps, and a
            # silent 30-minute run is indistinguishable from a hung one.
            el = time.perf_counter() - t_batch
            print(f"[eval] batch {goal_idx + 1}/{num_batches}: seeds "
                  f"[{base_seed}..{base_seed + cfg.num_envs - 1}] "
                  f"{steps_this_batch} steps, {el:.1f}s, "
                  f"final_cov mean {np.mean([info[i]['final_coverage'] for i in range(len(info))]):.4f}, "
                  f"policy {np.mean(lat_ms[-steps_this_batch:]):.0f}ms/step, "
                  f"enc {np.mean(enc_ms[-steps_this_batch:]):.0f}ms, "
                  f"env {np.mean(env_ms[-steps_this_batch:]):.0f}ms", flush=True)
        return final_covs, max_covs, lat_ms, enc_ms, env_ms, t2g_all, ep_steps, ep_rewards

    final_covs, max_covs, lat_ms, enc_ms, env_ms, t2g_all, ep_steps, ep_rewards = rollout()
    env.close()
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

    final_covs = np.array(final_covs)
    max_covs = np.array(max_covs)
    ep_rewards = np.array(ep_rewards)
    p_succ = float((final_covs >= 0.95).mean())

    def ci(mean, std, n):
        return None if n < 1 else [float(mean - 1.96 * std / math.sqrt(n)),
                                   float(mean + 1.96 * std / math.sqrt(n))]

    def ci_prop(p, n):
        se = math.sqrt(p * (1 - p) / n)
        return [float(p - 1.96 * se), float(p + 1.96 * se)]

    result = {
        "tag": TAG, "arm": "patch_policy_dp_orig", "encoder": ENC, "ckpt": CKPT,
        "n_evals": len(final_covs),
        "final_coverage_mean": float(final_covs.mean()),
        "final_coverage_std": float(final_covs.std()),
        "final_coverage_min": float(final_covs.min()),
        "final_coverage_max": float(final_covs.max()),
        "final_coverage_ci95": ci(float(final_covs.mean()), float(final_covs.std()), len(final_covs)),
        "max_coverage_mean": float(max_covs.mean()),
        "max_coverage_ci95": ci(float(max_covs.mean()), float(max_covs.std()), len(max_covs)),
        "success_rate_095": p_succ,
        "success_rate_ci95": ci_prop(p_succ, len(final_covs)),
        "reward_mean": float(ep_rewards.mean()),
        "reward_ci95": ci(float(ep_rewards.mean()), float(ep_rewards.std()), len(ep_rewards)),
        "n_episodes_reached_095": len(t2g_all),
        "time_to_goal_mean": float(np.mean(t2g_all)) if t2g_all else None,
        "episode_steps_mean": float(np.mean(ep_steps)),
        "latency_ms_per_action_mean": float(np.mean(lat_ms)),
        "latency_ms_per_action_std": float(np.std(lat_ms)),
        # The three per-step costs, separated: a DP step is DDPM-100, so the
        # policy term is the one that makes this arm's rollouts expensive, and
        # the split is what lets that be read off rather than guessed.
        "encoder_ms_per_step_mean": float(np.mean(enc_ms)),
        "env_ms_per_step_mean": float(np.mean(env_ms)),
        "total_ms_per_step_mean": float(np.mean(lat_ms) + np.mean(enc_ms) + np.mean(env_ms)),
        "rollout_wall_s": float(time.perf_counter() - T_START),
        "peak_gpu_mem_mb": float(peak_mem),
        "protocol": {
            "config_name": "train_pusht_diffusion",
            "eval_window_size": int(cfg.eval_window_size),
            "action_window_size": int(cfg.action_window_size),
            "pred_horizon": int(cfg.pred_horizon),
            "num_envs": int(cfg.num_envs),
            "seed": int(cfg.seed),
            "seeds_scheme": "v3: batch b -> [seed+10b .. seed+10b+9]",
            "noise_rng": "global torch/np/random seeded with cfg.seed before rollout",
            "inference": "DDPM 100 steps, EMA weights (upstream _predict, unmodified)",
            "action_ensemble": "temporal ensemble over the last action_window_size "
                               "chunks, shifted and re-meaned -- same as the TD/VQ-BeT arms",
            "not_this": "upstream's inline eval executes the chunk open-loop; "
                        "those coverage numbers are a progress signal, not comparable",
        },
    }
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
