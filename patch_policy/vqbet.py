"""BehaviorTransformer (VQ-BeT): GPT over [B,T,P,D] visual tokens -> VQ action head.

Baseline-compatible reimplementation of the Patch Policy VQ-BeT head:
  - GPT (block-causal, dynamic P) predicts per-timestep action codes + offsets
  - VQ-VAE tokenizer fit on collected actions, then frozen
  - loss = focal(code1)*5 + focal(code2)*0.5 + offset_multiplier * L1(offset)
  - policy gradients are zeroed until the VQ tokenizer is fit (baseline protocol)
"""

from __future__ import annotations

import tqdm
from typing import Dict, Optional, Sequence, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_policy.gpt import GPT, GPTConfig
from patch_policy.vqvae import VqVae


def batch_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index x along the leading dims matching idx.shape; returns idx.shape + trailing."""
    assert idx.shape == x.shape[: len(idx.shape)]
    remaining_shape = x.shape[len(idx.shape) + 1 :]
    x = x.flatten(start_dim=idx.dim() + 1)
    indices = []
    for i in range(len(idx.shape)):
        raw = torch.arange(idx.shape[i], device=idx.device)
        shape = [1] * len(idx.shape)
        shape[i] = idx.shape[i]
        indices.append(raw.reshape(shape))
    indices.append(idx)
    result = x[tuple(indices)]
    return result.reshape(idx.shape + remaining_shape)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()
        return (-((1 - pt) ** self.gamma) * logpt).mean()


def _repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 1) -> torch.Tensor:
    pad = length - x.shape[dim]
    if pad <= 0:
        return x
    first = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * x.ndim
    repeat_shape[dim] = pad
    return torch.cat([first.repeat(*repeat_shape), x], dim=dim)


class GroupedOptimizer:
    """Step multiple optimizers together; state_dict as {i: opt.state_dict()}."""

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none)

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[i])


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Sequence[int], final_relu: bool = True):
        """bin/offset head MLP.

        final_relu=True  (default, pre-fork patch_policy): ReLU after EVERY Linear incl. the
                        last — clamps logits/offsets to >=0.
        final_relu=False (native parity): torchvision-utils.MLP semantics — ReLU only
                        between hidden layers, NO activation on the final Linear, with
                        nn.Dropout(0.0) interleaved exactly as patch_policy
                        models/vq_behavior_transformer/utils.py MLP (dropout=0 -> identity
                        in train+eval, but keeps Sequential index layout == native so a
                        native state_dict loads by prefix swap). bin logits and px offsets
                        may then be negative.
        """
        super().__init__()
        hidden = list(hidden_channels)
        if final_relu:
            layers: list[nn.Module] = []
            in_dim = in_channels
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
        else:
            # torchvision-style MLP (native utils.MLP): Linear, ReLU, Dropout(0) between
            # hidden; final Linear + Dropout(0), no final activation.
            layers = []
            in_dim = in_channels
            for h in hidden[:-1]:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.0))
                in_dim = h
            layers.append(nn.Linear(in_dim, hidden[-1]))
            layers.append(nn.Dropout(0.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BehaviorTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int = 0,
        views: int = 1,
        obs_window_size: int = 5,
        act_window_size: int = 5,
        n_patches: int = 1,
        coord_decoder: Optional[nn.Module] = None,
        adaln: Optional[nn.Module] = None,  # WeMM-DINO AdaLN conditioning (spec §3-§4); None = DINO baseline
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 512,
        dropout: float = 0.0,
        vqvae_latent_dim: int = 512,
        vqvae_n_embed: int = 16,
        vqvae_groups: int = 2,
        vqvae_iters: int = 1000,
        vqvae_batch_size: int = 1024,
        vqvae_fit_steps: Optional[int] = None,
        act_scale: float = 500.0,
        offset_loss_multiplier: float = 10.0,
        secondary_code_multiplier: float = 0.5,
        gamma: float = 2.0,
        gpt_output_dim: Optional[int] = None,  # native parity: GPT lm_head readout dim (256 for DINO-P256)
        gelu: str = "exact",                    # native parity: "tanh" = GPT-2 new_gelu
        head_final_relu: bool = True,           # native parity: False = torchvision utils.MLP (no final act)
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.obs_window_size = obs_window_size
        self.act_window_size = act_window_size
        self.unconditional = goal_dim <= 0

        self.gpt = GPT(
            GPTConfig(
                input_dim=obs_dim + goal_dim,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                block_size=obs_window_size + act_window_size,
                n_patches=n_patches * views,
                dropout=dropout,
                output_dim=gpt_output_dim,  # native parity: 256 -> lm_head readout
                gelu=gelu,
            ),
            adaln=adaln,  # WeMM-DINO residual-AdaLN conditioning; None = DINO baseline
        )

        # Global-to-Spatial Coord decoder (optional): z [N,T,1,D] -> explicit
        # spatial slots [N,T,P,out]. Its output must feed GPT's wte directly.
        self.coord_decoder = coord_decoder
        if coord_decoder is not None:
            assert coord_decoder.out_dim == obs_dim + goal_dim, (
                f"coord_decoder out_dim {coord_decoder.out_dim} must equal GPT input "
                f"obs_dim+goal_dim {obs_dim + goal_dim}"
            )

        self.vqvae = VqVae(
            input_dim_h=act_window_size,
            input_dim_w=act_dim,
            n_latent_dims=vqvae_latent_dim,
            vqvae_n_embed=vqvae_n_embed,
            vqvae_groups=vqvae_groups,
            act_scale=act_scale,
        )
        self.vqvae_optim = torch.optim.Adam(self.vqvae.parameters(), lr=1e-3, weight_decay=1e-4)
        self.vqvae_fit_steps = vqvae_fit_steps
        self.vqvae_iters = vqvae_iters
        self.vqvae_batch_size = vqvae_batch_size
        self.vqvae_is_fit = False
        self._collected_actions = []
        # Shared pre-fitted VQ tokenizer (both arms of the controlled pair load the SAME
        # one). NOT installed here: the fit stays scheduled at the end of epoch 1 exactly
        # as in the original protocol, at which point these weights replace the 1000-iter
        # fit. See load_preset_vqvae / _maybe_fit_vq.
        self._vqvae_preset: Optional[dict] = None

        self.G = vqvae_groups
        self.C = vqvae_n_embed
        self.W = act_window_size
        self.A = act_dim
        self.offset_loss_multiplier = offset_loss_multiplier
        self.secondary_code_multiplier = secondary_code_multiplier
        self._criterion = FocalLoss(gamma=gamma)

        # Head input dim = GPT readout_dim: n_embd (no lm_head) or lm_head output (256, native parity).
        head_in = self.gpt.readout_dim
        self.bin_head = MLP(head_in, [1024, 1024, self.G * self.C], final_relu=head_final_relu)
        self.offset_head = MLP(head_in, [1024, 1024, self.G * self.C * self.W * self.A], final_relu=head_final_relu)

    # ---- VQ fit ------------------------------------------------------------

    def _unpack_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        """[N, total_w, A] -> [N, obs_w, W, A] with overlapping W-windows (baseline protocol)."""
        n, total_w, a = action_seq.shape
        obs_w = total_w + 1 - self.W
        result = action_seq.new_empty(n, obs_w, self.W, a)
        for i in range(obs_w):
            result[:, i] = action_seq[:, i : i + self.W]
        return result

    def _freeze_vqvae(self) -> None:
        """Fitted VQ tokenizer is a frozen target: eval mode + no grads + no stale grads.

        1. `eval()` — the EMA codebook is only updated while the module is in train
           mode, and `get_code()` runs under `no_grad` but that does NOT stop EMA
           updates. A fitted tokenizer must never be in train mode again.
        2. `requires_grad_(False)` + `grad = None` — `_sample_action` calls
           `vqvae.get_action_from_latent` (the decoder) OUTSIDE `no_grad`, so the
           decoder used to accumulate gradients that no optimizer ever consumed
           (VQ is in neither policy optimizer). Those residuals polluted the global
           grad_norm (which summed every param with a .grad). The decoded centers are
           already detached (`draw_code_forward` is `no_grad`), so with the params
           grad-free the decoder output no longer requires grad at all and only
           policy/action-head/AdaLN grads remain in the graph.
           Forward values and the loss are bit-identical — the VQ weights do not
           change; only the accounting does.
        """
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad_(False)
            p.grad = None

    def set_vqvae_fit(self, fitted: bool) -> None:
        """Set the VQ-fit flag AND enforce the matching mode/requires_grad state.

        Every path that flips `vqvae_is_fit` to True must go through here.

        Resume bug (1): `train.py --resume` used to assign `model.vqvae_is_fit = True`
        directly on a freshly built model, which defaults to TRAIN mode. The restored
        (already fitted) VQ therefore stayed in train mode until the first `evaluate()`
        called `model.train()` — so every policy forward in between ran `get_code()`
        against a LIVE EMA codebook and drifted the codebook/EMA buffers (observed:
        A1 epoch 100 -> 105 codebook change, 105 -> 110 unchanged). Setting the flag
        through this method restores eval + grad-free immediately, before any training
        forward, and the `train()` override below keeps it that way across all later
        eval()/train() switches.
        """
        self.vqvae_is_fit = bool(fitted)
        if self.vqvae_is_fit:
            self._freeze_vqvae()

    def load_preset_vqvae(self, state_dict: dict) -> None:
        """Register the shared pre-fitted VQ tokenizer (identical for both arms).

        Deliberately does NOT set `vqvae_is_fit`: the fit stays scheduled at the end of
        epoch 1 exactly as in the original protocol (epoch-1 loss is zeroed, then
        `finish_epoch` -> `_maybe_fit_vq(force=True)`), so neither arm gains extra
        policy-update budget from reusing a shared tokenizer. At that point these
        weights are installed instead of running the 1000-iteration fit.
        """
        self._vqvae_preset = state_dict

    def _maybe_fit_vq(self, force: bool = False):
        if self.vqvae_is_fit:
            return
        if not force:
            if self.vqvae_fit_steps is None:
                return
            if len(self._collected_actions) < self.vqvae_fit_steps:
                return

        # Shared-tokenizer path: same trigger point as a real fit (epoch 1 end), but the
        # weights come from the pre-fitted artifact instead of a fresh 1000-iter fit.
        if self._vqvae_preset is not None:
            self.vqvae.load_state_dict(self._vqvae_preset)
            self.set_vqvae_fit(True)
            self._collected_actions = []
            print("[VQ] shared preset tokenizer installed (pre-fitted on the TRAIN split "
                  "only, same protocol) -> VQ frozen: eval mode, requires_grad=False")
            return

        all_actions = torch.cat(self._collected_actions)
        all_actions = einops.rearrange(all_actions, "N T W A -> (N T) (W A)")
        all_actions = torch.unique(all_actions, dim=0)
        all_actions = einops.rearrange(all_actions, "... (W A) -> ... W A", W=self.W)
        print(f"[VQ] fitting on {all_actions.shape[0]} unique action chunks")

        for _ in tqdm.trange(self.vqvae_iters, desc="VQ fit"):
            shuffle = torch.randperm(len(all_actions), device=all_actions.device)
            for i in range(0, len(all_actions), self.vqvae_batch_size):
                batch = all_actions[shuffle[i : i + self.vqvae_batch_size]]
                loss, vq_code, loss_dict = self.vqvae(batch)
                self.vqvae_optim.zero_grad()
                loss.backward()
                self.vqvae_optim.step()

        print(f"[VQ] fit done. n_codes {vq_code.shape[-1]}, "
              f"loss_dict { {k: round(v, 4) for k, v in loss_dict.items()} }")
        # Mode + freeze in ONE place: eval immediately after the fit completes (the old
        # code only called .eval() here, leaving requires_grad=True on every VQ param).
        self.set_vqvae_fit(True)
        self._collected_actions = []

    def finish_epoch(self):
        if not self.vqvae_is_fit:
            self._maybe_fit_vq(force=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vqvae_is_fit:
            # Fitted VQ is a frozen target: re-enforce eval AND grad-free on every
            # model.train()/eval() switch (validation calls model.eval()/model.train()).
            self._freeze_vqvae()

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        cond_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        if (
            action_seq is not None
            and not self.vqvae_is_fit
            and self.training
        ):
            self._collected_actions.append(self._unpack_actions(action_seq))
            self._maybe_fit_vq()

        if obs_seq.shape[1] < self.obs_window_size:
            obs_seq = _repeat_start_to_length(obs_seq, self.obs_window_size, dim=1)

        # WeMM-DINO fusion: the cond history must be padded to the SAME length as
        # obs (repeat-start, matching FusionWindowDataset's start-virtual windows).
        # Rollout histories start at 1 frame and grow; without this, a short obs
        # history is padded to T but cond stays at its actual length -> adaln broadcast
        # mismatch (gpt dim1 = obs T vs cond T).
        if cond_seq is not None and cond_seq.shape[1] < self.obs_window_size:
            cond_seq = _repeat_start_to_length(cond_seq, self.obs_window_size, dim=1)

        if self.unconditional:
            gpt_input = obs_seq
        else:
            assert goal_seq is not None
            if goal_seq.shape[1] < self.obs_window_size:
                goal_seq = _repeat_start_to_length(goal_seq, self.obs_window_size, dim=1)
            gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)

        # Global-to-Spatial Coord decoder: z [N,T,1,D] -> explicit spatial slots
        # [N,T,P,out]. The ONLY interface change for this experiment.
        if self.coord_decoder is not None:
            gpt_input = self.coord_decoder(gpt_input)

        # WeMM-DINO AdaLN conditioning: cond is the WeMM cache token [N,T,1,D_cond]
        # -> [N,T,D_cond]; passed to gpt's AdaLN (between wte and wpe). None for baseline.
        if cond_seq is not None and cond_seq.shape[-2] == 1:
            cond_seq = cond_seq.squeeze(-2)  # [N,T,1,D_cond] -> [N,T,D_cond]
        gpt_output = self.gpt(gpt_input, cond_seq)  # [N, T, n_embd]
        cbet_logits, cbet_offsets = self._forward_heads(gpt_output)
        predicted_action, decoded_action, sampled_centers, sampled_offsets = self._sample_action(
            cbet_logits, cbet_offsets
        )

        if action_seq is not None:
            loss, loss_dict = self._calc_loss(
                action_seq, predicted_action, decoded_action, sampled_centers, cbet_logits
            )
        else:
            loss, loss_dict = None, {}
        return predicted_action, loss, loss_dict

    def _forward_heads(
        self, gpt_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cbet_logits = einops.rearrange(
            self.bin_head(gpt_output), "N T (G C) -> N T G C", G=self.G
        )
        cbet_offsets = einops.rearrange(
            self.offset_head(gpt_output),
            "N T (G C W A) -> N T G C W A",
            G=self.G, C=self.C, W=self.W, A=self.A,
        )
        return cbet_logits, cbet_offsets

    def _sample_action(
        self, cbet_logits: torch.Tensor, cbet_offsets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cbet_probs = torch.softmax(cbet_logits, dim=-1)  # [N,T,G,C]
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, self.C), num_samples=1),
            "(N T G) 1 -> N T G",
            N=cbet_probs.shape[0], T=cbet_probs.shape[1], G=self.G,
        )
        centers = self.vqvae.draw_code_forward(sampled_centers)
        decoded_action = self.vqvae.get_action_from_latent(centers)  # [N,T,W,A]
        sampled_offsets = batch_idx(cbet_offsets, sampled_centers)  # [N,T,G,W,A]
        sampled_offsets = sampled_offsets.sum(dim=2)  # [N,T,W,A]
        predicted_action = decoded_action + sampled_offsets
        return predicted_action, decoded_action, sampled_centers, sampled_offsets

    def _calc_loss(
        self,
        action_seq: torch.Tensor,
        predicted_action: torch.Tensor,
        decoded_action: torch.Tensor,
        sampled_centers: torch.Tensor,
        cbet_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        action_seq = self._unpack_actions(action_seq)  # [N,T,W,A]
        _, action_bins = self.vqvae.get_code(action_seq)  # [N,T,G]
        action_bins_flat = einops.rearrange(action_bins, "N T ... -> (N T) ...")
        cbet_logits_flat = einops.rearrange(cbet_logits, "N T ... -> (N T) ...")

        offset_loss = F.l1_loss(action_seq, predicted_action)
        cbet_loss1 = self._criterion(cbet_logits_flat[:, 0], action_bins_flat[:, 0])
        cbet_loss2 = self._criterion(cbet_logits_flat[:, 1], action_bins_flat[:, 1])
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self.secondary_code_multiplier

        loss = cbet_loss + self.offset_loss_multiplier * offset_loss
        loss_dict = {
            "classification_loss": cbet_loss.detach().item(),
            "offset_loss": offset_loss.detach().item(),
            "total_loss": loss.detach().item(),
        }
        if not self.vqvae_is_fit:
            loss = loss * 0.0  # baseline: don't train policy until VQ tokenizer is fit
        return loss, loss_dict

    # ---- optimizer ---------------------------------------------------------

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas) -> GroupedOptimizer:
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)
        # scope = GPT (+ the optional Coord decoder). Same decay/no-decay semantics
        # for both: Linear weight -> decay, bias/LayerNorm/Embedding -> no decay.
        scope = {"gpt": self.gpt}
        if self.coord_decoder is not None:
            scope["coord_decoder"] = self.coord_decoder
        for prefix, mod in scope.items():
            for mn, m in mod.named_modules():
                for pn, p in m.named_parameters():
                    fpn = f"{prefix}.{mn}.{pn}" if mn else f"{prefix}.{pn}"
                    if pn.endswith("bias"):
                        no_decay.add(fpn)
                    elif p.ndim == 0:
                        # 0-dim scalar params (AdaLN-zero gate `alpha`): weight-decay-free.
                        # Inert for clean/base (they have no 0-dim params) while the assert
                        # below keeps netting genuinely-unclassified non-scalar params.
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist):
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist):
                        no_decay.add(fpn)
        param_dict = {}
        for prefix, mod in scope.items():
            for pn, p in mod.named_parameters():
                param_dict[f"{prefix}.{pn}"] = p
        assert not (decay & no_decay)
        assert set(param_dict) == (decay | no_decay)

        # 确定性 param 序：decay/no_decay 是 set，其迭代顺序依赖字符串哈希且随进程
        # (PYTHONHASHSEED) 随机。resume 时 AdamW state 按 param 组内位置索引恢复，
        # 若新进程重建 optimizer 时顺序变了会错位（film->adaln 改名即触发，见
        # train.py resume 对齐校验）。sorted() 使跨进程顺序稳定，resume 可对齐。
        opt1 = torch.optim.AdamW(
            [
                {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
                {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )
        opt1.add_param_group({"params": self.bin_head.parameters()})
        opt2 = torch.optim.AdamW(
            self.offset_head.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return GroupedOptimizer([opt1, opt2])
