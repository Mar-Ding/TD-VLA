"""WeMM-Global Temporal Action Policy — v1 (no spatial tokens).

Replaces the patch GPT with a temporal transformer over
    [state_0..state_{T-1}, action_query_0..action_query_{H-1}]
(T=5 state tokens from the frozen WeMM global embedding, H=5 learned action
query tokens, temporal PE only). Output = the H action-query hidden states
[B, H, n_embd] fed to the EXISTING VQ-BeT bin/offset heads, so the action
layout [B, T, W, A] and the whole rollout/ensemble/training protocol are
bit-for-bit the same as the patch policy.

Mask (10x10, positions 0..4 = states, 5..9 = queries):
    states : attend to states only (bidirectional)   -- encoder context
    queries: attend to ALL 5 states AND all queries  -- decoder readout
Rationale in MD/GLOBAL_POLICY_ARCHITECTURE_AUDIT.md Part II §4.

Explicitly NOT used: block-causal patch mask, patch wpe, lm_head,
coord decoder, repeat expansion, spatial/2D positional embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import tqdm

from patch_policy.config import Config
from patch_policy.vqvae import VqVae
from patch_policy.vqbet import FocalLoss, GroupedOptimizer, MLP, _repeat_start_to_length, batch_idx


def build_global_mask(n_state: int, n_query: int) -> torch.Tensor:
    """[S, S] bool mask, S = n_state + n_query.

    Row = query position, col = key position:
      * state rows attend to states only (no query leakage into obs)
      * query rows attend to everything (all 5 states + all queries)
    """
    S = n_state + n_query
    mask = torch.zeros(S, S, dtype=torch.bool)
    mask[:n_state, :n_state] = True     # states <-> states (bidirectional)
    mask[n_state:, :] = True            # queries -> all
    return mask


class GlobalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_state: int, n_query: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.register_buffer("mask", build_global_mask(n_state, n_query))
        # Attention kernel selector -- see gpt.set_attn_impl. Default "current" keeps
        # every recorded global-policy run bit-identical; "sdpa" runs the same math
        # through F.scaled_dot_product_attention with the same bool mask (True == may
        # attend) and the same dropout placement.
        self.attn_impl = "current"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.attn_impl == "sdpa":
            # is_causal=False: the "states see everything, queries see all" structure
            # lives in the mask; the kernel must not add a second causal mask.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self.mask[:T, :T],
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            att = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
            att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class GlobalBlock(nn.Module):
    """pre-LN block (same structure as gpt.Block, custom mask)."""

    def __init__(self, n_embd: int, n_head: int, n_state: int, n_query: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = GlobalSelfAttention(n_embd, n_head, n_state, n_query, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GlobalTransformer(nn.Module):
    """n_layer pre-LN blocks over the [states; queries] sequence + ln_f."""

    def __init__(self, n_embd: int, n_layer: int, n_head: int,
                 n_state: int, n_query: int, dropout: float):
        super().__init__()
        self.n_state = n_state
        self.n_query = n_query
        self.h = nn.ModuleList(
            [GlobalBlock(n_embd, n_head, n_state, n_query, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class GlobalActionPolicy(nn.Module):
    """WeMM-Global: proj(obs) -> [T state tokens + H query tokens] -> H_action -> VQ-BeT heads.

    obs [B, T, P, D] (P=1 global token per frame; P>1 mean-pooled first) ->
    predicted_action [B, T, W, A], loss, loss_dict. Action protocol identical
    to BehaviorTransformer (VQ fit, focal 5/0.5 + 10*L1, multinomial sampling).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        p = cfg.policy
        g = cfg.global_policy
        enc = cfg.enc()
        rep_dim = enc.output_dim
        T = cfg.obs_window_size
        assert g.n_query == T, (
            f"global_policy.n_query {g.n_query} must equal obs_window_size {T} "
            "(one action query per timestep chunk)"
        )
        self.obs_window_size = T
        self.act_window_size = cfg.action_window_size
        self.rep_dim = rep_dim
        self.n_embd = g.n_embd

        # ---- representation -> temporal state tokens (only interface by rep) ----
        self.proj = nn.Linear(rep_dim, g.n_embd)        # rep-specific: WeMM 2048 / DINO 384 / Oracle 6
        self.ln_in = nn.LayerNorm(g.n_embd)             # shared input LN (stabilizes heterogeneous reps)
        self.time_pe = nn.Embedding(T, g.n_embd)        # temporal PE [5,512] — ONLY positional encoding
        nn.init.normal_(self.time_pe.weight, 0.0, 0.02)
        self.queries = nn.Parameter(torch.randn(g.n_query, g.n_embd) * 0.02)  # H distinct learned queries
        self.backbone = GlobalTransformer(
            n_embd=g.n_embd, n_layer=g.n_layer, n_head=g.n_head,
            n_state=T, n_query=g.n_query, dropout=g.dropout,
        )

        # ---- VQ-BeT action machinery (verbatim protocol from vqbet.py) --------
        self.vqvae = VqVae(
            input_dim_h=cfg.action_window_size,
            input_dim_w=2,                     # push-T act_dim
            n_latent_dims=p.vqvae_latent_dim,
            vqvae_n_embed=p.vqvae_n_embed,
            vqvae_groups=p.vqvae_groups,
            act_scale=p.act_scale,
        )
        self.vqvae_optim = torch.optim.Adam(self.vqvae.parameters(), lr=1e-3, weight_decay=1e-4)
        self.vqvae_fit_steps = p.vqvae_fit_steps
        self.vqvae_iters = p.vqvae_iters
        self.vqvae_batch_size = p.vqvae_batch_size
        self.vqvae_is_fit = False
        self._collected_actions = []

        self.G = p.vqvae_groups
        self.C = p.vqvae_n_embed
        self.W = cfg.action_window_size
        self.A = 2
        self.offset_loss_multiplier = p.offset_loss_multiplier
        self.secondary_code_multiplier = p.secondary_code_multiplier
        self._criterion = FocalLoss(gamma=p.gamma)

        self.bin_head = MLP(g.n_embd, [1024, 1024, self.G * self.C])
        self.offset_head = MLP(g.n_embd, [1024, 1024, self.G * self.C * self.W * self.A])

    # ---- VQ fit (verbatim vqbet.py protocol) --------------------------------

    def _unpack_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        n, total_w, a = action_seq.shape
        obs_w = total_w + 1 - self.W
        result = action_seq.new_empty(n, obs_w, self.W, a)
        for i in range(obs_w):
            result[:, i] = action_seq[:, i : i + self.W]
        return result

    def _maybe_fit_vq(self, force: bool = False):
        if self.vqvae_is_fit:
            return
        if not force:
            if self.vqvae_fit_steps is None:
                return
            if len(self._collected_actions) < self.vqvae_fit_steps:
                return

        all_actions = torch.cat(self._collected_actions)
        import einops
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

        self.vqvae.eval()
        print(f"[VQ] fit done. n_codes {vq_code.shape[-1]}, "
              f"loss_dict { {k: round(v, 4) for k, v in loss_dict.items()} }")
        self.vqvae_is_fit = True
        self._collected_actions = []

    def finish_epoch(self):
        if not self.vqvae_is_fit:
            self._maybe_fit_vq(force=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vqvae_is_fit:
            self.vqvae.eval()

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: torch.Tensor | None,
        action_seq: torch.Tensor | None,
    ):
        if action_seq is not None and not self.vqvae_is_fit and self.training:
            self._collected_actions.append(self._unpack_actions(action_seq))
            self._maybe_fit_vq()

        B, T, P, D = obs_seq.shape
        if T < self.obs_window_size:
            # rollout: history stack shorter than T=5 at episode start -> repeat-first
            # (identical to BehaviorTransformer._repeat_start_to_length; no-op in training)
            obs_seq = _repeat_start_to_length(obs_seq, self.obs_window_size, dim=1)
            T = self.obs_window_size
        if P > 1:
            # defensive: if a native P256 DINO cache is fed, exact mean-pool over P
            obs_seq = obs_seq.mean(dim=-2, keepdim=True)
        g = obs_seq.squeeze(-2)                                   # [B,T,D]
        s = self.ln_in(self.proj(g))                              # [B,T,n_embd]
        pos = torch.arange(T, device=s.device)
        s = s + self.time_pe(pos)                                 # temporal PE (states only)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)           # [B,H,n_embd]
        x = torch.cat([s, q], dim=1)                              # [B, T+H, n_embd]
        x = self.backbone(x)                                      # [B, T+H, n_embd]
        H_action = x[:, T:]                                       # query hidden states [B,H,n_embd]

        cbet_logits, cbet_offsets = self._forward_heads(H_action)
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

    def _forward_heads(self, gpt_output: torch.Tensor):
        cbet_logits = torch.reshape(
            self.bin_head(gpt_output), (*gpt_output.shape[:2], self.G, self.C)
        )
        cbet_offsets = torch.reshape(
            self.offset_head(gpt_output),
            (*gpt_output.shape[:2], self.G, self.C, self.W, self.A),
        )
        return cbet_logits, cbet_offsets

    def _sample_action(self, cbet_logits: torch.Tensor, cbet_offsets: torch.Tensor):
        cbet_probs = torch.softmax(cbet_logits, dim=-1)  # [N,T,G,C]
        sampled_centers = torch.reshape(
            torch.multinomial(cbet_probs.view(-1, self.C), num_samples=1),
            (cbet_probs.shape[0], cbet_probs.shape[1], self.G),
        )
        centers = self.vqvae.draw_code_forward(sampled_centers)
        decoded_action = self.vqvae.get_action_from_latent(centers)  # [N,T,W,A]
        sampled_offsets = batch_idx(cbet_offsets, sampled_centers)   # [N,T,G,W,A]
        sampled_offsets = sampled_offsets.sum(dim=2)                 # [N,T,W,A]
        predicted_action = decoded_action + sampled_offsets
        return predicted_action, decoded_action, sampled_centers, sampled_offsets

    def _calc_loss(self, action_seq, predicted_action, decoded_action, sampled_centers, cbet_logits):
        action_seq = self._unpack_actions(action_seq)  # [N,T,W,A]
        _, action_bins = self.vqvae.get_code(action_seq)  # [N,T,G]
        action_bins_flat = torch.reshape(action_bins, (-1, self.G))
        cbet_logits_flat = torch.reshape(cbet_logits, (-1, self.G, self.C))

        offset_loss = torch.nn.functional.l1_loss(action_seq, predicted_action)
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

    # ---- optimizer (same grouping semantics as vqbet.py) --------------------

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas):
        # module-based classification (mirror vqbet.py): Linear weight -> decay,
        # bias / LayerNorm / Embedding weight -> no decay; bare query vectors no decay.
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)
        skip = ("bin_head.", "offset_head.", "vqvae.")
        scope = {"proj": self.proj, "ln_in": self.ln_in, "time_pe": self.time_pe,
                 "backbone": self.backbone}
        for prefix, mod in scope.items():
            for mn, m in mod.named_modules():
                for pn, p in m.named_parameters():
                    fpn = f"{prefix}.{mn}.{pn}" if mn else f"{prefix}.{pn}"
                    if pn.endswith("bias"):
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist):
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist):
                        no_decay.add(fpn)
        no_decay.add("queries")  # bare learned query vectors (like an embedding)

        param_dict = {}
        for pn, p in self.named_parameters():
            if any(pn.startswith(s) for s in skip):
                continue
            param_dict[pn] = p
        assert not (decay & no_decay)
        assert set(param_dict) == (decay | no_decay), (
            f"unclassified params: {set(param_dict) - (decay | no_decay)}"
        )

        opt1 = torch.optim.AdamW(
            [
                {"params": [param_dict[pn] for pn in decay], "weight_decay": weight_decay},
                {"params": [param_dict[pn] for pn in no_decay], "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )
        opt1.add_param_group({"params": self.bin_head.parameters()})
        opt2 = torch.optim.AdamW(
            self.offset_head.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return GroupedOptimizer([opt1, opt2])
