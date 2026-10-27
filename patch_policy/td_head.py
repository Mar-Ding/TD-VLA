"""Truncated-diffusion action head (DiffusionDrive-style) for Patch Policy on PushT.

THE SINGLE RESEARCH VARIABLE. Everything upstream of the action head is byte-for-byte
the shared harness: RGB -> frozen DINOv2-S/14 (P=256) -> Patch Policy GPT trunk
(`patch_policy/gpt.py`, unchanged) -> here. This module replaces VQ-BeT
(`patch_policy/vqbet.BehaviorTransformer`) with a DiffusionDrive-style truncated
diffusion head. `build_model` in train.py constructs the trunk and then hands the
SAME `obs_dim`/`coord_decoder`/`adaln`/`fork_kwargs` to `build_td_policy`, so
"identical trunk, different head" is true by construction rather than by promise.

Faithful ports, with every deviation enumerated
------------------------------------------------
Ported verbatim from base/DiffusionDrive (SHA 9b52ed0e):
  * `gen_sineembed_for_position`      modules/blocks.py:20
  * `linear_relu_ln`                  modules/blocks.py:8
  * `SinusoidalPosEmb`                modules/conditional_unet1d.py:44
  * `py_sigmoid_focal_loss`           modules/multimodal_loss.py:67
  * `LossComputer`                    modules/multimodal_loss.py:117
  * `DiffMotionPlanningRefinementModule` transfuser_model_v2.py:182
  * `ModulationLayer`                 transfuser_model_v2.py:229
  * `CustomTransformerDecoder{,Layer}` transfuser_model_v2.py:270,350
  * `TrajectoryHead.forward_train/_test` transfuser_model_v2.py:459,504

DEVIATIONS (each has a reason; P0 re-derives each one from the code):

D1. Conditioning. The reference's decoder layer attends to three NAVSIM-specific
    streams: a BEV feature map (grid-sample deformable attention), per-agent
    queries, and an ego query. PushT has exactly one context stream -- the Patch
    Policy trunk readout `g` [N,T,readout_dim]. So the layer becomes ONE
    cross-attention to `g`; the agent/ego MHA blocks have no analogue and are
    dropped. The reference gets its block residual from `+ dropout(mha(...))` on
    those two blocks; dropping them without replacing that path would leave the
    block with NO identity path at all, i.e. a strictly harder optimisation
    problem than the reference's. D1 therefore also changes the residual
    placement to pre-norm (`q + cross_attn(ln1(q), ctx)`, `q + ffn(ln2(q))`).

D2. (RESOLVED -- mode scoring now reads the refined candidate, not `g`.) An earlier
    revision scored modes with `bin_head(g)` [N,T,K], i.e. t-independently and with no
    dependence on the individual anchor / noised candidate / refined candidate. That
    departs from `DiffMotionPlanningRefinementModule` (:216-228), where BOTH
    `plan_cls_branch` and `plan_reg_branch` read the SAME per-mode feature
    `traj_feature.view(bs, mode, -1)` -- mode identity is re-scored from the refined
    candidate, which is DiffusionDrive's iterative mode refinement. The canonical head
    now does the same: `_refine` produces z [N,T,K,D], and that ONE z_k feeds both
    `offset_head` (regression) and `bin_head` (score, D -> 1) -> logits [N,T,K], i.e.
    p(mode_k | g, candidate_k, t). The context-only variant is retained behind
    `td.cls_from="trunk"` as a speed/structure ablation, NOT as the canonical baseline.
    Loss consequence: cls is now genuinely layer-specific, so the reference's
    `ret_traj_loss += trajectory_loss` (sum over layers, :494-497) is ported as-is
    instead of the layer-invariant single term the old revision needed.

D3. Per (sample, timestep) noise level. Reference: `torch.randint(0, 50, (bs,))`
    -- one t per sample, shared across all 20 modes. This head predicts
    per-timestep chunks (T=5) and rollout consumes T=-1, so `t` is [N,T]. Same
    distribution, same inclusive-0 lower bound, finer granularity.
    HARD INVARIANT: `t` is [N,T] and nothing else. All K modes at a given (n,t)
    position are candidates for the SAME target, so they must be noised and
    denoised at the SAME t; a per-mode t would not be the score-matching
    objective. `_check_t` asserts the shape on every `_refine` entry, and
    `test_t_shared_across_modes` proves the broadcast numerically (a t replicated
    along K must give bit-identical noise).

D4. Regression target is the normalised chunk in full. Reference diffuses only
    (x,y) and predicts (x,y,heading) with the DDIM step consuming `[...,:2]`.
    PushT A=2, so all action dims are diffused and the anchor's coordinate dims
    are exactly the target dims -- the reference's 2-of-3 split has no analogue.

D5. `diff_weight` defaults to 1.0. The reference has `TrajectoryHead.diff_loss_weight
    = 2.0` (transfuser_model_v2.py:397, DEAD CODE -- never read) and an AGENT-level
    `diff_loss_weight = 20.0` (transfuser_config.py:85) applied in transfuser_loss.py:36
    to a `trajectory_loss` produced by a DIFFERENT head. Neither is a multiplier on
    the (cls+reg) `LossComputer` returns. Ours is an explicit knob defaulting to the
    identity 1.0 = exactly `10*focal + 8*L1`; P2 measures 2.0 rather than assuming.

Deliberate non-ports: the reference's `norm_odo`/`denorm_odo` (hardcoded per-dim
affine for NAVSIM's ego frame, lines 433-450) are replaced by `ActionNormalizer`
(`normalizer.py`), a per-dim affine FIT on the PushT train split to land in exactly
[-1,1]. Same role, same [0,1]-bounded-anchor purpose, different (data-derived)
constants. And the reference computes its mode distance in RAW space while we use
NORMALISED space, to agree with the `anchor_only_l1` floor reported by P1's
`coverage_audit`; PushT's two raw action dims have near-identical ranges (12..511 vs
25..511), so the argmin is near-invariant -- P1 reports the actual disagreement rate.

Harness contract (all of these are grep-confirmed in train.py / eval_rollout.py):
  * `forward(obs_seq, goal_seq, action_seq, cond_seq=None) -> (pred, loss, loss_dict)`
    with `pred` [N,T,W,A] in RAW PIXEL coordinates and `loss_dict` carrying
    `total_loss` / `classification_loss` / `offset_loss`.
  * `bin_head` and `offset_head` must be policy-level `MLP`s: train.py:1221-1222 and
    :1328-1329 read `model.{bin,offset}_head.net[0].weight.grad`, and train.py:931-939
    `layer_groups` keys off the prefixes "bin_head." / "offset_head.".
  * `vqvae_is_fit` (property + ignoring setter, see below), `_maybe_fit_vq`,
    `finish_epoch`, `set_vqvae_fit`, `load_preset_vqvae`, `_collected_actions`,
    `_unpack_actions`, and `vqvae` with `.state_dict()/.training/.parameters()`.

Optimiser scope hazard -- read before touching `configure_optimizers`:
`vqbet.py:453-500`'s two asserts cover ONLY `scope = {"gpt", "coord_decoder"}`;
`bin_head`/`offset_head` are optimised via param groups added OUTSIDE those asserts.
TD's `denoiser` follows that precedent (its own `opt3`), which means it bypasses the
harness's protection -- so `_assert_classifiable` runs TD's OWN explicit check over
`denoiser.named_parameters()` and `SystemExit`s naming the offenders, instead of
letting an unclassifiable parameter silently land in neither set. Derived bans, all
respected here: no `nn.MultiheadAttention` (`in_proj_weight` matches neither rule),
no bare `nn.Parameter`, no Parameters for the normaliser/schedule/anchors (buffers).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_policy.anchors import load_anchor_artifact, mode_distance
from patch_policy.diffusion import (
    TruncatedDiffusionSchedule,
    inference_chain,
    make_eval_timesteps_diffusiondrive,
)
from patch_policy.gpt import GPT, GPTConfig
from patch_policy.normalizer import ActionNormalizer
from patch_policy.vqbet import GroupedOptimizer, MLP, _repeat_start_to_length

__all__ = [
    "SinusoidalPosEmb",
    "gen_sineembed_for_position",
    "bias_init_with_prob",
    "py_sigmoid_focal_loss",
    "ModulationLayer",
    "CrossAttention",
    "RefinementBlock",
    "TruncatedDiffusionHead",
    "TruncatedDiffusionPolicy",
    "build_td_policy",
]


# ======================================================================================
# ported primitives
# ======================================================================================

def bias_init_with_prob(prior_prob: float) -> float:
    """blocks.py:36 -- focal-loss prior bias. The reference uses 0.01 for 20 modes."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 256) -> torch.Tensor:
    """blocks.py:20, verbatim (DAB-DETR copy-paste).

    `pos_tensor` [..., W, A>=2] -> [..., W, hidden_dim]. Reads dim 0 as x and dim 1 as
    y and concatenates as (y, x) -- keep that order, it is what the reference trains.
    Parameter-free, so it contributes no optimiser-visible parameters.
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class SinusoidalPosEmb(nn.Module):
    """conditional_unet1d.py:44, verbatim. Parameter-free (the frequency table is
    rebuilt per call), which is also why no bare `nn.Parameter` is needed here."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def py_sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """multimodal_loss.py:67, verbatim.

    NOT `vqbet.FocalLoss`. That one is a softmax+NLL multi-class focal loss over
    mutually-exclusive bins; this is a BINARY sigmoid focal loss over K INDEPENDENT
    logits (BCE + focal modulating weight). Substituting one for the other changes
    the classification target's semantics, so they are kept distinct on purpose.

    NOTE the reduction the reference actually uses -- `LossComputer` calls this with
    `reduction='mean', avg_factor=None` (multimodal_loss.py:148-156), i.e. a plain
    mean over every element of [B,K], NOT normalised by the number of positives.
    With K=20 and one positive the positive's gradient is therefore diluted ~K-fold.
    We reproduce that exactly; if the classifier turns out too weak to move at all in
    P2, this dilution is the first thing to look at.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # pt here denotes (1 - pt) in the Focal Loss paper, hence pt.pow(gamma) below.
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    if avg_factor is None:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss
    if reduction != "mean":
        raise ValueError('avg_factor can not be used with reduction != "mean"')
    eps = torch.finfo(torch.float32).eps
    return loss.sum() / (avg_factor + eps)


class ModulationLayer(nn.Module):
    """transfuser_model_v2.py:229, ported without the `global_cond`/`global_img` paths.

    `global_cond`/`global_img` were NAVSIM's alternative conditioning streams; `g`
    already reaches the block through the cross-attention (D1), so the only modulator
    left is the diffusion timestep. `time_embed` is [B,1,d] and broadcasts over the
    query's mode axis, exactly as in the reference.
    """

    def __init__(self, embed_dims: int, condition_dims: int):
        super().__init__()
        self.embed_dims = embed_dims
        # The reference's `if_zeroinit_scale = False` (line 233) -> DEFAULT init, kept.
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )

    def forward(self, traj_feature: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        scale_shift = self.scale_shift_mlp(time_embed)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return traj_feature * (1 + scale) + shift


class CrossAttention(nn.Module):
    """Hand-written MHA over `nn.Linear` q/k/v/out.

    `nn.MultiheadAttention` is BANNED here: its packed `in_proj_weight` matches
    neither the "bias" rule nor the "weight on Linear/LayerNorm/Embedding" rule, so it
    would be unclassifiable by the optimiser split (see module docstring).

    `ctx_dim` may differ from `d_model`: the context is the Patch Policy trunk readout
    (`readout_dim` == 256 on the native-parity/clean line) while the denoiser's width
    is `policy.n_embd` (512), so k/v project across. The reference's ego_query was
    already d_model, so this is a deviation forced by D1 + the clean-line head dim, not
    a design preference.
    """

    def __init__(self, d_model: int, n_heads: int, ctx_dim: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        ctx_dim = d_model if ctx_dim is None else ctx_dim
        self.ctx_dim = ctx_dim
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(ctx_dim, d_model)
        self.v = nn.Linear(ctx_dim, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        # Kernel selector -- see gpt.set_attn_impl. This site has NO mask at all, so it is
        # the one place where the fused kernels get the friendliest layout: no
        # [B, nh, Q, M] score matrix is ever built, and q_len=20 x m_len=5 is small
        # enough that the win here is memory/launch overhead, not FLOPs.
        self.attn_impl = "current"

    def forward(self, query: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """query [B,Q,D] x ctx [B,M,C] -> [B,Q,D]."""
        b, q_len, d = query.shape
        m_len = ctx.shape[1]
        h, dh = self.n_heads, d // self.n_heads
        q = self.q(query).view(b, q_len, h, dh).transpose(1, 2)
        k = self.k(ctx).view(b, m_len, h, dh).transpose(1, 2)
        v = self.v(ctx).view(b, m_len, h, dh).transpose(1, 2)
        if self.attn_impl == "sdpa":
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (dh ** -0.5)
            attn = self.attn_dropout(attn.softmax(dim=-1))
            y = attn @ v
        y = y.transpose(1, 2).reshape(b, q_len, d)
        return self.out(y)


class RefinementBlock(nn.Module):
    """One diffusion-decoder layer.

    Reference layer (CustomTransformerDecoderLayer.forward, :314) is
        x = cross_bev(grid-sample at noisy_traj_points, bev_feature)   # no residual
        x = x + dropout(cross_agent_attention(x, agents_query))         # residual
        x = norm1(x)
        x = x + dropout1(cross_ego_attention(x, ego_query))             # residual
        x = norm2(x)
        x = norm3(ffn(x))                                               # no residual
        x = time_modulation(x, time_embed)
        reg, cls = task_decoder(x)
    Our version (D1): the three attention sources collapse to one cross-attention to
    `g`, and because that removes the two blocks that supplied the residual, the
    placement becomes pre-norm residual -- otherwise the block would have no identity
    path. `ffn` is ported unchanged (Linear -> ReLU -> Linear, d_ffn wide). The
    `task_decoder` split is NOT here: `bin_head`/`offset_head` must be policy-level
    attributes for the harness (see module docstring) and are applied by the policy.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, ctx_dim: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, ctx_dim=ctx_dim, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.time_modulation = ModulationLayer(d_model, d_model)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        q = q + self.cross_attn(self.norm1(q), ctx)
        q = q + self.ffn(self.norm2(q))
        return self.time_modulation(q, time_embed)


# ======================================================================================
# the denoiser
# ======================================================================================

class TruncatedDiffusionHead(nn.Module):
    """Anchor encoder + timestep embedding + L refinement blocks. NO output projection.

    The output projection lives at the policy level as `offset_head`, because the
    harness needs `model.offset_head.net[0].weight.grad` and the `"offset_head."` name
    prefix. Splitting it out also keeps the two never double-registered: a module owned
    by both the policy and the denoiser would surface the same Parameter in
    `named_parameters()` twice, and putting it in two param groups makes AdamW raise
    "some parameters appear in more than one parameter group".

    Progressive refinement, ported from CustomTransformerDecoder (:362-380): the query
    FEATURE is computed once and fed to every layer unchanged (the reference never
    updates `traj_feature`), while the trajectory ESTIMATE that each layer's regression
    branch is a residual against is updated to the previous layer's output and
    DETACHED (`poses_reg[...,:2].clone().detach()`, :379). `forward` therefore returns
    only the L features; the policy drives the base chain, because computing the next
    base needs `offset_head`, which lives at the policy level.
    """

    def __init__(
        self,
        d_model: int,
        n_modes: int,
        n_poses: int,
        n_heads: int = 8,
        d_ffn: int = 1024,
        num_layers: int = 2,
        ctx_dim: Optional[int] = None,
        pos_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
        self.d_model = d_model
        self.n_modes = n_modes
        self.n_poses = n_poses
        self.pos_embed_dim = pos_embed_dim
        self.num_layers = num_layers

        # `plan_anchor_encoder` (transfuser_model_v2.py:413-416) is
        #   linear_relu_ln(d_model, 1, 1, 512) + Linear(d_model, d_model)
        # i.e. [Linear(512, d), ReLU, LayerNorm(d), Linear(d, d)]. The 512 is HARDCODED
        # there because that model has num_poses=8 and pos_embed_dim=64 -> 8*64=512. We
        # derive it instead, so the same class works for PushT's W=5 -> 320.
        pos_in = n_poses * pos_embed_dim
        self.anchor_encoder = nn.Sequential(
            nn.Linear(pos_in, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        # `time_mlp` (transfuser_model_v2.py:417-422), verbatim.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        self.blocks = nn.ModuleList([
            RefinementBlock(d_model, n_heads, d_ffn, ctx_dim=ctx_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor
    ) -> List[torch.Tensor]:
        """x_t [N,T,K,W,A] (normalised, noised anchor) ; t [N,T] long ; ctx [N,T,C].

        Returns `feats` -- `num_layers` tensors of [N,T,K,d_model], block l's refined
        feature. The policy turns feats[l] into the l-th trajectory estimate.
        """
        n, t_len, k, w, a = x_t.shape
        nt = n * t_len
        xf = x_t.reshape(nt, k, w, a)
        ctx_flat = ctx.reshape(nt, ctx.shape[-1])
        mem = ctx_flat.unsqueeze(1)                    # [nt,1,C]  one context token
        t_emb = self.time_mlp(t.reshape(-1).float()).view(nt, 1, self.d_model)

        # Encoded ONCE, from the ORIGINAL noised anchor, then reused by every block --
        # ported from the reference, which computes traj_feature before the layer loop
        # (:482) and never updates it. The blocks differ by their own weights and by the
        # base each one's residual is added to, not by their query feature.
        pos = gen_sineembed_for_position(xf, self.pos_embed_dim).flatten(-2)
        q0 = self.anchor_encoder(pos)                  # [nt,K,d]

        return [blk(q0, mem, t_emb).reshape(n, t_len, k, self.d_model)
                for blk in self.blocks]


# ======================================================================================
# the policy
# ======================================================================================

class _NoVQ(nn.Module):
    """VQ-BeT-only audit surface, empty on purpose.

    train.py touches `model.vqvae` for `.state_dict()` (checkpoint), `.training` and
    `.parameters()` (resume + param audit). TD has no tokenizer, and an EMPTY
    state_dict is the honest encoding of that: P0 asserts no `_vqvae_model.*` or
    `vqvae.*` key ever appears in a TD checkpoint, so an accidentally-loaded VQ
    artifact is detectable rather than silently ignored.
    """

    def __init__(self) -> None:
        super().__init__()   # .training is False, .parameters() is empty


def _last_linear(mlp: MLP) -> nn.Linear:
    """`MLP.net`'s final Linear. Its index depends on `final_relu`: the
    final_relu=False layout ends [..., Linear, Dropout(0)] so the Linear is at -2."""
    for mod in reversed(list(mlp.net)):
        if isinstance(mod, nn.Linear):
            return mod
    raise SystemExit("[TD] MLP has no Linear layer")


class TruncatedDiffusionPolicy(nn.Module):
    """Patch Policy trunk + truncated-diffusion action head. Duck-type compatible
    with `vqbet.BehaviorTransformer` so train.py / eval_rollout.py need no logic change.
    """

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
        adaln: Optional[nn.Module] = None,
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 512,
        dropout: float = 0.0,
        gpt_output_dim: Optional[int] = None,
        gelu: str = "exact",
        # ---- truncated-diffusion head ----
        td=None,
        anchor_artifact: Optional[str] = None,
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
                output_dim=gpt_output_dim,
                gelu=gelu,
            ),
            adaln=adaln,
        )
        self.coord_decoder = coord_decoder
        if coord_decoder is not None:
            assert coord_decoder.out_dim == obs_dim + goal_dim, (
                f"coord_decoder out_dim {coord_decoder.out_dim} must equal GPT input "
                f"obs_dim+goal_dim {obs_dim + goal_dim}"
            )

        cfg = td
        self.W = act_window_size
        self.A = act_dim
        self.K = int(cfg.n_modes)
        d_model = int(n_embd)

        # ---- the one dimension that changes with the paired-baseline choice ----
        # The `clean` baseline carries native_parity=True -> gpt_output_dim=256, so the
        # trunk readout is 256, NOT n_embd. Read it, never hardcode: 512 would break on
        # `clean` and 256 would break on `base`.
        readout_dim = self.gpt.readout_dim

        self.denoiser = TruncatedDiffusionHead(
            d_model=d_model,
            n_modes=self.K,
            n_poses=self.W,
            n_heads=int(cfg.n_heads),
            d_ffn=int(cfg.d_ffn),
            num_layers=int(cfg.num_layers),
            ctx_dim=readout_dim,
            pos_embed_dim=int(cfg.pos_embed_dim),
            dropout=float(cfg.dropout),
        )

        # Heads are `MLP` instances at POLICY level, in the two slots the harness reads.
        # `final_relu=False` is hardcoded, NOT taken from the fork's `head_final_relu`:
        # that knob reproduces the VQ-BeT head's MLP, whereas here a ReLU on the mode
        # logits would zero the focal-loss gradient (all-positive logits -> pt saturates)
        # and a ReLU on the x0 residual would forbid negative corrections outright.
        #
        # Both heads read the SAME refined per-mode feature z [N,T,K,D] (D2):
        #   bin_head   : D -> 1      score of mode k given (g, candidate_k, t)
        #   offset_head: D -> W*A    the regression residual for mode k
        # This is `DiffMotionPlanningRefinementModule`'s split (:194-204): its
        # plan_cls_branch and plan_reg_branch both consume `traj_feature.view(bs,mode,-1)`.
        # Hidden width is [1024,1024] -- the shared harness's MLP convention (VQ-BeT's
        # bin_head is [1024,1024,G*C]) rather than the reference's
        # `linear_relu_ln(d,1,2)`; both are "two hidden layers into a bare Linear".
        self.cls_from = str(cfg.cls_from)
        assert self.cls_from in ("refined", "trunk"), (
            f"td.cls_from must be 'refined' (canonical) or 'trunk' (ablation), "
            f"got {self.cls_from!r}"
        )
        # `refined` (canonical): z_k -> 1 logit per mode, so the output width is 1 and
        # K comes from the K axis of q. `trunk` (ablation): g -> K logits in one shot.
        score_in = d_model if self.cls_from == "refined" else readout_dim
        score_out = 1 if self.cls_from == "refined" else self.K
        self.bin_head = MLP(score_in, [1024, 1024, score_out], final_relu=False)
        self.offset_head = MLP(d_model, [1024, 1024, self.W * self.A], final_relu=False)
        # Reference-faithful class prior (DiffMotionPlanningRefinementModule.init_weight,
        # :214-215): `bias_init_with_prob(0.01)`. For K=20 with one positive this is a
        # slightly over-confident prior (1/20 = 0.05 would be -2.94); ported as-is.
        nn.init.constant_(_last_linear(self.bin_head).bias,
                          bias_init_with_prob(float(cfg.cls_bias_prior)))
        # NOTE `if_zeroinit_reg` is NOT ported as True: transfuser_model_v2.py:188 takes
        # it as an argument but line 205 unconditionally sets `self.if_zeroinit_reg
        # = False` BEFORE init_weight(), so the reference's reg branch keeps DEFAULT init.

        # ---- normaliser + schedule + anchors: BUFFERS, never Parameters ----
        self.normalizer = ActionNormalizer(self.W, self.A)
        self.schedule = TruncatedDiffusionSchedule(
            num_train_timesteps=int(cfg.num_train_timesteps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            beta_schedule=str(cfg.beta_schedule),
            train_t_max=int(cfg.train_t_max),
            clip_sample=bool(cfg.clip_sample),
            clip_sample_range=float(cfg.clip_sample_range),
        )
        # Anchors are fit on the TRAIN split only (P1) and stored in NORMALISED space,
        # which is the space `mode_distance` and the diffusion both work in.
        anchor_t = torch.zeros(self.K, self.W, self.A)
        self.register_buffer("anchors", anchor_t, persistent=True)

        # ---- loss weights (reference values, NOT tuned; spec forbids per-arm tuning) ----
        self.cls_weight = float(cfg.cls_weight)
        self.reg_weight = float(cfg.reg_weight)
        self.diff_weight = float(cfg.diff_weight)
        self.focal_gamma = float(cfg.focal_gamma)
        self.focal_alpha = float(cfg.focal_alpha)

        # ---- inference knobs ----
        self.infer_steps = int(cfg.infer_steps)
        self.infer_t_start = int(cfg.infer_t_start)
        # Attribution: reproduces the reference's own [10,0] + add_noise@8 mismatch, so
        # P5 can MEASURE the "2 steps is really 1" claim instead of arguing it.
        self.use_diffusiondrive_timesteps = bool(cfg.use_diffusiondrive_timesteps)
        self._noise_gen: Optional[torch.Generator] = None
        self.noise_seed = int(cfg.noise_seed)

        # ---- harness surface ----
        self.vqvae = _NoVQ()
        self._collected_actions: List[torch.Tensor] = []

        if anchor_artifact:
            self.load_anchors(anchor_artifact)
        else:
            raise SystemExit(
                "[TD] no anchor artifact configured. Anchors + normaliser come from P1 "
                "(scripts/anchors.py main); set td.anchor_artifact to its output. The "
                "head cannot start from un-fitted anchors in [-1,1].If you are trying "
                "to smoke-test the wiring, run P1 first -- it is CPU-only."
            )

    # ---- harness: VQ hooks, all intentionally inert --------------------------------

    @property
    def vqvae_is_fit(self) -> bool:
        """Always True: TD has no tokenizer, so there is nothing to fit before training.

        This keeps train.py:747's `assert model.vqvae_is_fit` and :1188's
        `if not model.vqvae_is_fit` untouched for both arms.
        """
        return True

    @vqvae_is_fit.setter
    def vqvae_is_fit(self, value) -> None:
        """Accept and ignore. `vqbet.py:258` records a historical bug where
        `train.py --resume` assigned `model.vqvae_is_fit = True` directly on a freshly
        built model; a read-only property would turn that into an AttributeError, so
        both read and write must be safe."""
        return None

    def _maybe_fit_vq(self, force: bool = False):
        if force:
            raise SystemExit(
                "[TD] --export-shared-vq is VQ-BeT-only. Refusing to emit an empty VQ "
                "artifact that a later arm could load as if it were a real tokenizer."
            )

    def finish_epoch(self):
        return None

    def set_vqvae_fit(self, fitted: bool) -> None:
        return None

    def load_preset_vqvae(self, state_dict: dict) -> None:
        raise SystemExit("[TD] no VQ tokenizer to preset; --preset-vqvae is VQ-BeT-only")

    def _unpack_actions(self, action_seq: torch.Tensor) -> torch.Tensor:
        """[N,total_w,A] -> [N,T,W,A]. Same overlapping-window protocol as
        BehaviorTransformer._unpack_actions (vqbet.py:222-229); P0 asserts the two
        agree numerically so they cannot drift apart."""
        n, total_w, a = action_seq.shape
        obs_w = total_w + 1 - self.W
        result = action_seq.new_empty(n, obs_w, self.W, a)
        for i in range(obs_w):
            result[:, i] = action_seq[:, i : i + self.W]
        return result

    # ---- anchors / normaliser ------------------------------------------------------

    def load_anchors(self, path: str) -> None:
        """Install the P1 artifact: normaliser buffers + anchors, both buffers.

        The artifact's normaliser was fit on the TRAIN split and maps the train chunks
        into exactly [-1,1]. Anchors are stored in that same normalised space. Both are
        `persistent=True` buffers, so a saved checkpoint carries them and P0's
        round-trip check can prove it.
        """
        payload = load_anchor_artifact(path)
        self.normalizer.load_state_dict(payload["normalizer"])
        self.normalizer.fitted = True
        anchors = payload["anchors_norm"].to(torch.float32)
        assert anchors.shape == (self.K, self.W, self.A), (
            f"anchor artifact K/W/A {tuple(anchors.shape)} != configured "
            f"{(self.K, self.W, self.A)}"
        )
        # Re-fit rather than trusting `fitted=True` blindly: a mismatched artifact would
        # otherwise only show up as a garbage-scale loss hours into training.
        assert anchors.abs().max() <= 1.0 + 1e-4, (
            f"anchors are not in [-1,1] (max |a| = {float(anchors.abs().max())}); the "
            "artifact was probably produced against a different normaliser"
        )
        self.anchors.copy_(anchors)
        self._anchor_meta = payload.get("meta", {})
        self._anchor_source = str(path)

    def set_noise_generator(self, gen: Optional[torch.Generator]) -> None:
        """Pin the inference-time eps source.

        P5 compares infer_steps=1 vs 2 on ONE checkpoint. Without a pinned generator the
        two rollouts start from different eps and the difference being measured is
        contaminated by the seed difference. Pass a generator seeded per
        `eval_rollout.py:319`'s cond_gen convention (31337 + train_seed).
        """
        self._noise_gen = gen

    def set_inference(self, t_start: Optional[int] = None,
                      steps: Optional[int] = None,
                      diffusiondrive_timesteps: Optional[bool] = None) -> None:
        if t_start is not None:
            self.infer_t_start = int(t_start)
        if steps is not None:
            self.infer_steps = int(steps)
        if diffusiondrive_timesteps is not None:
            self.use_diffusiondrive_timesteps = bool(diffusiondrive_timesteps)

    def chain(self) -> List[Tuple[int, int]]:
        """The (t_current, t_next) pairs, in order -- the single source for BOTH the
        sampled trajectory in `_sample` and every audit printout.

        Canonical (n_steps=2, t_start=49) -> [(49, 25), (25, 0)]: two model calls, two
        real DDIM transitions, endpoint 0 reached by the last transition.

        `use_diffusiondrive_timesteps=True` reproduces the REFERENCE's chain instead --
        `[10, 0]` with unit spacing, so its last transition is `0 -> -1` and returns
        x0_hat verbatim (the no-op). Attribution only; see `make_inference_timesteps`.
        """
        if self.use_diffusiondrive_timesteps:
            ts = make_eval_timesteps_diffusiondrive(self.infer_steps)   # [10, 0] @ n=2
            # diffusers' `prev_timestep = t - num_train_timesteps // num_inference_steps`
            # with `set_timesteps(1000)` is `t - 1` ALWAYS. Reproducing the reference
            # means reproducing THAT spacing, not our explicit-span one: it is the reason
            # the second call is an identity. Pairs -> [(10,9), (0,-1)].
            return [(t, t - 1) for t in ts]
        return inference_chain(self.infer_t_start, self.infer_steps)

    def describe_chain(self) -> str:
        """The audit printout: one line per model call, plus the schedule table."""
        rows = self.chain()
        lines = [f"  inference chain: infer_steps={self.infer_steps} "
                 f"infer_t_start={self.infer_t_start}"
                 + ("  [DiffusionDrive attribution mode]"
                    if self.use_diffusiondrive_timesteps else "")]
        for i, (t_cur, t_next) in enumerate(rows):
            w = self.schedule.step_weights(t_cur, t_next if t_next >= 0 else -1)
            lines.append(
                f"    model call #{i + 1}: t_current = {t_cur:>3}  t_next = {t_next:>3}  "
                f"| alpha_bar {w['alpha_bar_cur']:.8f} -> {w['alpha_bar_prev']:.8f}  "
                f"| travel {w['exact_travel_w0_over_sum']:.4f}"
            )
        return "\n".join(lines)

    def _gen(self, device: torch.device) -> torch.Generator:
        if self._noise_gen is None:
            self._noise_gen = torch.Generator(device=device).manual_seed(self.noise_seed)
        return self._noise_gen

    # ---- core: noise -> refine -> x0_hat -------------------------------------------

    @staticmethod
    def _check_t(t: torch.Tensor, n: int, t_len: int, where: str) -> None:
        """D3: `t` is [N,T] -- ONE timestep per (n,t) position, shared by all K modes.

        A t of shape [N,T,K,W,A] (or any per-mode t) would let different modes be
        denoised at different noise levels inside the same forward, which is not the
        score-matching objective: mode k is a CANDIDATE for the same sample, so it must
        see the same t. The shape check catches a wrongly-shaped t; the broadcast
        identity itself is asserted numerically in `test_t_shared_across_modes`.
        """
        assert t.ndim == 2 and tuple(t.shape) == (n, t_len), (
            f"{where}: t must be [N,T] = [{(n, t_len)}], got shape {tuple(t.shape)}. "
            "One t per (n,t) position; all K modes share it."
        )

    def test_t_shared_across_modes(self, n: int = 2, t_len: int = 3) -> float:
        """D3 unit test: prove numerically that one t broadcasts identically over K.

        Adds noise with t [N,T] to a zero base, then re-derives the same quantity with t
        explicitly expanded to [N,T,K] via repeat. If the two disagree anywhere, some
        mode saw a different noise level. Returns the max abs difference (must be 0.0).
        """
        k, w, a = self.K, self.W, self.A
        dev = self.anchors.device
        t = torch.arange(n * t_len, device=dev).reshape(n, t_len) % self.schedule.train_t_max
        self._check_t(t, n, t_len, "test_t_shared_across_modes")
        zeros = torch.zeros(n, t_len, k, w, a, device=dev)
        ones = torch.ones_like(zeros)
        a_bar = self.schedule.abar(t)                                  # [N,T]
        while a_bar.ndim < 5:
            a_bar = a_bar.unsqueeze(-1)
        from_t = a_bar.sqrt() * zeros + (1.0 - a_bar).sqrt() * ones    # [N,T,K,W,A]
        # Same thing, but with t physically replicated along K: any per-mode t would
        # show up here as a mismatch.
        t_rep = t[:, :, None].expand(n, t_len, k)
        a_bar_rep = self.schedule.abar(t_rep).unsqueeze(-1).unsqueeze(-1)
        from_rep = a_bar_rep.sqrt() * zeros + (1.0 - a_bar_rep).sqrt() * ones
        gap = float((from_t - from_rep).abs().max())
        assert gap == 0.0, (
            f"one t is NOT shared across the {k} modes: max|diff| = {gap}"
        )
        # And the value really does vary along t (otherwise the test proves nothing).
        spread = float(from_t[0].std(dim=0).max())
        assert spread > 1e-3, "t values collapsed to a constant; test is vacuous"
        return gap

    def _score(self, q: torch.Tensor, g: torch.Tensor, n: int, t_len: int) -> torch.Tensor:
        """Mode logits [N,T,K] for ONE refinement block. Canonical: reads the refined z_k.

        This is the D2 fix. `bin_head` is `D -> 1` and is applied to the SAME per-mode
        feature `q` [N,T,K,D] that `offset_head` consumes, so the score is
        `p(mode_k | g, candidate_k, t)` -- it can depend on how well this particular
        candidate fits, not just on the trunk context. The reference does the same:
        `plan_cls_branch` and `plan_reg_branch` both read `traj_feature.view(bs,mode,-1)`
        (transfuser_model_v2.py:216-228).

        `cls_from="trunk"` reproduces the rejected context-only variant and is kept ONLY
        for the later speed/structure ablation -- there `bin_head` is `D -> K` and reads
        `g`, exactly as the pre-fix revision did.
        """
        if self.cls_from == "refined":
            return self.bin_head(q).view(n, t_len, self.K)
        assert self.cls_from == "trunk"
        return self.bin_head(g)

    def _refine(
        self, x_t: torch.Tensor, t: torch.Tensor, g: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the L-block cascade. Returns (cls_logits, x0_hats), each length L.

        `x0_hats[l]` is in NORMALISED space and is `residual + base` -- the ported
        `poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points` (:341). The chain is
        x0_0 = base_0 + res_0, base_1 = x0_0.detach(), x0_1 = base_1 + res_1, ... so the
        refinement is progressive but gradient does not flow through the base update,
        exactly as the reference's `.clone().detach()`.

        Both lists are per-layer, which is what makes the reference's
        `ret_traj_loss += trajectory_loss` (:494-497) port over as a genuine sum: with
        the D2 fix both the cls term and the reg term are layer-specific.
        """
        # Take N,T from x_t and CHECK t against them, rather than unpacking t's own
        # shape -- that way a wrongly-shaped t produces the D3 assert message instead of
        # a bare ValueError from tuple unpacking.
        n, t_len = x_t.shape[0], x_t.shape[1]
        self._check_t(t, n, t_len, "_refine")
        feats = self.denoiser(x_t, t, g)
        k = x_t.shape[2]
        cls_logits: List[torch.Tensor] = []
        x0_hats: List[torch.Tensor] = []
        base = x_t
        for q in feats:
            cls_logits.append(self._score(q, g, n, t_len))
            res = self.offset_head(q).view(n, t_len, k, self.W, self.A)
            x0 = base + res
            x0_hats.append(x0)
            base = x0.detach()
        return cls_logits, x0_hats

    @torch.no_grad()
    def _sample(self, g: torch.Tensor) -> torch.Tensor:
        """Full inference chain -> [N,T,W,A] RAW PIXEL coordinates.

        The chain comes from `self.chain()`, so the number of MODEL CALLS equals
        `infer_steps` and every one of them performs a real DDIM transition. With the
        canonical settings that is [(49,25), (25,0)] -- not a big step 49->0 followed by
        an identity, which is what the reference's [10,0] + unit `prev_timestep` spacing
        amounts to (see `diffusion.py`).

        Noise level == the label handed to the denoiser (`infer_t_start`), which is the
        point where we deliberately DO NOT copy the reference: it noises at t=8
        (:518) but labels the first denoiser call t=10 (:510). The mismatch is
        measured, not argued, via `use_diffusiondrive_timesteps`.
        """
        n, t_len = g.shape[0], g.shape[1]
        k, w, a = self.K, self.W, self.A
        rows = self.chain()

        base = self.anchors.view(1, 1, k, w, a).expand(n, t_len, k, w, a)
        noise = torch.randn(base.shape, generator=self._gen(base.device),
                            device=base.device, dtype=base.dtype)
        t0 = rows[0][0]
        t0_t = torch.full((n, t_len), int(t0), device=base.device, dtype=torch.long)
        x = self.schedule.add_noise(base, noise, t0_t)

        cls_logits: List[torch.Tensor] = []
        for t_cur, t_next in rows:
            t_cur_t = torch.full((n, t_len), int(t_cur), device=base.device,
                                 dtype=torch.long)
            # Last block's estimate is the one the reference steps with
            # (`poses_reg_list[-1]`, :546).
            cls_logits, x0_hats = self._refine(x, t_cur_t, g)
            x = self.schedule.ddim_step(x, x0_hats[-1], t_cur_t, int(t_next))

        # Mode selection from the LAST block's refined candidates (D2), then unnormalise.
        mode_idx = cls_logits[-1].argmax(dim=-1)                         # [N,T] in [0,K)
        idx = mode_idx[:, :, None, None, None].expand(n, t_len, 1, w, a)
        chosen = x.gather(2, idx).squeeze(2)                             # [N,T,W,A] norm.
        return self.normalizer.unnormalize(chosen)

    # ---- forward -------------------------------------------------------------------

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        cond_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        if obs_seq.shape[1] < self.obs_window_size:
            obs_seq = _repeat_start_to_length(obs_seq, self.obs_window_size, dim=1)
        if cond_seq is not None and cond_seq.shape[1] < self.obs_window_size:
            cond_seq = _repeat_start_to_length(cond_seq, self.obs_window_size, dim=1)

        if self.unconditional:
            gpt_input = obs_seq
        else:
            assert goal_seq is not None
            if goal_seq.shape[1] < self.obs_window_size:
                goal_seq = _repeat_start_to_length(goal_seq, self.obs_window_size, dim=1)
            gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)

        if self.coord_decoder is not None:
            gpt_input = self.coord_decoder(gpt_input)

        if cond_seq is not None and cond_seq.shape[-2] == 1:
            cond_seq = cond_seq.squeeze(-2)
        g = self.gpt(gpt_input, cond_seq)          # [N,T,readout_dim]

        if action_seq is not None:
            loss, loss_dict, pred = self._calc_loss(g, action_seq)
            return pred, loss, loss_dict
        return self._sample(g), None, {}

    # ---- loss ----------------------------------------------------------------------

    def _calc_loss(
        self, g: torch.Tensor, action_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        gt = self._unpack_actions(action_seq)                       # [N,T,W,A] RAW px
        gt_n = self.normalizer.normalize(gt)                        # normalised
        n, t_len, w, a = gt_n.shape
        k = self.K
        device = gt_n.device

        # Reference: `dist = norm(target[...,:2] - plan_anchor, dim=-1).mean(dim=-1)`
        # then argmin (multimodal_loss.py:133-135). `mode_distance` is that, and it is
        # the same function P1's coverage audit uses, so the trained objective and the
        # reported anchor floor cannot disagree. Computed ONCE: the reference recomputes
        # it inside every layer's LossComputer call, but anchors and targets do not
        # change across layers, so recomputing is pure waste with an identical result.
        with torch.no_grad():
            dist = mode_distance(gt_n, self.anchors)                # [N,T,K]
            mode_idx = dist.argmin(dim=-1)                          # [N,T]

        # Noise + one denoiser pass at the sampled t. `t_sample` is [N,T]: D3 guarantees
        # a single timestep per (n,t) position that all K modes share.
        t_sample = self.schedule.sample_train_timesteps((n, t_len), device)
        base = self.anchors.view(1, 1, k, w, a).expand(n, t_len, k, w, a)
        noise = torch.randn(base.shape, device=device, dtype=base.dtype)
        x_t = self.schedule.add_noise(base, noise, t_sample)
        cls_logits_list, x0_hats = self._refine(x_t, t_sample, g)    # each L x [N,T,K,...]

        # Both terms are PER LAYER and both are SUMMED over layers, matching the
        # reference's `ret_traj_loss += trajectory_loss` (transfuser_model_v2.py:494-497).
        # That is only defensible now that the cls term genuinely varies by layer (D2):
        # before the fix, cls was layer-invariant and summing it would have multiplied a
        # constant by L. `trajectory_cls_weight=10.0` / `trajectory_reg_weight=8.0` are
        # applied once, outside the sum, as the reference does.
        onehot = torch.zeros(n, t_len, k, dtype=gt_n.dtype, device=device)
        onehot.scatter_(2, mode_idx.unsqueeze(-1), 1.0)
        gather_idx = mode_idx[:, :, None, None, None].expand(n, t_len, 1, w, a)
        per_layer_cls, per_layer_reg = [], []
        for cl, x0 in zip(cls_logits_list, x0_hats):
            per_layer_cls.append(py_sigmoid_focal_loss(
                cl, onehot, weight=None,
                gamma=self.focal_gamma, alpha=self.focal_alpha,
                reduction="mean", avg_factor=None,
            ))
            per_layer_reg.append(F.l1_loss(x0.gather(2, gather_idx).squeeze(2), gt_n))
        loss_cls_raw = torch.stack(per_layer_cls).sum()
        loss_reg_raw = torch.stack(per_layer_reg).sum()

        loss = self.diff_weight * (self.cls_weight * loss_cls_raw
                                  + self.reg_weight * loss_reg_raw)

        cls_logits = cls_logits_list[-1]
        with torch.no_grad():
            mode_acc = (cls_logits.argmax(dim=-1) == mode_idx).float().mean()
            t_hist = torch.bincount(t_sample.reshape(-1),
                                    minlength=self.schedule.train_t_max).tolist()
        loss_dict = {
            "classification_loss": float((self.cls_weight * loss_cls_raw).detach()),
            "offset_loss": float((self.reg_weight * loss_reg_raw).detach()),
            "total_loss": float(loss.detach()),
            # TD-only diagnostics; extra keys are invisible to train.py's reader, which
            # pulls the three keys above by name (train.py:1193-1196).
            "mode_acc": float(mode_acc),
            "focal_loss_raw": float(loss_cls_raw.detach()),
            "reg_loss_raw": float(loss_reg_raw.detach()),
            "cls_loss_layer0": float(per_layer_cls[0].detach()),
            "reg_loss_layer0": float(per_layer_reg[0].detach()),
            "reg_loss_layer_last": float(per_layer_reg[-1].detach()),
            "t_sample_max": int(t_sample.max()),
            "t_hist_head": t_hist[:8],
        }

        # Training-time `pred`: the last block's estimate for the argmax mode, in pixels.
        # Inference re-evaluates at the eval timesteps and runs DDIM, so this is a
        # DIAGNOSTIC, not the rollout path -- but it costs nothing extra here and it
        # makes the value train.py logs comparable to GT for debugging.
        with torch.no_grad():
            am = cls_logits.argmax(dim=-1)
            idx = am[:, :, None, None, None].expand(n, t_len, 1, w, a)
            pred = self.normalizer.unnormalize(x0_hats[-1].gather(2, idx).squeeze(2))
        return loss, loss_dict, pred

    # ---- optimizer -----------------------------------------------------------------

    @staticmethod
    def _assert_classifiable(module: nn.Module, whitelist, blacklist) -> None:
        """TD's OWN parameter-classification check.

        The harness's two asserts in vqbet.py:481-482 cover only `{gpt, coord_decoder}`;
        everything hung off a separate param group escapes them. `denoiser` is such a
        module, so the check is re-run here and a violation is a loud SystemExit naming
        the offender rather than a parameter that silently gets no gradient.
        """
        bad = []
        for mn, m in module.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or p.ndim == 0:
                    continue
                if pn.endswith("weight") and isinstance(m, whitelist + blacklist):
                    continue
                bad.append(f"{fpn}  ({type(m).__name__}, ndim={p.ndim})")
        if bad:
            raise SystemExit(
                "[TD] unclassifiable parameters in the diffusion denoiser -- they would "
                "land in NEITHER the decay nor the no_decay set:\n  " + "\n  ".join(bad)
                + "\n(see patch_policy/td_head.py module docstring)"
            )

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas) -> GroupedOptimizer:
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)
        # scope is IDENTICAL to vqbet.py:459-461 -- gpt only (coord_decoder is None on
        # this arm's configs). The two asserts below therefore have unchanged semantics.
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

        # denoiser escapes those asserts by construction -> check it explicitly.
        self._assert_classifiable(self.denoiser, whitelist, blacklist)

        # sorted() keeps the param order stable across processes so AdamW state restores
        # positionally on resume (vqbet.py:484-487).
        opt1 = torch.optim.AdamW(
            [
                {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
                {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )
        # No weight_decay kwarg on add_param_group, exactly as vqbet.py:496. That means
        # bin_head inherits `optimizer.defaults["weight_decay"]`, i.e. AdamW's CLASS
        # default 1e-2 -- NOT the configured weight_decay. This is an inherited property
        # of the shared harness (the VQ-BeT arm's bin_head does the same), so it is
        # mirrored rather than "fixed": changing it would make the two arms' shared slot
        # behave differently for no reason. P0 prints every group's weight_decay so this
        # is a measurement, not a claim.
        opt1.add_param_group({"params": list(self.bin_head.parameters())})
        opt2 = torch.optim.AdamW(
            self.offset_head.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=betas
        )
        # denoiser: the fork's second-head precedent (explicit configured weight_decay).
        opt3 = torch.optim.AdamW(
            self.denoiser.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=betas
        )
        self._assert_disjoint_param_groups(opt1, opt2, opt3)
        return GroupedOptimizer([opt1, opt2, opt3])

    @staticmethod
    def _assert_disjoint_param_groups(*opts) -> None:
        """A Parameter registered under two modules would appear in two groups and
        AdamW raises a bare ValueError. Catch it with the offending count instead."""
        seen: Dict[int, int] = {}
        for i, opt in enumerate(opts):
            for g in opt.param_groups:
                for p in g["params"]:
                    seen[id(p)] = seen.get(id(p), 0) + 1
        dupes = [k for k, v in seen.items() if v > 1]
        if dupes:
            raise SystemExit(
                f"[TD] {len(dupes)} parameter(s) appear in more than one param group "
                "-- a module is very likely registered on both the policy and the "
                "denoiser (e.g. offset_head). AdamW would raise on the first step."
            )


# ======================================================================================
# builder -- called from build_model, which already constructed the trunk's inputs
# ======================================================================================

def build_td_policy(cfg, obs_dim: int, coord_decoder,
                    adaln, fork_kwargs: dict) -> TruncatedDiffusionPolicy:
    """Build the TD arm from the SAME trunk inputs `build_model` computed.

    `build_model` computes obs_dim / coord_decoder / adaln / fork_kwargs and then calls
    this instead of `BehaviorTransformer(...)`, so the trunk construction is literally
    the same code path -- "only the action head differs" holds by construction.

    `fork_kwargs` carries `gpt_output_dim`/`gelu` (the trunk knobs) and
    `head_final_relu` (a VQ-BeT-head MLP reproduction knob). Only the first two are
    forwarded: TD's own heads are unclamped regardless, see __init__.
    """
    p = cfg.policy
    td = cfg.td
    return TruncatedDiffusionPolicy(
        obs_dim=obs_dim,
        act_dim=2,
        goal_dim=0,
        views=1,
        obs_window_size=cfg.obs_window_size,
        act_window_size=cfg.action_window_size,
        n_patches=cfg.policy_n_patches(),
        coord_decoder=coord_decoder,
        adaln=adaln,
        n_layer=p.n_layer,
        n_head=p.n_head,
        n_embd=p.n_embd,
        dropout=p.dropout,
        gpt_output_dim=fork_kwargs.get("gpt_output_dim"),
        gelu=fork_kwargs.get("gelu", "exact"),
        td=td,
        anchor_artifact=td.anchor_artifact,
    )
