"""Flat YAML config -> typed dataclass. No Hydra; single source of truth."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class WeMMConfig:
    model_name: str = "Tencent/WeMM-Embedding-2B"
    local_dir: Optional[str] = None
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    n_patches: int = 1
    output_dim: int = 2048
    freeze: bool = True
    encode_batch_size: int = 16
    # ---- WeMM-Local-P16 (spatial decomposition) ----
    # local_mode=True: 每个 224×224 帧拆成 4×4 固定 grid 的 overlapping crops
    #   (grid centers {28,84,140,196}，reflect-pad，window=crop_window)，
    #   每个 crop 独立走官方 model.embedding() -> [16,2048]（row-major，index=r*4+c）。
    # 这是唯一的 representation change（WeMM-P1 整图 -> 单 [1,2048]）。
    # encode_batch_size 在 local_mode 下解释为"每次 encode_images 的 crop 数"（限制 VRAM）。
    local_mode: bool = False
    crop_window: int = 112  # feasibility 推荐：84<96<112（112 上下文最充分）
    # ---- WeMM-Repeat-P256 (Global-to-Patch Token Expansion) ----
    # repeat_patches>1：仅改变 policy-facing P。cache 仍是 P=1（[B,T,1,2048]），
    # 在 policy 输入边界 repeat 成 [B,T,repeat_patches,2048]（纯复制，内容完全相同），
    # token 差异只来自原 Patch Policy 的 wpe positional embedding（token_i = W(z)+wpe(t*P+i)）。
    # 严禁：MLP/position-specific MLP/decoder/cross-attention/crop/intermediate tokens/新 2D 位置编码。
    repeat_patches: int = 1
    # ---- Global-to-Spatial Coord-MLP Decoder (explicit position-conditioned) ----
    # decoder="coord"：从 frozen global embedding z 解码 P 个显式 spatial slots——
    #   t_i = f_theta( concat(z, FourierPE(x_i,y_i)) )，f_theta 对所有位置共享。
    # cache 仍是 P=1；policy P = decoder_patches（64=8×8 / 256=16×16，row-major, [-1,1]²）。
    # 这是"global embedding -> policy tokens"接口的唯一改变；WeMM/GPT/Patch Policy/VQ-BeT/数据/rollout 不变。
    # 严禁：per-position MLP / cross-attention / transformer decoder / conv decoder / U-Net / 额外监督。
    # 这些槽是 position-conditioned decoded slots，不是真实 patch tokens，不是 native dense 特征。
    decoder: str = "none"           # "none" | "coord"
    decoder_patches: int = 256      # coord: P = 64 (8×8) | 256 (16×16)（必须为完全平方）
    decoder_pe_dim: int = 64        # coordinate Fourier PE 维度（x/y 对称、无参、deterministic）
    decoder_hidden: int = 512       # shared MLP hidden 维
    decoder_out_dim: int = 512      # slot dim 喂给 GPT（必须 == policy.n_embd）


@dataclass
class Qwen3VLEmbeddingConfig:
    """Qwen3-VL-Embedding-2B — 替换 WeMM-Embedding-2B 的 global embedding 条件流（唯一科学变量）。

    Qwen-A1 encoder replacement：同一 DINO-P256 dense main stream + 原 A1 residual AdaLN
    (adaln.py)，仅把 fusion.cond_encoder 从 'wemm' 换成 'qwen3vl'。字段接口镜像 WeMMConfig
    （encoder.py build_cond_encoder 依此构造），n_patches 恒为 1（每帧单个 global embedding）。
    output_dim 必须由 Phase-2 representation audit 实测后填充 —— 禁止按 WeMM=2048 硬编码；
    若填 0 且 fusion.cond_encoder='qwen3vl' 会在 build 处断言拒绝（防未实测即训练）。
    image_only=True = 本阶段契约：只喂图像，不提供任何 text/instruction/prompt（§5）。
    """

    model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    local_dir: Optional[str] = None
    dtype: str = "bf16"          # bf16 | fp16 | fp32
    n_patches: int = 1
    output_dim: int = 0          # audit 实测后填实际 embedding 维度（非 0 才能训练）
    freeze: bool = True
    encode_batch_size: int = 8   # 每次 encode_images 的 image 数（VRAM 约束，audit 校核）
    image_only: bool = True      # Phase-1 IMAGE-ONLY 契约（禁止加文本，除非官方强制并已报告）


@dataclass
class DINOConfig:
    """DINO-P1: DINOv2-S/14 patch tokens + mean pooling -> single [1,384] token.

    Mirrors base `models/encoder/dino.py` DinoV2Encoder with the
    `configs/encoder/dino_patch_avg_pool.yaml` settings exactly:
    same torch.hub weights, same ImageNet normalization, same
    feature_key='x_norm_patchtokens', same postprocess='avg_pool'.
    """

    name: str = "dinov2_vits14"
    feature_key: str = "x_norm_patchtokens"
    postprocess: str = "avg_pool"  # None (P=256 raw) | "avg_pool" (P=1)
    dtype: str = "fp32"  # DINOv2 native fp32
    n_patches: int = 1
    output_dim: int = 384
    freeze: bool = True
    encode_batch_size: int = 32


@dataclass
class AdaLNConfig:
    """WeMM-DINO AdaLN Fusion (spec §3-§4): residual adaptive-LN conditioning of DINO dense tokens.

        X_fused = X + gamma(LN_g(g)) * LN_local(X) + beta(LN_g(g))     (= X + ΔX)

    DINO keeps the real dense spatial structure (P=256 native tokens, frozen);
    WeMM official final global embedding g is the semantic conditioning stream
    (frozen, from disk cache). Identity init (HARD): the final Linear is
    zero-initialized (weight=bias=0) -> gamma=0, beta=0 -> X_fused == X exactly.
    Residual form (not (1+γ)LN(X)+β) keeps the DINO-P256 baseline input
    distribution. No alpha gate (dX_fused/dW_proj != 0 at init, so the AdaLN
    branch trains from step 0).

    alpha_gate=True selects the AdaLN-zero variant (patch_policy/adaln_gate.AdaLNGate):
        X_fused = X + alpha * (gamma*LN_local(X) + beta)
    where alpha is ONE learnable scalar init 0.0 and proj uses DEFAULT (non-zero)
    init -> identity start comes from alpha==0, NOT from zeroing proj. Set
    zero_init=False with alpha_gate=True (proj≡0 would permanently kill the
    branch AND the alpha gradient). See adaln_gate.py module docstring.
    Explicitly NOT: extra repeat / coord decoder / cross-attention / bilinear / concat.
    """

    enabled: bool = False
    cond_encoder: str = "wemm"       # conditioning stream (frozen, cache-sourced)
    cond_dim: int = 2048             # WeMM final global embedding dim
    hidden_dim: int = 1024           # Linear(cond_dim, hidden_dim); must == 2 * policy.n_embd
    cond_cache: str = "wemm_cache/pusht_wemm.pt"
    zero_init: bool = True           # HARD identity (AdaLN): proj weight=bias=0 -> gamma=beta=0
    ln_local_affine: bool = False    # LN_local pure normalization (AdaLN-style; gamma/beta are the affine)
    alpha_gate: bool = False         # True = AdaLN-zero scalar alpha-gate variant (see above); False = AdaLN
    alpha_init: float = 0.0          # alpha-gate init (0 = identity start, gate cold); alpha_gate only


@dataclass
class OracleConfig:
    """Oracle PushT simulator state (diagnostic only).

    states.pth [206, 246, 5] = per-frame [agent_x, agent_y, block_x, block_y,
    block_angle] (native sim state). The oracle cache normalizes to 6 dims:
    [ax,ay,bx,by]/512 (pixel->[0,1]) + [sin theta, cos theta] (angle wrap).
    Never used as a vision result — only to test whether the Global Policy
    backbone itself can solve PushT control.
    """

    name: str = "oracle_state"
    n_patches: int = 1
    output_dim: int = 6            # normalized oracle state dim fed to proj
    state_dim: int = 5             # raw states.pth per-frame dim
    coord_scale: float = 512.0     # pixel coordinate scale in the 512 sim frame
    freeze: bool = True


@dataclass
class GlobalPolicyConfig:
    """WeMM-Global Temporal Action Transformer v1 (policy_mode="global").

    Replaces the patch GPT with a temporal transformer over T state tokens +
    n_query learned action query tokens (temporal PE only, no spatial/patch PE).
    The VQ-BeT action machinery (VqVae, bin/offset heads, loss, sampling) is
    shared unchanged via cfg.policy.
    """

    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 512
    n_query: int = 5              # == obs_window_size (one action query per timestep chunk)
    dropout: float = 0.0


@dataclass
class VQBeTConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    vqvae_latent_dim: int = 512
    vqvae_n_embed: int = 16
    vqvae_groups: int = 2
    vqvae_iters: int = 1000
    vqvae_batch_size: int = 1024
    vqvae_fit_steps: Optional[int] = None
    act_scale: float = 500.0
    offset_loss_multiplier: float = 10.0
    secondary_code_multiplier: float = 0.5
    gamma: float = 2.0
    # ---- Native-parity faithful fork (patch_policy phase9 dino_s43, clean rebuild) ----
    # native_parity=True makes patch_policy a tensor-level fork of the historical native
    # DINO-P256 baseline: restores GPT lm_head Linear(512->256,bias=False) applied over
    # all positions then last-patch readout (head input 256, not 512), switches the
    # transformer MLP activation to GPT-2 tanh-GELU (native new_gelu), and rebuilds the
    # bin/offset head MLP to native torchvision utils.MLP semantics (ReLU only between
    # hidden layers, NO final activation). With these, loading the known-good dino_s43
    # weights into the fork gives bit-level forward parity. Default False preserves the
    # pre-fork patch_policy layout exactly (legacy ckpt load path unchanged).
    native_parity: bool = False


@dataclass
class TDConfig:
    """Truncated-diffusion action head (the single research variable).

    Defaults are the reference's values wherever the reference has one; where it does
    not, the default is the identity / no-op so a deviation has to be written down.
    Every deviation from base/DiffusionDrive is enumerated in patch_policy/td_head.py's
    module docstring -- do not add a field here without adding it there.
    """

    # ---- head geometry ----
    n_modes: int = 20              # K. spec: 第一版仍优先使用 K=20
    num_layers: int = 2            # CustomTransformerDecoder(diff_decoder_layer, 2)
    n_heads: int = 8               # reference tf_num_head
    d_ffn: int = 1024              # reference tf_d_ffn
    dropout: float = 0.0
    pos_embed_dim: int = 64        # gen_sineembed_for_position(hidden_dim=64), hardcoded there
    # Where the mode score comes from. "refined" is the CANONICAL setting: one refined
    # per-mode feature z_k [N,T,K,D] feeds BOTH the regression head and the score head,
    # so the score is p(mode_k | g, candidate_k, t) -- as in DiffusionDrive, where
    # plan_cls_branch and plan_reg_branch read the same `traj_feature.view(bs,mode,-1)`.
    # "trunk" is the rejected context-only variant (bin_head(g) -> [N,T,K]); it is kept
    # ONLY as a later speed/structure ablation and must not be used for the P6 baseline.
    cls_from: str = "refined"

    # ---- truncated diffusion schedule (P1b audits these against diffusers) ----
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "scaled_linear"
    # Training band is t ~ Uniform{0, ..., train_t_max-1} -- INCLUSIVE of 0, i.e.
    # `torch.randint(0, 50, ...)` (transfuser_model_v2.py:465), NOT 1..50.
    train_t_max: int = 50
    # diffusers' DDIMScheduler DEFAULTS. The reference builds its scheduler without
    # passing either, so it inherits `clip_sample=True, clip_sample_range=1.0` and clamps
    # x0_hat to [-1, 1] on every DDIM step. P1b cross-checks these against the installed
    # diffusers rather than against this comment.
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ---- inference ----
    # infer_t_start should be derived from P1's anchor error, not copied from the
    # reference's 8: the reference noises at t=8 while labelling the first denoiser call
    # t=10 (:518 vs :510), and t=8's noise scale (b_8 ~ 0.029) is likely far below the
    # anchor's own distance to the target, which leaves the later DDIM steps nothing to
    # do. P1's `recommend_t_start` writes the measured value into the anchor artifact's
    # meta; P1b turns that into the number used here.
    infer_steps: int = 2
    infer_t_start: int = 49
    # Attribution-only: reproduce the reference's [10, 0] via its own step_ratio formula
    # AND its add_noise@8, so P5 measures the "2 steps is really 1" claim. Never default.
    use_diffusiondrive_timesteps: bool = False
    noise_seed: int = 31337        # P5 pins the eps source; see set_noise_generator

    # ---- loss weights (reference values; spec forbids tuning per arm) ----
    cls_weight: float = 10.0       # transfuser_config.trajectory_cls_weight
    reg_weight: float = 8.0        # transfuser_config.trajectory_reg_weight
    # The reference's TrajectoryHead.diff_loss_weight=2.0 is DEAD CODE (:397, never
    # read); its AGENT-level diff_loss_weight=20.0 multiplies a different head's loss.
    # Neither is a multiplier on the (cls+reg) LossComputer returns, so the identity 1.0
    # is the faithful default and P2 measures 2.0 rather than assuming it.
    diff_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    cls_bias_prior: float = 0.01   # bias_init_with_prob(0.01), :214-215

    # ---- artifacts ----
    anchor_artifact: str = "anchors/pusht_kmeans_K20.pt"  # P1 output: normaliser + anchors


@dataclass
class Config:
    # seed 只控制训练随机性（torch/np/python/model init/dataloader shuffle/VQ-BeT）；
    # split_seed 固定控制 dataset train/val 划分（与 baseline 的 random_seed=42 解耦）。
    seed: int = 42
    split_seed: int = 42
    device: str = "cuda"
    # 视觉编码器选择："wemm" | "dino" | "oracle"。策略/VQ-BeT/训练协议三者完全一致，
    # 仅 encoder 与输入投影维度不同（WeMM 2048 / DINO 384 / Oracle 6 -> wte -> n_embd）。
    # oracle = diagnostic-only：真实 simulator state，只验证 Global Policy backbone 能力。
    encoder_name: str = "wemm"
    # 策略架构："patch"（现有 Patch Policy GPT）| "global"（WeMM-Global Temporal Transformer）。
    # 切换只改变 policy 骨架；数据/缓存/VQ-BeT 头/损失/训练协议/rollout 布局完全一致。
    policy_mode: str = "patch"
    # 动作头："vqbet"（BehaviorTransformer，原样）| "td"（截断扩散，本项目唯一研究变量）。
    # 这是两臂 yaml 之间**唯一**允许不同的一行；trunk/dataset/eval/协议全部共用同一份代码。
    action_head: str = "vqbet"
    # data
    dataset_root: str = "/path/to/pusht_dataset"   # <-- MUST be set (yaml overrides this)
    train_fraction: float = 0.9
    obs_window_size: int = 5
    action_window_size: int = 5
    use_relative_actions: bool = False
    # training
    batch_size: int = 64
    epochs: int = 400
    lr: float = 5.5e-5
    weight_decay: float = 2e-4
    betas: tuple = (0.9, 0.999)
    num_workers: int = 4
    eval_freq_epochs: int = 10
    # cache / output
    precompute_cache: str = "wemm_cache/pusht_wemm.pt"
    save_dir: str = "outputs/pusht"
    save_ckpt: bool = True
    # nested
    wemm: WeMMConfig = field(default_factory=WeMMConfig)
    qwen3vl: Qwen3VLEmbeddingConfig = field(default_factory=Qwen3VLEmbeddingConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    fusion: AdaLNConfig = field(default_factory=AdaLNConfig)
    policy: VQBeTConfig = field(default_factory=VQBeTConfig)
    global_policy: GlobalPolicyConfig = field(default_factory=GlobalPolicyConfig)
    td: TDConfig = field(default_factory=TDConfig)

    def __post_init__(self):
        """Fail loudly on an unknown head name.

        `build_model` dispatches on `action_head == "td"` and otherwise falls through to
        the untouched VQ-BeT path, so a typo (`action_head: diffusion`) would silently
        train the OTHER arm and the whole comparison would be two VQ-BeT runs. The two
        arms' yamls differ by exactly this one line, so it is worth an explicit gate.
        """
        if self.action_head not in ("vqbet", "td"):
            raise SystemExit(
                f"[config] unknown action_head {self.action_head!r} (vqbet | td). "
                "Anything else would silently run the VQ-BeT head."
            )

    def enc(self):
        """Active encoder config (wemm | dino | oracle) — the single source for policy dims."""
        if self.encoder_name == "oracle":
            return self.oracle
        return self.dino if self.encoder_name == "dino" else self.wemm

    def cond_config(self):
        """Conditioning-stream encoder config — selected by fusion.cond_encoder.

        'wemm'    -> WeMM-Embedding-2B（现有 WeMM-A1，原样）
        'qwen3vl' -> Qwen3-VL-Embedding-2B（本实验唯一科学变量）
        """
        c = self.fusion.cond_encoder
        if c == "wemm":
            return self.wemm
        if c == "qwen3vl":
            return self.qwen3vl
        raise ValueError(f"unknown fusion.cond_encoder: {c!r} (wemm | qwen3vl)")

    def cond_model_name(self) -> str:
        """Cache-meta key 'model_name' for the conditioning stream."""
        return self.cond_config().model_name

    def enc_model_name(self) -> str:
        """Cache meta key 'model_name' for the active encoder."""
        if self.encoder_name == "oracle":
            return self.oracle.name
        return self.enc().name if self.encoder_name == "dino" else self.enc().model_name

    def interface_mode(self) -> str:
        """P=1 cache -> policy tokens 的接口类型（决定数据集/模型如何把 global token 变成 slots）。

        'direct' : P=1 直接进 GPT（WeMM-P1 / DINO-P1）
        'repeat' : 纯复制成 P 份，仅靠 GPT 内部 wpe 区分（Repeat-P256）
        'coord'  : shared Coord-MLP decoder，显式 position-conditioned（本实验）
        """
        if self.encoder_name == "wemm":
            if self.wemm.decoder == "coord":
                return "coord"
            if self.wemm.repeat_patches and self.wemm.repeat_patches > 1:
                return "repeat"
        return "direct"

    def policy_n_patches(self) -> int:
        """Policy-facing patch count (GPT n_patches / input P).

        == enc().n_patches normally; for WeMM-Repeat-P256 (wemm.repeat_patches>1)
        returns repeat_patches — the cache stays P=1, the expansion is a pure
        policy-input transform (see dataset.expand_repeat_patches). For the
        Coord-MLP decoder (wemm.decoder="coord") returns wemm.decoder_patches —
        the cache also stays P=1, the model's CoordSpatialDecoder produces the
        P slots in forward.
        """
        mode = self.interface_mode()
        if mode == "repeat":
            return int(self.wemm.repeat_patches)
        if mode == "coord":
            return int(self.wemm.decoder_patches)
        return self.enc().n_patches

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        d = dict(d)
        wemm_d = dict(d.pop("wemm", {}) or {})
        qwen3vl_d = dict(d.pop("qwen3vl", {}) or {})
        dino_d = dict(d.pop("dino", {}) or {})
        oracle_d = dict(d.pop("oracle", {}) or {})
        policy_d = dict(d.pop("policy", {}) or {})
        global_d = dict(d.pop("global_policy", {}) or {})
        fusion_d = dict(d.pop("fusion", {}) or {})
        td_d = dict(d.pop("td", {}) or {})
        core = {k: v for k, v in d.items()
                if k not in ("wemm", "qwen3vl", "dino", "oracle", "policy",
                             "global_policy", "fusion", "td")}
        return cls(
            **core,
            wemm=WeMMConfig(**wemm_d),
            qwen3vl=Qwen3VLEmbeddingConfig(**qwen3vl_d),
            dino=DINOConfig(**dino_d),
            oracle=OracleConfig(**oracle_d),
            policy=VQBeTConfig(**policy_d),
            global_policy=GlobalPolicyConfig(**global_d),
            fusion=AdaLNConfig(**fusion_d),
            td=TDConfig(**td_d),
        )

    def to_dict(self):
        return dataclasses.asdict(self)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
