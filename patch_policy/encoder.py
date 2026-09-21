"""WeMM-Embedding-2B encoder -> per-timestep visual tokens [..., P, 2048].

Two modes (the ONLY representation difference, everything else frozen):
  * P=1 global  (local_mode=False): whole 224×224 image -> one model.embedding()
    -> [..., 1, 2048].
  * P=16 local  (local_mode=True): 4×4 fixed grid of overlapping 112×112 crops
    (centers {28,84,140,196}, reflect-pad) -> each crop independently through
    model.embedding() -> [..., 16, 2048] (row-major, token = r*4+c).

Uses the official transformers path (trust_remote_code):
    AutoProcessor + AutoModel, inputs built via apply_chat_template + process_vision_info,
    embedding extracted with model.embedding(**inputs)  (L2-normalized, 2048-dim).

Frozen: parameters detached, forward runs under inference_mode. No intermediate
vision tokens, no MLP patch recovery, no embedding duplication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# ---- WeMM-Local-P16 fixed crop grid (feasibility-verified, DO NOT change) ----
LOCAL_GRID = 4
LOCAL_CENTERS = [28, 84, 140, 196]  # cell centers in the 224×224 image coords


class WeMMEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Tencent/WeMM-Embedding-2B",
        local_dir: str | None = None,
        dtype: str = "bf16",
        n_patches: int = 1,
        output_dim: int = 2048,
        freeze: bool = True,
        encode_batch_size: int = 16,
        local_mode: bool = False,
        crop_window: int = 112,
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.n_patches = n_patches
        self.output_dim = output_dim
        self.dtype = DTYPES[dtype]
        self.local_mode = local_mode
        self.crop_window = crop_window
        self.encode_batch_size = encode_batch_size  # crops per encode_images call (local mode)
        path = local_dir or model_name

        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=self.dtype)
        self.model.eval()

        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        if self.local_mode:
            assert self.n_patches == LOCAL_GRID * LOCAL_GRID, (
                f"local_mode requires n_patches={LOCAL_GRID*LOCAL_GRID}, got {self.n_patches}"
            )
            assert self.crop_window % 2 == 0, f"crop_window must be even for reflect-pad, got {self.crop_window}"

    # ---- tensor helpers ----------------------------------------------------

    @staticmethod
    def _to_pil(images: torch.Tensor):
        """images: [N, 3, H, W] float 0..1 -> list[PIL.Image]."""
        from PIL import Image

        imgs = (images.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).round().to(
            torch.uint8
        ).cpu()
        return [Image.fromarray(a.numpy(), mode="RGB") for a in imgs]

    @staticmethod
    def build_crops(images: torch.Tensor, window: int) -> torch.Tensor:
        """images [N,3,224,224] -> [N, 16, 3, window, window].

        Identical per-image implementation to the feasibility diagnostic:
        reflect-pad p=window//2, then each grid cell (r,c) slices
        padded[:, cy:cy+window, cx:cx+window] at center (CENTERS[c], CENTERS[r]).
        Token order is row-major: idx = r*4+c.
        """
        assert images.shape[-1] == images.shape[-2] == 224, f"local mode expects 224×224, got {images.shape[-2:]}"
        p = window // 2
        pad = F.pad(images, (p, p, p, p), mode="reflect")  # [N,3,224+2p,224+2p]
        crops = torch.empty(
            images.shape[0], LOCAL_GRID * LOCAL_GRID, 3, window, window,
            dtype=images.dtype, device=images.device,
        )
        for r in range(LOCAL_GRID):
            for c in range(LOCAL_GRID):
                cx, cy = LOCAL_CENTERS[c], LOCAL_CENTERS[r]
                crops[:, r * LOCAL_GRID + c] = pad[:, :, cy : cy + window, cx : cx + window]
        return crops

    # ---- core forward ------------------------------------------------------

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: [N, 3, H, W] float 0..1 -> [N, output_dim] (L2-normalized)."""
        from qwen_vl_utils import process_vision_info

        pil = self._to_pil(images)
        messages = [
            [{"role": "user", "content": [{"type": "image", "image": img}]}]
            for img in pil
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages
        ]
        images_in, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True
        )
        proc_kwargs = dict(text=texts, images=images_in, videos=videos, return_tensors="pt")
        if videos is not None and len(videos) > 0:
            # 纯图像时 video_kwargs 含 fps=[]，会触发 processor 的严格字段校验，仅在有视频时传
            proc_kwargs.update(video_kwargs or {})
        inputs = self.processor(**proc_kwargs)
        device = next(self.model.parameters()).device  # 动态取设备（.to() 后仍正确）
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        with torch.autocast(device.type, dtype=self.dtype):
            emb = self.model.embedding(**inputs)  # [N, D]
        return emb.to(torch.float32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [..., 3, H, W] -> [..., P, output_dim].

        local_mode=False: whole image through model.embedding() -> [..., 1, 2048].
        local_mode=True:  16 crops/frame independently through model.embedding()
                          -> [..., 16, 2048]. Crops are chunked by
                          encode_batch_size (crops per encode_images call) to
                          bound VRAM; ordering stays row-major per frame.
        """
        prefix_shape = images.shape[:-3]
        if self.local_mode:
            flat = images.reshape(-1, *images.shape[-3:])  # [N,3,224,224]
            crops = self.build_crops(flat, self.crop_window)  # [N,16,3,w,w]
            crops_flat = crops.reshape(-1, 3, self.crop_window, self.crop_window)
            parts = []
            bs = self.encode_batch_size  # crops per encode_images call (VRAM bound)
            for s in range(0, crops_flat.shape[0], bs):
                parts.append(self.encode_images(crops_flat[s : s + bs]))  # fp32 [cb, D]
            emb = torch.cat(parts, 0).reshape(*prefix_shape, self.n_patches, self.output_dim)
        else:
            flat = images.reshape(-1, *images.shape[-3:])
            emb = self.encode_images(flat)  # [N, D]
            emb = emb.reshape(*prefix_shape, self.n_patches, self.output_dim)
        # build.md §19 assertions（[N,3,H,W]->[N,1,2048] ndim=3；[B,T,3,H,W]->[B,T,1,2048] ndim=4）
        assert emb.ndim >= 3
        assert emb.shape[-2] == self.n_patches
        assert emb.shape[-1] == self.output_dim
        assert torch.isfinite(emb).all(), "WeMM output contains NaN/Inf"
        return emb


class DINOEncoder(nn.Module):
    """DINOv2-S/14 patch tokens + mean pooling -> single [...,1,384] token.

    Mirrors base `models/encoder/dino.py` DinoV2Encoder exactly with the
    `configs/encoder/dino_patch_avg_pool.yaml` settings:
        torch.hub.load("facebookresearch/dinov2:b48308a", "dinov2_vits14")
        feature_key   = "x_norm_patchtokens"   (raw [...,256,384])
        postprocess   = "avg_pool"             -> mean(dim=-2) -> [...,384]
        latent_ndim=1 -> unsqueeze(prefix_ndim) -> [...,1,384]  (P=1)
    Same ImageNet normalization, same reshape of arbitrary leading dims.
    Frozen + inference_mode. Output fp32.
    """

    def __init__(
        self,
        name: str = "dinov2_vits14",
        feature_key: str = "x_norm_patchtokens",
        postprocess: str | None = "avg_pool",
        dtype: str = "fp32",
        n_patches: int = 1,
        output_dim: int = 384,
        freeze: bool = True,
        encode_batch_size: int = 32,
    ):
        super().__init__()
        import torchvision.transforms as transforms

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.name = name
        self.feature_key = feature_key
        self.postprocess = postprocess
        self.n_patches = n_patches
        self.output_dim = output_dim
        self.dtype = DTYPES[dtype]
        self.encode_batch_size = encode_batch_size

        # dinov2_vits14 cached at ~/.cache/torch/hub/ on the server (no network)
        self.base_model = torch.hub.load("facebookresearch/dinov2:b48308a", name)
        self.base_model.eval()

        self.emb_dim = self.base_model.num_features  # 384
        self.patch_size = self.base_model.patch_size  # 14
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")
        if postprocess == "avg_pool":
            self.latent_ndim = 1  # mean pool collapses the patch dim

        self.normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False

    # ---- core forward (mirror base DinoV2Encoder.forward line-for-line) ----

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 3, H, W] float 0..1 -> [..., P, D] (P=1 when avg_pool)."""
        assert x.dtype == torch.float32, f"DINO expects float32 input, got {x.dtype}"
        assert x.max() <= 1.0 and x.min() >= 0, "expect 0..1 range"
        x = self.normalization(x)

        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]
        prod_prefix = 1
        for d in prefix_shape:
            prod_prefix *= d
        x = x.reshape(prod_prefix, c, h, w)

        emb = self.base_model.forward_features(x)[self.feature_key]  # [N, 256, E]
        emb = emb.reshape(*prefix_shape, *emb.shape[1:])  # [...prefix, 256, E]

        if self.postprocess == "avg_pool":
            emb = torch.mean(emb, dim=-2)  # (...prefix, E)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape))  # (...prefix, 1, E)

        out = emb.float()
        assert out.ndim >= 3
        assert out.shape[-2] == self.n_patches, f"P mismatch: {out.shape} vs {self.n_patches}"
        assert out.shape[-1] == self.output_dim, f"D mismatch: {out.shape} vs {self.output_dim}"
        assert torch.isfinite(out).all(), "DINO output contains NaN/Inf"
        return out


class Qwen3VLEmbeddingEncoder(nn.Module):
    """Qwen3-VL-Embedding-2B IMAGE-ONLY -> [..., P=1, D] L2-normalized embedding.

    Thin nn.Module wrapper around the vendored official Qwen3VLEmbedder
    (qwen3vl_official.py — byte-copy of the HF repo script minus one import).
    External contract mirrors WeMMEncoder so build_cond_encoder / eval_rollout /
    precompute treat it identically:

      images [...,3,H,W] float 0..1  ->  [..., 1, D]  (one global embedding/frame)

    Official path per frame: system default instruction "Represent the user's input."
    (constant, invariant) + user image-only content, add_generation_prompt=True,
    processor smart_resize (224×224 stays unchanged: within [MIN,MAX] pixels and
    divisible by 32), last-non-pad-token pooling, F.normalize(p=2) — the official
    embedding. IMAGE-ONLY Phase-1: we never pass text/instruction; we only supply a
    bare image. We do NOT customize the default instruction (that would add our own
    prompt text).

    D (output_dim) is NOT assumed = 2048 here: measured in Phase-2 audit and filled
    into qwen3vl.output_dim. While output_dim<=0 (sentinel) forward records the real
    dim on first call (self._measured_dim) and skips the hard assert, so the audit
    probe can report print(embedding.shape) before config fill; once filled (>0) the
    assert is active exactly like WeMM/DINO.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        local_dir: str | None = None,
        dtype: str = "bf16",
        n_patches: int = 1,
        output_dim: int = 0,          # sentinel until Phase-2 audit
        freeze: bool = True,
        encode_batch_size: int = 8,
        image_only: bool = True,
    ):
        super().__init__()
        assert n_patches == 1, f"qwen3vl cond stream is P=1, got n_patches={n_patches}"
        assert image_only, "Phase-1 contract is IMAGE-ONLY (image_only=True); text unsupported"
        from patch_policy.qwen3vl_official import Qwen3VLEmbedder

        self.model_name = model_name
        self.n_patches = n_patches
        self.output_dim = output_dim
        self._measured_dim: int | None = None
        self.dtype = DTYPES[dtype]
        self.encode_batch_size = encode_batch_size
        self.image_only = image_only
        path = local_dir or model_name

        # Official loader. dtype= kwarg is the transformers>=5 API (torch_dtype legacy).
        # Qwen3VLEmbedder picks cuda:0 when available; registering self.model below
        # lets the caller's .to(device)/.eval() cascade reach the underlying weights.
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=path,
            dtype=self.dtype,
        )
        self.model = self.embedder.model  # nn.Module child -> .to()/.eval() work
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    # ---- tensor helpers (mirror WeMMEncoder) ---------------------------------

    @staticmethod
    def _to_pil(images: torch.Tensor):
        from PIL import Image

        imgs = (images.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).round().to(
            torch.uint8
        ).cpu()
        return [Image.fromarray(a.numpy(), mode="RGB") for a in imgs]

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: [N, 3, H, W] float 0..1 -> [N, D] L2-normalized (official path)."""
        pil = self._to_pil(images)
        with torch.autocast(self.model.device.type, dtype=self.dtype):
            # bare-image inputs only; official default system instruction is constant
            emb = self.embedder.process([{"image": im} for im in pil], normalize=True)
        emb = emb.to(torch.float32)
        if self.output_dim > 0:
            assert emb.shape[-1] == self.output_dim, (
                f"Qwen3-VL embedding dim {emb.shape[-1]} != qwen3vl.output_dim "
                f"{self.output_dim} (audit-fill mismatch)"
            )
        if self._measured_dim is None:
            self._measured_dim = int(emb.shape[-1])
        return emb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [..., 3, H, W] float 0..1 -> [..., P=1, D]."""
        prefix_shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])  # [N,3,H,W]
        emb = self.encode_images(flat)                 # [N, D]
        emb = emb.reshape(*prefix_shape, self.n_patches, emb.shape[-1])
        assert emb.ndim >= 3
        assert emb.shape[-2] == self.n_patches
        if self.output_dim > 0:
            assert emb.shape[-1] == self.output_dim
        assert torch.isfinite(emb).all(), "Qwen3-VL output contains NaN/Inf"
        return emb


def build_encoder(cfg) -> tuple[nn.Module, dict]:
    """Construct the active encoder (wemm | dino) per cfg.encoder_name.

    Returns (encoder, meta_dict). meta_dict's keys are the cache-meta contract
    (model_name / dtype / n_patches / output_dim) shared by precompute.py and
    train.py's check_meta.
    """
    enc = cfg.enc()
    if cfg.encoder_name == "oracle":
        # Oracle is cache-only (diagnostic). No vision encoder exists; training reads
        # oracle_cache/pusht_state.pt. A future oracle ROLLOUT would feed env simulator
        # state (info['pos_agent'] / info['block_pose']), not a vision encoder.
        raise NotImplementedError(
            "build_encoder does not support encoder_name='oracle': training is "
            "cache-only (patch_policy.build_oracle_cache); no vision encoder exists. "
            "Oracle rollout (env-state feeding) is a separate later phase."
        )
    enc_kwargs = dict(enc.__dict__)
    # WeMM-Repeat-P256 / Coord: repeat_patches & decoder* are POLICY-input interface
    # transforms (dataset.expand_repeat_patches / model CoordSpatialDecoder), not encoder
    # concerns — the frozen WeMMEncoder stays P=1.
    for _k in ("repeat_patches", "decoder", "decoder_patches", "decoder_pe_dim",
               "decoder_hidden", "decoder_out_dim"):
        enc_kwargs.pop(_k, None)
    if cfg.encoder_name == "dino":
        encoder = DINOEncoder(**enc_kwargs)
        meta = {
            "encoder": "dino",
            "model_name": enc.name,
            "dtype": enc.dtype,
            "n_patches": enc.n_patches,
            "output_dim": enc.output_dim,
            "feature_key": enc.feature_key,
            "postprocess": enc.postprocess,
        }
    else:
        encoder = WeMMEncoder(**enc_kwargs)
        meta = {
            "encoder": "wemm",
            "model_name": enc.model_name,
            "dtype": enc.dtype,
            "n_patches": enc.n_patches,
            "output_dim": enc.output_dim,
        }
    return encoder, meta


def build_cond_encoder(cfg) -> nn.Module:
    """Construct the fusion conditioning-stream encoder (fusion only; ONLINE).

    Dispatches on cfg.fusion.cond_encoder — 'wemm' -> WeMM-Embedding-2B (原样),
    'qwen3vl' -> Qwen3-VL-Embedding-2B (本实验唯一科学变量)。Training reads this
    stream from disk cache (load_embedding_cache); this builder exists for
    eval_rollout's ONLINE encode of live obs frames, so it must produce the exact
    same representation the cache was made with (same model_name / dtype / P=1).
    """
    cc = cfg.cond_config()
    if cfg.fusion.cond_encoder == "wemm":
        encoder = WeMMEncoder(
            model_name=cc.model_name, local_dir=cc.local_dir, dtype=cc.dtype,
            n_patches=cc.n_patches, output_dim=cc.output_dim, freeze=cc.freeze,
            encode_batch_size=cc.encode_batch_size,
            local_mode=cc.local_mode, crop_window=cc.crop_window,
        )
    elif cfg.fusion.cond_encoder == "qwen3vl":
        encoder = Qwen3VLEmbeddingEncoder(
            model_name=cc.model_name, local_dir=cc.local_dir, dtype=cc.dtype,
            n_patches=cc.n_patches, output_dim=cc.output_dim, freeze=cc.freeze,
            encode_batch_size=cc.encode_batch_size, image_only=cc.image_only,
        )
    else:
        raise ValueError(f"unknown fusion.cond_encoder: {cfg.fusion.cond_encoder!r} (wemm | qwen3vl)")
    if cfg.fusion.cond_dim != cc.output_dim:
        raise AssertionError(
            f"fusion.cond_dim {cfg.fusion.cond_dim} != cond_encoder output_dim "
            f"{cc.output_dim} ({cfg.fusion.cond_encoder})"
        )
    return encoder


def check_frozen(encoder: nn.Module) -> bool:
    mod = getattr(encoder, "model", None) or encoder.base_model
    return all(not p.requires_grad for p in mod.parameters())
