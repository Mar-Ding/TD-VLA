"""VQ-VAE action tokenizer (residual VQ, baseline-compatible).

Uses the same ResidualVQ (lucidrains vector-quantize-pytorch) as the base repo,
so codebook training behavior (EMA, L2-norm input) matches the baseline exactly.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


class EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, layer_num: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x))


class VqVae(nn.Module):
    def __init__(
        self,
        input_dim_h: int,      # action chunk length W
        input_dim_w: int,      # action dim A
        n_latent_dims: int,    # 512
        vqvae_n_embed: int,    # codebook size per group
        vqvae_groups: int,     # number of residual quantizers G
        encoder_loss_multiplier: float = 1.0,
        act_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.n_latent_dims = n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_groups = vqvae_groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        self.vq_layer = ResidualVQ(
            dim=n_latent_dims, num_quantizers=vqvae_groups, codebook_size=vqvae_n_embed
        )
        self.encoder = EncoderMLP(input_dim_w * input_dim_h, n_latent_dims)
        self.decoder = EncoderMLP(n_latent_dims, input_dim_w * input_dim_h)

    # ---- preprocess --------------------------------------------------------

    def preprocess(self, state: torch.Tensor) -> torch.Tensor:
        state = state / self.act_scale
        return einops.rearrange(state, "... T A -> ... (T A)")

    # ---- quantization paths ------------------------------------------------

    def get_code(self, state: torch.Tensor):
        """state: [..., W, A] -> (state_vq [..., latent], vq_code [..., G]) under no_grad."""
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_vq, vq_code, _ = self.vq_layer(state_rep)
        return state_vq, vq_code

    def draw_code_forward(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """codes: [..., G] -> latent [..., latent] (sum over residual quantizers)."""
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [..., latent] -> [..., W, A] (rescaled by act_scale)."""
        out = self.decoder(latent) * self.act_scale
        return einops.rearrange(out, "... (T A) -> ... T A", A=self.input_dim_w)

    # ---- training forward ---------------------------------------------------

    def forward(self, state: torch.Tensor):
        """state: [N, W, A] -> (loss, vq_code, loss_dict). Trains codebooks via EMA."""
        state = self.preprocess(state)  # [N, W*A]
        state_rep = self.encoder(state)  # [N, latent]
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.shape[0], -1, state_rep.shape[-1])
        state_vq, vq_code, vq_loss = self.vq_layer(state_rep_flat)
        state_vq = state_vq.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        dec_out = self.decoder(state_vq)

        vq_loss = vq_loss.sum()
        encoder_loss = (state - dec_out).abs().mean()
        vqvae_recon_loss = F.mse_loss(state, dec_out)
        loss = encoder_loss * self.encoder_loss_multiplier + vq_loss * 5
        loss_dict = {
            "vq_loss": vq_loss.detach().item(),
            "recon_loss": vqvae_recon_loss.detach().item(),
            "encoder_loss": encoder_loss.detach().item(),
        }
        return loss, vq_code, loss_dict
