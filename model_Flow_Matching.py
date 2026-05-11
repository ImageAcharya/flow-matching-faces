"""
model_Flow_Matching.py
--------
Defines the Flow Matching UNet architecture and the forward (noising) process.

Exports:
    - TimeEmbedding   : sinusoidal time embedding
    - SelfAttention   : lightweight multi-head self-attention block
    - ResBlock        : residual conv block conditioned on time
    - Down / Up       : encoder / decoder stages
    - FlowUNet        : full UNet velocity network
    - forward_process : adds noise to an image at time t  →  (I_t, v_true)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Forward Process (Flow Matching)
# ─────────────────────────────────────────────────────────────
def forward_process(I: torch.Tensor, t: torch.Tensor):
    """
    Linearly interpolate between clean image I (t=0) and Gaussian noise (t=1).

    Args:
        I : (B, C, H, W) clean images normalised to [-1, 1]
        t : (B,)          timesteps in [0, 1]

    Returns:
        I_t    : noisy image at time t
        v_true : target velocity  (noise - image)
    """
    eps    = torch.randn_like(I)
    t_view = t.view(-1, 1, 1, 1)
    I_t    = (1 - t_view) * I + t_view * eps
    v_true = eps - I
    return I_t, v_true


# ─────────────────────────────────────────────────────────────
# Sinusoidal Time Embedding
# ─────────────────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    """Maps scalar timestep t ∈ [0,1] to a fixed-dim sinusoidal vector."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device)
            * torch.log(torch.tensor(10_000.0)) / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


# ─────────────────────────────────────────────────────────────
# Self-Attention Block  (bottleneck only)
# ─────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    """Lightweight spatial self-attention with GroupNorm + residual."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)        # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h                                      # residual


# ─────────────────────────────────────────────────────────────
# Residual Block  (time-conditioned)
# ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Two-layer conv block with time injection and GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1     = nn.GroupNorm(8, out_ch)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.act       = nn.SiLU()
        self.res       = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return self.act(h + self.res(x))


# ─────────────────────────────────────────────────────────────
# Encoder / Decoder Stages
# ─────────────────────────────────────────────────────────────
class Down(nn.Module):
    """ResBlock  →  stride-2 conv downsample.  Returns (downsampled, skip)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim)
        self.down  = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        x = self.block(x, t)
        return self.down(x), x          # (downsampled, skip connection)


class Up(nn.Module):
    """Transposed-conv upsample  →  skip concat  →  ResBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t)


# ─────────────────────────────────────────────────────────────
# Flow Matching UNet
# ─────────────────────────────────────────────────────────────
class FlowUNet(nn.Module):
    """
    4-level UNet that predicts the velocity field v(x_t, t).

    Architecture:
        Encoder : 3 → 64 → 128 → 256 → 512
        Bottleneck: 512 → 512 + SelfAttention
        Decoder : 512 → 256 → 128 → 64 → 3
    """

    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        # Encoder
        self.inc   = ResBlock(3,   64,  time_dim)
        self.down1 = Down(64,  128, time_dim)
        self.down2 = Down(128, 256, time_dim)
        self.down3 = Down(256, 512, time_dim)

        # Bottleneck
        self.mid  = ResBlock(512, 512, time_dim)
        self.attn = SelfAttention(512)

        # Decoder
        self.up3 = Up(512, 512, 256, time_dim)
        self.up2 = Up(256, 256, 128, time_dim)
        self.up1 = Up(128, 128, 64,  time_dim)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        x1 = self.inc(x, t_emb)
        x2, s1 = self.down1(x1, t_emb)
        x3, s2 = self.down2(x2, t_emb)
        x4, s3 = self.down3(x3, t_emb)

        x = self.mid(x4, t_emb)
        x = self.attn(x)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        return self.out(x)