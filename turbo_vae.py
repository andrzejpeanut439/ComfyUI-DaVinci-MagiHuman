"""
TurboVAED - Distilled decode-only VAE for daVinci-MagiHuman.
Based on Wan2.2 VAE with lightweight architecture.
Supports sliding window temporal decoding for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import Optional, List


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution for efficiency."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_dw=False, dw_kernel_size=5):
        super().__init__()
        if is_dw and in_channels == out_channels:
            dw_pad = dw_kernel_size // 2
            self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=dw_kernel_size,
                                        stride=stride, padding=dw_pad, groups=in_channels)
            self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.depthwise = None
            self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, x):
        if self.depthwise is not None:
            x = self.depthwise(x)
        return self.pointwise(x)


class TurboRMSNorm3d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
        return (x * norm).to(dtype) * self.weight


class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, is_dw=False, dw_kernel_size=5):
        super().__init__()
        self.norm1 = TurboRMSNorm3d(in_ch)
        self.conv1 = DepthwiseSeparableConv3d(in_ch, out_ch, 3, 1, 1, is_dw, dw_kernel_size)
        self.norm2 = TurboRMSNorm3d(out_ch)
        self.conv2 = DepthwiseSeparableConv3d(out_ch, out_ch, 3, 1, 1, is_dw, dw_kernel_size)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Upsample3d(nn.Module):
    def __init__(self, channels, spatial=True, temporal=True):
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.conv = nn.Conv3d(channels, channels, 3, 1, 1)

    def forward(self, x):
        scale = []
        if self.temporal:
            scale = [2, 1, 1]
        if self.spatial:
            if scale:
                scale = [scale[0], 2, 2]
            else:
                scale = [1, 2, 2]
        if not scale:
            return self.conv(x)
        x = F.interpolate(x, scale_factor=scale, mode='nearest')
        return self.conv(x)


class TurboVAEDecoder(nn.Module):
    """Turbo VAE Decoder for fast video decoding."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        latent_channels = config.get("latent_channels", 48)
        block_out_channels = config.get("decoder_block_out_channels", [64, 128, 256, 512])
        layers_per_block = config.get("decoder_layers_per_block", [2, 2, 2, 3, 3])
        patch_size = config.get("patch_size", 2)
        patch_size_t = config.get("patch_size_t", 1)
        spatio_temporal_scaling = config.get("decoder_spatio_temporal_scaling", [False, True, True, True])
        spatio_only = config.get("decoder_spatio_only", [False, True, False, False])
        is_dw_conv = config.get("decoder_is_dw_conv", [False, False, False, False, False])
        dw_kernel_size = config.get("decoder_dw_kernel_size", 5)
        use_unpatchify = config.get("use_unpatchify", True)
        out_channels = config.get("out_channels", 3)

        self.spatial_compression_ratio = config.get("spatial_compression_ratio", 16)
        self.temporal_compression_ratio = config.get("temporal_compression_ratio", 4)
        self.first_chunk_size = config.get("first_chunk_size", 7)
        self.step_size = config.get("step_size", 7)
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        # Channel normalization constants (from Wan2.2 VAE)
        self.register_buffer('latent_mean', torch.zeros(latent_channels))
        self.register_buffer('latent_std', torch.ones(latent_channels))

        # Initial conv from latent space
        reversed_channels = list(reversed(block_out_channels))
        self.conv_in = nn.Conv3d(latent_channels, reversed_channels[0], 3, 1, 1)

        # Decoder blocks (going from deepest to shallowest)
        self.blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        in_ch = reversed_channels[0]
        # First set of res blocks (no upsampling)
        block_resblocks = nn.ModuleList()
        for _ in range(layers_per_block[0]):
            block_resblocks.append(ResBlock3d(in_ch, in_ch, is_dw_conv[0], dw_kernel_size))
        self.blocks.append(block_resblocks)
        self.upsamplers.append(None)

        # Remaining blocks with upsampling
        for i, out_ch in enumerate(reversed_channels[1:]):
            block_resblocks = nn.ModuleList()
            for j in range(layers_per_block[i + 1]):
                ch_in = in_ch if j == 0 else out_ch
                block_resblocks.append(ResBlock3d(ch_in, out_ch, is_dw_conv[min(i+1, len(is_dw_conv)-1)], dw_kernel_size))
            self.blocks.append(block_resblocks)

            temporal = spatio_temporal_scaling[i] if i < len(spatio_temporal_scaling) else True
            spatial = not spatio_only[i] if i < len(spatio_only) else True
            self.upsamplers.append(Upsample3d(out_ch, spatial=spatial, temporal=temporal))
            in_ch = out_ch

        # Final conv to pixel space
        self.norm_out = TurboRMSNorm3d(block_out_channels[0])
        if use_unpatchify:
            self.conv_out = nn.Conv3d(block_out_channels[0], out_channels * patch_size * patch_size * patch_size_t, 3, 1, 1)
        else:
            self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, 3, 1, 1)
        self.use_unpatchify = use_unpatchify

    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a chunk of latent frames."""
        h = self.conv_in(z)

        for block, upsampler in zip(self.blocks, self.upsamplers):
            for resblock in block:
                h = resblock(h)
            if upsampler is not None:
                h = upsampler(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        if self.use_unpatchify:
            # Unpatchify: reshape patch dimensions back to spatial
            B, C, T, H, W = h.shape
            out_c = 3  # RGB
            h = h.reshape(B, out_c, self.patch_size_t, self.patch_size, self.patch_size, T, H, W)
            h = h.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            h = h.reshape(B, out_c, T * self.patch_size_t, H * self.patch_size, W * self.patch_size)

        return h

    def forward(self, z: torch.Tensor, output_offload: bool = False) -> torch.Tensor:
        """
        Decode latent video to pixel space with sliding window.

        Args:
            z: [B, C, T, H, W] latent video tensor
            output_offload: if True, move decoded chunks to CPU immediately (for 1080p)

        Returns:
            Decoded video [B, 3, T_out, H_out, W_out]
        """
        # Normalize latents
        z = (z - self.latent_mean.view(1, -1, 1, 1, 1)) / self.latent_std.view(1, -1, 1, 1, 1)

        T = z.shape[2]
        if T <= self.first_chunk_size:
            return self.decode_chunk(z)

        # Sliding window decode
        chunks = []

        # First chunk
        first = self.decode_chunk(z[:, :, :self.first_chunk_size])
        if output_offload:
            first = first.cpu()
        chunks.append(first)

        # Remaining chunks
        pos = self.first_chunk_size
        while pos < T:
            end = min(pos + self.step_size, T)
            chunk = self.decode_chunk(z[:, :, pos:end])
            if output_offload:
                chunk = chunk.cpu()
            chunks.append(chunk)
            pos = end

        return torch.cat(chunks, dim=2)


def load_turbo_vae(model_dir: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> TurboVAEDecoder:
    """Load TurboVAED from config JSON and checkpoint."""
    # Find config JSON
    config_path = None
    ckpt_path = None
    for f in os.listdir(model_dir):
        if f.endswith('.json') and 'index' not in f:
            config_path = os.path.join(model_dir, f)
        if f.endswith('.ckpt') or f.endswith('.safetensors'):
            ckpt_path = os.path.join(model_dir, f)

    if config_path is None:
        raise FileNotFoundError(f"No config JSON found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model = TurboVAEDecoder(config)

    if ckpt_path and os.path.exists(ckpt_path):
        if ckpt_path.endswith('.ckpt'):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Handle EMA state dict
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            elif 'ema_state_dict' in ckpt:
                ckpt = ckpt['ema_state_dict']
            model.load_state_dict(ckpt, strict=False)
        else:
            from safetensors.torch import load_file
            ckpt = load_file(ckpt_path, device=device)
            model.load_state_dict(ckpt, strict=False)

    model = model.to(dtype)
    return model
