"""
Data proxy for daVinci-MagiHuman.
Handles patchification of video latents and token packing for the single-stream transformer.
"""

import torch
import math
import numpy as np
from typing import Tuple, Optional


# Modality IDs
VIDEO_ID = 0
AUDIO_ID = 1
TEXT_ID = 2


class MagiDataProxy:
    """Handles packing video/audio/text tokens into a single sequence for the DiT."""

    def __init__(
        self,
        z_dim: int = 48,
        patch_size: int = 2,
        t_patch_size: int = 1,
        vae_stride: Tuple[int, int, int] = (4, 16, 16),
        fps: int = 25,
    ):
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.vae_stride = vae_stride
        self.fps = fps

        # Video channels after patchification
        self.video_channels = z_dim * patch_size * patch_size  # 48 * 2 * 2 = 192
        self.audio_channels = 64  # Fixed audio latent dim

    def get_latent_shape(
        self,
        height: int,
        width: int,
        num_frames: int,
    ) -> Tuple[int, int, int, int, int]:
        """Get the latent tensor shape for given video dimensions.

        Returns: (batch, channels, latent_t, latent_h, latent_w)
        """
        latent_t = (num_frames - 1) // self.vae_stride[0] + 1
        latent_h = height // self.vae_stride[1]
        latent_w = width // self.vae_stride[2]
        return (1, self.z_dim, latent_t, latent_h, latent_w)

    def patchify_video(self, video_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert video latent [B, C, T, H, W] to patched tokens [B, num_tokens, patch_channels].

        Returns: (tokens, coords)
        """
        B, C, T, H, W = video_latent.shape
        p = self.patch_size
        tp = self.t_patch_size

        # Ensure dimensions are divisible by patch size
        assert H % p == 0 and W % p == 0, f"H={H}, W={W} must be divisible by patch_size={p}"

        pH = H // p
        pW = W // p
        pT = T // tp

        # Reshape: [B, C, T, H, W] -> [B, C, pT, tp, pH, p, pW, p]
        x = video_latent.reshape(B, C, pT, tp, pH, p, pW, p)
        # -> [B, pT, pH, pW, C*tp*p*p]
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        tokens = x.reshape(B, pT * pH * pW, C * tp * p * p)

        # Generate coordinates for each patch
        # coords: [t, y, x] for each token
        coords = torch.zeros(B, pT * pH * pW, 3, device=video_latent.device, dtype=video_latent.dtype)
        idx = 0
        for t in range(pT):
            for h in range(pH):
                for w in range(pW):
                    coords[:, idx, 0] = t  # temporal
                    coords[:, idx, 1] = h  # height
                    coords[:, idx, 2] = w  # width
                    idx += 1

        return tokens, coords

    def unpatchify_video(
        self,
        tokens: torch.Tensor,
        T: int, H: int, W: int,
    ) -> torch.Tensor:
        """Convert patched tokens back to video latent.

        tokens: [B, num_tokens, patch_channels]
        Returns: [B, C, T, H, W]
        """
        B = tokens.shape[0]
        p = self.patch_size
        tp = self.t_patch_size
        C = self.z_dim

        pH = H // p
        pW = W // p
        pT = T // tp

        # [B, pT*pH*pW, C*tp*p*p] -> [B, pT, pH, pW, C, tp, p, p]
        x = tokens.reshape(B, pT, pH, pW, C, tp, p, p)
        # -> [B, C, pT, tp, pH, p, pW, p]
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(B, C, T, H, W)

    def prepare_audio_tokens(
        self,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create audio token placeholders and coordinates.

        Audio has ~4.17 tokens per video frame at 25fps (100 audio tokens/sec with stride 160 at 16kHz).
        For simplicity, we create audio_length = num_frames * audio_rate / fps tokens.
        """
        # Audio: 16kHz sample rate, hop=160 -> 100 frames/sec
        # Video: 25 fps
        # Audio tokens per video frame: 100/25 = 4
        video_duration = num_frames / self.fps  # seconds
        audio_token_count = int(video_duration * 100)  # 100 audio frames/sec

        tokens = torch.randn(1, audio_token_count, self.audio_channels, device=device, dtype=dtype)
        coords = torch.zeros(1, audio_token_count, 3, device=device, dtype=dtype)
        for i in range(audio_token_count):
            coords[:, i, 0] = i  # temporal position only

        return tokens, coords

    def prepare_text_coords(
        self,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Create text coordinates (all zeros - text has no spatial/temporal position)."""
        B, S, _ = text_tokens.shape
        return torch.zeros(B, S, 3, device=text_tokens.device, dtype=text_tokens.dtype)

    def build_sequence(
        self,
        video_tokens: torch.Tensor,
        video_coords: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_coords: torch.Tensor,
        text_tokens: torch.Tensor,
        text_coords: torch.Tensor,
    ) -> dict:
        """Pack all modality tokens into a single sequence with metadata.

        Returns dict with:
            - video_tokens, audio_tokens, text_tokens
            - video_coords, audio_coords, text_coords
            - modality_ids: [B, total_seq_len] with 0=video, 1=audio, 2=text
            - video_mask, audio_mask: boolean masks
        """
        B = video_tokens.shape[0]
        sv = video_tokens.shape[1]
        sa = audio_tokens.shape[1]
        st = text_tokens.shape[1]
        total = sv + sa + st
        device = video_tokens.device

        modality_ids = torch.zeros(B, total, dtype=torch.long, device=device)
        modality_ids[:, sv:sv + sa] = AUDIO_ID
        modality_ids[:, sv + sa:] = TEXT_ID

        video_mask = torch.zeros(B, total, dtype=torch.bool, device=device)
        video_mask[:, :sv] = True

        audio_mask = torch.zeros(B, total, dtype=torch.bool, device=device)
        audio_mask[:, sv:sv + sa] = True

        return {
            "video_tokens": video_tokens,
            "audio_tokens": audio_tokens,
            "text_tokens": text_tokens,
            "video_coords": video_coords,
            "audio_coords": audio_coords,
            "text_coords": text_coords,
            "modality_ids": modality_ids,
            "video_mask": video_mask,
            "audio_mask": audio_mask,
            "video_shape": (sv,),
            "audio_shape": (sa,),
        }
