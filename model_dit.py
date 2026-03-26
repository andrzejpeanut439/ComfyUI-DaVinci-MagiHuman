"""
daVinci-MagiHuman DiT Model - 15B Single-Stream Transformer
Architecture: 40 layers, hidden=5120, GQA 40q/8kv heads, head_dim=128
Sandwich layers: 0-3 and 36-39 have multi-modality norms (3 experts)
Shared layers: 4-35 have single norms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm).to(dtype) * self.weight


class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with per-modality weights for sandwich layers."""
    def __init__(self, dim: int, num_modality: int = 1, eps: float = 1e-6):
        super().__init__()
        self.num_modality = num_modality
        self.eps = eps
        if num_modality > 1:
            self.weight = nn.Parameter(torch.ones(num_modality, dim))
        else:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, modality_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * norm

        if self.num_modality > 1 and modality_ids is not None:
            # Gather per-token modality weights
            w = self.weight[modality_ids]  # [seq_len, dim]
            return x.to(dtype) * w
        else:
            return x.to(dtype) * self.weight


class ElementWiseFourierEmbed(nn.Module):
    """RoPE-style positional embedding using learned frequency bands."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.bands = nn.Parameter(torch.randn(dim // 2))

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        coords: [batch, seq_len, num_coords] - spatial/temporal coordinates
        Returns: (cos, sin) each [batch, seq_len, dim]
        """
        # coords shape: [B, S, C] where C is number of coordinate dimensions
        # bands shape: [dim//2]
        # We compute freqs = coords @ bands_matrix to get [B, S, dim//2]
        freqs = torch.einsum('bsc,d->bsd', coords.float(), self.bands.float())
        freqs = freqs * 2 * math.pi
        return freqs.cos().to(coords.dtype), freqs.sin().to(coords.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 5120,
        num_query_heads: int = 40,
        num_kv_groups: int = 8,
        head_dim: int = 128,
        num_modality: int = 1,
        enable_gating: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.enable_gating = enable_gating

        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality)

        # QKV + gating projection
        # Q: num_query_heads * head_dim
        # K: num_kv_groups * head_dim
        # V: num_kv_groups * head_dim
        # Gate: num_query_heads (if gating enabled)
        qkv_dim = (num_query_heads + 2 * num_kv_groups) * head_dim
        if enable_gating:
            qkv_dim += num_query_heads
        self.linear_qkv = nn.Linear(hidden_size, qkv_dim, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.linear_proj = nn.Linear(num_query_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        # Pre-norm
        normed = self.pre_norm(x, modality_ids)

        # QKV projection
        qkv = self.linear_qkv(normed)

        # Split into Q, K, V (and optionally gate)
        q_dim = self.num_query_heads * self.head_dim
        k_dim = self.num_kv_groups * self.head_dim
        v_dim = k_dim

        q = qkv[..., :q_dim].reshape(B, S, self.num_query_heads, self.head_dim)
        k = qkv[..., q_dim:q_dim + k_dim].reshape(B, S, self.num_kv_groups, self.head_dim)
        v = qkv[..., q_dim + k_dim:q_dim + k_dim + v_dim].reshape(B, S, self.num_kv_groups, self.head_dim)

        if self.enable_gating:
            gate = qkv[..., q_dim + k_dim + v_dim:]  # [B, S, num_query_heads]
            gate = torch.sigmoid(gate)

        # Per-head RMSNorm on Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope_kv(k, rope_cos, rope_sin, self.num_query_heads, self.num_kv_groups)

        # GQA: repeat K,V for query groups
        if self.num_kv_groups < self.num_query_heads:
            repeat_factor = self.num_query_heads // self.num_kv_groups
            k = k.unsqueeze(3).expand(-1, -1, -1, repeat_factor, -1).reshape(B, S, self.num_query_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, repeat_factor, -1).reshape(B, S, self.num_query_heads, self.head_dim)

        # Transpose for attention: [B, heads, S, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        # [B, heads, S, dim] -> [B, S, heads*dim]
        out = out.transpose(1, 2).reshape(B, S, self.num_query_heads * self.head_dim)

        # Per-head gating
        if self.enable_gating:
            out = out.reshape(B, S, self.num_query_heads, self.head_dim)
            out = out * gate.unsqueeze(-1)
            out = out.reshape(B, S, self.num_query_heads * self.head_dim)

        # Output projection
        out = self.linear_proj(out)

        return out


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 5120,
        num_modality: int = 1,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.pre_norm = MultiModalityRMSNorm(hidden_size, num_modality)

        if use_swiglu:
            # SwiGLU: hidden -> 2*intermediate (gate+up) -> hidden
            intermediate = int(hidden_size * 4 * 2 / 3)
            # Round to nearest multiple of 4
            intermediate = ((intermediate + 3) // 4) * 4
            self.up_gate_proj = nn.Linear(hidden_size, intermediate * 2, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        else:
            # GELU: hidden -> 4*hidden -> hidden
            intermediate = hidden_size * 4
            self.up_gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, modality_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.pre_norm(x, modality_ids)

        if self.use_swiglu:
            up_gate = self.up_gate_proj(normed)
            gate, up = up_gate.chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)
        else:
            return self.down_proj(F.gelu(self.up_gate_proj(normed)))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 5120,
        num_query_heads: int = 40,
        num_kv_groups: int = 8,
        head_dim: int = 128,
        mm_layers: list = None,
        enable_gating: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        mm_layers = mm_layers or [0, 1, 2, 3, 36, 37, 38, 39]
        num_modality = 3 if layer_idx in mm_layers else 1
        # Layers 0-3 use GELU, layers 4+ use SwiGLU
        use_swiglu = layer_idx >= 4

        self.attention = Attention(
            hidden_size=hidden_size,
            num_query_heads=num_query_heads,
            num_kv_groups=num_kv_groups,
            head_dim=head_dim,
            num_modality=num_modality,
            enable_gating=enable_gating,
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            num_modality=num_modality,
            use_swiglu=use_swiglu,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(x, rope_cos, rope_sin, modality_ids, attention_mask)
        x = x + self.mlp(x, modality_ids)
        return x


class DiTModel(nn.Module):
    """daVinci-MagiHuman 15B DiT Model"""

    def __init__(
        self,
        num_layers: int = 40,
        hidden_size: int = 5120,
        num_query_heads: int = 40,
        num_kv_groups: int = 8,
        head_dim: int = 128,
        video_in_channels: int = 192,  # 48 * 4 (z_dim * patch_size^2)
        audio_in_channels: int = 64,
        text_in_channels: int = 3584,  # T5Gemma embedding dim
        mm_layers: list = None,
        enable_gating: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        # Input adapters
        self.video_embedder = nn.Linear(video_in_channels, hidden_size)
        self.audio_embedder = nn.Linear(audio_in_channels, hidden_size)
        self.text_embedder = nn.Linear(text_in_channels, hidden_size)

        # RoPE
        self.rope = ElementWiseFourierEmbed(head_dim)

        # Transformer blocks
        mm_layers = mm_layers or [0, 1, 2, 3, 36, 37, 38, 39]
        self.layers = nn.ModuleList([
            TransformerLayer(
                layer_idx=i,
                hidden_size=hidden_size,
                num_query_heads=num_query_heads,
                num_kv_groups=num_kv_groups,
                head_dim=head_dim,
                mm_layers=mm_layers,
                enable_gating=enable_gating,
            )
            for i in range(num_layers)
        ])

        # Output heads
        self.final_norm_video = RMSNorm(hidden_size)
        self.final_norm_audio = RMSNorm(hidden_size)
        self.final_linear_video = nn.Linear(hidden_size, video_in_channels, bias=False)
        self.final_linear_audio = nn.Linear(hidden_size, audio_in_channels, bias=False)

    def forward(
        self,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        text_coords: torch.Tensor,
        modality_ids: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the single-stream transformer.

        Args:
            video_tokens: [B, Sv, video_in_channels] patchified video latents
            audio_tokens: [B, Sa, audio_in_channels] audio latents
            text_tokens: [B, St, text_in_channels] text embeddings
            video_coords: [B, Sv, num_coord_dims] video spatial-temporal coords
            audio_coords: [B, Sa, num_coord_dims] audio coords
            text_coords: [B, St, num_coord_dims] text coords
            modality_ids: [B, S] modality id per token (0=video, 1=audio, 2=text)
            video_mask: [B, S] boolean mask for video tokens
            audio_mask: [B, S] boolean mask for audio tokens
        """
        # Embed each modality
        v = self.video_embedder(video_tokens)
        a = self.audio_embedder(audio_tokens)
        t = self.text_embedder(text_tokens)

        # Concatenate into single sequence
        x = torch.cat([v, a, t], dim=1)  # [B, S, hidden]
        coords = torch.cat([video_coords, audio_coords, text_coords], dim=1)

        # Compute RoPE
        rope_cos, rope_sin = self.rope(coords)

        # Run through transformer layers
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, modality_ids, attention_mask)

        # Extract video and audio predictions
        video_out = self.final_linear_video(self.final_norm_video(x[video_mask]))
        audio_out = self.final_linear_audio(self.final_norm_audio(x[audio_mask]))

        return video_out, audio_out


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to query tensor.
    x: [B, S, heads, dim]
    cos, sin: [B, S, dim]
    """
    dim = x.shape[-1]
    cos = cos[..., :dim].unsqueeze(2)  # [B, S, 1, dim]
    sin = sin[..., :dim].unsqueeze(2)

    x1, x2 = x[..., :dim // 2], x[..., dim // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def apply_rope_kv(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   num_q_heads: int, num_kv_groups: int) -> torch.Tensor:
    """Apply RoPE to key tensor (may have fewer heads)."""
    dim = k.shape[-1]
    cos = cos[..., :dim].unsqueeze(2)
    sin = sin[..., :dim].unsqueeze(2)

    k1, k2 = k[..., :dim // 2], k[..., dim // 2:]
    rotated = torch.cat([-k2, k1], dim=-1)
    return k * cos + rotated * sin


def load_dit_from_sharded(model_dir: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> DiTModel:
    """Load DiTModel from sharded safetensors checkpoint."""
    import json
    import os
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Create model
    model = DiTModel()

    # Remap weight names from checkpoint to our model
    # Checkpoint: adapter.*, block.layers.*, final_*
    # Our model: video_embedder, audio_embedder, text_embedder, rope, layers.*, final_*
    key_map = {
        "adapter.video_embedder.": "video_embedder.",
        "adapter.audio_embedder.": "audio_embedder.",
        "adapter.text_embedder.": "text_embedder.",
        "adapter.rope.": "rope.",
        "block.layers.": "layers.",
        "final_norm_video.": "final_norm_video.",
        "final_norm_audio.": "final_norm_audio.",
        "final_linear_video.": "final_linear_video.",
        "final_linear_audio.": "final_linear_audio.",
    }

    # Group weights by shard file
    shard_to_keys = {}
    for key, shard_file in index["weight_map"].items():
        if shard_file not in shard_to_keys:
            shard_to_keys[shard_file] = []
        shard_to_keys[shard_file].append(key)

    # Load shard by shard
    state_dict = {}
    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            continue
        shard_data = load_file(shard_path, device=device)
        for key in shard_to_keys[shard_file]:
            if key in shard_data:
                # Remap key
                new_key = key
                for old_prefix, new_prefix in key_map.items():
                    if key.startswith(old_prefix):
                        new_key = new_prefix + key[len(old_prefix):]
                        break
                state_dict[new_key] = shard_data[key].to(dtype)
        del shard_data

    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype)
    return model
