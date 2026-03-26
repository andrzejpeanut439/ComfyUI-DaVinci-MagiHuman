"""
daVinci-MagiHuman DiT Model - 15B Single-Stream Transformer
Architecture: 40 layers, hidden=5120, GQA 40q/8kv heads, head_dim=128

Sandwich layers (0-3, 36-39): weights are 3x concatenated for 3 modalities (video/audio/text)
Shared layers (4-35): single set of weights

Weight shapes from checkpoint:
  Sandwich attention:
    pre_norm.weight:   [15360]         = 3 * 5120 (flat concat)
    linear_qkv.weight: [21624, 5120]   = 3 * 7208 (Q:5120 + K:1024 + V:1024 + Gate:40 = 7208 per modality)
    q_norm.weight:     [384]           = 3 * 128
    k_norm.weight:     [384]           = 3 * 128
    linear_proj.weight:[15360, 5120]   = 3 * 5120

  Sandwich MLP (layers 0-3, GELU):
    pre_norm.weight:     [15360]         = 3 * 5120
    up_gate_proj.weight: [61440, 5120]   = 3 * 20480 (GELU, intermediate=20480)
    down_proj.weight:    [15360, 20480]  = 3 * 5120

  Sandwich MLP (layers 36-39, SwiGLU):
    pre_norm.weight:     [15360]         = 3 * 5120
    up_gate_proj.weight: [81912, 5120]   = 3 * 27304 (SwiGLU, 2*13652)
    down_proj.weight:    [15360, 13652]  = 3 * 5120

  Shared attention:
    pre_norm.weight:   [5120]
    linear_qkv.weight: [7208, 5120]
    q_norm.weight:     [128]
    k_norm.weight:     [128]
    linear_proj.weight:[5120, 5120]

  Shared MLP (SwiGLU):
    pre_norm.weight:     [5120]
    up_gate_proj.weight: [27304, 5120]  (2 * 13652)
    down_proj.weight:    [5120, 13652]

  RoPE: bands [16]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Architecture constants
HIDDEN = 5120
NUM_Q_HEADS = 40
NUM_KV_GROUPS = 8
HEAD_DIM = 128
ROPE_DIM = 16  # bands shape [16]
NUM_MODALITIES = 3
QKV_PER_MOD = NUM_Q_HEADS * HEAD_DIM + 2 * NUM_KV_GROUPS * HEAD_DIM + NUM_Q_HEADS  # 5120+1024+1024+40=7208
SWIGLU_INTERMEDIATE = 13652  # actual value from checkpoint
GELU_INTERMEDIATE = HIDDEN * 4  # 20480

MM_LAYERS = {0, 1, 2, 3, 36, 37, 38, 39}
GELU_LAYERS = {0, 1, 2, 3}  # sandwich layers that use GELU


class Attention(nn.Module):
    """Attention block that stores weights in checkpoint-native format."""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.is_mm = layer_idx in MM_LAYERS
        nm = NUM_MODALITIES if self.is_mm else 1

        # Weights stored flat-concatenated for multi-modality layers
        self.pre_norm = nn.Parameter(torch.ones(nm * HIDDEN))
        self.linear_qkv = nn.Linear(HIDDEN, nm * QKV_PER_MOD, bias=False)
        self.q_norm = nn.Parameter(torch.ones(nm * HEAD_DIM))
        self.k_norm = nn.Parameter(torch.ones(nm * HEAD_DIM))
        self.linear_proj = nn.Linear(HIDDEN, nm * HIDDEN, bias=False)
        # Note: linear_proj checkpoint shape is [nm*HIDDEN, HIDDEN] for out_features x in_features

    def _rms_norm(self, x, weight, eps=1e-6):
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (x * norm).to(dtype) * weight

    def forward(self, x, rope_cos, rope_sin, modality_ids=None, attention_mask=None):
        B, S, _ = x.shape

        if self.is_mm and modality_ids is not None:
            # Per-token modality norm: weight is [3*5120], reshape to [3, 5120], gather per token
            norm_w = self.pre_norm.view(NUM_MODALITIES, HIDDEN)
            w = norm_w[modality_ids]  # [B, S, HIDDEN]
            normed = self._rms_norm(x, w)

            # Per-modality QKV: weight is [3*7208, 5120]
            # Route each token through its modality's projection
            qkv_w = self.linear_qkv.weight.view(NUM_MODALITIES, QKV_PER_MOD, HIDDEN)
            qkv = torch.zeros(B, S, QKV_PER_MOD, device=x.device, dtype=x.dtype)
            for m in range(NUM_MODALITIES):
                mask = (modality_ids == m)
                if mask.any():
                    tokens_m = normed[mask]  # [N_m, HIDDEN]
                    qkv[mask] = F.linear(tokens_m, qkv_w[m])

            # Per-modality Q/K norms
            q_norm_w = self.q_norm.view(NUM_MODALITIES, HEAD_DIM)
            k_norm_w = self.k_norm.view(NUM_MODALITIES, HEAD_DIM)
        else:
            normed = self._rms_norm(x, self.pre_norm)
            qkv = self.linear_qkv(normed)
            q_norm_w = self.q_norm.unsqueeze(0)  # [1, HEAD_DIM]
            k_norm_w = self.k_norm.unsqueeze(0)

        # Split QKV + gate
        q_dim = NUM_Q_HEADS * HEAD_DIM  # 5120
        k_dim = NUM_KV_GROUPS * HEAD_DIM  # 1024
        v_dim = k_dim
        g_dim = NUM_Q_HEADS  # 40

        q = qkv[..., :q_dim].reshape(B, S, NUM_Q_HEADS, HEAD_DIM)
        k = qkv[..., q_dim:q_dim + k_dim].reshape(B, S, NUM_KV_GROUPS, HEAD_DIM)
        v = qkv[..., q_dim + k_dim:q_dim + k_dim + v_dim].reshape(B, S, NUM_KV_GROUPS, HEAD_DIM)
        gate = torch.sigmoid(qkv[..., q_dim + k_dim + v_dim:q_dim + k_dim + v_dim + g_dim])

        # Per-head RMSNorm on Q and K
        if self.is_mm and modality_ids is not None:
            # Per-token modality norm for Q
            qn = q_norm_w[modality_ids]  # [B, S, HEAD_DIM]
            kn = k_norm_w[modality_ids]
            q = self._rms_norm(q, qn.unsqueeze(2))  # broadcast over heads
            k = self._rms_norm(k, kn.unsqueeze(2))
        else:
            q = self._rms_norm(q, q_norm_w)
            k = self._rms_norm(k, k_norm_w)

        # Apply RoPE
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # GQA expand
        repeat = NUM_Q_HEADS // NUM_KV_GROUPS  # 5
        k = k.unsqueeze(3).expand(-1, -1, -1, repeat, -1).reshape(B, S, NUM_Q_HEADS, HEAD_DIM)
        v = v.unsqueeze(3).expand(-1, -1, -1, repeat, -1).reshape(B, S, NUM_Q_HEADS, HEAD_DIM)

        # SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, S, NUM_Q_HEADS * HEAD_DIM)

        # Per-head gating
        out = out.reshape(B, S, NUM_Q_HEADS, HEAD_DIM)
        out = out * gate.unsqueeze(-1)
        out = out.reshape(B, S, NUM_Q_HEADS * HEAD_DIM)

        # Output projection
        if self.is_mm and modality_ids is not None:
            proj_w = self.linear_proj.weight.view(NUM_MODALITIES, HIDDEN, HIDDEN)
            result = torch.zeros(B, S, HIDDEN, device=x.device, dtype=x.dtype)
            for m in range(NUM_MODALITIES):
                mask = (modality_ids == m)
                if mask.any():
                    result[mask] = F.linear(out[mask], proj_w[m])
            return result
        else:
            return self.linear_proj(out)


class MLP(nn.Module):
    """MLP block that stores weights in checkpoint-native format."""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.is_mm = layer_idx in MM_LAYERS
        self.use_gelu = layer_idx in GELU_LAYERS
        nm = NUM_MODALITIES if self.is_mm else 1

        if self.use_gelu:
            intermediate = GELU_INTERMEDIATE  # 20480
        else:
            intermediate = SWIGLU_INTERMEDIATE  # 13652

        self.intermediate = intermediate

        self.pre_norm = nn.Parameter(torch.ones(nm * HIDDEN))

        if self.use_gelu:
            # GELU: up_gate_proj is [nm * intermediate, HIDDEN]
            self.up_gate_proj = nn.Linear(HIDDEN, nm * intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, nm * HIDDEN, bias=False)
        else:
            # SwiGLU: up_gate_proj is [nm * 2*intermediate, HIDDEN]
            self.up_gate_proj = nn.Linear(HIDDEN, nm * 2 * intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, nm * HIDDEN, bias=False)

    def _rms_norm(self, x, weight, eps=1e-6):
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (x * norm).to(dtype) * weight

    def forward(self, x, modality_ids=None):
        B, S, _ = x.shape
        nm = NUM_MODALITIES if self.is_mm else 1
        inter = self.intermediate

        if self.is_mm and modality_ids is not None:
            norm_w = self.pre_norm.view(NUM_MODALITIES, HIDDEN)
            w = norm_w[modality_ids]
            normed = self._rms_norm(x, w)

            if self.use_gelu:
                up_w = self.up_gate_proj.weight.view(NUM_MODALITIES, inter, HIDDEN)
                down_w = self.down_proj.weight.view(NUM_MODALITIES, HIDDEN, inter)
            else:
                up_w = self.up_gate_proj.weight.view(NUM_MODALITIES, 2 * inter, HIDDEN)
                down_w = self.down_proj.weight.view(NUM_MODALITIES, HIDDEN, inter)

            result = torch.zeros(B, S, HIDDEN, device=x.device, dtype=x.dtype)
            for m in range(NUM_MODALITIES):
                mask = (modality_ids == m)
                if not mask.any():
                    continue
                h = F.linear(normed[mask], up_w[m])
                if self.use_gelu:
                    h = F.gelu(h)
                else:
                    gate, up = h.chunk(2, dim=-1)
                    h = F.silu(gate) * up
                result[mask] = F.linear(h, down_w[m])
            return result
        else:
            normed = self._rms_norm(x, self.pre_norm)
            h = self.up_gate_proj(normed)
            if self.use_gelu:
                h = F.gelu(h)
            else:
                gate, up = h.chunk(2, dim=-1)
                h = F.silu(gate) * up
            return self.down_proj(h)


class TransformerLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.attention = Attention(layer_idx)
        self.mlp = MLP(layer_idx)

    def forward(self, x, rope_cos, rope_sin, modality_ids=None, attention_mask=None):
        x = x + self.attention(x, rope_cos, rope_sin, modality_ids, attention_mask)
        x = x + self.mlp(x, modality_ids)
        return x


class ElementWiseFourierEmbed(nn.Module):
    """RoPE with learned frequency bands. bands shape: [16]."""
    def __init__(self):
        super().__init__()
        self.bands = nn.Parameter(torch.randn(ROPE_DIM))

    def forward(self, coords):
        """coords: [B, S, num_coords] -> (cos, sin) each [B, S, ROPE_DIM * num_coords]"""
        # For each coordinate dimension, compute freqs using bands
        # coords shape: [B, S, C] where C = number of coordinate dims (e.g., 3 for t,h,w)
        B, S, C = coords.shape
        # bands: [16]
        # Compute: for each coord dim, freqs = coord * bands -> [B, S, 16] per dim
        # Concatenate across dims -> [B, S, 16*C]
        coords_f = coords.float()
        bands_f = self.bands.float()
        # [B, S, C, 1] * [1, 1, 1, 16] -> [B, S, C, 16]
        freqs = coords_f.unsqueeze(-1) * bands_f.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        freqs = freqs.reshape(B, S, C * ROPE_DIM)  # [B, S, 48] for 3 coords
        return freqs.cos().to(coords.dtype), freqs.sin().to(coords.dtype)


def _rms_norm(x, weight, eps=1e-6):
    dtype = x.dtype
    x = x.float()
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * norm).to(dtype) * weight


class DiTModel(nn.Module):
    """daVinci-MagiHuman 15B DiT Model - matches checkpoint format exactly."""

    def __init__(self, num_layers=40):
        super().__init__()
        self.num_layers = num_layers

        # Adapters
        self.video_embedder = nn.Linear(192, HIDDEN)   # 48*4=192 -> 5120
        self.audio_embedder = nn.Linear(64, HIDDEN)    # 64 -> 5120
        self.text_embedder = nn.Linear(3584, HIDDEN)   # T5Gemma dim -> 5120

        # RoPE
        self.rope = ElementWiseFourierEmbed()

        # Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(i) for i in range(num_layers)])

        # Output heads
        self.final_norm_video = nn.Parameter(torch.ones(HIDDEN))
        self.final_norm_audio = nn.Parameter(torch.ones(HIDDEN))
        self.final_linear_video = nn.Linear(HIDDEN, 192, bias=False)
        self.final_linear_audio = nn.Linear(HIDDEN, 64, bias=False)

    def forward(self, video_tokens, audio_tokens, text_tokens,
                video_coords, audio_coords, text_coords,
                modality_ids, video_mask, audio_mask, attention_mask=None):
        v = self.video_embedder(video_tokens)
        a = self.audio_embedder(audio_tokens)
        t = self.text_embedder(text_tokens)

        x = torch.cat([v, a, t], dim=1)
        coords = torch.cat([video_coords, audio_coords, text_coords], dim=1)

        rope_cos, rope_sin = self.rope(coords)

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, modality_ids, attention_mask)

        # Output
        video_out = self.final_linear_video(_rms_norm(x[video_mask], self.final_norm_video))
        audio_out = self.final_linear_audio(_rms_norm(x[audio_mask], self.final_norm_audio))

        return video_out, audio_out


def apply_rope(x, cos, sin):
    """Apply RoPE. x: [B, S, heads, head_dim], cos/sin: [B, S, rope_total_dim]."""
    rope_dim = cos.shape[-1]
    # Only apply to first rope_dim dimensions of head_dim
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    cos = cos.unsqueeze(2)  # [B, S, 1, rope_dim]
    sin = sin.unsqueeze(2)

    x1, x2 = x_rope[..., :rope_dim // 2], x_rope[..., rope_dim // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    x_rope = x_rope * cos + rotated * sin

    return torch.cat([x_rope, x_pass], dim=-1)


def load_dit_from_sharded(model_dir, dtype=torch.bfloat16, device="cpu"):
    """Load DiTModel from sharded safetensors checkpoint."""
    import json
    import os
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    model = DiTModel()

    # Key remapping: checkpoint -> model
    key_map = {
        "adapter.video_embedder.": "video_embedder.",
        "adapter.audio_embedder.": "audio_embedder.",
        "adapter.text_embedder.": "text_embedder.",
        "adapter.rope.": "rope.",
        "block.layers.": "layers.",
        "final_linear_video.": "final_linear_video.",
        "final_linear_audio.": "final_linear_audio.",
    }
    # Special keys: checkpoint "final_norm_video.weight" -> model "final_norm_video" (bare Parameter)
    special_remap = {
        "final_norm_video.weight": "final_norm_video",
        "final_norm_audio.weight": "final_norm_audio",
    }

    shard_to_keys = {}
    for key, shard_file in index["weight_map"].items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    state_dict = {}
    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            continue
        print(f"  Loading {shard_file}...")
        shard_data = load_file(shard_path, device=device)
        for key in shard_to_keys[shard_file]:
            if key in shard_data:
                # Check special remaps first
                if key in special_remap:
                    new_key = special_remap[key]
                else:
                    new_key = key
                    for old_prefix, new_prefix in key_map.items():
                        if key.startswith(old_prefix):
                            new_key = new_prefix + key[len(old_prefix):]
                            break
                state_dict[new_key] = shard_data[key].to(dtype)
        del shard_data

    # Fix bare nn.Parameter keys: checkpoint has "foo.weight" but model has "foo"
    # Build set of model keys for lookup
    model_keys = set(model.state_dict().keys())
    fixed_dict = {}
    for k, v in state_dict.items():
        if k in model_keys:
            fixed_dict[k] = v
        elif k.endswith(".weight"):
            stripped = k[:-len(".weight")]
            if stripped in model_keys:
                fixed_dict[stripped] = v
            else:
                fixed_dict[k] = v
        else:
            fixed_dict[k] = v
    state_dict = fixed_dict

    # Load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    model = model.to(dtype)
    return model
