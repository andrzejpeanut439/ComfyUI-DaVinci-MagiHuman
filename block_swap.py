"""
Block swapping for VRAM-constrained GPUs (RTX 5090 32GB).
The 15B DiT model is ~30GB in bf16 - doesn't fit fully in 32GB VRAM.
This module handles layer-by-layer CPU<->GPU offloading during forward pass.
"""

import torch
import torch.nn as nn
from typing import Optional
import gc


class BlockSwapManager:
    """Manages CPU<->GPU block swapping for the DiT transformer layers.

    Strategy: Keep N blocks on GPU at a time, swap others to CPU.
    Uses CUDA streams for async prefetching of the next block.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks_on_gpu: int = 8,
        device: torch.device = None,
        offload_device: torch.device = None,
    ):
        """
        Args:
            model: DiTModel with .layers attribute
            blocks_on_gpu: Number of transformer blocks to keep on GPU simultaneously
            device: GPU device
            offload_device: CPU device for offloaded blocks
        """
        self.model = model
        self.blocks_on_gpu = blocks_on_gpu
        self.device = device or torch.device("cuda")
        self.offload_device = offload_device or torch.device("cpu")
        self.num_layers = len(model.layers)

        # Prefetch stream for async transfers
        if self.device.type == "cuda":
            self.prefetch_stream = torch.cuda.Stream(device=self.device)
        else:
            self.prefetch_stream = None

        self._gpu_blocks = set()

    def setup(self):
        """Move adapter/final layers to GPU, transformer blocks to CPU."""
        # Keep embedders and output heads on GPU (small)
        self.model.video_embedder.to(self.device)
        self.model.audio_embedder.to(self.device)
        self.model.text_embedder.to(self.device)
        self.model.rope.to(self.device)
        self.model.final_norm_video.data = self.model.final_norm_video.data.to(self.device)
        self.model.final_norm_audio.data = self.model.final_norm_audio.data.to(self.device)
        self.model.final_linear_video.to(self.device)
        self.model.final_linear_audio.to(self.device)

        # Move all blocks to CPU
        for i, layer in enumerate(self.model.layers):
            layer.to(self.offload_device)
        self._gpu_blocks.clear()

        # Pre-load first N blocks to GPU
        for i in range(min(self.blocks_on_gpu, self.num_layers)):
            self._move_to_gpu(i)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _move_to_gpu(self, block_idx: int):
        """Move a block to GPU."""
        if block_idx not in self._gpu_blocks:
            self.model.layers[block_idx].to(self.device, non_blocking=True)
            self._gpu_blocks.add(block_idx)

    def _move_to_cpu(self, block_idx: int):
        """Move a block to CPU."""
        if block_idx in self._gpu_blocks:
            self.model.layers[block_idx].to(self.offload_device, non_blocking=True)
            self._gpu_blocks.discard(block_idx)

    def prefetch_block(self, block_idx: int):
        """Async prefetch a block to GPU."""
        if block_idx >= self.num_layers or block_idx in self._gpu_blocks:
            return

        if self.prefetch_stream is not None:
            with torch.cuda.stream(self.prefetch_stream):
                self._move_to_gpu(block_idx)
        else:
            self._move_to_gpu(block_idx)

    def execute_block(self, block_idx: int, *args, **kwargs):
        """Execute a transformer block with automatic swap management."""
        # Ensure current block is on GPU
        if block_idx not in self._gpu_blocks:
            self._move_to_gpu(block_idx)

        # Sync prefetch stream
        if self.prefetch_stream is not None:
            self.prefetch_stream.synchronize()

        # Start prefetching next block
        next_idx = block_idx + 1
        if next_idx < self.num_layers:
            self.prefetch_block(next_idx)

        # Execute the block
        result = self.model.layers[block_idx](*args, **kwargs)

        # Evict old blocks to stay within budget
        evict_idx = block_idx - self.blocks_on_gpu + 1
        if evict_idx >= 0:
            self._move_to_cpu(evict_idx)

        return result

    def forward_with_swap(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        modality_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        callback=None,
    ) -> torch.Tensor:
        """Run all transformer layers with block swapping."""
        for i in range(self.num_layers):
            x = self.execute_block(i, x, rope_cos, rope_sin, modality_ids, attention_mask)
            if callback:
                callback(i, self.num_layers)
        return x

    def cleanup(self):
        """Move everything back to CPU and free VRAM."""
        for i in list(self._gpu_blocks):
            self._move_to_cpu(i)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
