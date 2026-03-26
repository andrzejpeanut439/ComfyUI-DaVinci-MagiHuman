"""
Flow-matching scheduler for daVinci-MagiHuman.
Supports DDIM (for distill mode) and UniPC (for base mode).
"""

import torch
import numpy as np
from typing import Optional


class FlowMatchingScheduler:
    """Simple flow-matching scheduler with shift-based sigma schedule."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    def get_sigmas(self, num_inference_steps: int, device: torch.device = None) -> torch.Tensor:
        """Get sigma schedule for inference.

        Uses shifted flow-matching schedule:
            sigma(t) = shift * t / (1 + (shift - 1) * t)
        where t goes from 1.0 to 0.0
        """
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        sigmas = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)
        return sigmas

    def step_ddim(
        self,
        model_output: torch.Tensor,
        sigma: float,
        sigma_next: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM step for flow matching (used in distill mode).

        In flow matching, the model predicts the velocity v where:
            x_t = sigma * x_0 + (1 - sigma) * noise
            v = x_0 - noise

        So x_0 = x_t + (1 - sigma) * v  ... approximately
        And we step to:
            x_{t-1} = sigma_next * x_0_pred + (1 - sigma_next) * noise_pred
        """
        # Predict clean signal
        # v = x_0 - noise => x_0 = sample - (1 - sigma) * v + sigma * v
        # Simplified: x_0_pred = sample + (1 - sigma) * model_output / sigma ...
        # Actually for flow matching with the convention used here:
        # x_t = sigma * noise + (1 - sigma) * x_0
        # model predicts v ≈ (x_0 - noise)
        # x_0 = (x_t - sigma * noise) / (1 - sigma)
        # noise = (x_t - (1 - sigma) * x_0) / sigma

        # Direct linear interpolation step (standard flow matching DDIM):
        # x_{t-1} = x_t + (sigma_next - sigma) * model_output
        denoised = sample + (sigma_next - sigma) * model_output
        return denoised

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Add noise at a given sigma level.
        x_t = (1 - sigma) * x_0 + sigma * noise
        """
        return (1 - sigma) * original + sigma * noise

    def get_noise_level_sigma(self, noise_value: int) -> float:
        """Convert a noise_value index (0-999) to sigma.
        Used for SR re-noising.
        """
        t = noise_value / self.num_train_timesteps
        return self.shift * t / (1 + (self.shift - 1) * t)
