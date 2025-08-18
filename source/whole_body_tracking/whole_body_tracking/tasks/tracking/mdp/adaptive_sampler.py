"""
Adaptive Sampling Mechanism for BeyondMimic Motion Tracking

This module implements the adaptive sampling mechanism described in Section III-F of the BeyondMimic paper.
The key idea is to sample difficult motion segments more frequently based on empirical failure statistics,
enabling efficient training on long, multi-motion sequences.

Mathematical Implementation based on:
- Bin division of motion timeline (1-second bins)
- Failure tracking per bin with exponential smoothing
- Non-causal convolution for probability calculation (Equation 3)
- Weighted sampling with uniform mixing to prevent catastrophic forgetting
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional
import wandb


class AdaptiveSampler:
    """
    Adaptive sampling mechanism that focuses training on difficult motion segments.
    
    Implements the algorithm from BeyondMimic paper Section III-F:
    1. Divide motion into discrete bins (default: 1-second intervals)
    2. Track failure rates per bin with exponential smoothing
    3. Apply non-causal convolution to calculate sampling probabilities
    4. Mix with uniform distribution to prevent catastrophic forgetting
    """
    
    def __init__(
        self,
        motion_fps: int,
        motion_duration: float,
        bin_size_seconds: float = 1.0,
        gamma: float = 0.9,
        lambda_uniform: float = 0.1,
        K: int = 5,
        alpha_smooth: float = 0.1,
        device: str = "cpu",
        motion_id: int = 0,
        motion_name: str = "unknown"
    ):
        """
        Initialize adaptive sampler.
        
        Args:
            motion_fps: Frames per second of the motion
            motion_duration: Total duration of motion in seconds
            bin_size_seconds: Size of each bin in seconds (default: 1.0)
            gamma: Decay rate for convolution kernel (default: 0.9)
            lambda_uniform: Mixing ratio with uniform distribution (default: 0.1)
            K: Convolution kernel size (default: 5)
            alpha_smooth: EMA smoothing factor for failure rates (default: 0.1)
            device: Device for tensor operations
            motion_id: Unique identifier for this motion (for multi-motion training)
            motion_name: Human-readable name for this motion
        """
        self.motion_fps = motion_fps
        self.motion_duration = motion_duration
        self.bin_size_seconds = bin_size_seconds
        self.gamma = gamma
        self.lambda_uniform = lambda_uniform
        self.K = K
        self.alpha_smooth = alpha_smooth
        self.device = device
        self.motion_id = motion_id
        self.motion_name = motion_name
        
        # Calculate number of bins (S in paper)
        self.num_bins = int(np.ceil(motion_duration / bin_size_seconds))
        self.frames_per_bin = int(motion_fps * bin_size_seconds)
        self.total_frames = int(motion_fps * motion_duration)
        
        # Statistics tracking tensors
        self.episode_counts = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        self.failure_counts = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        self.smoothed_failure_rates = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        self.sampling_probabilities = torch.ones(self.num_bins, device=device, dtype=torch.float32) / self.num_bins
        
        # Statistics for monitoring
        self.total_episodes = 0
        self.total_failures = 0
        self.update_counter = 0
        
        print(f"AdaptiveSampler initialized:")
        print(f"  Motion: {motion_name} (ID: {motion_id})")
        print(f"  Duration: {motion_duration:.1f}s @ {motion_fps}fps = {self.total_frames} frames")
        print(f"  Bins: {self.num_bins} bins of {bin_size_seconds}s each ({self.frames_per_bin} frames/bin)")
        print(f"  Parameters: γ={gamma}, λ={lambda_uniform}, K={K}, α={alpha_smooth}")
    
    def frame_to_bin(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Convert frame indices to bin indices."""
        return torch.clamp(frame_indices // self.frames_per_bin, 0, self.num_bins - 1)
    
    def bin_to_frame_range(self, bin_index: int) -> Tuple[int, int]:
        """Get frame range for a given bin index."""
        start_frame = bin_index * self.frames_per_bin
        end_frame = min((bin_index + 1) * self.frames_per_bin, self.total_frames)
        return start_frame, end_frame
    
    def update_failure_statistics(self, starting_frames: torch.Tensor, failures: torch.Tensor):
        """
        Update failure statistics for completed episodes.
        
        Args:
            starting_frames: (N,) tensor of starting frame indices for episodes
            failures: (N,) tensor of boolean failure indicators
        """
        if len(starting_frames) == 0:
            return
            
        # Convert frames to bins
        starting_bins = self.frame_to_bin(starting_frames)
        
        # Update episode and failure counts per bin
        for bin_idx in range(self.num_bins):
            bin_mask = starting_bins == bin_idx
            if bin_mask.any():
                bin_episodes = bin_mask.sum().float()
                bin_failures = (bin_mask & failures).sum().float()
                
                self.episode_counts[bin_idx] += bin_episodes
                self.failure_counts[bin_idx] += bin_failures
        
        # Update global statistics
        self.total_episodes += len(starting_frames)
        self.total_failures += failures.sum().item()
        self.update_counter += 1
        
        # Update smoothed failure rates using exponential moving average
        self._update_smoothed_failure_rates()
    
    def _update_smoothed_failure_rates(self):
        """Update smoothed failure rates using exponential moving average."""
        # Calculate raw failure rates (avoid division by zero)
        raw_failure_rates = self.failure_counts / torch.clamp(self.episode_counts, min=1.0)
        
        # Apply exponential moving average smoothing
        # Only update bins that have received episodes
        valid_bins = self.episode_counts > 0
        self.smoothed_failure_rates[valid_bins] = (
            self.alpha_smooth * raw_failure_rates[valid_bins] +
            (1 - self.alpha_smooth) * self.smoothed_failure_rates[valid_bins]
        )
    
    def compute_convolved_failure_rates(self) -> torch.Tensor:
        """
        Apply non-causal convolution with exponentially decaying kernel.
        
        Implements Equation 3 from the paper:
        k(u) = γ^u to assign greater weight to recent past failures
        
        Returns:
            convolved_rates: (num_bins,) tensor of convolved failure rates
        """
        convolved_rates = torch.zeros_like(self.smoothed_failure_rates)
        
        for s in range(self.num_bins):
            weighted_sum = 0.0
            normalization = 0.0
            
            # Apply convolution kernel looking forward in time
            for u in range(self.K):
                if s + u < self.num_bins:
                    weight = self.gamma ** u
                    weighted_sum += weight * self.smoothed_failure_rates[s + u]
                    normalization += weight
            
            # Normalize to prevent scaling issues
            convolved_rates[s] = weighted_sum / max(normalization, 1e-8)
        
        return convolved_rates
    
    def update_sampling_probabilities(self):
        """
        Calculate final sampling probabilities with uniform mixing.
        
        Implements: p's = λ(1/S) + (1-λ)ps where ps from Equation 3
        """
        # Get convolved failure rates
        convolved_rates = self.compute_convolved_failure_rates()
        
        # Normalize to get pure adaptive probabilities
        total_rate = torch.clamp(convolved_rates.sum(), min=1e-8)
        adaptive_probs = convolved_rates / total_rate
        
        # Mix with uniform distribution to prevent catastrophic forgetting
        uniform_probs = torch.ones(self.num_bins, device=self.device) / self.num_bins
        self.sampling_probabilities = (
            self.lambda_uniform * uniform_probs + 
            (1 - self.lambda_uniform) * adaptive_probs
        )
        
        # Ensure probabilities sum to 1 (numerical stability)
        self.sampling_probabilities = self.sampling_probabilities / self.sampling_probabilities.sum()
    
    def sample_starting_frames(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample starting frame indices using adaptive probabilities.
        
        Args:
            batch_size: Number of starting frames to sample
            
        Returns:
            starting_frames: (batch_size,) tensor of starting frame indices
            starting_bins: (batch_size,) tensor of corresponding bin indices
        """
        if batch_size == 0:
            return torch.empty(0, device=self.device, dtype=torch.long), torch.empty(0, device=self.device, dtype=torch.long)
        
        # Sample bins according to learned probabilities
        bin_indices = torch.multinomial(
            self.sampling_probabilities, 
            batch_size, 
            replacement=True
        )
        
        # Sample uniformly within selected bins
        starting_frames = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        
        for i, bin_idx in enumerate(bin_indices):
            start_frame, end_frame = self.bin_to_frame_range(bin_idx.item())
            # Sample uniformly within the bin
            frame_offset = torch.randint(0, end_frame - start_frame, (1,), device=self.device)
            starting_frames[i] = start_frame + frame_offset
        
        return starting_frames, bin_indices
    
    def sample_starting_phases(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample starting phases [0, 1] using adaptive probabilities.
        
        Args:
            batch_size: Number of starting phases to sample
            
        Returns:
            phases: (batch_size,) tensor of starting phases [0, 1]
            starting_bins: (batch_size,) tensor of corresponding bin indices
        """
        starting_frames, starting_bins = self.sample_starting_frames(batch_size)
        
        # Convert frames to phases [0, 1]
        phases = starting_frames.float() / max(self.total_frames - 1, 1)
        
        return phases, starting_bins
    
    def get_statistics(self) -> dict:
        """Get current statistics for monitoring and logging."""
        overall_failure_rate = self.total_failures / max(self.total_episodes, 1)
        
        # Calculate entropy of sampling distribution (higher = more uniform)
        eps = 1e-8
        entropy = -torch.sum(self.sampling_probabilities * torch.log(self.sampling_probabilities + eps))
        max_entropy = np.log(self.num_bins)  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy
        
        # Find most and least sampled bins
        max_prob_bin = torch.argmax(self.sampling_probabilities).item()
        min_prob_bin = torch.argmin(self.sampling_probabilities).item()
        
        return {
            "motion_id": self.motion_id,
            "motion_name": self.motion_name,
            "total_episodes": self.total_episodes,
            "total_failures": self.total_failures,
            "overall_failure_rate": overall_failure_rate,
            "sampling_entropy": entropy.item(),
            "normalized_entropy": normalized_entropy.item(),
            "max_prob_bin": max_prob_bin,
            "min_prob_bin": min_prob_bin,
            "max_probability": self.sampling_probabilities[max_prob_bin].item(),
            "min_probability": self.sampling_probabilities[min_prob_bin].item(),
            "update_counter": self.update_counter,
            "bin_episode_counts": self.episode_counts.cpu().numpy(),
            "bin_failure_rates": (self.failure_counts / torch.clamp(self.episode_counts, min=1)).cpu().numpy(),
            "smoothed_failure_rates": self.smoothed_failure_rates.cpu().numpy(),
            "sampling_probabilities": self.sampling_probabilities.cpu().numpy(),
        }
    
    def log_to_wandb(self, prefix: str = "adaptive_sampling"):
        """Log sampling statistics to Weights & Biases."""
        if wandb.run is None:
            return
            
        stats = self.get_statistics()
        
        # Log scalar metrics
        wandb.log({
            f"{prefix}/total_episodes": stats["total_episodes"],
            f"{prefix}/total_failures": stats["total_failures"],
            f"{prefix}/overall_failure_rate": stats["overall_failure_rate"],
            f"{prefix}/sampling_entropy": stats["sampling_entropy"],
            f"{prefix}/normalized_entropy": stats["normalized_entropy"],
            f"{prefix}/max_probability": stats["max_probability"],
            f"{prefix}/min_probability": stats["min_probability"],
        })
        
        # Log per-bin metrics (limit to avoid too much data)
        if self.num_bins <= 20:  # Only log individual bins for short motions
            for i in range(self.num_bins):
                wandb.log({
                    f"{prefix}/bin_{i}_failure_rate": stats["bin_failure_rates"][i],
                    f"{prefix}/bin_{i}_sampling_prob": stats["sampling_probabilities"][i],
                    f"{prefix}/bin_{i}_episode_count": stats["bin_episode_counts"][i],
                })
    
    def reset_statistics(self):
        """Reset all statistics (useful for testing or restarting training)."""
        self.episode_counts.zero_()
        self.failure_counts.zero_()
        self.smoothed_failure_rates.zero_()
        self.sampling_probabilities.fill_(1.0 / self.num_bins)
        self.total_episodes = 0
        self.total_failures = 0
        self.update_counter = 0
        
        print("AdaptiveSampler statistics reset")
    
    def save_state(self, filepath: str):
        """Save sampler state for checkpointing."""
        state = {
            "episode_counts": self.episode_counts.cpu(),
            "failure_counts": self.failure_counts.cpu(),
            "smoothed_failure_rates": self.smoothed_failure_rates.cpu(),
            "sampling_probabilities": self.sampling_probabilities.cpu(),
            "total_episodes": self.total_episodes,
            "total_failures": self.total_failures,
            "update_counter": self.update_counter,
            "config": {
                "motion_fps": self.motion_fps,
                "motion_duration": self.motion_duration,
                "bin_size_seconds": self.bin_size_seconds,
                "gamma": self.gamma,
                "lambda_uniform": self.lambda_uniform,
                "K": self.K,
                "alpha_smooth": self.alpha_smooth,
            }
        }
        torch.save(state, filepath)
        print(f"AdaptiveSampler state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load sampler state from checkpoint."""
        state = torch.load(filepath, map_location=self.device)
        
        self.episode_counts = state["episode_counts"].to(self.device)
        self.failure_counts = state["failure_counts"].to(self.device)
        self.smoothed_failure_rates = state["smoothed_failure_rates"].to(self.device)
        self.sampling_probabilities = state["sampling_probabilities"].to(self.device)
        self.total_episodes = state["total_episodes"]
        self.total_failures = state["total_failures"]
        self.update_counter = state["update_counter"]
        
        print(f"AdaptiveSampler state loaded from {filepath}")
        print(f"  Loaded {self.total_episodes} episodes with {self.total_failures} failures")