"""
Motion Library for Multi-Motion Training

This module implements the MotionLibrary class that manages multiple motion files
and their corresponding adaptive samplers for multi-motion training as described
in the BeyondMimic paper.

Key features:
- Load and manage multiple motion files simultaneously
- Maintain separate adaptive sampling statistics per motion
- Provide unified interface for motion and phase sampling
- Support motion-aware failure tracking and statistics
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict, Sequence
import wandb

from .adaptive_sampler import AdaptiveSampler
from .commands import MotionLoader, AdaptiveSamplingCfg


class MotionLibrary:
    """
    Manages multiple motions and their corresponding adaptive samplers for multi-motion training.
    
    This class enables a single policy to train on multiple diverse motions by:
    1. Loading multiple motion files and creating MotionLoader instances
    2. Creating separate AdaptiveSampler instances for each motion
    3. Providing unified sampling interface for motion selection and phase sampling
    4. Managing motion-specific failure tracking and statistics
    """
    
    def __init__(
        self,
        motion_files: List[str],
        body_indexes: Sequence[int],
        adaptive_sampling_cfg: AdaptiveSamplingCfg,
        device: str = "cpu"
    ):
        """
        Initialize motion library with multiple motion files.
        
        Args:
            motion_files: List of paths to motion NPZ files
            body_indexes: Body indices for motion loading
            adaptive_sampling_cfg: Configuration for adaptive sampling
            device: Device for tensor operations
        """
        self.motion_files = motion_files
        self.num_motions = len(motion_files)
        self.device = device
        self.adaptive_sampling_cfg = adaptive_sampling_cfg
        
        if self.num_motions == 0:
            raise ValueError("Motion library must contain at least one motion file")
        
        # Load all motions and create adaptive samplers
        self.motions = []
        self.adaptive_samplers = []
        self.motion_names = []
        
        print(f"Loading motion library with {self.num_motions} motions...")
        
        for i, motion_file in enumerate(motion_files):
            print(f"  Loading motion {i+1}/{self.num_motions}: {motion_file}")
            
            # Load motion
            motion = MotionLoader(motion_file, body_indexes, device)
            motion_duration = motion.time_step_total / motion.fps
            
            # Create adaptive sampler for this motion
            sampler = AdaptiveSampler(
                motion_fps=int(motion.fps),
                motion_duration=motion_duration,
                bin_size_seconds=adaptive_sampling_cfg.bin_size_seconds,
                gamma=adaptive_sampling_cfg.gamma,
                lambda_uniform=adaptive_sampling_cfg.lambda_uniform,
                K=adaptive_sampling_cfg.K,
                alpha_smooth=adaptive_sampling_cfg.alpha_smooth,
                device=device
            )
            
            # Extract motion name from file path
            motion_name = motion_file.split('/')[-1].replace('.npz', '')
            
            self.motions.append(motion)
            self.adaptive_samplers.append(sampler)
            self.motion_names.append(motion_name)
            
            print(f"    Duration: {motion_duration:.1f}s, FPS: {motion.fps}, Frames: {motion.time_step_total}")
        
        # Motion sampling statistics (for potential future adaptive motion selection)
        self.motion_episode_counts = torch.zeros(self.num_motions, device=device, dtype=torch.float32)
        self.motion_failure_counts = torch.zeros(self.num_motions, device=device, dtype=torch.float32)
        
        print(f"Motion library initialized successfully with {self.num_motions} motions")
    
    def sample_motion_and_phase(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample motions and starting phases for a batch of environments.
        
        Args:
            batch_size: Number of environments to sample for
            
        Returns:
            motion_ids: (batch_size,) tensor of motion indices
            starting_phases: (batch_size,) tensor of starting phases [0, 1]
            starting_bins: (batch_size,) tensor of corresponding bin indices
        """
        if batch_size == 0:
            return (torch.empty(0, device=self.device, dtype=torch.long),
                   torch.empty(0, device=self.device, dtype=torch.float32),
                   torch.empty(0, device=self.device, dtype=torch.long))
        
        # Sample motions uniformly for now (can be made adaptive later)
        motion_ids = torch.randint(0, self.num_motions, (batch_size,), device=self.device)
        
        starting_phases = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        starting_bins = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        
        # Sample starting phase for each selected motion using its adaptive sampler
        for motion_id in range(self.num_motions):
            mask = motion_ids == motion_id
            if mask.any():
                num_samples = mask.sum().item()
                phases, bins = self.adaptive_samplers[motion_id].sample_starting_phases(num_samples)
                starting_phases[mask] = phases
                starting_bins[mask] = bins
        
        return motion_ids, starting_phases, starting_bins
    
    def update_failure_statistics(self, motion_ids: torch.Tensor, starting_frames: torch.Tensor, 
                                failures: torch.Tensor):
        """
        Update failure statistics for completed episodes.
        
        Args:
            motion_ids: (N,) tensor of motion indices for completed episodes
            starting_frames: (N,) tensor of starting frame indices
            failures: (N,) tensor of boolean failure indicators
        """
        if len(motion_ids) == 0:
            return
        
        # Update motion-level statistics
        for motion_id in range(self.num_motions):
            motion_mask = motion_ids == motion_id
            if motion_mask.any():
                motion_episodes = motion_mask.sum().float()
                motion_failures = (motion_mask & failures).sum().float()
                
                self.motion_episode_counts[motion_id] += motion_episodes
                self.motion_failure_counts[motion_id] += motion_failures
                
                # Update adaptive sampler for this motion
                motion_starting_frames = starting_frames[motion_mask]
                motion_failures_bool = failures[motion_mask]
                
                self.adaptive_samplers[motion_id].update_failure_statistics(
                    motion_starting_frames, motion_failures_bool
                )
    
    def update_sampling_probabilities(self):
        """Update sampling probabilities for all motion adaptive samplers."""
        for sampler in self.adaptive_samplers:
            sampler.update_sampling_probabilities()
    
    def get_motion_statistics(self) -> Dict:
        """Get comprehensive statistics for all motions."""
        stats = {
            "num_motions": self.num_motions,
            "motion_names": self.motion_names,
            "total_episodes": self.motion_episode_counts.sum().item(),
            "total_failures": self.motion_failure_counts.sum().item(),
        }
        
        # Per-motion statistics
        for i, (name, sampler) in enumerate(zip(self.motion_names, self.adaptive_samplers)):
            motion_stats = sampler.get_statistics()
            motion_failure_rate = (self.motion_failure_counts[i] / 
                                 torch.clamp(self.motion_episode_counts[i], min=1)).item()
            
            stats[f"motion_{i}_{name}"] = {
                "episodes": self.motion_episode_counts[i].item(),
                "failures": self.motion_failure_counts[i].item(), 
                "failure_rate": motion_failure_rate,
                "sampling_entropy": motion_stats["normalized_entropy"],
                "total_bins": motion_stats["bin_episode_counts"].sum()
            }
        
        return stats
    
    def log_to_wandb(self, prefix: str = "motion_library"):
        """Log motion library statistics to Weights & Biases."""
        if wandb.run is None:
            return
        
        stats = self.get_motion_statistics()
        
        # Log overall statistics
        wandb.log({
            f"{prefix}/num_motions": stats["num_motions"],
            f"{prefix}/total_episodes": stats["total_episodes"],
            f"{prefix}/total_failures": stats["total_failures"],
            f"{prefix}/overall_failure_rate": stats["total_failures"] / max(stats["total_episodes"], 1)
        })
        
        # Log per-motion statistics
        for i, name in enumerate(self.motion_names):
            motion_key = f"motion_{i}_{name}"
            if motion_key in stats:
                motion_stats = stats[motion_key]
                wandb.log({
                    f"{prefix}/{name}/episodes": motion_stats["episodes"],
                    f"{prefix}/{name}/failure_rate": motion_stats["failure_rate"],
                    f"{prefix}/{name}/sampling_entropy": motion_stats["sampling_entropy"]
                })
        
        # Log individual sampler statistics
        for i, (name, sampler) in enumerate(zip(self.motion_names, self.adaptive_samplers)):
            sampler.log_to_wandb(prefix=f"{prefix}/{name}")
    
    def get_motion_data(self, motion_ids: torch.Tensor, frame_indices: torch.Tensor, 
                       property_name: str) -> torch.Tensor:
        """
        Get motion data for specified motion IDs and frame indices.
        
        Args:
            motion_ids: (N,) tensor of motion indices
            frame_indices: (N,) tensor of frame indices
            property_name: Name of property to retrieve ('joint_pos', 'joint_vel', etc.)
            
        Returns:
            Data tensor with motion-specific values
        """
        if len(motion_ids) == 0:
            # Return empty tensor with correct shape
            sample_motion = self.motions[0]
            sample_data = getattr(sample_motion, property_name)
            empty_shape = (0,) + sample_data.shape[1:]
            return torch.empty(empty_shape, device=self.device, dtype=sample_data.dtype)
        
        # Get sample shape from first motion
        sample_motion = self.motions[0]
        sample_data = getattr(sample_motion, property_name)
        result_shape = (len(motion_ids),) + sample_data.shape[1:]
        result = torch.zeros(result_shape, device=self.device, dtype=sample_data.dtype)
        
        # Fill data for each motion
        for motion_id in range(self.num_motions):
            mask = motion_ids == motion_id
            if mask.any():
                motion = self.motions[motion_id]
                motion_data = getattr(motion, property_name)
                motion_frames = frame_indices[mask]
                
                # Clamp frame indices to valid range
                motion_frames = torch.clamp(motion_frames, 0, motion_data.shape[0] - 1)
                
                result[mask] = motion_data[motion_frames]
        
        return result
    
    def save_state(self, filepath: str):
        """Save motion library state for checkpointing."""
        state = {
            "motion_files": self.motion_files,
            "motion_names": self.motion_names,
            "num_motions": self.num_motions,
            "motion_episode_counts": self.motion_episode_counts.cpu(),
            "motion_failure_counts": self.motion_failure_counts.cpu(),
            "adaptive_sampling_cfg": {
                "bin_size_seconds": self.adaptive_sampling_cfg.bin_size_seconds,
                "gamma": self.adaptive_sampling_cfg.gamma,
                "lambda_uniform": self.adaptive_sampling_cfg.lambda_uniform,
                "K": self.adaptive_sampling_cfg.K,
                "alpha_smooth": self.adaptive_sampling_cfg.alpha_smooth,
            }
        }
        
        # Save individual sampler states
        sampler_states = []
        for sampler in self.adaptive_samplers:
            sampler_state = {
                "episode_counts": sampler.episode_counts.cpu(),
                "failure_counts": sampler.failure_counts.cpu(),
                "smoothed_failure_rates": sampler.smoothed_failure_rates.cpu(),
                "sampling_probabilities": sampler.sampling_probabilities.cpu(),
                "total_episodes": sampler.total_episodes,
                "total_failures": sampler.total_failures,
                "update_counter": sampler.update_counter,
            }
            sampler_states.append(sampler_state)
        
        state["sampler_states"] = sampler_states
        
        torch.save(state, filepath)
        print(f"MotionLibrary state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load motion library state from checkpoint."""
        state = torch.load(filepath, map_location=self.device)
        
        # Verify compatibility
        if state["num_motions"] != self.num_motions:
            raise ValueError(f"Checkpoint has {state['num_motions']} motions, "
                           f"but current library has {self.num_motions}")
        
        # Load motion-level statistics
        self.motion_episode_counts = state["motion_episode_counts"].to(self.device)
        self.motion_failure_counts = state["motion_failure_counts"].to(self.device)
        
        # Load individual sampler states
        sampler_states = state["sampler_states"]
        for i, (sampler, sampler_state) in enumerate(zip(self.adaptive_samplers, sampler_states)):
            sampler.episode_counts = sampler_state["episode_counts"].to(self.device)
            sampler.failure_counts = sampler_state["failure_counts"].to(self.device)
            sampler.smoothed_failure_rates = sampler_state["smoothed_failure_rates"].to(self.device)
            sampler.sampling_probabilities = sampler_state["sampling_probabilities"].to(self.device)
            sampler.total_episodes = sampler_state["total_episodes"]
            sampler.total_failures = sampler_state["total_failures"]
            sampler.update_counter = sampler_state["update_counter"]
        
        print(f"MotionLibrary state loaded from {filepath}")
        print(f"  Loaded {self.motion_episode_counts.sum().item()} total episodes "
              f"with {self.motion_failure_counts.sum().item()} failures")