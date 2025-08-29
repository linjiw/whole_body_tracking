"""Trajectory dataset for diffusion model training.

This module implements the data structures and dataset classes for Stage 2 of BeyondMimic.
Based on the paper's specifications (N=4 history, H=32 for states, H=16 for actions).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset


@dataclass
class StateRepresentation:
    """Body-position state representation.
    
    Critical choice based on paper ablation (100% vs 72% success rate).
    Body positions provide direct spatial grounding for guidance.
    """
    # Global states (relative to current frame)
    root_pos: torch.Tensor      # (3,) - relative position
    root_vel: torch.Tensor      # (3,) - linear velocity  
    root_rot: torch.Tensor      # (3,) - rotation vector
    
    # Local states (in character frame)  
    body_positions: torch.Tensor   # (num_bodies, 3)
    body_velocities: torch.Tensor  # (num_bodies, 3)
    
    @property
    def dim(self) -> int:
        """Total state dimension."""
        return 9 + self.body_positions.numel() + self.body_velocities.numel()
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to flat tensor representation."""
        return torch.cat([
            self.root_pos.flatten(),
            self.root_vel.flatten(),
            self.root_rot.flatten(),
            self.body_positions.flatten(),
            self.body_velocities.flatten()
        ])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, num_bodies: int = 23):
        """Reconstruct from flat tensor."""
        idx = 0
        root_pos = tensor[idx:idx+3]
        idx += 3
        root_vel = tensor[idx:idx+3]
        idx += 3
        root_rot = tensor[idx:idx+3]
        idx += 3
        body_positions = tensor[idx:idx+num_bodies*3].reshape(num_bodies, 3)
        idx += num_bodies * 3
        body_velocities = tensor[idx:idx+num_bodies*3].reshape(num_bodies, 3)
        
        return cls(
            root_pos=root_pos,
            root_vel=root_vel,
            root_rot=root_rot,
            body_positions=body_positions,
            body_velocities=body_velocities
        )


@dataclass
class Trajectory:
    """Single trajectory with history and future.
    
    Following paper specifications:
    - History: N=4 past state-action pairs
    - Future: H=32 states, H=16 actions (though actions masked to 8 in loss)
    """
    # History: O_t = [s_{t-N}, a_{t-N}, ..., s_t]
    history_states: torch.Tensor   # (N+1, state_dim) where N=4
    history_actions: torch.Tensor  # (N, action_dim)
    
    # Future: τ_t = [a_t, s_{t+1}, ..., s_{t+H}, a_{t+H}]
    future_states: torch.Tensor    # (H, state_dim) where H=32
    future_actions: torch.Tensor   # (H, action_dim) where H=16
    
    # Metadata
    motion_id: str
    timestep: int
    success: bool
    
    # Optional: for debugging and analysis
    episode_id: Optional[int] = None
    
    def get_observation_history(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get observation history in format for model input."""
        return self.history_states, self.history_actions
    
    def get_future_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get future trajectory as separate states and actions.
        
        Returns:
            Tuple of (future_states, future_actions)
            
        Note: States and actions have different dimensions, so they are returned
        separately rather than interleaved. The diffusion model will handle
        the interleaving internally with proper embedding layers.
        """
        return self.future_states, self.future_actions
    
    def get_interleaved_indices(self) -> List[Tuple[str, int]]:
        """Get indices for interleaving states and actions.
        
        Returns:
            List of tuples (type, index) where type is 'state' or 'action'
        """
        indices = []
        min_len = min(len(self.future_states), len(self.future_actions))
        
        for i in range(min_len):
            indices.append(('action', i))
            indices.append(('state', i))
        
        # Add remaining states if H_states > H_actions  
        for i in range(len(self.future_actions), len(self.future_states)):
            indices.append(('state', i))
            
        return indices


class TrajectoryDataset(Dataset):
    """PyTorch dataset for trajectory data."""
    
    def __init__(
        self,
        trajectories: List[Trajectory],
        transform: Optional[callable] = None,
        augment_noise: bool = False,
        noise_scale: float = 0.01
    ):
        """Initialize dataset.
        
        Args:
            trajectories: List of trajectory objects
            transform: Optional transform to apply to trajectories
            augment_noise: Whether to add noise during training
            noise_scale: Scale of noise to add
        """
        self.trajectories = trajectories
        self.transform = transform
        self.augment_noise = augment_noise
        self.noise_scale = noise_scale
        
        # Compute statistics for normalization
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute mean and std for normalization."""
        if len(self.trajectories) == 0:
            self.state_mean = torch.zeros(1)
            self.state_std = torch.ones(1)
            self.action_mean = torch.zeros(1)
            self.action_std = torch.ones(1)
            return
            
        # Collect all states and actions
        all_states = []
        all_actions = []
        
        for traj in self.trajectories:
            all_states.append(traj.history_states)
            all_states.append(traj.future_states)
            all_actions.append(traj.history_actions)
            all_actions.append(traj.future_actions)
        
        # Compute statistics
        all_states = torch.cat([s.flatten() for s in all_states])
        all_actions = torch.cat([a.flatten() for a in all_actions])
        
        self.state_mean = all_states.mean()
        self.state_std = all_states.std() + 1e-6
        self.action_mean = all_actions.mean()
        self.action_std = all_actions.std() + 1e-6
    
    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize states."""
        return (states - self.state_mean) / self.state_std
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions."""
        return (actions - self.action_mean) / self.action_std
    
    def denormalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Denormalize states."""
        return states * self.state_std + self.state_mean
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions."""
        return actions * self.action_std + self.action_mean
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trajectory.
        
        Returns:
            Dictionary with keys:
            - 'history_states': (N+1, state_dim)
            - 'history_actions': (N, action_dim)
            - 'future_states': (H, state_dim)
            - 'future_actions': (H, action_dim)
            - 'motion_id': str
            - 'success': bool
        """
        traj = self.trajectories[idx]
        
        # Get data
        history_states = traj.history_states.clone()
        history_actions = traj.history_actions.clone()
        future_states = traj.future_states.clone()
        future_actions = traj.future_actions.clone()
        
        # Add noise if augmenting
        if self.augment_noise and self.training:
            history_states += torch.randn_like(history_states) * self.noise_scale
            history_actions += torch.randn_like(history_actions) * self.noise_scale
            future_states += torch.randn_like(future_states) * self.noise_scale
            future_actions += torch.randn_like(future_actions) * self.noise_scale
        
        # Apply transform if specified
        if self.transform:
            history_states, history_actions, future_states, future_actions = \
                self.transform(history_states, history_actions, future_states, future_actions)
        
        return {
            'history_states': history_states,
            'history_actions': history_actions,
            'future_states': future_states,
            'future_actions': future_actions,
            'motion_id': traj.motion_id,
            'success': traj.success,
            'timestep': traj.timestep
        }
    
    def save(self, filepath: str):
        """Save dataset to disk."""
        data = {
            'trajectories': self.trajectories,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std
        }
        torch.save(data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrajectoryDataset':
        """Load dataset from disk."""
        # Use weights_only=False for now since we're loading our own data structures
        # In production, we should implement a safer serialization method
        data = torch.load(filepath, weights_only=False)
        dataset = cls(data['trajectories'])
        dataset.state_mean = data['state_mean']
        dataset.state_std = data['state_std']
        dataset.action_mean = data['action_mean']
        dataset.action_std = data['action_std']
        return dataset