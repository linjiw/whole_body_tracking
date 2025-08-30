"""Data collection pipeline for diffusion model training.

This module collects state-action trajectories from trained Stage 1 tracking policies.
Critical features:
- Action delay randomization (0-100ms) for handling inference latency
- State noise injection for robustness
- Trajectory chunking with proper history/future windows
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from collections import deque
import random

from .trajectory_dataset import Trajectory, StateRepresentation, TrajectoryDataset


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    # Trajectory windows
    history_length: int = 4  # N=4 from paper
    future_length_states: int = 32  # H=32 for states
    future_length_actions: int = 16  # H=16 for actions
    
    # Augmentation
    action_noise_scale: float = 0.01
    state_noise_scale: float = 0.005
    action_delay_prob: float = 0.5  # Probability of applying delay
    action_delay_range: tuple = (0, 100)  # milliseconds
    
    # Collection parameters
    episodes_per_motion: int = 10
    max_episode_length: int = 1000
    min_trajectory_length: int = 50  # Minimum length to save a trajectory
    
    # Domain randomization (matching Stage 1)
    friction_range: tuple = (0.5, 2.0)
    mass_range: tuple = (0.8, 1.2)
    com_offset_range: float = 0.05
    
    # State representation
    num_bodies: int = 23  # For SMPL/humanoid skeleton
    state_dim: int = 165  # From paper
    action_dim: int = 69  # Joint positions


class ActionDelayBuffer:
    """Buffer for simulating action delays during data collection."""
    
    def __init__(self, max_delay_steps: int = 3):
        """Initialize delay buffer.
        
        Args:
            max_delay_steps: Maximum delay in timesteps (at 30Hz, 3 steps = 100ms)
        """
        self.buffer = deque(maxlen=max_delay_steps)
        self.max_delay = max_delay_steps
    
    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def add_action(self, action: torch.Tensor, delay_steps: int = 0):
        """Add action with specified delay."""
        for _ in range(delay_steps):
            if len(self.buffer) < self.max_delay:
                self.buffer.append(torch.zeros_like(action))
        self.buffer.append(action)
    
    def get_delayed_action(self) -> Optional[torch.Tensor]:
        """Get the next action from buffer."""
        if len(self.buffer) > 0:
            return self.buffer.popleft()
        return None


class MotionDataCollector:
    """Collects state-action trajectories from trained tracking policies."""
    
    def __init__(
        self,
        env: Optional[Any] = None,  # Will be ManagerBasedRLEnv when available
        policies: Optional[Dict[str, Any]] = None,  # Will be OnPolicyRunner instances
        cfg: Optional[DataCollectionConfig] = None
    ):
        """Initialize data collector.
        
        Args:
            env: Isaac Lab environment (optional for now)
            policies: Dictionary of trained policies
            cfg: Data collection configuration
        """
        self.env = env
        self.policies = policies or {}
        self.cfg = cfg or DataCollectionConfig()
        
        # Buffers for trajectory collection
        self.state_buffer = deque(maxlen=self.cfg.history_length + self.cfg.future_length_states + 1)
        self.action_buffer = deque(maxlen=self.cfg.history_length + self.cfg.future_length_actions)
        self.delay_buffer = ActionDelayBuffer(max_delay_steps=3)
        
        # Collected trajectories
        self.trajectories = []
    
    def collect_trajectories(
        self,
        num_episodes: int,
        add_noise: bool = True,
        save_path: Optional[str] = None
    ) -> TrajectoryDataset:
        """Collect trajectories with domain randomization.
        
        Key features from paper:
        - Action delay randomization (0-100ms) for inference latency
        - State noise injection for robustness
        - Random policy switching between motion skills
        - Include failure recovery sequences for OOD handling
        - Domain randomization matching Stage 1 training
        
        Args:
            num_episodes: Number of episodes to collect
            add_noise: Whether to add noise augmentation
            save_path: Optional path to save dataset
            
        Returns:
            TrajectoryDataset containing collected trajectories
        """
        if self.env is None or len(self.policies) == 0:
            # For testing without Isaac Sim, generate mock data
            print("Warning: No environment or policies available. Generating mock data for testing.")
            return self._generate_mock_dataset(num_episodes)
        
        # Actual collection logic (will be implemented when Isaac Sim is available)
        for episode_idx in range(num_episodes):
            motion_id = random.choice(list(self.policies.keys()))
            policy = self.policies[motion_id]
            
            # Reset environment with domain randomization
            self._reset_with_randomization()
            
            # Collect episode
            self._collect_episode(policy, motion_id, episode_idx, add_noise)
        
        # Create dataset
        dataset = TrajectoryDataset(self.trajectories, augment_noise=add_noise)
        
        # Save if requested
        if save_path:
            dataset.save(save_path)
            print(f"Saved dataset with {len(dataset)} trajectories to {save_path}")
        
        return dataset
    
    def _collect_episode(
        self,
        policy: Any,
        motion_id: str,
        episode_id: int,
        add_noise: bool
    ):
        """Collect a single episode."""
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.delay_buffer.reset()
        
        obs = self.env.reset()
        episode_length = 0
        episode_success = True
        
        for timestep in range(self.cfg.max_episode_length):
            # Get action from policy
            action = policy(obs)
            
            # Add action noise if enabled
            if add_noise:
                action += torch.randn_like(action) * self.cfg.action_noise_scale
            
            # Apply action delay with probability
            if random.random() < self.cfg.action_delay_prob:
                delay_ms = random.uniform(*self.cfg.action_delay_range)
                delay_steps = int(delay_ms / 33.3)  # Convert to steps (30Hz)
                self.delay_buffer.add_action(action, delay_steps)
                action = self.delay_buffer.get_delayed_action() or action
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Convert observation to state representation
            state = self._obs_to_state(obs)
            next_state = self._obs_to_state(next_obs)
            
            # Add to buffers
            self.state_buffer.append(state)
            self.action_buffer.append(action)
            
            # Extract trajectories if we have enough history
            if len(self.state_buffer) >= self.cfg.history_length + self.cfg.future_length_states:
                self._extract_trajectory(motion_id, timestep, episode_success, episode_id)
            
            obs = next_obs
            episode_length += 1
            
            if done:
                episode_success = info.get('success', False)
                break
    
    def _extract_trajectory(
        self,
        motion_id: str,
        timestep: int,
        success: bool,
        episode_id: int
    ):
        """Extract a trajectory from the current buffers."""
        # Get history (last N+1 states, last N actions)
        history_states = torch.stack(list(self.state_buffer)[:self.cfg.history_length + 1])
        history_actions = torch.stack(list(self.action_buffer)[:self.cfg.history_length])
        
        # Get future
        future_states = torch.stack(
            list(self.state_buffer)[self.cfg.history_length + 1:
                                   self.cfg.history_length + 1 + self.cfg.future_length_states]
        )
        future_actions = torch.stack(
            list(self.action_buffer)[self.cfg.history_length:
                                    self.cfg.history_length + self.cfg.future_length_actions]
        )
        
        # Create trajectory
        trajectory = Trajectory(
            history_states=history_states,
            history_actions=history_actions,
            future_states=future_states,
            future_actions=future_actions,
            motion_id=motion_id,
            timestep=timestep,
            success=success,
            episode_id=episode_id
        )
        
        self.trajectories.append(trajectory)
    
    def _obs_to_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert observation to state representation.
        
        This will be implemented to match the actual observation structure
        from Stage 1. For now, returns mock state.
        """
        # Placeholder - will extract body positions, velocities, etc.
        return obs
    
    def _reset_with_randomization(self):
        """Reset environment with domain randomization.
        
        Matches Stage 1 training domain randomization:
        - Friction coefficients (0.5-2.0x)
        - Link masses (0.8-1.2x)
        - Center of mass offsets (±5cm)
        - Joint damping/stiffness (0.9-1.1x)
        - External forces (random perturbations)
        - Initial state noise
        """
        if self.env is None:
            return
        
        # Apply domain randomization matching Stage 1
        friction = random.uniform(*self.cfg.friction_range)
        mass_scale = random.uniform(*self.cfg.mass_range)
        com_offset = torch.randn(3) * self.cfg.com_offset_range
        
        # Additional randomizations from Stage 1
        joint_damping_scale = random.uniform(0.9, 1.1)
        joint_stiffness_scale = random.uniform(0.9, 1.1)
        joint_armature_scale = random.uniform(0.9, 1.1)
        
        # Random external forces (impulses)
        apply_external_force = random.random() < 0.1  # 10% chance
        if apply_external_force:
            force_magnitude = random.uniform(50, 200)  # N
            force_direction = torch.randn(3)
            force_direction = force_direction / torch.norm(force_direction)
            external_force = force_direction * force_magnitude
        else:
            external_force = None
        
        # Initial state noise (joint positions)
        initial_joint_noise = torch.randn(self.cfg.action_dim) * 0.05  # ±5% noise
        
        # Apply randomizations when Isaac Lab integration is available
        # These will be implemented with actual Isaac Lab API:
        # self.env.set_friction_coefficients(friction)
        # self.env.set_link_masses(mass_scale)
        # self.env.set_com_offsets(com_offset)
        # self.env.set_joint_damping(joint_damping_scale)
        # self.env.set_joint_stiffness(joint_stiffness_scale)
        # self.env.set_joint_armature(joint_armature_scale)
        # if external_force is not None:
        #     self.env.apply_external_force(external_force)
        # self.env.add_initial_joint_noise(initial_joint_noise)
        
        # Store randomization parameters for logging
        self._current_randomization = {
            'friction': friction,
            'mass_scale': mass_scale,
            'com_offset': com_offset.numpy() if hasattr(com_offset, 'numpy') else com_offset,
            'joint_damping_scale': joint_damping_scale,
            'joint_stiffness_scale': joint_stiffness_scale,
            'joint_armature_scale': joint_armature_scale,
            'external_force': external_force.numpy() if external_force is not None else None,
            'initial_joint_noise_std': 0.05
        }
    
    def _generate_mock_dataset(self, num_episodes: int) -> TrajectoryDataset:
        """Generate mock dataset for testing without Isaac Sim."""
        print(f"Generating {num_episodes} mock trajectories for testing...")
        
        mock_trajectories = []
        
        for episode_id in range(num_episodes):
            # Generate multiple trajectories per episode
            episode_length = random.randint(100, 500)
            
            for timestep in range(0, episode_length - self.cfg.future_length_states, 10):
                # Create mock states and actions
                history_states = torch.randn(
                    self.cfg.history_length + 1, self.cfg.state_dim
                ) * 0.1
                
                history_actions = torch.randn(
                    self.cfg.history_length, self.cfg.action_dim
                ) * 0.1
                
                future_states = torch.randn(
                    self.cfg.future_length_states, self.cfg.state_dim
                ) * 0.1
                
                future_actions = torch.randn(
                    self.cfg.future_length_actions, self.cfg.action_dim
                ) * 0.1
                
                # Add some structure to make it more realistic
                # States should evolve smoothly
                for i in range(1, len(future_states)):
                    future_states[i] = future_states[i-1] + torch.randn_like(future_states[i]) * 0.01
                
                # Actions should be somewhat correlated
                for i in range(1, len(future_actions)):
                    future_actions[i] = 0.9 * future_actions[i-1] + 0.1 * future_actions[i]
                
                trajectory = Trajectory(
                    history_states=history_states,
                    history_actions=history_actions,
                    future_states=future_states,
                    future_actions=future_actions,
                    motion_id=f"mock_motion_{episode_id % 5}",
                    timestep=timestep,
                    success=random.random() > 0.2,  # 80% success rate
                    episode_id=episode_id
                )
                
                mock_trajectories.append(trajectory)
        
        print(f"Generated {len(mock_trajectories)} mock trajectories")
        return TrajectoryDataset(mock_trajectories)