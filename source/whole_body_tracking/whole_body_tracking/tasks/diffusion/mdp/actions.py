"""
Action functions for diffusion data collection.
Includes action delay randomization and trajectory storage.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def delayed_action(
    env: ManagerBasedRLEnv,
    action: torch.Tensor,
    delay_range_ms: tuple[int, int] = (0, 100),
    control_freq_hz: float = 20.0
) -> torch.Tensor:
    """
    Apply action delay randomization as described in paper Section VI.A.
    
    This is critical for handling the ~20ms inference latency during deployment.
    During training, actions are delayed by 0-100ms to simulate this latency.
    
    Args:
        env: The environment instance
        action: Current action from policy
        delay_range_ms: Range of delay in milliseconds (0-100ms from paper)
        control_freq_hz: Control frequency in Hz (20Hz = 50ms timesteps)
        
    Returns:
        Delayed action tensor
    """
    # Convert delay from milliseconds to timesteps
    dt_ms = 1000.0 / control_freq_hz  # timestep in milliseconds
    min_delay_steps = int(delay_range_ms[0] / dt_ms)
    max_delay_steps = int(delay_range_ms[1] / dt_ms)
    
    # Sample random delay for each environment
    delay_steps = torch.randint(
        min_delay_steps,
        max_delay_steps + 1,
        (env.num_envs,),
        device=env.device
    )
    
    # TODO: Implement proper action buffer management
    # This requires maintaining per-environment action buffers
    # For now, return the original action as placeholder
    
    return action


def action_trajectory_sequence(
    env: ManagerBasedRLEnv,
    current_action: torch.Tensor,
    future_horizon: int = 16
) -> torch.Tensor:
    """
    Create action sequence for trajectory τ_t = [a_t, s_{t+1}, ..., a_{t+H}].
    
    During data collection, this would be populated by rolling out
    the expert policy for the specified horizon.
    
    Args:
        env: The environment instance
        current_action: Current action from policy
        future_horizon: Number of future actions to predict
        
    Returns:
        Action sequence tensor
    """
    batch_size = env.num_envs
    action_dim = current_action.shape[1]
    
    # Create sequence starting with current action
    action_sequence = torch.zeros(
        batch_size, future_horizon + 1, action_dim,
        device=env.device
    )
    action_sequence[:, 0] = current_action
    
    # TODO: Fill in future actions by rolling out expert policy
    # For now, repeat current action as placeholder
    for i in range(1, future_horizon + 1):
        action_sequence[:, i] = current_action
    
    return action_sequence