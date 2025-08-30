"""
Observation functions for diffusion data collection.
Extends tracking observations to support Body-Pos state representation.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pos_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Extract Body-Pos state representation following paper Section VI.D.
    
    This is the critical state representation that achieves 100% vs 72% success rate
    compared to Joint-Pos representation in the paper's ablation study.
    
    Returns:
        Body-Pos state: [root_pos_rel, root_lin_vel, root_ang_vel, body_pos_local, body_vel_local]
    """
    # Get robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Global states (relative to current frame) 
    root_pos_w = asset.data.root_pos_w
    root_lin_vel_w = asset.data.root_lin_vel_w
    root_ang_vel_w = asset.data.root_ang_vel_w
    root_quat_w = asset.data.root_quat_w
    
    # Make root position relative to environment origin
    root_pos_rel = root_pos_w - env.scene.env_origins
    
    # Local states (in character frame) - body positions and velocities
    body_pos_w = asset.data.body_pos_w  # World positions of all bodies
    body_lin_vel_w = asset.data.body_lin_vel_w  # World velocities of all bodies
    
    # Convert body positions to character frame (relative to root)
    # Subtract root position to get relative positions  
    body_pos_rel = body_pos_w - root_pos_w.unsqueeze(1)
    
    # Convert to root frame using inverse root rotation
    root_quat_inv = torch.stack([
        root_quat_w[:, 3], -root_quat_w[:, 0], -root_quat_w[:, 1], -root_quat_w[:, 2]
    ], dim=1)
    
    # Apply inverse rotation to get body positions in character frame
    body_pos_local = quat_apply_inverse(root_quat_w, body_pos_rel.view(-1, 3))
    body_pos_local = body_pos_local.view(body_pos_w.shape)
    
    # Convert body velocities to character frame
    body_vel_rel = body_lin_vel_w - root_lin_vel_w.unsqueeze(1) 
    body_vel_local = quat_apply_inverse(root_quat_w, body_vel_rel.view(-1, 3))
    body_vel_local = body_vel_local.view(body_lin_vel_w.shape)
    
    # Concatenate all components into Body-Pos representation
    # Following the paper's structure: [global_states, local_states]
    state = torch.cat([
        root_pos_rel,                                # (N, 3) - relative root position
        root_lin_vel_w,                             # (N, 3) - root linear velocity  
        root_ang_vel_w,                             # (N, 3) - root angular velocity
        body_pos_local.flatten(start_dim=1),        # (N, num_bodies*3) - body positions in character frame
        body_vel_local.flatten(start_dim=1),        # (N, num_bodies*3) - body velocities in character frame
    ], dim=1)
    
    return state


def trajectory_history(
    env: ManagerBasedRLEnv, 
    history_length: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Extract trajectory history O_t = [s_{t-N}, a_{t-N}, ..., s_t] for diffusion model.
    
    Args:
        env: The environment instance
        history_length: Number of historical timesteps (N in paper)
        asset_cfg: Scene entity configuration for the robot
        
    Returns:
        History tensor containing past states and actions
    """
    # This would require maintaining a history buffer in the environment
    # For now, return current state as placeholder
    current_state = body_pos_state(env, asset_cfg)
    
    # TODO: Implement proper history tracking
    # This would need to be integrated with the environment's step function
    # to maintain a rolling buffer of past states and actions
    
    # Placeholder: repeat current state for history length
    batch_size = current_state.shape[0]
    state_dim = current_state.shape[1]
    action_dim = env.action_space.shape[0]
    
    # Create placeholder history
    history_states = current_state.unsqueeze(1).repeat(1, history_length + 1, 1)
    history_actions = torch.zeros(batch_size, history_length, action_dim, device=env.device)
    
    # Flatten and concatenate 
    history = torch.cat([
        history_states.flatten(start_dim=1),
        history_actions.flatten(start_dim=1)
    ], dim=1)
    
    return history


def future_trajectory_target(
    env: ManagerBasedRLEnv,
    horizon: int = 16,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Generate future trajectory targets τ_t = [a_t, s_{t+1}, ..., s_{t+H}, a_{t+H}].
    
    This would be used during training data collection to create ground truth
    future trajectories from the expert tracking policies.
    
    Args:
        env: The environment instance  
        horizon: Future trajectory horizon (H in paper)
        asset_cfg: Scene entity configuration for the robot
        
    Returns:
        Future trajectory tensor
    """
    # This is a placeholder - in actual data collection, this would be
    # populated by rolling out the expert policy for H timesteps
    
    batch_size = env.num_envs
    state_dim = body_pos_state(env, asset_cfg).shape[1]
    action_dim = env.action_space.shape[0]
    
    # Placeholder future trajectory
    future_actions = torch.zeros(batch_size, horizon + 1, action_dim, device=env.device)
    future_states = torch.zeros(batch_size, horizon, state_dim, device=env.device)
    
    # Interleave actions and states: [a_t, s_{t+1}, a_{t+1}, s_{t+2}, ...]
    trajectory = torch.cat([
        future_actions.flatten(start_dim=1),
        future_states.flatten(start_dim=1)
    ], dim=1)
    
    return trajectory