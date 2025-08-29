"""Converters between Isaac Lab observation/action spaces and diffusion model formats."""

import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObservationSpaceConfig:
    """Configuration for observation space dimensions."""
    
    # Based on tracking_env_cfg.py PolicyCfg observation terms
    num_joints: int = 19  # For G1 robot (can be 69 for full body)
    command_dim: int = None  # num_joints * 2 (pos + vel)
    motion_anchor_pos_dim: int = 3
    motion_anchor_ori_dim: int = 6  # First 2 columns of rotation matrix
    base_lin_vel_dim: int = 3
    base_ang_vel_dim: int = 3
    joint_pos_dim: int = None  # num_joints
    joint_vel_dim: int = None  # num_joints
    actions_dim: int = None  # num_joints
    
    # Body-Pos representation dimensions (for diffusion model)
    body_pos_state_dim: int = 48  # Default from diffusion model
    
    def __post_init__(self):
        """Initialize dependent dimensions."""
        if self.command_dim is None:
            self.command_dim = self.num_joints * 2
        if self.joint_pos_dim is None:
            self.joint_pos_dim = self.num_joints
        if self.joint_vel_dim is None:
            self.joint_vel_dim = self.num_joints
        if self.actions_dim is None:
            self.actions_dim = self.num_joints
    
    @property
    def total_obs_dim(self) -> int:
        """Total dimension of concatenated observation."""
        return (
            self.command_dim + 
            self.motion_anchor_pos_dim + 
            self.motion_anchor_ori_dim +
            self.base_lin_vel_dim +
            self.base_ang_vel_dim +
            self.joint_pos_dim +
            self.joint_vel_dim +
            self.actions_dim
        )


class ObservationConverter:
    """
    Converts between Isaac Lab observations and diffusion model state representation.
    
    Isaac Lab provides observations as a dictionary with 'policy' group containing
    concatenated terms. The diffusion model expects Body-Pos state representation.
    """
    
    def __init__(self, config: Optional[ObservationSpaceConfig] = None):
        """
        Initialize observation converter.
        
        Args:
            config: Observation space configuration
        """
        self.config = config or ObservationSpaceConfig()
        
        # Compute index ranges for parsing concatenated observation
        idx = 0
        self.command_range = (idx, idx + self.config.command_dim)
        idx += self.config.command_dim
        
        self.motion_anchor_pos_range = (idx, idx + self.config.motion_anchor_pos_dim)
        idx += self.config.motion_anchor_pos_dim
        
        self.motion_anchor_ori_range = (idx, idx + self.config.motion_anchor_ori_dim)
        idx += self.config.motion_anchor_ori_dim
        
        self.base_lin_vel_range = (idx, idx + self.config.base_lin_vel_dim)
        idx += self.config.base_lin_vel_dim
        
        self.base_ang_vel_range = (idx, idx + self.config.base_ang_vel_dim)
        idx += self.config.base_ang_vel_dim
        
        self.joint_pos_range = (idx, idx + self.config.joint_pos_dim)
        idx += self.config.joint_pos_dim
        
        self.joint_vel_range = (idx, idx + self.config.joint_vel_dim)
        idx += self.config.joint_vel_dim
        
        self.actions_range = (idx, idx + self.config.actions_dim)
        idx += self.config.actions_dim
        
        logger.info(f"ObservationConverter initialized with total dim: {idx}")
    
    def isaac_to_diffusion(
        self,
        isaac_obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert Isaac Lab observation to diffusion model state representation.
        
        Args:
            isaac_obs: Isaac Lab observation dictionary with 'policy' key
            
        Returns:
            State tensor for diffusion model [batch_size, state_dim]
        """
        # Get the concatenated policy observation
        if isinstance(isaac_obs, dict):
            policy_obs = isaac_obs.get("policy", isaac_obs.get("obs", None))
            if policy_obs is None:
                raise ValueError("No 'policy' or 'obs' key found in observation dict")
        else:
            # Assume it's already the concatenated tensor
            policy_obs = isaac_obs
        
        batch_size = policy_obs.shape[0]
        device = policy_obs.device
        
        # Parse individual components
        command = policy_obs[:, self.command_range[0]:self.command_range[1]]
        motion_anchor_pos = policy_obs[:, self.motion_anchor_pos_range[0]:self.motion_anchor_pos_range[1]]
        motion_anchor_ori = policy_obs[:, self.motion_anchor_ori_range[0]:self.motion_anchor_ori_range[1]]
        base_lin_vel = policy_obs[:, self.base_lin_vel_range[0]:self.base_lin_vel_range[1]]
        base_ang_vel = policy_obs[:, self.base_ang_vel_range[0]:self.base_ang_vel_range[1]]
        joint_pos = policy_obs[:, self.joint_pos_range[0]:self.joint_pos_range[1]]
        joint_vel = policy_obs[:, self.joint_vel_range[0]:self.joint_vel_range[1]]
        last_action = policy_obs[:, self.actions_range[0]:self.actions_range[1]]
        
        # Convert to Body-Pos representation
        # This is a simplified version - in practice, you'd compute actual body positions
        state = self._create_body_pos_state(
            motion_anchor_pos=motion_anchor_pos,
            motion_anchor_ori=motion_anchor_ori,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            command=command,
            device=device,
        )
        
        return state
    
    def _create_body_pos_state(
        self,
        motion_anchor_pos: torch.Tensor,
        motion_anchor_ori: torch.Tensor,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        command: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """
        Create Body-Pos state representation.
        
        This is a placeholder that creates a state compatible with the diffusion model.
        In practice, this would compute actual Cartesian body positions from joint angles.
        
        Args:
            Various observation components
            
        Returns:
            Body-Pos state [batch_size, state_dim]
        """
        batch_size = motion_anchor_pos.shape[0]
        
        # For now, create a state by concatenating available information
        # and padding/projecting to match expected dimension
        components = [
            motion_anchor_pos,  # 3
            base_lin_vel,  # 3
            motion_anchor_ori[:, :3],  # 3 (rotation vector approximation)
            # We need to add body positions here
            # For now, use a projection of joint positions
        ]
        
        # Concatenate what we have
        partial_state = torch.cat(components, dim=-1)
        
        # Project or pad to expected dimension
        if partial_state.shape[-1] < self.config.body_pos_state_dim:
            # Pad with projected joint information
            padding_dim = self.config.body_pos_state_dim - partial_state.shape[-1]
            
            # Create a simple projection of joint positions to approximate body positions
            # In reality, this would use forward kinematics
            joint_features = torch.cat([joint_pos, joint_vel], dim=-1)
            
            # Simple linear projection (learned in practice)
            projection = torch.nn.Linear(
                joint_features.shape[-1],
                padding_dim,
                device=device,
            )
            
            with torch.no_grad():
                # Initialize with small random weights
                projection.weight.data.normal_(0, 0.01)
                projection.bias.data.zero_()
            
            padding = projection(joint_features)
            state = torch.cat([partial_state, padding], dim=-1)
        else:
            # Truncate if too large
            state = partial_state[:, :self.config.body_pos_state_dim]
        
        return state
    
    def create_history_tensor(
        self,
        obs_buffer: list,
        action_buffer: list,
    ) -> torch.Tensor:
        """
        Create observation history tensor for diffusion model.
        
        Args:
            obs_buffer: List of recent observations
            action_buffer: List of recent actions
            
        Returns:
            History tensor [batch_size, history_len, state_dim + action_dim]
        """
        if not obs_buffer:
            raise ValueError("Observation buffer is empty")
        
        history = []
        for i, obs in enumerate(obs_buffer):
            # Convert observation to state
            state = self.isaac_to_diffusion(obs)
            
            # Get corresponding action (or zeros if not available)
            if i < len(action_buffer):
                action = action_buffer[i]
            else:
                action = torch.zeros(
                    state.shape[0],
                    self.config.actions_dim,
                    device=state.device,
                )
            
            # Concatenate state and action
            combined = torch.cat([state, action], dim=-1)
            history.append(combined)
        
        # Stack into history tensor
        history_tensor = torch.stack(history, dim=1)
        
        return history_tensor


class ActionConverter:
    """
    Converts between diffusion model actions and Isaac Lab action format.
    
    The diffusion model outputs actions in its own scale/format,
    while Isaac Lab expects joint position targets with specific scaling.
    """
    
    def __init__(
        self,
        action_scale: Optional[Dict[str, float]] = None,
        num_actions: int = 19,
    ):
        """
        Initialize action converter.
        
        Args:
            action_scale: Scaling factors for actions (e.g., G1_ACTION_SCALE)
            num_actions: Number of action dimensions
        """
        self.action_scale = action_scale or {}
        self.num_actions = num_actions
        
        logger.info(f"ActionConverter initialized with {num_actions} actions")
    
    def diffusion_to_isaac(
        self,
        diffusion_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert diffusion model action to Isaac Lab format.
        
        Args:
            diffusion_action: Action from diffusion model [batch_size, action_dim]
            
        Returns:
            Action for Isaac Lab environment [batch_size, num_actions]
        """
        # Ensure correct shape
        if diffusion_action.dim() == 1:
            diffusion_action = diffusion_action.unsqueeze(0)
        
        batch_size = diffusion_action.shape[0]
        device = diffusion_action.device
        
        # Truncate or pad to match expected action dimension
        if diffusion_action.shape[-1] > self.num_actions:
            isaac_action = diffusion_action[:, :self.num_actions]
        elif diffusion_action.shape[-1] < self.num_actions:
            padding = torch.zeros(
                batch_size,
                self.num_actions - diffusion_action.shape[-1],
                device=device,
            )
            isaac_action = torch.cat([diffusion_action, padding], dim=-1)
        else:
            isaac_action = diffusion_action
        
        # Apply action scaling if provided
        if self.action_scale:
            # This would apply joint-specific scaling
            # For now, we'll skip this as it requires joint name mapping
            pass
        
        return isaac_action
    
    def isaac_to_diffusion(
        self,
        isaac_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert Isaac Lab action to diffusion model format.
        
        Args:
            isaac_action: Action from Isaac Lab [batch_size, num_actions]
            
        Returns:
            Action for diffusion model [batch_size, action_dim]
        """
        # For now, just pass through
        # In practice, might need to unscale or transform
        return isaac_action.clone()