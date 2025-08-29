"""Wrapper to integrate diffusion policy with Isaac Lab environment."""

import torch
from typing import Dict, Optional, Any, Tuple
from collections import deque
import logging

from isaaclab.envs import ManagerBasedRLEnv
from ..models.diffusion_model import StateActionDiffusionModel
from ..guidance import (
    ClassifierGuidance,
    GuidanceConfig,
    RollingGuidanceInference,
    RollingConfig,
)
from .space_converters import ObservationConverter, ActionConverter, ObservationSpaceConfig

logger = logging.getLogger(__name__)


class IsaacLabDiffusionWrapper:
    """
    Wrapper that enables using a diffusion policy with Isaac Lab environments.
    
    This wrapper handles:
    - Observation history management
    - Space conversion between Isaac Lab and diffusion model
    - Rolling inference with guidance
    - Action filtering and smoothing
    """
    
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        diffusion_model: StateActionDiffusionModel,
        guidance_config: Optional[GuidanceConfig] = None,
        rolling_config: Optional[RollingConfig] = None,
        obs_config: Optional[ObservationSpaceConfig] = None,
        history_length: int = 4,
        device: str = "cuda",
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: Isaac Lab environment
            diffusion_model: Trained diffusion model
            guidance_config: Configuration for classifier guidance
            rolling_config: Configuration for rolling inference
            obs_config: Observation space configuration
            history_length: Length of observation history
            device: Device for computation
        """
        self.env = env
        self.device = device
        self.history_length = history_length
        
        # Initialize converters
        self.obs_config = obs_config or ObservationSpaceConfig()
        self.obs_converter = ObservationConverter(self.obs_config)
        self.action_converter = ActionConverter(num_actions=self.obs_config.num_joints)
        
        # Initialize guidance and rolling inference
        guidance_config = guidance_config or GuidanceConfig()
        self.guidance = ClassifierGuidance(diffusion_model, guidance_config, device)
        
        rolling_config = rolling_config or RollingConfig()
        self.rolling_inference = RollingGuidanceInference(
            diffusion_model,
            self.guidance,
            rolling_config,
            device,
        )
        
        # History buffers
        self.obs_buffer = deque(maxlen=history_length)
        self.action_buffer = deque(maxlen=history_length)
        
        # Statistics tracking
        self.step_count = 0
        self.total_cost = 0.0
        
        logger.info(
            f"IsaacLabDiffusionWrapper initialized with history_length={history_length}"
        )
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset the environment and wrapper state.
        
        Returns:
            Initial observation from environment
        """
        # Reset environment
        obs, _ = self.env.reset()
        
        # Clear history buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        
        # Fill initial history with current observation
        for _ in range(self.history_length):
            self.obs_buffer.append(obs)
            # Add zero actions initially
            zero_action = torch.zeros(
                self.env.num_envs,
                self.obs_config.num_joints,
                device=self.device,
            )
            self.action_buffer.append(zero_action)
        
        # Reset rolling inference
        self.rolling_inference.reset()
        
        # Reset statistics
        self.step_count = 0
        self.total_cost = 0.0
        
        logger.info("Environment and wrapper reset")
        
        return obs
    
    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        cost_function: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Get action from diffusion policy with guidance.
        
        Args:
            obs: Current observation from environment
            cost_function: Optional cost function for guidance
            
        Returns:
            Action to execute in environment
        """
        # Update observation buffer
        self.obs_buffer.append(obs)
        
        # Create history tensor for diffusion model
        history_tensor = self.obs_converter.create_history_tensor(
            list(self.obs_buffer),
            list(self.action_buffer),
        )
        
        # Get action from rolling inference
        if cost_function is not None:
            # Use guided inference
            result = self.rolling_inference.step(
                observation_history=history_tensor,
                cost_function=cost_function,
            )
            action = result["action"]
            cost = result["cost"]
            self.total_cost += cost
        else:
            # Use unguided inference (just diffusion model)
            with torch.no_grad():
                # Sample from diffusion model without guidance
                states, actions = self.guidance.model.model.sample(
                    history_states=history_tensor[:, :, :self.obs_config.body_pos_state_dim],
                    history_actions=history_tensor[:, :, self.obs_config.body_pos_state_dim:],
                )
                # Take first action
                action = actions[:, 0]
        
        # Convert action to Isaac Lab format
        isaac_action = self.action_converter.diffusion_to_isaac(action)
        
        # Update action buffer
        self.action_buffer.append(isaac_action)
        
        self.step_count += 1
        
        return isaac_action
    
    def step(
        self,
        action: Optional[torch.Tensor] = None,
        cost_function: Optional[Any] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Take a step in the environment using diffusion policy.
        
        Args:
            action: Optional action to execute (if None, gets from policy)
            cost_function: Optional cost function for guidance
            
        Returns:
            Tuple of (obs, reward, done, info)
        """
        # Get current observation
        current_obs = self.env.get_observations()[0]
        
        # Get action if not provided
        if action is None:
            action = self.get_action(current_obs, cost_function)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add step statistics to info
        info["diffusion_step_count"] = self.step_count
        if cost_function is not None:
            info["diffusion_avg_cost"] = self.total_cost / max(1, self.step_count)
        
        return obs, reward, terminated, truncated, info
    
    def get_observations(self) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Get current observations from environment.
        
        Returns:
            Tuple of (observations, info)
        """
        return self.env.get_observations()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.env.num_envs
    
    @property
    def unwrapped(self):
        """Get unwrapped environment."""
        return self.env.unwrapped