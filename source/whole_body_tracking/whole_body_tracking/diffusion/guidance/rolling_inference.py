"""Rolling inference with FIFO buffer for temporal consistency."""

import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .classifier_guidance import ClassifierGuidance, GuidanceConfig
from ..models.diffusion_model import StateActionDiffusionModel
from .model_adapter import DiffusionModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class RollingConfig:
    """Configuration for rolling inference."""
    
    horizon: int = 16  # Total trajectory horizon
    replan_frequency: int = 1  # Replan every N steps
    warm_start_steps: int = 8  # Number of clean steps to keep
    noise_injection_start: int = 4  # Where to start adding noise
    temporal_smoothing: float = 0.9  # Blend factor with previous trajectory
    use_action_filtering: bool = True  # Apply low-pass filter to actions


class TrajectoryBuffer:
    """
    FIFO buffer for maintaining temporal consistency in trajectories.
    
    Key innovation from BeyondMimic:
    - Maintains a rolling window of partially-denoised trajectory
    - Executed actions are removed, new noisy future is added
    - Previous decisions persist, providing temporal coherence
    """
    
    def __init__(self, horizon: int, dim: int, device: str = "cuda"):
        """
        Initialize trajectory buffer.
        
        Args:
            horizon: Trajectory horizon length
            dim: Dimension of trajectory (state_dim + action_dim)
            device: Device for tensors
        """
        self.horizon = horizon
        self.dim = dim
        self.device = device
        
        # Initialize with noise
        self.buffer = torch.randn(1, horizon, dim, device=device)
        self.noise_levels = torch.ones(horizon, device=device)  # Track noise level per step
        
        # History of executed actions for smoothing
        self.action_history = []
        self.max_history = 5
    
    def shift_and_extend(
        self,
        num_executed: int = 1,
        new_observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shift buffer after executing actions and extend with noise.
        
        Args:
            num_executed: Number of actions that were executed
            new_observation: Optional new observation to incorporate
            
        Returns:
            Updated buffer tensor
        """
        if num_executed >= self.horizon:
            # Full reset if we've executed everything
            self.buffer = torch.randn_like(self.buffer)
            self.noise_levels = torch.ones_like(self.noise_levels)
            return self.buffer
        
        # Shift buffer left (remove executed actions)
        self.buffer = torch.cat([
            self.buffer[:, num_executed:],
            torch.randn(1, num_executed, self.dim, device=self.device)
        ], dim=1)
        
        # Update noise levels
        self.noise_levels = torch.cat([
            self.noise_levels[num_executed:],
            torch.ones(num_executed, device=self.device)
        ])
        
        # Optionally incorporate new observation
        if new_observation is not None:
            # Blend observation into first position
            self.buffer[:, 0] = 0.5 * self.buffer[:, 0] + 0.5 * new_observation
        
        return self.buffer
    
    def get_clean_portion(self, num_steps: int) -> torch.Tensor:
        """
        Get the clean (low-noise) portion of the buffer.
        
        Args:
            num_steps: Number of clean steps to retrieve
            
        Returns:
            Clean trajectory portion
        """
        num_clean = min(num_steps, (self.noise_levels < 0.3).sum().item())
        if num_clean > 0:
            return self.buffer[:, :num_clean]
        return None
    
    def update_with_denoised(
        self,
        denoised_trajectory: torch.Tensor,
        blend_factor: float = 0.9,
    ):
        """
        Update buffer with newly denoised trajectory.
        
        Args:
            denoised_trajectory: New denoised trajectory
            blend_factor: How much to blend with existing buffer
        """
        # Temporal smoothing - blend new with old
        self.buffer = (
            blend_factor * denoised_trajectory +
            (1 - blend_factor) * self.buffer
        )
        
        # Update noise levels (gradually decrease for denoised parts)
        self.noise_levels = self.noise_levels * 0.5
    
    def add_action_to_history(self, action: torch.Tensor):
        """
        Add executed action to history for analysis.
        
        Args:
            action: Executed action tensor
        """
        self.action_history.append(action.detach().cpu())
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)


class RollingGuidanceInference:
    """
    Rolling inference with guided diffusion for real-time control.
    
    Implements the key ideas from BeyondMimic:
    1. Maintain trajectory buffer across timesteps
    2. Only execute first action from planned trajectory
    3. Shift and replan with temporal consistency
    4. Use guidance to achieve task objectives
    """
    
    def __init__(
        self,
        model: StateActionDiffusionModel,
        guidance: ClassifierGuidance,
        config: Optional[RollingConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize rolling inference.
        
        Args:
            model: Trained diffusion model
            guidance: Classifier guidance module
            config: Rolling inference configuration
            device: Device for computation
        """
        # Wrap model with adapter if needed
        if not isinstance(model, DiffusionModelAdapter):
            self.model = DiffusionModelAdapter(model)
        else:
            self.model = model
        self.guidance = guidance
        self.config = config or RollingConfig()
        self.device = device
        
        # Initialize trajectory buffer
        traj_dim = self.model.state_dim + self.model.action_dim
        self.buffer = TrajectoryBuffer(
            horizon=self.config.horizon,
            dim=traj_dim,
            device=device,
        )
        
        # Step counter for replanning
        self.step_count = 0
        
        # Action filter for smoothing
        self.action_filter = None
        if self.config.use_action_filtering:
            self.action_filter = ActionLowPassFilter(alpha=0.7)
        
        logger.info("Initialized RollingGuidanceInference")
    
    def step(
        self,
        observation_history: torch.Tensor,
        cost_function,
        current_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Execute one step of rolling inference.
        
        Args:
            observation_history: Recent observations [1, history_len, obs_dim]
            cost_function: Task-specific cost function for guidance
            current_state: Optional current state for validation
            
        Returns:
            Dictionary with:
                - 'action': Action to execute [action_dim]
                - 'trajectory': Full planned trajectory
                - 'cost': Current cost value
                - 'info': Additional information
        """
        # Check if we need to replan
        should_replan = (self.step_count % self.config.replan_frequency) == 0
        
        if should_replan:
            # Get clean portion of buffer for warm-start
            clean_portion = self.buffer.get_clean_portion(
                self.config.warm_start_steps
            )
            
            # Perform guided denoising
            if clean_portion is not None:
                # Warm-start from clean portion
                initial_trajectory = torch.cat([
                    clean_portion,
                    torch.randn(
                        1,
                        self.config.horizon - clean_portion.shape[1],
                        self.buffer.dim,
                        device=self.device,
                    )
                ], dim=1)
            else:
                # Start from scratch
                initial_trajectory = self.buffer.buffer
            
            # Run guided sampling with warm-start
            results = self._guided_denoise_with_warmstart(
                initial_trajectory=initial_trajectory,
                observation_history=observation_history,
                cost_function=cost_function,
            )
            
            # Update buffer with new trajectory
            self.buffer.update_with_denoised(
                results["trajectory"],
                blend_factor=self.config.temporal_smoothing,
            )
        else:
            # Use existing buffer without replanning
            results = {
                "trajectory": self.buffer.buffer,
                "states": None,
                "actions": None,
                "costs": torch.tensor([0.0]),
            }
        
        # Extract first action to execute
        action = self._extract_first_action(self.buffer.buffer)
        
        # Apply action filtering if enabled
        if self.action_filter is not None:
            action = self.action_filter.filter(action)
        
        # Record action in history
        self.buffer.add_action_to_history(action)
        
        # Shift buffer for next step
        self.buffer.shift_and_extend(
            num_executed=1,
            new_observation=current_state,
        )
        
        # Increment step counter
        self.step_count += 1
        
        return {
            "action": action.squeeze(0),  # Remove batch dimension
            "trajectory": self.buffer.buffer.clone(),
            "cost": results["costs"].item(),
            "info": {
                "step_count": self.step_count,
                "replanned": should_replan,
                "noise_levels": self.buffer.noise_levels.clone(),
            }
        }
    
    def _guided_denoise_with_warmstart(
        self,
        initial_trajectory: torch.Tensor,
        observation_history: torch.Tensor,
        cost_function,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform guided denoising with warm-start trajectory.
        
        Args:
            initial_trajectory: Starting trajectory (partially clean)
            observation_history: Observation history
            cost_function: Cost function for guidance
            
        Returns:
            Denoising results
        """
        trajectory = initial_trajectory.clone()
        num_steps = self.model.num_steps
        
        # Determine starting noise level based on buffer state
        start_level = int(self.buffer.noise_levels[0].item() * num_steps)
        start_level = min(start_level, num_steps - 1)
        
        # Reverse diffusion from appropriate noise level
        for k in reversed(range(start_level + 1)):
            with torch.no_grad():
                # Model prediction
                clean_pred = self.model.denoise_step(
                    noisy_trajectory=trajectory,
                    observation_history=observation_history,
                    noise_level=k,
                )
            
            # Apply guidance if not at final step
            if k > 0:
                guidance_grad = self.guidance.compute_guidance_gradient(
                    trajectory=clean_pred,
                    cost_function=cost_function,
                    noise_level=k,
                    total_steps=num_steps,
                )
                clean_pred = clean_pred - guidance_grad
            
            # DDPM reverse step
            if k > 0:
                trajectory = self.model.reverse_step(
                    x_t=trajectory,
                    x_0_pred=clean_pred,
                    t=k,
                )
            else:
                trajectory = clean_pred
        
        # Extract states and actions
        states, actions = self.model._split_trajectory(trajectory)
        
        # Compute final cost
        with torch.no_grad():
            final_cost = cost_function(trajectory)
        
        return {
            "trajectory": trajectory,
            "states": states,
            "actions": actions,
            "costs": final_cost,
        }
    
    def _extract_first_action(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract the first action from trajectory.
        
        Args:
            trajectory: Full trajectory [1, horizon, state_dim + action_dim]
            
        Returns:
            First action [1, action_dim]
        """
        # Actions are at even indices in interleaved format
        # First element should be first action
        state_dim = self.model.state_dim
        action_dim = self.model.action_dim
        
        # Extract action portion
        first_action = trajectory[:, 0, state_dim:state_dim + action_dim]
        
        return first_action
    
    def reset(self):
        """Reset the rolling inference state."""
        # Reinitialize buffer
        traj_dim = self.model.state_dim + self.model.action_dim
        self.buffer = TrajectoryBuffer(
            horizon=self.config.horizon,
            dim=traj_dim,
            device=self.device,
        )
        
        # Reset counters
        self.step_count = 0
        
        # Reset action filter
        if self.action_filter is not None:
            self.action_filter.reset()
        
        logger.info("Reset rolling inference state")


class ActionLowPassFilter:
    """
    Low-pass filter for smoothing actions.
    
    Reduces high-frequency noise in action commands.
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize filter.
        
        Args:
            alpha: Filter coefficient (0=no filtering, 1=no change)
        """
        self.alpha = alpha
        self.prev_action = None
    
    def filter(self, action: torch.Tensor) -> torch.Tensor:
        """
        Apply low-pass filter to action.
        
        Args:
            action: Raw action tensor
            
        Returns:
            Filtered action
        """
        if self.prev_action is None:
            self.prev_action = action.clone()
            return action
        
        # Exponential moving average
        filtered = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = filtered.clone()
        
        return filtered
    
    def reset(self):
        """Reset filter state."""
        self.prev_action = None