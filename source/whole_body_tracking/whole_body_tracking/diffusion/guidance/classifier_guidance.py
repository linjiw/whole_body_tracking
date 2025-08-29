"""Classifier guidance for steering diffusion trajectories at test time."""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from ..models.diffusion_model import StateActionDiffusionModel
from .model_adapter import DiffusionModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class GuidanceConfig:
    """Configuration for classifier guidance."""
    
    guidance_scale: float = 1.0  # Strength of guidance gradient
    gradient_clipping: float = 10.0  # Max gradient norm
    warmup_steps: int = 5  # Steps before applying guidance
    cooldown_steps: int = 3  # Steps to reduce guidance at end
    use_gradient_checkpointing: bool = False  # For memory efficiency
    normalize_gradients: bool = True  # Normalize gradients before applying


class ClassifierGuidance:
    """
    Implements classifier guidance for test-time control of diffusion policies.
    
    Based on the BeyondMimic paper equation:
    τ_{k-1} = denoise(τ_k) - guidance_scale * ∇_τ G_c(τ_k)
    
    This allows steering the generated trajectories toward task objectives
    without any retraining, using only gradients from differentiable cost functions.
    """
    
    def __init__(
        self,
        model: StateActionDiffusionModel,
        config: Optional[GuidanceConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize classifier guidance.
        
        Args:
            model: Trained diffusion model
            config: Guidance configuration
            device: Device for computation
        """
        # Wrap model with adapter
        self.model = DiffusionModelAdapter(model)
        self.config = config or GuidanceConfig()
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.model.to(device)
        self.model.model.eval()
        
        logger.info(f"Initialized ClassifierGuidance with scale={self.config.guidance_scale}")
    
    def compute_guidance_gradient(
        self,
        trajectory: torch.Tensor,
        cost_function: Callable[[torch.Tensor], torch.Tensor],
        noise_level: int,
        total_steps: int,
    ) -> torch.Tensor:
        """
        Compute the guidance gradient for steering the trajectory.
        
        Args:
            trajectory: Current noisy trajectory [batch_size, horizon, state_dim + action_dim]
            cost_function: Differentiable cost function G_c(τ)
            noise_level: Current denoising step k
            total_steps: Total number of denoising steps K
            
        Returns:
            Guidance gradient [batch_size, horizon, state_dim + action_dim]
        """
        # Enable gradients for trajectory
        trajectory = trajectory.detach().requires_grad_(True)
        
        # Compute cost
        cost = cost_function(trajectory)
        
        # Ensure cost is scalar or reduce if needed
        if cost.dim() > 0:
            cost = cost.mean()
        
        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=cost,
            inputs=trajectory,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        # Apply warmup/cooldown schedule
        scale = self._compute_guidance_schedule(noise_level, total_steps)
        
        # Normalize gradients if configured
        if self.config.normalize_gradients:
            gradient = self._normalize_gradient(gradient)
        
        # Clip gradients to prevent instability
        if self.config.gradient_clipping > 0:
            gradient = self._clip_gradient(gradient)
        
        return scale * self.config.guidance_scale * gradient
    
    def guided_sampling(
        self,
        observation_history: torch.Tensor,
        cost_function: Callable[[torch.Tensor], torch.Tensor],
        num_samples: int = 1,
        return_all_steps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate guided trajectories using classifier guidance.
        
        Args:
            observation_history: Past observations [batch_size, history_len, obs_dim]
            cost_function: Task-specific cost function
            num_samples: Number of trajectories to generate
            return_all_steps: Whether to return intermediate denoising steps
            
        Returns:
            Dictionary containing:
                - 'trajectory': Final clean trajectory
                - 'states': Extracted state sequence
                - 'actions': Extracted action sequence
                - 'costs': Final cost values
                - 'all_steps': (optional) All intermediate trajectories
        """
        batch_size = num_samples
        horizon = self.model.horizon
        traj_dim = self.model.state_dim + self.model.action_dim
        
        # Expand observation history if needed
        if observation_history.shape[0] == 1 and batch_size > 1:
            observation_history = observation_history.expand(batch_size, -1, -1)
        
        # Initialize from pure noise
        trajectory = torch.randn(
            batch_size, horizon, traj_dim,
            device=self.device
        )
        
        all_steps = [] if return_all_steps else None
        
        # Reverse diffusion with guidance
        num_steps = self.model.num_steps
        
        for k in reversed(range(num_steps)):
            with torch.no_grad():
                # Get model prediction without guidance
                clean_pred = self.model.denoise_step(
                    noisy_trajectory=trajectory,
                    observation_history=observation_history,
                    noise_level=k,
                )
            
            # Compute guidance gradient if not at final step
            if k > 0 and k > self.config.cooldown_steps:
                guidance_grad = self.compute_guidance_gradient(
                    trajectory=clean_pred,
                    cost_function=cost_function,
                    noise_level=k,
                    total_steps=num_steps,
                )
                
                # Apply guidance to clean prediction
                clean_pred = clean_pred - guidance_grad
            
            # DDPM reverse step (with or without guidance applied)
            if k > 0:
                trajectory = self.model.reverse_step(
                    x_t=trajectory,
                    x_0_pred=clean_pred,
                    t=k,
                )
            else:
                trajectory = clean_pred
            
            if return_all_steps:
                all_steps.append(trajectory.clone())
        
        # Extract states and actions
        states, actions = self.model._split_trajectory(trajectory)
        
        # Compute final costs
        with torch.no_grad():
            final_costs = cost_function(trajectory)
        
        results = {
            "trajectory": trajectory,
            "states": states,
            "actions": actions,
            "costs": final_costs,
        }
        
        if return_all_steps:
            results["all_steps"] = torch.stack(all_steps, dim=0)
        
        return results
    
    def _compute_guidance_schedule(self, noise_level: int, total_steps: int) -> float:
        """
        Compute guidance strength schedule with warmup and cooldown.
        
        Args:
            noise_level: Current denoising step (K to 0)
            total_steps: Total number of steps K
            
        Returns:
            Guidance scale multiplier
        """
        # Warmup phase (early denoising)
        if noise_level > total_steps - self.config.warmup_steps:
            progress = (total_steps - noise_level) / self.config.warmup_steps
            return progress  # Linear warmup
        
        # Cooldown phase (final denoising)
        if noise_level < self.config.cooldown_steps:
            progress = noise_level / self.config.cooldown_steps
            return progress  # Linear cooldown
        
        return 1.0  # Full guidance
    
    def _normalize_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Normalize gradient to unit norm.
        
        Args:
            gradient: Raw gradient tensor
            
        Returns:
            Normalized gradient
        """
        grad_norm = torch.norm(gradient, dim=-1, keepdim=True)
        grad_norm = torch.clamp(grad_norm, min=1e-8)  # Prevent division by zero
        return gradient / grad_norm
    
    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Clip gradient norm to prevent instability.
        
        Args:
            gradient: Raw gradient tensor
            
        Returns:
            Clipped gradient
        """
        grad_norm = torch.norm(gradient)
        if grad_norm > self.config.gradient_clipping:
            gradient = gradient * (self.config.gradient_clipping / grad_norm)
        return gradient
    
    def compose_cost_functions(
        self,
        cost_functions: Dict[str, Tuple[Callable, float]],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Compose multiple cost functions with weights.
        
        Args:
            cost_functions: Dictionary of {name: (function, weight)}
            
        Returns:
            Composed cost function
        """
        def composed_cost(trajectory: torch.Tensor) -> torch.Tensor:
            total_cost = 0.0
            for name, (func, weight) in cost_functions.items():
                cost = func(trajectory)
                total_cost = total_cost + weight * cost
                logger.debug(f"Cost {name}: {cost.item():.4f} (weight={weight})")
            return total_cost
        
        return composed_cost