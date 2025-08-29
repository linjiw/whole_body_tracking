"""State-action diffusion model for humanoid control.

This module implements the joint state-action diffusion model following
the BeyondMimic paper specifications with differentiated attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .embeddings import (
    SinusoidalPositionEmbeddings,
    ObservationHistoryEmbedding,
    FutureTrajectoryEmbedding,
    ClassifierGuidanceEmbedding
)
from .transformer import TransformerWithHistory, DifferentiatedTransformer


class NoiseSchedule(nn.Module):
    """Noise schedule for diffusion process.
    
    Implements independent schedules for states and actions as specified
    in the paper (k_s and k_a can be different).
    """
    
    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        """Initialize noise schedule.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: Type of schedule ("linear", "cosine", "quadratic")
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Generate beta schedule
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule from improved DDPM
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        elif schedule_type == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Store as buffers (not trainable parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Pre-compute for q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Pre-compute for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", 
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0).
        
        Args:
            x_0: Original data
            t: Timestep indices
            noise: Optional pre-generated noise
            
        Returns:
            Noisy samples at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
    
    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Original data
            x_t: Noisy data at timestep t
            t: Timestep indices
            
        Returns:
            Tuple of (posterior_mean, posterior_log_variance)
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].unsqueeze(-1).unsqueeze(-1) * x_0 +
            self.posterior_mean_coef2[t].unsqueeze(-1).unsqueeze(-1) * x_t
        )
        posterior_log_variance = self.posterior_log_variance[t].unsqueeze(-1).unsqueeze(-1)
        
        return posterior_mean, posterior_log_variance


class StateActionDiffusionModel(nn.Module):
    """Joint state-action diffusion model.
    
    Key features from the paper:
    - Joint modeling of states and actions p(τ) where τ = [s, a]
    - Differentiated attention mechanisms
    - Independent noise schedules for states and actions
    - Classifier guidance support
    """
    
    def __init__(
        self,
        state_dim: int = 165,
        action_dim: int = 69,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        history_length: int = 4,
        future_length_states: int = 32,
        future_length_actions: int = 16,
        num_timesteps: int = 100,
        state_schedule_type: str = "cosine",
        action_schedule_type: str = "linear",
        dropout: float = 0.1
    ):
        """Initialize diffusion model.
        
        Args:
            state_dim: Dimension of state vectors
            action_dim: Dimension of action vectors
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            history_length: Length of observation history (N)
            future_length_states: Length of future states (H_s)
            future_length_actions: Length of future actions (H_a)
            num_timesteps: Number of diffusion timesteps
            state_schedule_type: Noise schedule for states
            action_schedule_type: Noise schedule for actions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.future_length_states = future_length_states
        self.future_length_actions = future_length_actions
        self.num_timesteps = num_timesteps
        
        # Embedding layers
        self.time_embed = SinusoidalPositionEmbeddings(hidden_dim)
        self.history_embed = ObservationHistoryEmbedding(
            state_dim, action_dim, hidden_dim, history_length, dropout
        )
        self.trajectory_embed = FutureTrajectoryEmbedding(
            state_dim, action_dim, hidden_dim,
            future_length_states, future_length_actions, dropout
        )
        
        # Main transformer
        base_transformer = DifferentiatedTransformer(
            hidden_dim, num_layers, num_heads,
            mlp_ratio=4, dropout=dropout,
            state_dim=state_dim, action_dim=action_dim
        )
        self.transformer = TransformerWithHistory(
            base_transformer, hidden_dim, dropout
        )
        
        # Independent noise schedules
        self.state_noise_schedule = NoiseSchedule(
            num_timesteps, schedule_type=state_schedule_type
        )
        self.action_noise_schedule = NoiseSchedule(
            num_timesteps, schedule_type=action_schedule_type
        )
    
    def forward(
        self,
        future_states: torch.Tensor,
        future_actions: torch.Tensor,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through diffusion model.
        
        Args:
            future_states: (batch_size, H_s, state_dim) noisy future states
            future_actions: (batch_size, H_a, action_dim) noisy future actions
            history_states: (batch_size, N+1, state_dim) observation history states
            history_actions: (batch_size, N, action_dim) observation history actions
            timesteps: (batch_size,) diffusion timesteps
            guidance: Optional (batch_size, guidance_dim) guidance features
            
        Returns:
            Tuple of:
                - (batch_size, H_s, state_dim) predicted noise for states
                - (batch_size, H_a, action_dim) predicted noise for actions
        """
        batch_size = future_states.shape[0]
        
        # Embed time
        time_embeds = self.time_embed(timesteps)  # (B, hidden_dim)
        
        # Embed observation history
        history_embeds = self.history_embed(history_states, history_actions)  # (B, 2N+1, hidden)
        
        # Embed future trajectory
        state_embeds, action_embeds = self.trajectory_embed(
            future_states, future_actions
        )  # (B, H_s, hidden), (B, H_a, hidden)
        
        # Process through transformer
        pred_states, pred_actions = self.transformer(
            state_embeds, action_embeds, history_embeds, time_embeds
        )
        
        return pred_states, pred_actions
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss_components: bool = False
    ) -> torch.Tensor:
        """Single training step.
        
        Args:
            batch: Batch of training data with keys:
                - 'history_states': (B, N+1, state_dim)
                - 'history_actions': (B, N, action_dim)
                - 'future_states': (B, H_s, state_dim)
                - 'future_actions': (B, H_a, action_dim)
            return_loss_components: Whether to return individual loss components
            
        Returns:
            Loss value (and optionally loss components)
        """
        batch_size = batch['future_states'].shape[0]
        device = batch['future_states'].device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Add noise to future trajectories
        state_noise = torch.randn_like(batch['future_states'])
        action_noise = torch.randn_like(batch['future_actions'])
        
        noisy_states = self.state_noise_schedule.q_sample(
            batch['future_states'], t, state_noise
        )
        noisy_actions = self.action_noise_schedule.q_sample(
            batch['future_actions'], t, action_noise
        )
        
        # Predict noise
        pred_state_noise, pred_action_noise = self.forward(
            noisy_states, noisy_actions,
            batch['history_states'], batch['history_actions'],
            t
        )
        
        # Compute losses
        state_loss = F.mse_loss(pred_state_noise, batch['future_states'])
        
        # Action loss with horizon masking (only first 8 actions)
        action_mask = torch.zeros_like(pred_action_noise)
        action_mask[:, :8] = 1.0  # Only compute loss on first 8 actions
        masked_action_loss = F.mse_loss(
            pred_action_noise * action_mask,
            batch['future_actions'] * action_mask
        )
        
        # Total loss
        total_loss = state_loss + masked_action_loss
        
        if return_loss_components:
            return total_loss, {
                'total_loss': total_loss.item(),
                'state_loss': state_loss.item(),
                'action_loss': masked_action_loss.item()
            }
        
        return total_loss
    
    @torch.no_grad()
    def sample(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample future trajectories using DDPM.
        
        Args:
            history_states: (batch_size, N+1, state_dim) observation history
            history_actions: (batch_size, N, action_dim) action history
            guidance: Optional guidance features
            guidance_scale: Scale for classifier guidance
            return_intermediates: Whether to return intermediate samples
            
        Returns:
            Tuple of:
                - (batch_size, H_s, state_dim) sampled future states
                - (batch_size, H_a, action_dim) sampled future actions
        """
        batch_size = history_states.shape[0]
        device = history_states.device
        
        # Start from pure noise
        states = torch.randn(
            batch_size, self.future_length_states, self.state_dim,
            device=device
        )
        actions = torch.randn(
            batch_size, self.future_length_actions, self.action_dim,
            device=device
        )
        
        intermediates = []
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            pred_state_noise, pred_action_noise = self.forward(
                states, actions, history_states, history_actions, t_batch, guidance
            )
            
            # Apply classifier guidance if provided
            if guidance is not None and guidance_scale != 1.0:
                # This would require a separate classifier network
                # For now, we'll skip the guidance gradient computation
                pass
            
            # Remove noise (DDPM update)
            if t > 0:
                # Add noise for next step
                state_noise = torch.randn_like(states)
                action_noise = torch.randn_like(actions)
                
                # Compute x_{t-1} from x_t and predicted noise
                states = self._ddpm_step(
                    states, pred_state_noise, t, self.state_noise_schedule, state_noise
                )
                actions = self._ddpm_step(
                    actions, pred_action_noise, t, self.action_noise_schedule, action_noise
                )
            else:
                # Final step without noise
                states = self._ddpm_step(
                    states, pred_state_noise, t, self.state_noise_schedule, None
                )
                actions = self._ddpm_step(
                    actions, pred_action_noise, t, self.action_noise_schedule, None
                )
            
            if return_intermediates:
                intermediates.append((states.clone(), actions.clone()))
        
        if return_intermediates:
            return states, actions, intermediates
        
        return states, actions
    
    def _ddpm_step(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        t: int,
        noise_schedule: NoiseSchedule,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single DDPM denoising step.
        
        Args:
            x_t: Current noisy sample
            pred_noise: Predicted noise
            t: Current timestep
            noise_schedule: Noise schedule to use
            noise: Optional noise for next step
            
        Returns:
            Denoised sample x_{t-1}
        """
        # Predict x_0
        alpha_cumprod = noise_schedule.alphas_cumprod[t]
        sqrt_recip_alpha_cumprod = 1.0 / torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod[t]
        
        pred_x_0 = sqrt_recip_alpha_cumprod * (x_t - sqrt_one_minus_alpha_cumprod * pred_noise)
        
        # Clip predictions
        pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
        
        # Compute x_{t-1}
        if t > 0:
            posterior_mean, posterior_log_variance = noise_schedule.q_posterior(
                pred_x_0, x_t, torch.tensor([t])
            )
            
            if noise is not None:
                return posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
            else:
                return posterior_mean
        else:
            return pred_x_0