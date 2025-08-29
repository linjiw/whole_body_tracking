"""Adapter to make the diffusion model compatible with guidance."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DiffusionModelAdapter:
    """
    Adapter class to make StateActionDiffusionModel compatible with guidance.
    
    This wraps the model to provide a unified interface for guidance.
    """
    
    def __init__(self, model):
        """
        Initialize adapter.
        
        Args:
            model: StateActionDiffusionModel instance
        """
        self.model = model
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        self.horizon = model.future_length_states
        self.num_steps = model.num_timesteps
        
        # Access noise schedules
        self.state_noise_schedule = model.state_noise_schedule
        self.action_noise_schedule = model.action_noise_schedule
    
    def denoise_step(
        self,
        noisy_trajectory: torch.Tensor,
        observation_history: torch.Tensor,
        noise_level: int,
    ) -> torch.Tensor:
        """
        Perform one denoising step.
        
        Args:
            noisy_trajectory: Noisy trajectory [batch_size, horizon, state_dim + action_dim]
            observation_history: Observation history [batch_size, history_len, obs_dim]
            noise_level: Current noise level (timestep)
            
        Returns:
            Predicted clean trajectory
        """
        batch_size = noisy_trajectory.shape[0]
        device = noisy_trajectory.device
        
        # Split trajectory into states and actions
        states, actions = self._split_trajectory(noisy_trajectory)
        
        # Extract history states and actions from observation_history
        # Assuming observation_history is [batch_size, history_len, state_dim + action_dim]
        history_states = observation_history[:, :, :self.state_dim]
        history_actions = observation_history[:, :, self.state_dim:self.state_dim + self.action_dim]
        
        # Create timestep tensor
        t = torch.full((batch_size,), noise_level, device=device, dtype=torch.long)
        
        # Forward pass through model
        with torch.no_grad():
            pred_state_noise, pred_action_noise = self.model(
                states, actions, 
                history_states, history_actions,
                t, guidance=None
            )
        
        # Compute predicted clean trajectory (x_0) from noise prediction
        # Using the formula: x_0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)
        sqrt_alpha = torch.sqrt(self.state_noise_schedule.alphas_cumprod[noise_level])
        sqrt_one_minus_alpha = torch.sqrt(1 - self.state_noise_schedule.alphas_cumprod[noise_level])
        
        # Denoise states
        clean_states = (states - sqrt_one_minus_alpha * pred_state_noise) / sqrt_alpha
        
        # Denoise actions  
        sqrt_alpha_a = torch.sqrt(self.action_noise_schedule.alphas_cumprod[noise_level])
        sqrt_one_minus_alpha_a = torch.sqrt(1 - self.action_noise_schedule.alphas_cumprod[noise_level])
        clean_actions = (actions - sqrt_one_minus_alpha_a * pred_action_noise) / sqrt_alpha_a
        
        # Recombine into trajectory
        clean_trajectory = self._combine_trajectory(clean_states, clean_actions)
        
        return clean_trajectory
    
    def reverse_step(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        DDPM reverse diffusion step.
        
        Args:
            x_t: Current noisy sample
            x_0_pred: Predicted clean sample
            t: Current timestep
            
        Returns:
            x_{t-1}: Less noisy sample
        """
        if t == 0:
            return x_0_pred
        
        device = x_t.device
        batch_size = x_t.shape[0]
        
        # Get schedule parameters
        beta_t = self.state_noise_schedule.betas[t]
        alpha_t = self.state_noise_schedule.alphas[t]
        alpha_bar_t = self.state_noise_schedule.alphas_cumprod[t]
        alpha_bar_t_prev = self.state_noise_schedule.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        # Compute coefficients for posterior mean
        beta_tilde = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
        mu_coeff1 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        mu_coeff2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        
        # Compute posterior mean
        mu = mu_coeff1 * x_0_pred + mu_coeff2 * x_t
        
        # Add noise (except for t=0)
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_tilde)
        
        return mu + sigma * noise
    
    def _split_trajectory(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split trajectory into states and actions.
        
        Args:
            trajectory: Combined trajectory [batch_size, horizon, state_dim + action_dim]
            
        Returns:
            Tuple of (states, actions)
        """
        states = trajectory[:, :, :self.state_dim]
        actions = trajectory[:, :, self.state_dim:self.state_dim + self.action_dim]
        
        # Adjust dimensions if needed
        if states.shape[1] > self.model.future_length_states:
            states = states[:, :self.model.future_length_states]
        if actions.shape[1] > self.model.future_length_actions:
            actions = actions[:, :self.model.future_length_actions]
        
        return states, actions
    
    def _combine_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine states and actions into trajectory.
        
        Args:
            states: State sequence [batch_size, horizon_s, state_dim]
            actions: Action sequence [batch_size, horizon_a, action_dim]
            
        Returns:
            Combined trajectory
        """
        # Use the shorter horizon
        horizon = min(states.shape[1], actions.shape[1])
        
        # Truncate to same length
        states = states[:, :horizon]
        actions = actions[:, :horizon]
        
        # Concatenate along feature dimension
        trajectory = torch.cat([states, actions], dim=-1)
        
        return trajectory
    
    def extract_states_from_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Extract states from combined trajectory."""
        return trajectory[:, :, :self.state_dim]
    
    def extract_actions_from_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Extract actions from combined trajectory."""
        return trajectory[:, :, self.state_dim:self.state_dim + self.action_dim]