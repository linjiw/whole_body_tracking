"""Model configuration for BeyondMimic Stage 2 diffusion model.

This module defines the configuration dataclasses for the diffusion model
architecture and hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class TransformerConfig:
    """Configuration for the transformer architecture."""
    
    hidden_dim: int = 512
    """Hidden dimension for transformer layers."""
    
    num_layers: int = 6
    """Number of transformer blocks."""
    
    num_heads: int = 8
    """Number of attention heads."""
    
    mlp_ratio: int = 4
    """MLP hidden dimension ratio (mlp_hidden = hidden_dim * mlp_ratio)."""
    
    dropout: float = 0.1
    """Dropout probability."""
    
    attention_dropout: float = 0.3
    """Attention dropout (higher for robustness as per paper)."""
    
    use_layer_norm: bool = True
    """Whether to use layer normalization."""
    
    activation: str = "gelu"
    """Activation function type."""


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion process."""
    
    num_timesteps: int = 100
    """Number of diffusion timesteps (T in DDPM)."""
    
    state_schedule_type: Literal["linear", "cosine", "quadratic"] = "cosine"
    """Noise schedule type for states."""
    
    action_schedule_type: Literal["linear", "cosine", "quadratic"] = "linear"
    """Noise schedule type for actions (typically less noise than states)."""
    
    state_beta_start: float = 1e-4
    """Starting beta value for state noise schedule."""
    
    state_beta_end: float = 0.02
    """Ending beta value for state noise schedule."""
    
    action_beta_start: float = 1e-4
    """Starting beta value for action noise schedule."""
    
    action_beta_end: float = 0.01
    """Ending beta value for action noise schedule (less than states)."""
    
    prediction_type: str = "epsilon"
    """What the model predicts: 'epsilon' (noise) or 'x0' (clean data)."""
    
    clip_sample: bool = True
    """Whether to clip predicted samples to [-1, 1]."""
    
    clip_sample_range: float = 1.0
    """Range for clipping samples."""


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory representation."""
    
    state_dim: int = 165
    """Dimension of state vectors (body-pos representation)."""
    
    action_dim: int = 69
    """Dimension of action vectors (joint positions)."""
    
    history_length: int = 4
    """Number of past timesteps in observation history (N)."""
    
    future_length_states: int = 32
    """Number of future states to predict (H_s)."""
    
    future_length_actions: int = 16
    """Number of future actions to predict (H_a)."""
    
    action_horizon_mask: int = 8
    """Only compute loss on first K actions (prevents long-horizon instability)."""
    
    num_bodies: int = 23
    """Number of bodies in the robot model (for body-pos representation)."""


@dataclass
class StateActionDiffusionConfig:
    """Complete configuration for the state-action diffusion model."""
    
    transformer: TransformerConfig = None
    """Transformer architecture configuration."""
    
    diffusion: DiffusionConfig = None
    """Diffusion process configuration."""
    
    trajectory: TrajectoryConfig = None
    """Trajectory representation configuration."""
    
    # Additional model settings
    use_ema: bool = True
    """Whether to use exponential moving average for inference."""
    
    ema_decay: float = 0.9999
    """EMA decay rate."""
    
    use_differentiated_attention: bool = True
    """Whether to use differentiated attention (critical for success)."""
    
    guidance_scale_range: tuple = (0.0, 2.0)
    """Range for classifier guidance scale during inference."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Initialize defaults if not provided
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if self.diffusion is None:
            self.diffusion = DiffusionConfig()
        if self.trajectory is None:
            self.trajectory = TrajectoryConfig()
            
        # Ensure action horizon mask doesn't exceed action length
        if self.trajectory.action_horizon_mask > self.trajectory.future_length_actions:
            self.trajectory.action_horizon_mask = self.trajectory.future_length_actions
        
        # Ensure hidden dim is divisible by num heads
        if self.transformer.hidden_dim % self.transformer.num_heads != 0:
            raise ValueError(
                f"Hidden dim ({self.transformer.hidden_dim}) must be divisible "
                f"by num heads ({self.transformer.num_heads})"
            )


# Pre-defined configurations for different model sizes
def get_small_config() -> StateActionDiffusionConfig:
    """Get configuration for a small model (fast training/testing)."""
    config = StateActionDiffusionConfig()
    config.transformer.hidden_dim = 256
    config.transformer.num_layers = 3
    config.transformer.num_heads = 4
    config.diffusion.num_timesteps = 50
    return config


def get_base_config() -> StateActionDiffusionConfig:
    """Get configuration for the base model (paper specification)."""
    return StateActionDiffusionConfig()


def get_large_config() -> StateActionDiffusionConfig:
    """Get configuration for a large model (better quality, slower)."""
    config = StateActionDiffusionConfig()
    config.transformer.hidden_dim = 768
    config.transformer.num_layers = 8
    config.transformer.num_heads = 12
    config.diffusion.num_timesteps = 200
    return config