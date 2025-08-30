"""Embedding layers for diffusion model.

This module implements the embedding layers for states, actions, and time encoding
following the BeyondMimic paper specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time encoding.
    
    Following standard transformer positional encoding but adapted for
    continuous timesteps in diffusion models.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        """Initialize sinusoidal embeddings.
        
        Args:
            dim: Embedding dimension (must be even)
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even for sinusoidal encoding"
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: (batch_size,) tensor of timesteps
            
        Returns:
            (batch_size, dim) tensor of embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # Create frequency bands
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        
        # Apply sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class StateEmbedding(nn.Module):
    """Embedding layer for state representations.
    
    Maps state vectors to higher-dimensional embeddings suitable for
    transformer processing. Includes normalization for stability.
    """
    
    def __init__(
        self,
        state_dim: int = 165,  # From paper
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize state embedding.
        
        Args:
            state_dim: Dimension of state vector
            hidden_dim: Output embedding dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Build MLP
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(state_dim)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Embed state vectors.
        
        Args:
            states: (..., state_dim) tensor of states
            
        Returns:
            (..., hidden_dim) tensor of embeddings
        """
        # Normalize input
        states = self.input_norm(states)
        
        # Apply MLP
        return self.mlp(states)


class ActionEmbedding(nn.Module):
    """Embedding layer for action representations.
    
    Maps action vectors to embeddings. Actions are typically lower-dimensional
    than states and represent joint positions/velocities.
    """
    
    def __init__(
        self,
        action_dim: int = 69,  # From paper
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize action embedding.
        
        Args:
            action_dim: Dimension of action vector
            hidden_dim: Output embedding dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Build MLP
        layers = []
        in_dim = action_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(action_dim)
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed action vectors.
        
        Args:
            actions: (..., action_dim) tensor of actions
            
        Returns:
            (..., hidden_dim) tensor of embeddings
        """
        # Normalize input
        actions = self.input_norm(actions)
        
        # Apply MLP
        return self.mlp(actions)


class ObservationHistoryEmbedding(nn.Module):
    """Embedding for observation history O_t.
    
    Processes the interleaved sequence of past states and actions
    [s_{t-N}, a_{t-N}, ..., s_{t-1}, a_{t-1}, s_t].
    """
    
    def __init__(
        self,
        state_dim: int = 165,
        action_dim: int = 69,
        hidden_dim: int = 512,
        history_length: int = 4,  # N from paper
        dropout: float = 0.1
    ):
        """Initialize observation history embedding.
        
        Args:
            state_dim: Dimension of state vectors
            action_dim: Dimension of action vectors
            hidden_dim: Output embedding dimension
            history_length: Number of past timesteps (N)
            dropout: Dropout probability
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        
        # Separate embeddings for states and actions
        self.state_embed = StateEmbedding(state_dim, hidden_dim, dropout=dropout)
        self.action_embed = ActionEmbedding(action_dim, hidden_dim, dropout=dropout)
        
        # Learnable position embeddings for history
        # 2N+1 positions: N state-action pairs + current state
        self.position_embed = nn.Parameter(
            torch.randn(2 * history_length + 1, hidden_dim) * 0.02
        )
        
        # Type embeddings to distinguish states from actions
        self.state_type_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.action_type_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor
    ) -> torch.Tensor:
        """Embed observation history.
        
        Args:
            history_states: (batch_size, N+1, state_dim) past states
            history_actions: (batch_size, N, action_dim) past actions
            
        Returns:
            (batch_size, 2N+1, hidden_dim) embedded history sequence
        """
        batch_size = history_states.shape[0]
        
        # Embed states and actions
        state_embeds = self.state_embed(history_states)  # (B, N+1, hidden)
        action_embeds = self.action_embed(history_actions)  # (B, N, hidden)
        
        # Interleave states and actions: [s0, a0, s1, a1, ..., sN]
        embeddings = []
        pos_idx = 0
        
        for i in range(self.history_length):
            # Add state
            state_emb = state_embeds[:, i] + self.state_type_embed + self.position_embed[pos_idx]
            embeddings.append(state_emb)
            pos_idx += 1
            
            # Add action
            action_emb = action_embeds[:, i] + self.action_type_embed + self.position_embed[pos_idx]
            embeddings.append(action_emb)
            pos_idx += 1
        
        # Add final state
        final_state_emb = state_embeds[:, -1] + self.state_type_embed + self.position_embed[pos_idx]
        embeddings.append(final_state_emb)
        
        # Stack embeddings
        embeddings = torch.stack(embeddings, dim=1)  # (B, 2N+1, hidden)
        
        # Apply normalization and dropout
        embeddings = self.output_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class FutureTrajectoryEmbedding(nn.Module):
    """Embedding for future trajectory τ_t.
    
    Handles the interleaved sequence of future states and actions
    with proper position and type embeddings.
    """
    
    def __init__(
        self,
        state_dim: int = 165,
        action_dim: int = 69,
        hidden_dim: int = 512,
        future_length_states: int = 32,  # H_s
        future_length_actions: int = 16,  # H_a
        dropout: float = 0.1
    ):
        """Initialize future trajectory embedding.
        
        Args:
            state_dim: Dimension of state vectors
            action_dim: Dimension of action vectors
            hidden_dim: Output embedding dimension
            future_length_states: Number of future states (H_s)
            future_length_actions: Number of future actions (H_a)
            dropout: Dropout probability
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.future_length_states = future_length_states
        self.future_length_actions = future_length_actions
        
        # Separate embeddings for states and actions
        self.state_embed = StateEmbedding(state_dim, hidden_dim, dropout=dropout)
        self.action_embed = ActionEmbedding(action_dim, hidden_dim, dropout=dropout)
        
        # Maximum sequence length for position embeddings
        max_seq_len = future_length_states + future_length_actions
        self.position_embed = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.02
        )
        
        # Type embeddings
        self.state_type_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.action_type_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        future_states: torch.Tensor,
        future_actions: torch.Tensor,
        noisy_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed future trajectory.
        
        Returns separate embeddings for states and actions to support
        differentiated attention mechanisms.
        
        Args:
            future_states: (batch_size, H_s, state_dim) future states
            future_actions: (batch_size, H_a, action_dim) future actions
            noisy_mask: Optional (batch_size, seq_len) mask for noisy tokens
            
        Returns:
            Tuple of:
                - (batch_size, H_s, hidden_dim) state embeddings
                - (batch_size, H_a, hidden_dim) action embeddings
        """
        batch_size = future_states.shape[0]
        
        # Embed states and actions
        state_embeds = self.state_embed(future_states)  # (B, H_s, hidden)
        action_embeds = self.action_embed(future_actions)  # (B, H_a, hidden)
        
        # Add position embeddings
        # For interleaved sequences, we need to handle positions carefully
        # States and actions are interleaved but have different lengths
        # Ensure positions are within bounds of position_embed
        state_positions = torch.arange(
            0, min(self.future_length_states, self.position_embed.shape[0]),
            device=state_embeds.device
        )
        action_positions = torch.arange(
            0, min(self.future_length_actions, self.position_embed.shape[0]),
            device=action_embeds.device
        )
        
        # Add position and type embeddings
        # Only add position embeddings for the available positions
        state_embeds = state_embeds + self.state_type_embed
        if state_positions.numel() > 0:
            state_embeds[:, :len(state_positions)] = (
                state_embeds[:, :len(state_positions)] + 
                self.position_embed[state_positions].unsqueeze(0)
            )
        
        action_embeds = action_embeds + self.action_type_embed
        if action_positions.numel() > 0:
            action_embeds[:, :len(action_positions)] = (
                action_embeds[:, :len(action_positions)] + 
                self.position_embed[action_positions].unsqueeze(0)
            )
        
        # Apply normalization and dropout
        state_embeds = self.output_norm(state_embeds)
        state_embeds = self.dropout(state_embeds)
        
        action_embeds = self.output_norm(action_embeds)
        action_embeds = self.dropout(action_embeds)
        
        return state_embeds, action_embeds


class ClassifierGuidanceEmbedding(nn.Module):
    """Embedding for classifier guidance signals.
    
    Processes guidance features (e.g., target positions, orientations)
    for conditioning the diffusion model.
    """
    
    def __init__(
        self,
        guidance_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize guidance embedding.
        
        Args:
            guidance_dim: Dimension of guidance features
            hidden_dim: Output embedding dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()
        self.guidance_dim = guidance_dim
        self.hidden_dim = hidden_dim
        
        # Build MLP
        layers = []
        in_dim = guidance_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, guidance: torch.Tensor) -> torch.Tensor:
        """Embed guidance features.
        
        Args:
            guidance: (batch_size, guidance_dim) tensor of guidance features
            
        Returns:
            (batch_size, hidden_dim) tensor of embeddings
        """
        return self.mlp(guidance)