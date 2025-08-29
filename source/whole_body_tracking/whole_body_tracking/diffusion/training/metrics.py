"""Metrics for evaluating diffusion model performance.

This module implements various metrics for training and validation including:
- Reconstruction metrics (MSE, MAE)
- Motion quality metrics (smoothness, physical plausibility)
- Diversity metrics (trajectory variance)
- Task-specific metrics (tracking accuracy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricResults:
    """Container for metric results."""
    
    # Loss metrics
    total_loss: float
    state_loss: float
    action_loss: float
    
    # Reconstruction metrics
    state_mse: float
    action_mse: float
    state_mae: float
    action_mae: float
    
    # Motion quality metrics
    trajectory_smoothness: float
    action_smoothness: float
    velocity_error: float
    acceleration_error: float
    
    # Physical plausibility
    joint_limit_violations: float
    contact_violations: float
    
    # Diversity metrics (optional)
    trajectory_variance: Optional[float] = None
    action_variance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DiffusionMetrics:
    """Compute metrics for diffusion model evaluation."""
    
    def __init__(
        self,
        state_dim: int = 165,
        action_dim: int = 69,
        num_bodies: int = 23,
        joint_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """Initialize metrics calculator.
        
        Args:
            state_dim: Dimension of state vectors
            action_dim: Dimension of action vectors
            num_bodies: Number of robot bodies
            joint_limits: Optional (lower, upper) joint limit tensors
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_bodies = num_bodies
        self.joint_limits = joint_limits
    
    def compute_reconstruction_metrics(
        self,
        pred_states: torch.Tensor,
        pred_actions: torch.Tensor,
        target_states: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute reconstruction metrics.
        
        Args:
            pred_states: Predicted states (B, H, state_dim)
            pred_actions: Predicted actions (B, H, action_dim)
            target_states: Target states
            target_actions: Target actions
            mask: Optional mask for valid timesteps
            
        Returns:
            Dictionary of reconstruction metrics
        """
        metrics = {}
        
        # MSE metrics
        state_mse = F.mse_loss(pred_states, target_states, reduction='none')
        action_mse = F.mse_loss(pred_actions, target_actions, reduction='none')
        
        if mask is not None:
            state_mse = (state_mse * mask.unsqueeze(-1)).sum() / mask.sum()
            action_mse = (action_mse * mask.unsqueeze(-1)).sum() / mask.sum()
        else:
            state_mse = state_mse.mean()
            action_mse = action_mse.mean()
        
        metrics['state_mse'] = state_mse.item()
        metrics['action_mse'] = action_mse.item()
        
        # MAE metrics
        state_mae = F.l1_loss(pred_states, target_states, reduction='none')
        action_mae = F.l1_loss(pred_actions, target_actions, reduction='none')
        
        if mask is not None:
            state_mae = (state_mae * mask.unsqueeze(-1)).sum() / mask.sum()
            action_mae = (action_mae * mask.unsqueeze(-1)).sum() / mask.sum()
        else:
            state_mae = state_mae.mean()
            action_mae = action_mae.mean()
        
        metrics['state_mae'] = state_mae.item()
        metrics['action_mae'] = action_mae.item()
        
        return metrics
    
    def compute_smoothness_metrics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute trajectory smoothness metrics.
        
        Measures how smooth the predicted trajectories are by computing
        the variance of velocities and accelerations.
        
        Args:
            states: State trajectory (B, H, state_dim)
            actions: Action trajectory (B, H, action_dim)
            
        Returns:
            Dictionary of smoothness metrics
        """
        metrics = {}
        
        # State smoothness (velocity variance)
        if states.shape[1] > 1:
            state_vel = states[:, 1:] - states[:, :-1]
            state_smoothness = state_vel.var(dim=1).mean()
            metrics['trajectory_smoothness'] = state_smoothness.item()
            
            # Acceleration smoothness
            if states.shape[1] > 2:
                state_acc = state_vel[:, 1:] - state_vel[:, :-1]
                acc_smoothness = state_acc.var(dim=1).mean()
                metrics['acceleration_smoothness'] = acc_smoothness.item()
        
        # Action smoothness
        if actions.shape[1] > 1:
            action_vel = actions[:, 1:] - actions[:, :-1]
            action_smoothness = action_vel.var(dim=1).mean()
            metrics['action_smoothness'] = action_smoothness.item()
        
        return metrics
    
    def compute_velocity_metrics(
        self,
        pred_states: torch.Tensor,
        target_states: torch.Tensor
    ) -> Dict[str, float]:
        """Compute velocity-based metrics.
        
        Args:
            pred_states: Predicted states (B, H, state_dim)
            target_states: Target states
            
        Returns:
            Dictionary of velocity metrics
        """
        metrics = {}
        
        if pred_states.shape[1] <= 1:
            return metrics
        
        # Extract velocities (assuming velocity is part of state)
        # This is a simplified version - actual implementation would
        # depend on the specific state representation
        
        # Compute finite differences as proxy for velocity
        pred_vel = pred_states[:, 1:] - pred_states[:, :-1]
        target_vel = target_states[:, 1:] - target_states[:, :-1]
        
        vel_error = F.mse_loss(pred_vel, target_vel)
        metrics['velocity_error'] = vel_error.item()
        
        # Acceleration error
        if pred_states.shape[1] > 2:
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            target_acc = target_vel[:, 1:] - target_vel[:, :-1]
            acc_error = F.mse_loss(pred_acc, target_acc)
            metrics['acceleration_error'] = acc_error.item()
        
        return metrics
    
    def compute_physical_plausibility(
        self,
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute physical plausibility metrics.
        
        Args:
            actions: Action trajectory (B, H, action_dim)
            
        Returns:
            Dictionary of plausibility metrics
        """
        metrics = {}
        
        # Joint limit violations
        if self.joint_limits is not None:
            lower_limits, upper_limits = self.joint_limits
            violations = ((actions < lower_limits) | (actions > upper_limits)).float()
            violation_rate = violations.mean()
            metrics['joint_limit_violations'] = violation_rate.item()
        
        # Contact violations (simplified - would need actual contact info)
        # This is a placeholder that could be expanded with actual physics checks
        metrics['contact_violations'] = 0.0
        
        return metrics
    
    def compute_diversity_metrics(
        self,
        trajectories: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute diversity metrics across multiple trajectories.
        
        Args:
            trajectories: List of trajectory tensors
            
        Returns:
            Dictionary of diversity metrics
        """
        if len(trajectories) < 2:
            return {}
        
        metrics = {}
        
        # Stack trajectories
        stacked = torch.stack(trajectories, dim=0)
        
        # Compute variance across trajectories
        traj_variance = stacked.var(dim=0).mean()
        metrics['trajectory_variance'] = traj_variance.item()
        
        # Compute pairwise distances (simplified)
        n_traj = len(trajectories)
        distances = []
        for i in range(n_traj):
            for j in range(i + 1, n_traj):
                dist = F.mse_loss(trajectories[i], trajectories[j])
                distances.append(dist.item())
        
        if distances:
            metrics['mean_pairwise_distance'] = np.mean(distances)
            metrics['std_pairwise_distance'] = np.std(distances)
        
        return metrics
    
    def compute_all_metrics(
        self,
        pred_states: torch.Tensor,
        pred_actions: torch.Tensor,
        target_states: torch.Tensor,
        target_actions: torch.Tensor,
        loss: float,
        state_loss: float,
        action_loss: float
    ) -> MetricResults:
        """Compute all metrics.
        
        Args:
            pred_states: Predicted states
            pred_actions: Predicted actions
            target_states: Target states
            target_actions: Target actions
            loss: Total loss value
            state_loss: State loss value
            action_loss: Action loss value
            
        Returns:
            MetricResults object containing all metrics
        """
        # Reconstruction metrics
        recon_metrics = self.compute_reconstruction_metrics(
            pred_states, pred_actions, target_states, target_actions
        )
        
        # Smoothness metrics
        smooth_metrics = self.compute_smoothness_metrics(
            pred_states, pred_actions
        )
        
        # Velocity metrics
        vel_metrics = self.compute_velocity_metrics(
            pred_states, target_states
        )
        
        # Physical plausibility
        phys_metrics = self.compute_physical_plausibility(pred_actions)
        
        # Combine all metrics
        return MetricResults(
            total_loss=loss,
            state_loss=state_loss,
            action_loss=action_loss,
            state_mse=recon_metrics.get('state_mse', 0.0),
            action_mse=recon_metrics.get('action_mse', 0.0),
            state_mae=recon_metrics.get('state_mae', 0.0),
            action_mae=recon_metrics.get('action_mae', 0.0),
            trajectory_smoothness=smooth_metrics.get('trajectory_smoothness', 0.0),
            action_smoothness=smooth_metrics.get('action_smoothness', 0.0),
            velocity_error=vel_metrics.get('velocity_error', 0.0),
            acceleration_error=vel_metrics.get('acceleration_error', 0.0),
            joint_limit_violations=phys_metrics.get('joint_limit_violations', 0.0),
            contact_violations=phys_metrics.get('contact_violations', 0.0)
        )


class MetricTracker:
    """Track metrics over time for logging and early stopping."""
    
    def __init__(self, window_size: int = 100):
        """Initialize metric tracker.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.metrics_history = {}
        self.best_metrics = {}
        self.step = 0
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            self.metrics_history[key].append(value)
            
            # Keep only window_size most recent values
            if len(self.metrics_history[key]) > self.window_size:
                self.metrics_history[key].pop(0)
            
            # Track best value (assuming lower is better)
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
        
        self.step += 1
    
    def get_moving_average(self, metric_name: str) -> float:
        """Get moving average of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Moving average value
        """
        if metric_name not in self.metrics_history:
            return 0.0
        
        history = self.metrics_history[metric_name]
        if not history:
            return 0.0
        
        return sum(history) / len(history)
    
    def get_best(self, metric_name: str) -> float:
        """Get best value of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Best value seen so far
        """
        return self.best_metrics.get(metric_name, float('inf'))
    
    def is_improving(
        self,
        metric_name: str,
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> bool:
        """Check if metric is improving.
        
        Args:
            metric_name: Name of the metric
            patience: Number of steps to look back
            min_delta: Minimum change to consider as improvement
            
        Returns:
            True if metric is improving
        """
        if metric_name not in self.metrics_history:
            return True
        
        history = self.metrics_history[metric_name]
        if len(history) < patience:
            return True
        
        recent = history[-patience:]
        old_avg = sum(recent[:patience//2]) / (patience//2)
        new_avg = sum(recent[patience//2:]) / (patience//2)
        
        return old_avg - new_avg > min_delta