"""Task-specific cost functions for guidance."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class CostFunction(ABC):
    """Base class for differentiable cost functions."""
    
    def __init__(self, model_adapter=None):
        """Initialize with optional model adapter for dimension info."""
        self.model_adapter = model_adapter
        self.state_dim = model_adapter.state_dim if model_adapter else 48
        self.action_dim = model_adapter.action_dim if model_adapter else 19
    
    @abstractmethod
    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute cost for a trajectory.
        
        Args:
            trajectory: Trajectory tensor [batch_size, horizon, state_dim + action_dim]
            
        Returns:
            Cost tensor [batch_size] or scalar
        """
        pass
    
    def extract_states(
        self, 
        trajectory: torch.Tensor,
        state_dim: int = None,
    ) -> torch.Tensor:
        """
        Extract state sequence from trajectory.
        
        Args:
            trajectory: Full trajectory [batch_size, horizon, state_dim + action_dim]
            state_dim: Dimension of state representation (uses self.state_dim if None)
            
        Returns:
            States [batch_size, horizon, state_dim]
        """
        if state_dim is None:
            state_dim = self.state_dim
        
        # If we have a model adapter, use its extraction method
        if self.model_adapter and hasattr(self.model_adapter, 'extract_states_from_trajectory'):
            return self.model_adapter.extract_states_from_trajectory(trajectory)
        
        # Otherwise use default extraction
        return trajectory[:, :, :state_dim]
    
    def extract_actions(
        self,
        trajectory: torch.Tensor,
        state_dim: int = None,
        action_dim: int = None,
    ) -> torch.Tensor:
        """
        Extract action sequence from trajectory.
        
        Args:
            trajectory: Full trajectory [batch_size, horizon, state_dim + action_dim]
            state_dim: Dimension of state representation (uses self.state_dim if None)
            action_dim: Dimension of action space (uses self.action_dim if None)
            
        Returns:
            Actions [batch_size, horizon, action_dim]
        """
        if state_dim is None:
            state_dim = self.state_dim
        if action_dim is None:
            action_dim = self.action_dim
        
        # If we have a model adapter, use its extraction method
        if self.model_adapter and hasattr(self.model_adapter, 'extract_actions_from_trajectory'):
            return self.model_adapter.extract_actions_from_trajectory(trajectory)
        
        # Otherwise use default extraction
        return trajectory[:, :, state_dim:state_dim + action_dim]


class JoystickCost(CostFunction):
    """
    Cost function for joystick velocity control.
    
    Penalizes deviation from desired root velocity:
    G_c(τ) = ||v_xy - v_goal||²
    """
    
    def __init__(
        self,
        goal_velocity: torch.Tensor,
        velocity_weight: float = 1.0,
        acceleration_penalty: float = 0.1,
        model_adapter=None,
    ):
        """
        Initialize joystick cost.
        
        Args:
            goal_velocity: Target velocity [2] or [batch_size, 2] for (vx, vy)
            velocity_weight: Weight for velocity tracking
            acceleration_penalty: Penalty for acceleration changes
        """
        super().__init__(model_adapter)
        self.goal_velocity = goal_velocity
        self.velocity_weight = velocity_weight
        self.acceleration_penalty = acceleration_penalty
    
    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute joystick control cost.
        
        Args:
            trajectory: Trajectory [batch_size, horizon, state_dim + action_dim]
            
        Returns:
            Cost [batch_size]
        """
        # Extract states
        states = self.extract_states(trajectory)
        batch_size = states.shape[0]
        
        # Extract root velocities (assuming indices 3:6 are root velocities)
        root_velocities = states[:, :, 3:6]  # [batch_size, horizon, 3]
        xy_velocities = root_velocities[:, :, :2]  # [batch_size, horizon, 2]
        
        # Expand goal if needed
        if self.goal_velocity.dim() == 1:
            goal = self.goal_velocity.unsqueeze(0).unsqueeze(0)
            goal = goal.expand(batch_size, xy_velocities.shape[1], -1)
        else:
            goal = self.goal_velocity.unsqueeze(1)
            goal = goal.expand(-1, xy_velocities.shape[1], -1)
        
        # Velocity tracking cost
        velocity_error = (xy_velocities - goal).pow(2).sum(dim=-1)  # [batch_size, horizon]
        velocity_cost = velocity_error.mean(dim=1) * self.velocity_weight
        
        # Acceleration penalty (smoothness)
        if self.acceleration_penalty > 0:
            accelerations = xy_velocities[:, 1:] - xy_velocities[:, :-1]
            accel_cost = accelerations.pow(2).sum(dim=-1).mean(dim=1) * self.acceleration_penalty
            total_cost = velocity_cost + accel_cost
        else:
            total_cost = velocity_cost
        
        return total_cost


class WaypointCost(CostFunction):
    """
    Cost function for waypoint navigation.
    
    From BeyondMimic Eq. 8:
    G_c(τ) = w_pos * ||p - p_goal||² + w_vel * ||v||²
    
    where weights change based on distance to goal.
    """
    
    def __init__(
        self,
        goal_position: torch.Tensor,
        distance_threshold: float = 0.5,
        position_weight_far: float = 1.0,
        position_weight_near: float = 10.0,
        velocity_weight_far: float = 0.1,
        velocity_weight_near: float = 1.0,
        model_adapter=None,
    ):
        """
        Initialize waypoint cost.
        
        Args:
            goal_position: Target position [3] or [batch_size, 3]
            distance_threshold: Distance to switch weight schedule
            position_weight_far: Position weight when far from goal
            position_weight_near: Position weight when near goal
            velocity_weight_far: Velocity weight when far (encourage movement)
            velocity_weight_near: Velocity weight when near (encourage stopping)
        """
        super().__init__(model_adapter)
        self.goal_position = goal_position
        self.distance_threshold = distance_threshold
        self.position_weight_far = position_weight_far
        self.position_weight_near = position_weight_near
        self.velocity_weight_far = velocity_weight_far
        self.velocity_weight_near = velocity_weight_near
    
    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute waypoint navigation cost.
        
        Args:
            trajectory: Trajectory [batch_size, horizon, state_dim + action_dim]
            
        Returns:
            Cost [batch_size]
        """
        # Extract states
        states = self.extract_states(trajectory)
        batch_size = states.shape[0]
        
        # Extract root position and velocity (assuming indices 0:3 and 3:6)
        root_positions = states[:, :, 0:3]  # [batch_size, horizon, 3]
        root_velocities = states[:, :, 3:6]  # [batch_size, horizon, 3]
        
        # Only consider XY position for navigation
        xy_positions = root_positions[:, :, :2]
        xy_velocities = root_velocities[:, :, :2]
        
        # Expand goal if needed
        if self.goal_position.dim() == 1:
            goal = self.goal_position[:2].unsqueeze(0).unsqueeze(0)
            goal = goal.expand(batch_size, xy_positions.shape[1], -1)
        else:
            goal = self.goal_position[:, :2].unsqueeze(1)
            goal = goal.expand(-1, xy_positions.shape[1], -1)
        
        # Compute distance to goal
        position_error = xy_positions - goal  # [batch_size, horizon, 2]
        distances = torch.norm(position_error, dim=-1)  # [batch_size, horizon]
        
        # Adaptive weights based on distance
        # Use smooth transition instead of hard threshold
        alpha = torch.sigmoid((self.distance_threshold - distances) * 10)  # Smooth transition
        
        position_weight = (
            alpha * self.position_weight_near + 
            (1 - alpha) * self.position_weight_far
        )
        
        velocity_weight = (
            alpha * self.velocity_weight_near +
            (1 - alpha) * self.velocity_weight_far
        )
        
        # Position cost
        position_cost = (position_error.pow(2).sum(dim=-1) * position_weight).mean(dim=1)
        
        # Velocity cost (should slow down near goal)
        velocity_cost = (xy_velocities.pow(2).sum(dim=-1) * velocity_weight).mean(dim=1)
        
        total_cost = position_cost + velocity_cost
        
        return total_cost


class ObstacleAvoidanceCost(CostFunction):
    """
    Cost function for obstacle avoidance using SDF.
    
    From BeyondMimic Eq. 10:
    G_c(τ) = Σ_i B(sdf(p_i), δ)
    
    where B is a relaxed barrier function.
    """
    
    def __init__(
        self,
        sdf_function: Optional[nn.Module] = None,
        safety_margin: float = 0.1,
        barrier_weight: float = 10.0,
        barrier_delta: float = 0.05,
        model_adapter=None,
    ):
        """
        Initialize obstacle avoidance cost.
        
        Args:
            sdf_function: Signed distance field module (callable)
            safety_margin: Minimum distance to maintain from obstacles
            barrier_weight: Weight for barrier function
            barrier_delta: Relaxation parameter for barrier function
        """
        super().__init__(model_adapter)
        self.sdf_function = sdf_function
        self.safety_margin = safety_margin
        self.barrier_weight = barrier_weight
        self.barrier_delta = barrier_delta
    
    def relaxed_barrier(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute relaxed barrier function B(x, δ).
        
        Smooth approximation of hard barrier that's differentiable.
        
        Args:
            distance: Distance to obstacle
            
        Returns:
            Barrier cost
        """
        # Relaxed log barrier: -log(x + δ) for x > 0
        # Quadratic extension for x <= 0
        safe_dist = distance - self.safety_margin
        
        # Use smooth approximation
        barrier = torch.where(
            safe_dist > self.barrier_delta,
            -torch.log(safe_dist),  # Log barrier when far
            0.5 * ((safe_dist - 2 * self.barrier_delta) / self.barrier_delta).pow(2),  # Quadratic when close
        )
        
        return barrier * self.barrier_weight
    
    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute obstacle avoidance cost.
        
        Args:
            trajectory: Trajectory [batch_size, horizon, state_dim + action_dim]
            
        Returns:
            Cost [batch_size]
        """
        if self.sdf_function is None:
            # Return zero cost if no SDF provided
            return torch.zeros(trajectory.shape[0], device=trajectory.device)
        
        # Extract states
        states = self.extract_states(trajectory)
        
        # Extract body positions (assuming they start at index 9)
        # Format: [root_pos(3), root_vel(3), root_rot(3), body_positions(...)]
        body_positions = states[:, :, 9:]  # [batch_size, horizon, num_bodies * 3]
        
        # Reshape to [batch_size * horizon * num_bodies, 3]
        batch_size, horizon, body_dim = body_positions.shape
        num_bodies = body_dim // 3
        body_positions = body_positions.reshape(batch_size, horizon, num_bodies, 3)
        
        # Compute SDF distances for all body parts
        body_positions_flat = body_positions.reshape(-1, 3)
        sdf_distances = self.sdf_function(body_positions_flat)  # [batch_size * horizon * num_bodies]
        sdf_distances = sdf_distances.reshape(batch_size, horizon, num_bodies)
        
        # Apply barrier function
        barrier_costs = self.relaxed_barrier(sdf_distances)
        
        # Sum over bodies and average over time
        total_cost = barrier_costs.sum(dim=-1).mean(dim=1)
        
        return total_cost


class CompositeCost(CostFunction):
    """
    Compose multiple cost functions with weights.
    
    Useful for multi-objective tasks like "navigate to waypoint while avoiding obstacles".
    """
    
    def __init__(self, cost_functions: dict, model_adapter=None):
        """
        Initialize composite cost.
        
        Args:
            cost_functions: Dictionary of {name: (cost_fn, weight)}
        """
        super().__init__(model_adapter)
        self.cost_functions = cost_functions
    
    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted sum of costs.
        
        Args:
            trajectory: Trajectory [batch_size, horizon, state_dim + action_dim]
            
        Returns:
            Total cost [batch_size]
        """
        total_cost = 0.0
        for name, (cost_fn, weight) in self.cost_functions.items():
            cost = cost_fn(trajectory)
            total_cost = total_cost + weight * cost
        
        return total_cost