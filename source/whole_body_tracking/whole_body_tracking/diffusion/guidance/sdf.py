"""Signed Distance Field (SDF) for obstacle representation."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class Obstacle:
    """Base class for obstacles."""
    position: torch.Tensor  # Center position [3]
    
    def distance(self, points: torch.Tensor) -> torch.Tensor:
        """Compute signed distance to points."""
        raise NotImplementedError


@dataclass
class SphereObstacle(Obstacle):
    """Spherical obstacle."""
    radius: float
    
    def distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to sphere.
        
        Args:
            points: Query points [N, 3]
            
        Returns:
            Signed distances [N]
        """
        # Distance to center minus radius
        center_dist = torch.norm(points - self.position.unsqueeze(0), dim=-1)
        return center_dist - self.radius


@dataclass  
class BoxObstacle(Obstacle):
    """Axis-aligned box obstacle."""
    half_extents: torch.Tensor  # Half-widths [3]
    
    def distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to box.
        
        Args:
            points: Query points [N, 3]
            
        Returns:
            Signed distances [N]
        """
        # Transform to box-local coordinates
        local_points = points - self.position.unsqueeze(0)
        
        # Distance to box surface
        d = torch.abs(local_points) - self.half_extents.unsqueeze(0)
        
        # Outside distance
        outside = torch.norm(torch.clamp(d, min=0.0), dim=-1)
        
        # Inside distance (negative)
        inside = torch.min(d, dim=-1)[0].clamp(max=0.0)
        
        return outside + inside


@dataclass
class CylinderObstacle(Obstacle):
    """Vertical cylinder obstacle."""
    radius: float
    height: float
    
    def distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to cylinder.
        
        Args:
            points: Query points [N, 3]
            
        Returns:
            Signed distances [N]
        """
        # Transform to cylinder-local coordinates
        local_points = points - self.position.unsqueeze(0)
        
        # Distance in XY plane
        xy_dist = torch.norm(local_points[:, :2], dim=-1) - self.radius
        
        # Distance in Z
        z_dist = torch.abs(local_points[:, 2]) - self.height / 2
        
        # Combined distance (approximation)
        outside = torch.norm(torch.stack([
            torch.clamp(xy_dist, min=0.0),
            torch.clamp(z_dist, min=0.0)
        ], dim=-1), dim=-1)
        
        inside = torch.max(xy_dist, z_dist).clamp(max=0.0)
        
        return outside + inside


class SignedDistanceField(nn.Module):
    """
    Signed Distance Field for multiple obstacles.
    
    Provides differentiable distance queries for obstacle avoidance.
    """
    
    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        resolution: Optional[int] = None,
    ):
        """
        Initialize SDF.
        
        Args:
            obstacles: List of obstacles in the environment
            bounds: (min_corner, max_corner) for precomputed grid
            resolution: Grid resolution for acceleration structure
        """
        super().__init__()
        
        self.obstacles = obstacles or []
        self.bounds = bounds
        self.resolution = resolution
        
        # Optionally precompute grid for faster queries
        self.grid = None
        if bounds is not None and resolution is not None:
            self._precompute_grid()
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the field."""
        self.obstacles.append(obstacle)
        self.grid = None  # Invalidate grid
    
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []
        self.grid = None
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query SDF at given points.
        
        Args:
            points: Query points [N, 3] or [batch_size, N, 3]
            
        Returns:
            Signed distances [N] or [batch_size, N]
        """
        original_shape = points.shape
        if points.dim() == 3:
            batch_size, num_points, _ = points.shape
            points = points.reshape(-1, 3)
        else:
            batch_size = None
        
        if self.grid is not None:
            # Use precomputed grid with interpolation
            distances = self._query_grid(points)
        else:
            # Direct computation
            distances = self._compute_distances(points)
        
        if batch_size is not None:
            distances = distances.reshape(batch_size, num_points)
        
        return distances
    
    def _compute_distances(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute minimum distance to all obstacles.
        
        Args:
            points: Query points [N, 3]
            
        Returns:
            Minimum signed distances [N]
        """
        if len(self.obstacles) == 0:
            # No obstacles - return large positive distance
            return torch.full((points.shape[0],), 1000.0, device=points.device)
        
        # Compute distance to each obstacle
        distances = []
        for obstacle in self.obstacles:
            dist = obstacle.distance(points)
            distances.append(dist)
        
        # Return minimum distance (closest obstacle)
        distances = torch.stack(distances, dim=-1)
        min_distances, _ = torch.min(distances, dim=-1)
        
        return min_distances
    
    def _precompute_grid(self):
        """Precompute SDF on a regular grid for acceleration."""
        if self.bounds is None or self.resolution is None:
            return
        
        min_corner, max_corner = self.bounds
        
        # Create grid points
        x = torch.linspace(min_corner[0], max_corner[0], self.resolution)
        y = torch.linspace(min_corner[1], max_corner[1], self.resolution)
        z = torch.linspace(min_corner[2], max_corner[2], self.resolution)
        
        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        # Compute SDF values
        with torch.no_grad():
            sdf_values = self._compute_distances(grid_points)
        
        # Reshape to grid
        self.grid = sdf_values.reshape(self.resolution, self.resolution, self.resolution)
        self.grid_min = min_corner
        self.grid_max = max_corner
    
    def _query_grid(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query precomputed grid with trilinear interpolation.
        
        Args:
            points: Query points [N, 3]
            
        Returns:
            Interpolated distances [N]
        """
        # Normalize points to [0, 1]
        normalized = (points - self.grid_min) / (self.grid_max - self.grid_min)
        
        # Convert to grid coordinates
        grid_coords = normalized * (self.resolution - 1)
        
        # Trilinear interpolation
        # This is a simplified version - full implementation would use
        # torch.nn.functional.grid_sample for efficiency
        
        # Get integer coordinates
        x0 = torch.floor(grid_coords[:, 0]).long().clamp(0, self.resolution - 2)
        y0 = torch.floor(grid_coords[:, 1]).long().clamp(0, self.resolution - 2)
        z0 = torch.floor(grid_coords[:, 2]).long().clamp(0, self.resolution - 2)
        
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        # Get fractional parts
        fx = grid_coords[:, 0] - x0.float()
        fy = grid_coords[:, 1] - y0.float()
        fz = grid_coords[:, 2] - z0.float()
        
        # Get corner values
        v000 = self.grid[x0, y0, z0]
        v001 = self.grid[x0, y0, z1]
        v010 = self.grid[x0, y1, z0]
        v011 = self.grid[x0, y1, z1]
        v100 = self.grid[x1, y0, z0]
        v101 = self.grid[x1, y0, z1]
        v110 = self.grid[x1, y1, z0]
        v111 = self.grid[x1, y1, z1]
        
        # Trilinear interpolation
        v00 = v000 * (1 - fx) + v100 * fx
        v01 = v001 * (1 - fx) + v101 * fx
        v10 = v010 * (1 - fx) + v110 * fx
        v11 = v011 * (1 - fx) + v111 * fx
        
        v0 = v00 * (1 - fy) + v10 * fy
        v1 = v01 * (1 - fy) + v11 * fy
        
        v = v0 * (1 - fz) + v1 * fz
        
        return v
    
    @staticmethod
    def create_random_obstacles(
        num_obstacles: int = 5,
        bounds: Tuple[float, float, float] = (-5.0, 5.0, 0.0, 5.0, 0.0, 3.0),
        min_size: float = 0.2,
        max_size: float = 1.0,
        device: str = "cuda",
    ) -> List[Obstacle]:
        """
        Create random obstacles for testing.
        
        Args:
            num_obstacles: Number of obstacles to create
            bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
            min_size: Minimum obstacle size
            max_size: Maximum obstacle size
            device: Device for tensors
            
        Returns:
            List of random obstacles
        """
        obstacles = []
        
        for _ in range(num_obstacles):
            # Random position within bounds
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[2], bounds[3])
            z = np.random.uniform(bounds[4], bounds[5])
            position = torch.tensor([x, y, z], device=device)
            
            # Random obstacle type
            obstacle_type = np.random.choice(['sphere', 'box', 'cylinder'])
            
            if obstacle_type == 'sphere':
                radius = np.random.uniform(min_size, max_size)
                obstacle = SphereObstacle(position=position, radius=radius)
            
            elif obstacle_type == 'box':
                half_extents = torch.tensor([
                    np.random.uniform(min_size, max_size),
                    np.random.uniform(min_size, max_size),
                    np.random.uniform(min_size, max_size),
                ], device=device)
                obstacle = BoxObstacle(position=position, half_extents=half_extents)
            
            else:  # cylinder
                radius = np.random.uniform(min_size, max_size)
                height = np.random.uniform(min_size * 2, max_size * 2)
                obstacle = CylinderObstacle(
                    position=position,
                    radius=radius,
                    height=height,
                )
            
            obstacles.append(obstacle)
        
        return obstacles