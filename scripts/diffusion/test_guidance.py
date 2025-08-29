#!/usr/bin/env python3
"""Test script for guidance implementation."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking"))

from whole_body_tracking.diffusion.models.diffusion_model import (
    StateActionDiffusionModel,
    DiffusionConfig,
)
from whole_body_tracking.diffusion.guidance import (
    ClassifierGuidance,
    GuidanceConfig,
    JoystickCost,
    WaypointCost,
    ObstacleAvoidanceCost,
    SignedDistanceField,
)
from whole_body_tracking.diffusion.guidance.sdf import (
    SphereObstacle,
    BoxObstacle,
)
from whole_body_tracking.diffusion.guidance.rolling_inference import (
    RollingGuidanceInference,
    RollingConfig,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_model():
    """Create a mock diffusion model for testing."""
    config = DiffusionConfig(
        state_dim=48,
        action_dim=19,
        history_len=4,
        horizon=16,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
    )
    
    model = StateActionDiffusionModel(config)
    
    # Initialize with random weights
    for param in model.parameters():
        param.data.normal_(0, 0.02)
    
    return model


def test_joystick_guidance():
    """Test joystick velocity control with guidance."""
    logger.info("Testing joystick guidance...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and guidance
    model = create_mock_model().to(device)
    guidance_config = GuidanceConfig(
        guidance_scale=1.0,
        gradient_clipping=10.0,
    )
    guidance = ClassifierGuidance(model, guidance_config, device)
    
    # Create joystick cost function
    goal_velocity = torch.tensor([1.0, 0.0], device=device)  # Move forward
    cost_fn = JoystickCost(goal_velocity, velocity_weight=1.0)
    
    # Create mock observation history
    obs_history = torch.randn(1, 4, 48 + 19, device=device)
    
    # Run guided sampling
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=cost_fn,
        num_samples=1,
        return_all_steps=True,
    )
    
    # Check results
    assert "trajectory" in results
    assert "actions" in results
    assert "costs" in results
    
    logger.info(f"Joystick guidance test passed. Final cost: {results['costs'].item():.4f}")
    
    return results


def test_waypoint_guidance():
    """Test waypoint navigation with guidance."""
    logger.info("Testing waypoint guidance...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and guidance
    model = create_mock_model().to(device)
    guidance = ClassifierGuidance(model, device=device)
    
    # Create waypoint cost function
    goal_position = torch.tensor([2.0, 1.0, 0.0], device=device)
    cost_fn = WaypointCost(
        goal_position,
        distance_threshold=0.5,
        position_weight_far=1.0,
        position_weight_near=10.0,
    )
    
    # Create mock observation history
    obs_history = torch.randn(1, 4, 48 + 19, device=device)
    
    # Run guided sampling
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=cost_fn,
        num_samples=1,
    )
    
    logger.info(f"Waypoint guidance test passed. Final cost: {results['costs'].item():.4f}")
    
    return results


def test_obstacle_avoidance():
    """Test obstacle avoidance with SDF."""
    logger.info("Testing obstacle avoidance...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create obstacles
    obstacles = [
        SphereObstacle(
            position=torch.tensor([1.0, 0.0, 0.5], device=device),
            radius=0.3,
        ),
        BoxObstacle(
            position=torch.tensor([-1.0, 1.0, 0.5], device=device),
            half_extents=torch.tensor([0.2, 0.3, 0.4], device=device),
        ),
    ]
    
    # Create SDF
    sdf = SignedDistanceField(obstacles)
    
    # Test SDF queries
    test_points = torch.tensor([
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],  # At sphere center
        [2.0, 0.0, 0.5],
    ], device=device)
    
    distances = sdf(test_points)
    logger.info(f"SDF distances: {distances}")
    
    # Create model and guidance
    model = create_mock_model().to(device)
    guidance = ClassifierGuidance(model, device=device)
    
    # Create obstacle avoidance cost
    cost_fn = ObstacleAvoidanceCost(
        sdf_function=sdf,
        safety_margin=0.1,
        barrier_weight=10.0,
    )
    
    # Create mock observation history
    obs_history = torch.randn(1, 4, 48 + 19, device=device)
    
    # Run guided sampling
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=cost_fn,
        num_samples=1,
    )
    
    logger.info(f"Obstacle avoidance test passed. Final cost: {results['costs'].item():.4f}")
    
    return results


def test_rolling_inference():
    """Test rolling inference with temporal consistency."""
    logger.info("Testing rolling inference...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and guidance
    model = create_mock_model().to(device)
    guidance = ClassifierGuidance(model, device=device)
    
    # Create rolling inference
    rolling_config = RollingConfig(
        horizon=16,
        replan_frequency=2,
        warm_start_steps=8,
        temporal_smoothing=0.9,
    )
    rolling = RollingGuidanceInference(
        model=model,
        guidance=guidance,
        config=rolling_config,
        device=device,
    )
    
    # Create cost function (joystick control)
    goal_velocity = torch.tensor([1.0, 0.5], device=device)
    cost_fn = JoystickCost(goal_velocity)
    
    # Simulate multiple steps
    actions = []
    costs = []
    
    for step in range(10):
        # Create mock observation history
        obs_history = torch.randn(1, 4, 48 + 19, device=device)
        
        # Get action from rolling inference
        result = rolling.step(
            observation_history=obs_history,
            cost_function=cost_fn,
        )
        
        actions.append(result["action"].cpu())
        costs.append(result["cost"])
        
        logger.info(f"Step {step}: Cost={result['cost']:.4f}, Replanned={result['info']['replanned']}")
    
    logger.info("Rolling inference test passed")
    
    return actions, costs


def test_composed_guidance():
    """Test composing multiple cost functions."""
    logger.info("Testing composed guidance...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and guidance
    model = create_mock_model().to(device)
    guidance = ClassifierGuidance(model, device=device)
    
    # Create multiple cost functions
    waypoint_cost = WaypointCost(
        goal_position=torch.tensor([2.0, 0.0, 0.0], device=device)
    )
    
    # Create obstacles
    obstacles = [
        SphereObstacle(
            position=torch.tensor([1.0, 0.0, 0.5], device=device),
            radius=0.3,
        ),
    ]
    sdf = SignedDistanceField(obstacles)
    obstacle_cost = ObstacleAvoidanceCost(sdf_function=sdf)
    
    # Compose cost functions
    composed_cost = guidance.compose_cost_functions({
        "waypoint": (waypoint_cost, 1.0),
        "obstacle": (obstacle_cost, 0.5),
    })
    
    # Create mock observation history
    obs_history = torch.randn(1, 4, 48 + 19, device=device)
    
    # Run guided sampling
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=composed_cost,
        num_samples=1,
    )
    
    logger.info(f"Composed guidance test passed. Final cost: {results['costs'].item():.4f}")
    
    return results


def visualize_trajectory(results):
    """Visualize a trajectory."""
    if "all_steps" in results:
        # Plot denoising progress
        all_steps = results["all_steps"].cpu()
        num_steps = all_steps.shape[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot trajectory evolution
        for i in range(0, num_steps, num_steps // 4):
            traj = all_steps[i, 0]  # First sample
            
            # Extract some dimensions for visualization
            axes[0, 0].plot(traj[:, 0].numpy(), alpha=0.5, label=f"Step {i}")
            axes[0, 1].plot(traj[:, 1].numpy(), alpha=0.5, label=f"Step {i}")
        
        axes[0, 0].set_title("Trajectory Dim 0")
        axes[0, 0].legend()
        axes[0, 1].set_title("Trajectory Dim 1")
        axes[0, 1].legend()
        
        # Plot final trajectory
        final_traj = results["trajectory"][0].cpu()
        axes[1, 0].plot(final_traj[:, 0].numpy())
        axes[1, 0].set_title("Final Trajectory Dim 0")
        
        axes[1, 1].plot(final_traj[:, 1].numpy())
        axes[1, 1].set_title("Final Trajectory Dim 1")
        
        plt.tight_layout()
        plt.savefig("guidance_trajectory.png")
        logger.info("Saved trajectory visualization to guidance_trajectory.png")


def main():
    """Run all tests."""
    logger.info("Starting guidance tests...")
    
    # Test individual components
    joystick_results = test_joystick_guidance()
    waypoint_results = test_waypoint_guidance()
    obstacle_results = test_obstacle_avoidance()
    
    # Test rolling inference
    actions, costs = test_rolling_inference()
    
    # Test composed guidance
    composed_results = test_composed_guidance()
    
    # Visualize results
    visualize_trajectory(joystick_results)
    
    logger.info("\n" + "="*50)
    logger.info("All guidance tests passed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()