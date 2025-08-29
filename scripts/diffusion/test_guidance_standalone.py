#!/usr/bin/env python3
"""Standalone test script for guidance implementation."""

import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import logging

# Add diffusion modules directly to path
project_root = Path(__file__).resolve().parents[2]
diffusion_path = project_root / "source/whole_body_tracking/whole_body_tracking/diffusion"
sys.path.insert(0, str(diffusion_path.parent))

# Now import directly
from diffusion.models.diffusion_model import StateActionDiffusionModel
from diffusion.guidance.classifier_guidance import ClassifierGuidance, GuidanceConfig
from diffusion.guidance.cost_functions import JoystickCost, WaypointCost, ObstacleAvoidanceCost
from diffusion.guidance.sdf import SignedDistanceField, SphereObstacle, BoxObstacle
from diffusion.guidance.rolling_inference import RollingGuidanceInference, RollingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_model():
    """Create a mock diffusion model for testing."""
    model = StateActionDiffusionModel(
        state_dim=48,
        action_dim=19,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        history_length=4,
        future_length_states=16,
        future_length_actions=16,
    )
    
    # Initialize with random weights
    for param in model.parameters():
        param.data.normal_(0, 0.02)
    
    return model


def test_all_components():
    """Test all guidance components."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Testing on device: {device}")
    
    # Create model
    logger.info("Creating diffusion model...")
    model = create_mock_model().to(device)
    
    # Create guidance
    logger.info("Creating classifier guidance...")
    guidance_config = GuidanceConfig(
        guidance_scale=1.0,
        gradient_clipping=10.0,
    )
    guidance = ClassifierGuidance(model, guidance_config, device)
    
    # Test 1: Joystick control
    logger.info("\n1. Testing joystick control...")
    goal_velocity = torch.tensor([1.0, 0.0], device=device)
    joystick_cost = JoystickCost(goal_velocity, velocity_weight=1.0)
    
    obs_history = torch.randn(1, 4, 48 + 19, device=device)
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=joystick_cost,
        num_samples=1,
    )
    logger.info(f"   Joystick cost: {results['costs'].item():.4f}")
    
    # Test 2: Waypoint navigation
    logger.info("\n2. Testing waypoint navigation...")
    goal_position = torch.tensor([2.0, 1.0, 0.0], device=device)
    waypoint_cost = WaypointCost(goal_position)
    
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=waypoint_cost,
        num_samples=1,
    )
    logger.info(f"   Waypoint cost: {results['costs'].item():.4f}")
    
    # Test 3: Obstacle avoidance
    logger.info("\n3. Testing obstacle avoidance...")
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
    sdf = SignedDistanceField(obstacles)
    obstacle_cost = ObstacleAvoidanceCost(sdf_function=sdf)
    
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=obstacle_cost,
        num_samples=1,
    )
    logger.info(f"   Obstacle cost: {results['costs'].item():.4f}")
    
    # Test 4: Rolling inference
    logger.info("\n4. Testing rolling inference...")
    rolling_config = RollingConfig(
        horizon=16,
        replan_frequency=2,
        warm_start_steps=8,
    )
    rolling = RollingGuidanceInference(
        model=model,
        guidance=guidance,
        config=rolling_config,
        device=device,
    )
    
    actions = []
    for step in range(5):
        result = rolling.step(
            observation_history=obs_history,
            cost_function=joystick_cost,
        )
        actions.append(result["action"].cpu())
        logger.info(f"   Step {step}: Cost={result['cost']:.4f}, Replanned={result['info']['replanned']}")
    
    # Test 5: Composed guidance
    logger.info("\n5. Testing composed guidance...")
    composed_cost = guidance.compose_cost_functions({
        "waypoint": (waypoint_cost, 1.0),
        "obstacle": (obstacle_cost, 0.5),
    })
    
    results = guidance.guided_sampling(
        observation_history=obs_history,
        cost_function=composed_cost,
        num_samples=1,
    )
    logger.info(f"   Composed cost: {results['costs'].item():.4f}")
    
    logger.info("\n" + "="*50)
    logger.info("All tests passed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    test_all_components()