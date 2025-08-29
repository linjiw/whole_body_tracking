#!/usr/bin/env python3
"""
Play script for guided diffusion policy.

This script loads a trained diffusion model and uses classifier guidance
to control the robot in real-time with various tasks:
- Joystick velocity control
- Waypoint navigation  
- Obstacle avoidance
- Composed multi-objective control
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import diffusion modules
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking"))

from whole_body_tracking.diffusion.models.diffusion_model import StateActionDiffusionModel
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
    CylinderObstacle,
)
from whole_body_tracking.diffusion.guidance.rolling_inference import (
    RollingGuidanceInference,
    RollingConfig,
)


def load_model(checkpoint_path: str, device: str = "cuda") -> StateActionDiffusionModel:
    """
    Load trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    
    # Create model
    model = StateActionDiffusionModel(
        state_dim=model_config.get("state_dim", 48),
        action_dim=model_config.get("action_dim", 19),
        hidden_dim=model_config.get("hidden_dim", 512),
        num_layers=model_config.get("num_layers", 6),
        num_heads=model_config.get("num_heads", 8),
        history_length=model_config.get("history_length", 4),
        future_length_states=model_config.get("future_length_states", 32),
        future_length_actions=model_config.get("future_length_actions", 16),
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def create_sdf_from_config(config: Dict[str, Any], device: str) -> SignedDistanceField:
    """
    Create SDF from configuration.
    
    Args:
        config: Obstacle configuration
        device: Device for tensors
        
    Returns:
        Configured SDF
    """
    obstacles = []
    
    for obs_config in config.get("obstacles", []):
        obs_type = obs_config["type"]
        position = torch.tensor(obs_config["position"], device=device)
        
        if obs_type == "sphere":
            obstacle = SphereObstacle(
                position=position,
                radius=obs_config["radius"]
            )
        elif obs_type == "box":
            obstacle = BoxObstacle(
                position=position,
                half_extents=torch.tensor(obs_config["half_extents"], device=device)
            )
        elif obs_type == "cylinder":
            obstacle = CylinderObstacle(
                position=position,
                radius=obs_config["radius"],
                height=obs_config["height"]
            )
        else:
            logger.warning(f"Unknown obstacle type: {obs_type}")
            continue
        
        obstacles.append(obstacle)
    
    # Create SDF with optional grid acceleration
    bounds = config.get("bounds")
    resolution = config.get("resolution")
    
    if bounds:
        bounds = (
            torch.tensor(bounds[0], device=device),
            torch.tensor(bounds[1], device=device)
        )
    
    sdf = SignedDistanceField(obstacles, bounds, resolution)
    
    return sdf


def create_cost_function(task: str, config: Dict[str, Any], sdf: Optional[SignedDistanceField], device: str):
    """
    Create cost function based on task type.
    
    Args:
        task: Task name
        config: Task configuration
        sdf: Optional SDF for obstacle avoidance
        device: Device for tensors
        
    Returns:
        Cost function
    """
    if task == "joystick":
        velocity = torch.tensor(config["velocity"], device=device)
        return JoystickCost(
            goal_velocity=velocity,
            velocity_weight=config.get("velocity_weight", 1.0),
            acceleration_penalty=config.get("acceleration_penalty", 0.1),
        )
    
    elif task == "waypoint":
        position = torch.tensor(config["position"], device=device)
        return WaypointCost(
            goal_position=position,
            distance_threshold=config.get("distance_threshold", 0.5),
            position_weight_far=config.get("position_weight_far", 1.0),
            position_weight_near=config.get("position_weight_near", 10.0),
            velocity_weight_far=config.get("velocity_weight_far", 0.1),
            velocity_weight_near=config.get("velocity_weight_near", 1.0),
        )
    
    elif task == "obstacle":
        if sdf is None:
            raise ValueError("SDF required for obstacle avoidance")
        return ObstacleAvoidanceCost(
            sdf_function=sdf,
            safety_margin=config.get("safety_margin", 0.1),
            barrier_weight=config.get("barrier_weight", 10.0),
            barrier_delta=config.get("barrier_delta", 0.05),
        )
    
    else:
        raise ValueError(f"Unknown task: {task}")


def run_guided_control(
    model: StateActionDiffusionModel,
    task_config: Dict[str, Any],
    num_steps: int = 100,
    device: str = "cuda",
):
    """
    Run guided control loop.
    
    Args:
        model: Trained diffusion model
        task_config: Task configuration
        num_steps: Number of control steps
        device: Device for computation
    """
    # Create guidance
    guidance_config = GuidanceConfig(
        guidance_scale=task_config.get("guidance_scale", 1.0),
        gradient_clipping=task_config.get("gradient_clipping", 10.0),
        warmup_steps=task_config.get("warmup_steps", 5),
        cooldown_steps=task_config.get("cooldown_steps", 3),
    )
    guidance = ClassifierGuidance(model, guidance_config, device)
    
    # Create rolling inference
    rolling_config = RollingConfig(
        horizon=task_config.get("horizon", 16),
        replan_frequency=task_config.get("replan_frequency", 1),
        warm_start_steps=task_config.get("warm_start_steps", 8),
        temporal_smoothing=task_config.get("temporal_smoothing", 0.9),
    )
    rolling = RollingGuidanceInference(model, guidance, rolling_config, device)
    
    # Create SDF if needed
    sdf = None
    if "obstacles" in task_config:
        sdf = create_sdf_from_config(task_config, device)
    
    # Create cost function(s)
    tasks = task_config.get("tasks", [])
    if len(tasks) == 1:
        # Single task
        cost_function = create_cost_function(
            tasks[0]["type"],
            tasks[0],
            sdf,
            device
        )
    else:
        # Composed tasks
        cost_functions = {}
        for task in tasks:
            cost_fn = create_cost_function(task["type"], task, sdf, device)
            weight = task.get("weight", 1.0)
            cost_functions[task["type"]] = (cost_fn, weight)
        
        cost_function = guidance.compose_cost_functions(cost_functions)
    
    # Initialize observation history (mock for now)
    history_len = 4
    obs_dim = model.state_dim + model.action_dim
    observation_history = torch.randn(1, history_len, obs_dim, device=device)
    
    # Control loop
    logger.info(f"Starting control loop for {num_steps} steps...")
    actions = []
    costs = []
    
    for step in range(num_steps):
        # Get action from guided policy
        result = rolling.step(
            observation_history=observation_history,
            cost_function=cost_function,
        )
        
        action = result["action"]
        cost = result["cost"]
        
        actions.append(action.cpu().numpy())
        costs.append(cost)
        
        # Log progress
        if step % 10 == 0:
            logger.info(f"Step {step:3d}: Cost={cost:.4f}, Replanned={result['info']['replanned']}")
        
        # Update observation history (mock - in real system, get from robot)
        # Shift history and add new observation
        observation_history = torch.cat([
            observation_history[:, 1:],
            torch.randn(1, 1, obs_dim, device=device)
        ], dim=1)
    
    logger.info(f"Control loop completed. Average cost: {np.mean(costs):.4f}")
    
    return actions, costs


def main():
    parser = argparse.ArgumentParser(description="Play guided diffusion policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to task configuration YAML",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of control steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--save_trajectory",
        type=str,
        help="Path to save trajectory",
    )
    
    args = parser.parse_args()
    
    # Load task configuration
    with open(args.config, "r") as f:
        task_config = yaml.safe_load(f)
    
    logger.info(f"Loaded task configuration from {args.config}")
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Run guided control
    actions, costs = run_guided_control(
        model=model,
        task_config=task_config,
        num_steps=args.num_steps,
        device=args.device,
    )
    
    # Save trajectory if requested
    if args.save_trajectory:
        trajectory_data = {
            "actions": np.array(actions),
            "costs": np.array(costs),
            "config": task_config,
        }
        np.savez(args.save_trajectory, **trajectory_data)
        logger.info(f"Saved trajectory to {args.save_trajectory}")


if __name__ == "__main__":
    main()