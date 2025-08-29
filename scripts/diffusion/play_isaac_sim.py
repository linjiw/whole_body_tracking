#!/usr/bin/env python3
"""
Deployment script for diffusion policy in Isaac Sim.

This script loads a trained diffusion model and deploys it in the Isaac Lab
tracking environment with optional classifier guidance.
"""

import argparse
import sys
import os
import pathlib
import yaml
import torch
from typing import Optional, Dict, Any

from isaaclab.app import AppLauncher

# Parse arguments first
parser = argparse.ArgumentParser(description="Deploy diffusion policy in Isaac Sim")

# Model and configuration
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to diffusion model checkpoint",
)
parser.add_argument(
    "--task",
    type=str,
    default="Tracking-Flat-G1-v0",
    help="Name of the Isaac Lab task",
)
parser.add_argument(
    "--guidance_config",
    type=str,
    help="Path to guidance configuration YAML",
)

# Environment settings
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments",
)
parser.add_argument(
    "--motion_file",
    type=str,
    help="Path to reference motion NPZ file",
)

# Visualization settings
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record videos during deployment",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of recorded video in steps",
)

# Performance settings
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device for computation",
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    default=False,
    help="Run performance benchmarking",
)

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the implementation follows after app launch."""

import gymnasium as gym
import numpy as np
import time
import logging
from collections import defaultdict

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import whole body tracking tasks
import whole_body_tracking.tasks  # noqa: F401

# Import diffusion modules
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
)
from whole_body_tracking.diffusion.integration import (
    IsaacLabDiffusionWrapper,
    ObservationSpaceConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_diffusion_model(checkpoint_path: str, device: str) -> StateActionDiffusionModel:
    """
    Load trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded diffusion model
    """
    logger.info(f"Loading diffusion model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration
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
    
    logger.info("Diffusion model loaded successfully")
    return model


def create_cost_function_from_config(
    config: Dict[str, Any],
    device: str,
) -> Optional[Any]:
    """
    Create cost function from configuration.
    
    Args:
        config: Guidance configuration dictionary
        device: Device for tensors
        
    Returns:
        Cost function or None
    """
    if not config or "tasks" not in config:
        return None
    
    tasks = config["tasks"]
    
    if len(tasks) == 1:
        # Single task
        task = tasks[0]
        task_type = task["type"]
        
        if task_type == "joystick":
            velocity = torch.tensor(task["velocity"], device=device)
            return JoystickCost(
                goal_velocity=velocity,
                velocity_weight=task.get("velocity_weight", 1.0),
            )
        elif task_type == "waypoint":
            position = torch.tensor(task["position"], device=device)
            return WaypointCost(
                goal_position=position,
                distance_threshold=task.get("distance_threshold", 0.5),
            )
        elif task_type == "obstacle":
            # Create obstacles from config
            obstacles = []
            for obs_config in task.get("obstacles", []):
                if obs_config["type"] == "sphere":
                    obstacles.append(
                        SphereObstacle(
                            position=torch.tensor(obs_config["position"], device=device),
                            radius=obs_config["radius"],
                        )
                    )
                elif obs_config["type"] == "box":
                    obstacles.append(
                        BoxObstacle(
                            position=torch.tensor(obs_config["position"], device=device),
                            half_extents=torch.tensor(obs_config["half_extents"], device=device),
                        )
                    )
            
            sdf = SignedDistanceField(obstacles)
            return ObstacleAvoidanceCost(sdf_function=sdf)
    else:
        # Multiple tasks - create composite
        logger.warning("Composite cost functions not yet implemented in Isaac Lab integration")
        return None
    
    return None


@hydra_task_config(args_cli.task, "env_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg):
    """
    Main deployment function.
    
    Args:
        env_cfg: Environment configuration from Hydra
    """
    # Update environment configuration
    env_cfg.scene.num_envs = args_cli.num_envs
    
    if args_cli.motion_file:
        logger.info(f"Using motion file: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    
    # Create Isaac Lab environment
    logger.info(f"Creating environment: {args_cli.task}")
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    
    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", "diffusion_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        logger.info("Recording video during deployment")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Load diffusion model
    diffusion_model = load_diffusion_model(args_cli.checkpoint, args_cli.device)
    
    # Load guidance configuration if provided
    guidance_config = None
    cost_function = None
    if args_cli.guidance_config:
        with open(args_cli.guidance_config, "r") as f:
            guidance_yaml = yaml.safe_load(f)
        
        # Create guidance configuration
        guidance_config = GuidanceConfig(
            guidance_scale=guidance_yaml.get("guidance_scale", 1.0),
            gradient_clipping=guidance_yaml.get("gradient_clipping", 10.0),
        )
        
        # Create cost function
        cost_function = create_cost_function_from_config(guidance_yaml, args_cli.device)
        
        logger.info(f"Loaded guidance configuration from {args_cli.guidance_config}")
    
    # Determine number of joints from environment
    # Get a sample action to determine dimensions
    sample_action = env.action_space.sample()
    num_joints = sample_action.shape[-1] if hasattr(sample_action, 'shape') else 19
    
    # Create observation space configuration
    obs_config = ObservationSpaceConfig(num_joints=num_joints)
    
    # Create diffusion wrapper
    logger.info("Creating IsaacLabDiffusionWrapper")
    wrapper = IsaacLabDiffusionWrapper(
        env=env.unwrapped,
        diffusion_model=diffusion_model,
        guidance_config=guidance_config,
        obs_config=obs_config,
        device=args_cli.device,
    )
    
    # Reset environment
    logger.info("Resetting environment")
    obs = wrapper.reset()
    
    # Performance tracking
    if args_cli.benchmark:
        step_times = []
        inference_times = []
    
    # Main control loop
    logger.info("Starting control loop")
    step_count = 0
    total_reward = 0.0
    
    try:
        while simulation_app.is_running():
            # Time the step if benchmarking
            if args_cli.benchmark:
                step_start = time.time()
            
            # Get action from diffusion policy
            with torch.no_grad():
                if args_cli.benchmark:
                    inference_start = time.time()
                
                action = wrapper.get_action(obs, cost_function)
                
                if args_cli.benchmark:
                    inference_times.append(time.time() - inference_start)
            
            # Step environment
            obs, reward, terminated, truncated, info = wrapper.step(action)
            
            # Track statistics
            total_reward += reward.mean().item()
            step_count += 1
            
            if args_cli.benchmark:
                step_times.append(time.time() - step_start)
            
            # Log progress periodically
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                logger.info(
                    f"Step {step_count}: Avg Reward={avg_reward:.4f}, "
                    f"Cost={info.get('diffusion_avg_cost', 0.0):.4f}"
                )
            
            # Check termination
            if args_cli.video and step_count >= args_cli.video_length:
                logger.info("Video recording complete")
                break
            
            if terminated.any() or truncated.any():
                logger.info("Episode terminated or truncated")
                obs = wrapper.reset()
                total_reward = 0.0
    
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    
    # Print benchmarking results
    if args_cli.benchmark and step_times:
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE BENCHMARKING RESULTS")
        logger.info("="*50)
        
        avg_step_time = np.mean(step_times) * 1000  # Convert to ms
        std_step_time = np.std(step_times) * 1000
        avg_inference_time = np.mean(inference_times) * 1000
        std_inference_time = np.std(inference_times) * 1000
        
        logger.info(f"Average step time: {avg_step_time:.2f} ± {std_step_time:.2f} ms")
        logger.info(f"Average inference time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        logger.info(f"Control frequency: {1000/avg_step_time:.1f} Hz")
        logger.info(f"Target latency (<20ms): {'✓ PASS' if avg_inference_time < 20 else '✗ FAIL'}")
        logger.info("="*50)
    
    # Clean up
    logger.info("Closing environment")
    wrapper.close()


if __name__ == "__main__":
    # Run main function
    main()
    
    # Close simulation
    simulation_app.close()