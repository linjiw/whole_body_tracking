#!/usr/bin/env python
"""
GUI-friendly data collection test script.
Runs for a very short time to debug GUI freezing issues.
"""

import argparse
import os
import sys
import torch
import numpy as np

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test data collection with GUI")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--policy_path", type=str, required=True, help="Path to trained policy directory")
parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps to run (default: 100)")

# Import CLI args and append AppLauncher args
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim with GUI (remove headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim launch"""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import tracking tasks
import whole_body_tracking.tasks


def extract_body_pos_state(env, obs):
    """Extract Body-Pos state representation from environment observations."""
    # Get robot state
    robot = env.unwrapped.scene["robot"]
    
    # Global states (relative to current frame)
    root_pos = robot.data.root_pos_w - env.unwrapped.scene.env_origins  # Relative position
    root_lin_vel = robot.data.root_lin_vel_w  # Linear velocity
    root_ang_vel = robot.data.root_ang_vel_w  # Angular velocity
    
    # Local states (in character frame) - use existing body positions
    body_pos_w = robot.data.body_pos_w
    root_pos_w = robot.data.root_pos_w
    body_lin_vel_w = robot.data.body_lin_vel_w
    root_lin_vel_w = robot.data.root_lin_vel_w
    
    # Convert to character frame (relative to root)
    body_pos_rel = body_pos_w - root_pos_w.unsqueeze(1)
    body_vel_rel = body_lin_vel_w - root_lin_vel_w.unsqueeze(1)
    
    # Concatenate into Body-Pos representation
    state = torch.cat([
        root_pos.flatten(start_dim=1),        # (num_envs, 3)
        root_lin_vel.flatten(start_dim=1),    # (num_envs, 3) 
        root_ang_vel.flatten(start_dim=1),    # (num_envs, 3)
        body_pos_rel.flatten(start_dim=1),    # (num_envs, num_bodies*3)
        body_vel_rel.flatten(start_dim=1),    # (num_envs, num_bodies*3)
    ], dim=1)
    
    return state


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """GUI-friendly data collection test."""
    
    # Update environment config
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # CRITICAL: Relax termination conditions for data collection
    if hasattr(env_cfg.terminations, 'anchor_pos'):
        env_cfg.terminations.anchor_pos.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, 'anchor_ori'):
        env_cfg.terminations.anchor_ori.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, 'ee_body_pos'):
        env_cfg.terminations.ee_body_pos.params["threshold"] = 10.0
    
    # Extract motion file from saved training config
    policy_path = args_cli.policy_path
    env_config_path = os.path.join(policy_path, "params", "env.yaml")
    
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            content = f.read()
        for line in content.split('\n'):
            if 'motion_file:' in line and '.npz' in line:
                motion_file = line.split('motion_file:')[1].strip()
                break
        else:
            raise ValueError("Could not find motion_file in saved configuration")
    else:
        raise FileNotFoundError(f"Environment configuration not found at: {env_config_path}")
    
    print(f"[INFO] Using motion file: {motion_file}")
    env_cfg.commands.motion.motion_file = motion_file
    
    # Create environment with GUI rendering
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env)
    
    # Load trained policy
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    model_files = [f for f in os.listdir(policy_path) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {policy_path}")
    
    latest_checkpoint = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    resume_path = os.path.join(policy_path, latest_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get correct action dimension
    with torch.inference_mode():
        sample_action = policy(obs)
    action_dim = sample_action.shape[1]
    state_dim = extract_body_pos_state(env, obs).shape[1]
    
    print(f"[INFO] State dimension: {state_dim}")
    print(f"[INFO] Action dimension: {action_dim}")
    print(f"[INFO] Running for maximum {args_cli.max_steps} steps...")
    
    # Store some trajectory data for verification
    states_collected = []
    actions_collected = []
    
    timestep = 0
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        # Extract current state
        current_state = extract_body_pos_state(env, obs)
        states_collected.append(current_state[0])  # Take first environment
        
        # Get action from policy
        with torch.inference_mode():
            action = policy(obs)
        
        actions_collected.append(action[0])  # Take first environment
        
        # Step environment
        obs, rewards, dones, info = env.step(action)
        
        timestep += 1
        
        # Print progress every 20 steps
        if timestep % 20 == 0:
            print(f"[INFO] Step {timestep}/{args_cli.max_steps}")
            print(f"        State range: [{current_state.min():.3f}, {current_state.max():.3f}]")
            print(f"        Action range: [{action.min():.3f}, {action.max():.3f}]")
            
        # Check if episode is done
        if isinstance(dones, dict):
            terminated = dones.get("terminated", torch.zeros_like(rewards, dtype=torch.bool))
            truncated = dones.get("time_outs", torch.zeros_like(rewards, dtype=torch.bool))
        else:
            terminated = dones
            truncated = torch.zeros_like(dones, dtype=torch.bool)
            
        done = terminated | truncated
        
        if torch.any(done):
            print(f"[INFO] Episode reset at timestep {timestep}")
            obs, _ = env.reset()
    
    print(f"[INFO] Collected {len(states_collected)} states and {len(actions_collected)} actions")
    print(f"[INFO] Data collection test completed successfully!")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation
    simulation_app.close()