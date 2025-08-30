#!/usr/bin/env python
"""
Simple script to test trained model using official play.py patterns from README.
This helps debug if GUI freezing is specific to data collection or general model issues.
"""

import argparse
import os
import pathlib
import sys
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments following README and play.py pattern
parser = argparse.ArgumentParser(description="Test trained model with GUI")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
parser.add_argument("--policy_path", type=str, required=True, help="Path to trained policy directory")
parser.add_argument("--duration", type=int, default=10, help="Duration to run in seconds")

# Import CLI args and append AppLauncher args
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim with GUI (no headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim launch"""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import tracking tasks
import whole_body_tracking.tasks


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Test trained model with GUI following README patterns."""
    
    # Update environment config
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Extract motion file from saved training config (same logic as data collection)
    policy_path = args_cli.policy_path
    env_config_path = os.path.join(policy_path, "params", "env.yaml")
    
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            content = f.read()
        # Extract motion file path using simple string search
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
    
    # Wrap environment (following play.py pattern)
    env = RslRlVecEnvWrapper(env)
    
    # Load trained policy (following play.py pattern)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # Find latest checkpoint in policy directory
    model_files = [f for f in os.listdir(policy_path) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {policy_path}")
    
    latest_checkpoint = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    resume_path = os.path.join(policy_path, latest_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    
    # Create policy runner
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    # Get inference policy
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    # Reset environment
    obs, _ = env.get_observations()
    
    print(f"[INFO] Running trained policy for {args_cli.duration} seconds...")
    print("[INFO] Policy network:")
    print(ppo_runner.alg.policy.actor)
    
    # Simulate environment with GUI
    timestep = 0
    max_timesteps = int(args_cli.duration * 20)  # 20Hz control frequency
    
    while simulation_app.is_running() and timestep < max_timesteps:
        # Run in inference mode
        with torch.inference_mode():
            # Agent stepping
            actions = policy(obs)
            # Environment stepping
            obs, rewards, dones, info = env.step(actions)
            
        timestep += 1
        
        # Print progress every 2 seconds
        if timestep % 40 == 0:
            print(f"[INFO] Running... {timestep//20}/{args_cli.duration}s")
            
        # Check if episode is done
        if isinstance(dones, dict):
            terminated = dones.get("terminated", torch.zeros_like(rewards, dtype=torch.bool))
            truncated = dones.get("time_outs", torch.zeros_like(rewards, dtype=torch.bool))
        else:
            terminated = dones
            truncated = torch.zeros_like(dones, dtype=torch.bool)
            
        done = terminated | truncated
        
        if torch.any(done):
            print(f"[INFO] Episode finished at timestep {timestep}")
            # Reset environment
            obs, _ = env.reset()
    
    print(f"[INFO] Test completed successfully after {timestep} timesteps")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation
    simulation_app.close()