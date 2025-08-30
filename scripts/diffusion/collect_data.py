#!/usr/bin/env python
"""
Stage 2 Data Collection Script for BeyondMimic Diffusion Training

Collects state-action trajectory data from trained tracking policies following
the existing codebase patterns from play.py and the tracking environment.
"""

import argparse
import os
import pathlib
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from dataclasses import dataclass

from isaaclab.app import AppLauncher

# Add argparse arguments following play.py pattern
parser = argparse.ArgumentParser(description="Collect diffusion training data from trained tracking policies")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments for data collection")
parser.add_argument("--policy_paths", type=str, nargs="+", required=True, 
                    help="Paths to trained policy directories (e.g., logs/rsl_rl/g1_flat/run1)")
parser.add_argument("--output_dir", type=str, default="data/diffusion", help="Output directory for collected data")
parser.add_argument("--episodes_per_policy", type=int, default=500, help="Episodes to collect per policy")
parser.add_argument("--horizon", type=int, default=16, help="Future trajectory horizon (H)")
parser.add_argument("--history_length", type=int, default=4, help="History length (N)")
parser.add_argument("--action_delay_range", type=Tuple[int, int], default=(0, 100), 
                    help="Action delay randomization range in ms")

# Import CLI args and append AppLauncher args
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Isaac Sim launch"""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import tracking tasks
import whole_body_tracking.tasks


@dataclass
class TrajectoryData:
    """Data structure for storing trajectory information following paper format."""
    # History: O_t = [s_{t-N}, a_{t-N}, ..., s_t] 
    history_states: torch.Tensor      # (episode_length, N+1, state_dim)
    history_actions: torch.Tensor     # (episode_length, N, action_dim)
    
    # Future: τ_t = [a_t, s_{t+1}, ..., s_{t+H}, a_{t+H}]
    future_actions: torch.Tensor      # (episode_length, H+1, action_dim) 
    future_states: torch.Tensor       # (episode_length, H, state_dim)
    
    # Metadata
    motion_file: str
    policy_path: str
    episode_id: int
    timesteps: int


class Stage2DataCollector:
    """Data collector for Stage 2 diffusion training following existing codebase patterns."""
    
    def __init__(
        self, 
        env: ManagerBasedRLEnv,
        policy_paths: List[str],
        horizon: int = 16,
        history_length: int = 4,
        action_delay_range: Tuple[int, int] = (0, 100)
    ):
        self.env = env
        self.horizon = horizon
        self.history_length = history_length
        self.action_delay_range = action_delay_range
        self.device = env.device
        
        # Load all trained policies following play.py pattern
        self.policies = self._load_policies(policy_paths)
        
        # Initialize trajectory storage
        self.trajectories: List[TrajectoryData] = []
        
        # Action delay randomization (convert ms to timesteps)
        # Assuming 20Hz control frequency (50ms timestep)
        self.min_delay_steps = action_delay_range[0] // 50
        self.max_delay_steps = action_delay_range[1] // 50
        
    def _load_policies(self, policy_paths: List[str]) -> Dict[str, torch.nn.Module]:
        """Load trained policies following the play.py pattern."""
        policies = {}
        
        for policy_path in policy_paths:
            print(f"[INFO] Loading policy from: {policy_path}")
            
            # Get config from the log directory (following play.py pattern)
            agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
            
            # Find latest checkpoint - need to look directly in the policy_path directory
            # List model checkpoints in the policy directory
            model_files = [f for f in os.listdir(policy_path) if f.startswith("model_") and f.endswith(".pt")]
            if not model_files:
                raise FileNotFoundError(f"No model checkpoints found in {policy_path}")
            
            # Get the latest checkpoint (highest number)
            latest_checkpoint = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            resume_path = os.path.join(policy_path, latest_checkpoint)
            print(f"[INFO] Loading checkpoint: {resume_path}")
            
            # Create policy runner (following play.py)
            ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            ppo_runner.load(resume_path)
            
            # Get inference policy
            policy = ppo_runner.get_inference_policy(device=self.env.device)
            policies[policy_path] = policy
            
        return policies
    
    def _extract_body_pos_state(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract Body-Pos state representation from environment observations.
        Following paper Section VI.D - Body-Pos representation.
        """
        # This needs to be adapted based on the actual observation structure
        # For now, assuming we can extract the required components from obs
        
        # Get robot state - need to access the underlying environment
        robot = self.env.unwrapped.scene["robot"]
        
        # Global states (relative to current frame)
        root_pos = robot.data.root_pos_w - self.env.unwrapped.scene.env_origins  # Relative position
        root_lin_vel = robot.data.root_lin_vel_w  # Linear velocity
        root_ang_vel = robot.data.root_ang_vel_w  # Angular velocity (rotation vector)
        
        # Local states (in character frame) - compute body positions from joint states
        # This would need forward kinematics computation
        body_positions = self._compute_body_positions_from_joints(robot.data.joint_pos)
        body_velocities = self._compute_body_velocities_from_joints(robot.data.joint_vel)
        
        # Concatenate into Body-Pos representation
        state = torch.cat([
            root_pos.flatten(start_dim=1),        # (num_envs, 3)
            root_lin_vel.flatten(start_dim=1),    # (num_envs, 3) 
            root_ang_vel.flatten(start_dim=1),    # (num_envs, 3)
            body_positions.flatten(start_dim=1),  # (num_envs, num_bodies*3)
            body_velocities.flatten(start_dim=1), # (num_envs, num_bodies*3)
        ], dim=1)
        
        return state
    
    def _compute_body_positions_from_joints(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Compute body positions in character frame from joint positions."""
        # This requires forward kinematics - for now return placeholder
        # In real implementation, would use robot's forward kinematics
        robot = self.env.unwrapped.scene["robot"]
        # Use existing body position data relative to root
        body_pos_w = robot.data.body_pos_w
        root_pos_w = robot.data.root_pos_w
        
        # Convert to character frame (relative to root)
        body_pos_rel = body_pos_w - root_pos_w.unsqueeze(1)
        return body_pos_rel
    
    def _compute_body_velocities_from_joints(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Compute body velocities in character frame from joint velocities."""
        # Similar to positions, this requires forward kinematics
        robot = self.env.unwrapped.scene["robot"]
        body_lin_vel_w = robot.data.body_lin_vel_w
        root_lin_vel_w = robot.data.root_lin_vel_w
        
        # Convert to character frame 
        body_vel_rel = body_lin_vel_w - root_lin_vel_w.unsqueeze(1)
        return body_vel_rel
        
    def collect_trajectories(self, episodes_per_policy: int = 500) -> List[TrajectoryData]:
        """
        Collect trajectory data from all loaded policies with domain randomization.
        Following the paper's data collection procedure (Section VI.A).
        """
        print(f"[INFO] Collecting {episodes_per_policy} episodes per policy...")
        
        for policy_path, policy in self.policies.items():
            print(f"[INFO] Collecting data from policy: {policy_path}")
            
            for episode in range(episodes_per_policy):
                trajectory_data = self._collect_single_episode(policy, policy_path, episode)
                if trajectory_data is not None:
                    self.trajectories.append(trajectory_data)
                
                if (episode + 1) % 50 == 0:
                    print(f"[INFO] Collected {episode + 1}/{episodes_per_policy} episodes")
        
        print(f"[INFO] Total trajectories collected: {len(self.trajectories)}")
        return self.trajectories
    
    def _collect_single_episode(
        self, 
        policy: torch.nn.Module, 
        policy_path: str, 
        episode_id: int
    ) -> Optional[TrajectoryData]:
        """Collect data from a single episode with action delay randomization."""
        
        # Reset environment 
        obs, _ = self.env.reset()
        
        # Initialize trajectory storage
        max_episode_length = int(self.env.max_episode_length)
        state_dim = self._extract_body_pos_state(obs).shape[1] 
        
        # Get a sample action to determine correct action_dim
        with torch.inference_mode():
            sample_action = policy(obs)
        
        # Use the actual action tensor shape instead of action_space
        action_dim = sample_action.shape[1]  # [num_envs, action_dim]
        
        # Store full episode trajectory first
        episode_states = []
        episode_actions = []
        episode_obs = []
        
        # Action delay buffer for randomization
        action_buffer = []
        
        episode_length = 0
        
        while episode_length < max_episode_length:
            # Extract current state
            current_state = self._extract_body_pos_state(obs)
            episode_states.append(current_state[0])  # Take first environment only
            episode_obs.append(obs)
            
            # Get action from policy (with inference mode)
            with torch.inference_mode():
                action = policy(obs)
            
            # Apply action delay randomization (following paper Section VI.A)
            delay_steps = torch.randint(
                self.min_delay_steps, 
                self.max_delay_steps + 1, 
                (1,), 
                device=self.device
            ).item()
            
            # Buffer the action (take first environment only)
            action_buffer.append(action[0].clone())
            if len(action_buffer) < delay_steps:
                # Use previous action or zero action for initial steps
                actual_action = action_buffer[0] if action_buffer else torch.zeros_like(action[0])
            else:
                # Use delayed action
                actual_action = action_buffer.pop(0)
                
            episode_actions.append(actual_action)
            
            # Expand actual_action back to batch size for environment step
            actual_action_batch = actual_action.unsqueeze(0)  # Add batch dimension
            
            # Step environment - following play.py pattern
            obs, rewards, dones, info = self.env.step(actual_action_batch)
            
            # RSL-RL wrapper returns dones as dict, extract terminated/truncated
            if isinstance(dones, dict):
                terminated = dones.get("terminated", torch.zeros_like(rewards, dtype=torch.bool))
                truncated = dones.get("time_outs", torch.zeros_like(rewards, dtype=torch.bool))
            else:
                # If dones is a tensor, treat as terminated
                terminated = dones
                truncated = torch.zeros_like(dones, dtype=torch.bool)
                
            done = terminated | truncated
            
            episode_length += 1
            
            # Break if any environment is done
            if torch.any(done):
                break
        
        # Now extract trajectory data following paper format
        # Only process timesteps where we can form complete history and future
        valid_timesteps = episode_length - self.history_length - self.horizon
        if valid_timesteps <= 0:
            print(f"[WARN] Episode {episode_id} too short for trajectory extraction: {episode_length} steps")
            return None
            
        # Check if we have enough data for trajectory extraction
        print(f"[INFO] Episode {episode_id}: {episode_length} steps, {valid_timesteps} valid trajectory segments")
            
        # Storage tensors for valid trajectory segments
        history_states = torch.zeros(
            (valid_timesteps, self.history_length + 1, state_dim), 
            device=self.device
        )
        history_actions = torch.zeros(
            (valid_timesteps, self.history_length, action_dim),
            device=self.device  
        )
        future_states = torch.zeros(
            (valid_timesteps, self.horizon, state_dim),
            device=self.device
        )
        future_actions = torch.zeros(
            (valid_timesteps, self.horizon + 1, action_dim),
            device=self.device
        )
        
        # Convert lists to tensors for easier manipulation
        episode_states_tensor = torch.stack(episode_states)  # (episode_length, state_dim)
        episode_actions_tensor = torch.stack(episode_actions)  # (episode_length, action_dim)
        
        # Extract trajectory segments
        for t in range(valid_timesteps):
            timestep = t + self.history_length  # Start after history_length steps
            
            # History: O_t = [s_{t-N}, a_{t-N}, ..., s_t]
            # States from t-N to t (inclusive) -> N+1 states
            for i in range(self.history_length + 1):
                state_idx = timestep - self.history_length + i
                history_states[t, i] = episode_states_tensor[state_idx]
                
            # Actions from t-N to t-1 (inclusive) -> N actions  
            for i in range(self.history_length):
                action_idx = timestep - self.history_length + i
                history_actions[t, i] = episode_actions_tensor[action_idx]
            
            # Future: τ_t = [a_t, s_{t+1}, ..., s_{t+H}, a_{t+H}]
            # Actions from t to t+H (inclusive) -> H+1 actions
            for i in range(self.horizon + 1):
                action_idx = timestep + i
                if action_idx < episode_actions_tensor.shape[0]:
                    future_actions[t, i] = episode_actions_tensor[action_idx]
                else:
                    # Use last action if we run out of data
                    future_actions[t, i] = episode_actions_tensor[-1]
                    
            # States from t+1 to t+H (inclusive) -> H states
            for i in range(self.horizon):
                state_idx = timestep + 1 + i
                if state_idx < episode_states_tensor.shape[0]:
                    future_states[t, i] = episode_states_tensor[state_idx]
                else:
                    # Use last state if we run out of data  
                    future_states[t, i] = episode_states_tensor[-1]
        
        # Get motion file from environment command (if available)
        motion_file = getattr(self.env.unwrapped.command_manager._terms["motion"].cfg, "motion_file", "unknown")
        
        return TrajectoryData(
            history_states=history_states,
            history_actions=history_actions, 
            future_actions=future_actions,
            future_states=future_states,
            motion_file=motion_file,
            policy_path=policy_path,
            episode_id=episode_id,
            timesteps=valid_timesteps  # Number of valid trajectory segments
        )
    
    def save_dataset(self, output_dir: str):
        """Save collected trajectories to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy for saving
        dataset = {
            "trajectories": [],
            "metadata": {
                "horizon": self.horizon,
                "history_length": self.history_length,
                "action_delay_range": self.action_delay_range,
                "total_episodes": len(self.trajectories)
            }
        }
        
        for traj in self.trajectories:
            dataset["trajectories"].append({
                "history_states": traj.history_states.cpu().numpy(),
                "history_actions": traj.history_actions.cpu().numpy(),
                "future_states": traj.future_states.cpu().numpy(), 
                "future_actions": traj.future_actions.cpu().numpy(),
                "motion_file": traj.motion_file,
                "policy_path": traj.policy_path,
                "episode_id": traj.episode_id,
                "timesteps": traj.timesteps
            })
        
        # Save dataset
        output_path = os.path.join(output_dir, "diffusion_dataset.npz")
        np.savez_compressed(output_path, **dataset)
        print(f"[INFO] Dataset saved to: {output_path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Main data collection function following play.py pattern."""
    
    # Update environment config
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # CRITICAL: Relax termination conditions for successful data collection
    # Disable strict termination conditions that cause immediate episode end
    if hasattr(env_cfg.terminations, 'anchor_pos'):
        env_cfg.terminations.anchor_pos.params["threshold"] = 10.0  # Very high threshold
    if hasattr(env_cfg.terminations, 'anchor_ori'):
        env_cfg.terminations.anchor_ori.params["threshold"] = 10.0  # Very high threshold
    if hasattr(env_cfg.terminations, 'ee_body_pos'):
        env_cfg.terminations.ee_body_pos.params["threshold"] = 10.0  # Very high threshold
    
    # Set motion file based on the trained model name 
    # This is necessary because the tracking environment requires a motion file
    # For now, manually set it based on the model directory name
    first_policy_path = args_cli.policy_paths[0]
    policy_dir_name = os.path.basename(first_policy_path)
    
    # Extract motion name from directory name (e.g., "2025-08-29_17-55-04_wal3_subject4_0829_1754" -> "wal3_subject4")
    if "wal3_subject4" in policy_dir_name:
        motion_file = "/home/linji/nfs/whole_body_tracking/artifacts/walk3_subject4:v1/motion.npz"
    else:
        # Try to read motion file from a simple text search in env.yaml
        env_config_path = os.path.join(first_policy_path, "params", "env.yaml")
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
    
    # Create environment 
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap environment (following play.py pattern)
    env = RslRlVecEnvWrapper(env)
    
    # Create data collector
    collector = Stage2DataCollector(
        env=env,
        policy_paths=args_cli.policy_paths,
        horizon=args_cli.horizon,
        history_length=args_cli.history_length,
        action_delay_range=args_cli.action_delay_range
    )
    
    # Collect trajectories
    trajectories = collector.collect_trajectories(args_cli.episodes_per_policy)
    
    # Save dataset
    collector.save_dataset(args_cli.output_dir)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation
    simulation_app.close()