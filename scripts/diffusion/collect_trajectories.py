#!/usr/bin/env python3
"""Collect expert trajectories from Stage 1 tracking policies.

This script runs trained motion tracking policies and collects state-action
trajectories for training the Stage 2 diffusion model.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
import pickle
import json

# Add project path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking"))

# Import diffusion data modules
from whole_body_tracking.diffusion.data import (
    Trajectory,
    TrajectoryDataset,
    StateRepresentation
)
from whole_body_tracking.diffusion.data.data_collection import (
    MotionDataCollector,
    DataCollectionConfig,
    ActionDelayBuffer
)

# Try to import Isaac Lab components
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
        LocomotionVelocityRoughEnvCfg
    )
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Warning: Isaac Lab not available. Will generate mock data.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect trajectories from Stage 1 policies"
    )
    
    # Environment arguments
    parser.add_argument(
        "--task",
        type=str,
        default="Tracking-Flat-G1-v0",
        help="Isaac Lab task name"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments"
    )
    
    # Policy arguments
    parser.add_argument(
        "--policy_dir",
        type=str,
        required=False,
        help="Directory containing trained policies"
    )
    parser.add_argument(
        "--policy_names",
        type=str,
        nargs="+",
        default=["walk", "run", "jump", "dance"],
        help="Names of motion policies to collect from"
    )
    
    # Collection arguments
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Total number of episodes to collect"
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=1000,
        help="Maximum episode length"
    )
    parser.add_argument(
        "--episodes_per_motion",
        type=int,
        default=100,
        help="Episodes to collect per motion type"
    )
    
    # Data augmentation
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise augmentation during collection"
    )
    parser.add_argument(
        "--state_noise_scale",
        type=float,
        default=0.005,
        help="Scale of state noise"
    )
    parser.add_argument(
        "--action_noise_scale",
        type=float,
        default=0.01,
        help="Scale of action noise"
    )
    parser.add_argument(
        "--action_delay_prob",
        type=float,
        default=0.5,
        help="Probability of applying action delay"
    )
    parser.add_argument(
        "--action_delay_ms",
        type=int,
        nargs=2,
        default=[0, 100],
        help="Range of action delay in milliseconds"
    )
    
    # Trajectory parameters
    parser.add_argument(
        "--history_length",
        type=int,
        default=4,
        help="Length of observation history (N)"
    )
    parser.add_argument(
        "--future_length_states",
        type=int,
        default=32,
        help="Number of future states (H_s)"
    )
    parser.add_argument(
        "--future_length_actions",
        type=int,
        default=16,
        help="Number of future actions (H_a)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/trajectories",
        help="Directory to save collected data"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="collected_trajectories",
        help="Name for output file"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["pt", "npz", "pkl"],
        default="pt",
        help="Format for saving data"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Generate mock data for testing"
    )
    
    return parser.parse_args()


def load_policies(policy_dir: str, policy_names: List[str], device: str) -> Dict:
    """Load trained Stage 1 policies.
    
    Args:
        policy_dir: Directory containing policy checkpoints
        policy_names: List of policy names to load
        device: Device to load policies on
        
    Returns:
        Dictionary mapping policy names to loaded models
    """
    policies = {}
    
    if not ISAAC_AVAILABLE or not Path(policy_dir).exists():
        print("Warning: Cannot load real policies. Using mock policies.")
        return {}
    
    for name in policy_names:
        policy_path = Path(policy_dir) / f"{name}_policy.pt"
        if policy_path.exists():
            try:
                # Load policy checkpoint
                checkpoint = torch.load(policy_path, map_location=device)
                
                # Create policy network (would need actual architecture)
                # This is a placeholder - actual implementation would load
                # the specific policy architecture used in Stage 1
                policy = checkpoint.get('model', None)
                if policy:
                    policy.eval()
                    policies[name] = policy
                    print(f"Loaded policy: {name}")
            except Exception as e:
                print(f"Failed to load policy {name}: {e}")
    
    return policies


def collect_with_isaac(args, policies: Dict) -> TrajectoryDataset:
    """Collect trajectories using Isaac Lab environment.
    
    Args:
        args: Command line arguments
        policies: Dictionary of loaded policies
        
    Returns:
        Dataset of collected trajectories
    """
    if not ISAAC_AVAILABLE:
        raise RuntimeError("Isaac Lab not available")
    
    # Create environment
    env_cfg = LocomotionVelocityRoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Create data collector
    collection_cfg = DataCollectionConfig(
        history_length=args.history_length,
        future_length_states=args.future_length_states,
        future_length_actions=args.future_length_actions,
        episodes_per_motion=args.episodes_per_motion,
        max_episode_length=args.max_episode_length,
        action_noise_scale=args.action_noise_scale,
        state_noise_scale=args.state_noise_scale,
        action_delay_prob=args.action_delay_prob,
        action_delay_range=tuple(args.action_delay_ms)
    )
    
    collector = MotionDataCollector(
        env=env,
        policies=policies,
        cfg=collection_cfg
    )
    
    # Collect trajectories
    print(f"Collecting {args.num_episodes} episodes...")
    dataset = collector.collect_trajectories(
        num_episodes=args.num_episodes,
        add_noise=args.add_noise
    )
    
    env.close()
    
    return dataset


def collect_mock_data(args) -> TrajectoryDataset:
    """Generate mock trajectories for testing.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dataset of mock trajectories
    """
    print("Generating mock trajectories for testing...")
    
    collection_cfg = DataCollectionConfig(
        history_length=args.history_length,
        future_length_states=args.future_length_states,
        future_length_actions=args.future_length_actions,
        episodes_per_motion=args.episodes_per_motion,
        max_episode_length=args.max_episode_length,
        state_dim=165,  # From paper
        action_dim=69,   # From paper
        num_bodies=23    # SMPL/humanoid skeleton
    )
    
    collector = MotionDataCollector(cfg=collection_cfg)
    
    # Generate mock data
    dataset = collector.collect_trajectories(
        num_episodes=args.num_episodes,
        add_noise=args.add_noise
    )
    
    return dataset


def save_dataset(dataset: TrajectoryDataset, args):
    """Save collected dataset to file.
    
    Args:
        dataset: Dataset to save
        args: Command line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.output_name}.{args.save_format}"
    
    print(f"Saving dataset to {output_path}")
    print(f"  Total trajectories: {len(dataset)}")
    print(f"  State dimension: {dataset.trajectories[0].history_states.shape[-1]}")
    print(f"  Action dimension: {dataset.trajectories[0].history_actions.shape[-1]}")
    
    if args.save_format == "pt":
        dataset.save(str(output_path))
    
    elif args.save_format == "npz":
        # Convert to numpy arrays for npz format
        data_dict = {
            'history_states': [],
            'history_actions': [],
            'future_states': [],
            'future_actions': [],
            'motion_ids': [],
            'timesteps': [],
            'success': []
        }
        
        for traj in dataset.trajectories:
            data_dict['history_states'].append(traj.history_states.numpy())
            data_dict['history_actions'].append(traj.history_actions.numpy())
            data_dict['future_states'].append(traj.future_states.numpy())
            data_dict['future_actions'].append(traj.future_actions.numpy())
            data_dict['motion_ids'].append(traj.motion_id)
            data_dict['timesteps'].append(traj.timestep)
            data_dict['success'].append(traj.success)
        
        # Stack arrays
        for key in ['history_states', 'history_actions', 'future_states', 'future_actions']:
            data_dict[key] = np.stack(data_dict[key])
        
        np.savez_compressed(output_path, **data_dict)
    
    elif args.save_format == "pkl":
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    # Save metadata
    metadata = {
        'num_trajectories': len(dataset),
        'history_length': args.history_length,
        'future_length_states': args.future_length_states,
        'future_length_actions': args.future_length_actions,
        'state_dim': dataset.trajectories[0].history_states.shape[-1],
        'action_dim': dataset.trajectories[0].history_actions.shape[-1],
        'collection_args': vars(args)
    }
    
    metadata_path = output_dir / f"{args.output_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")


def main():
    """Main collection function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "="*60)
    print("BeyondMimic Stage 2 - Trajectory Collection")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}/{args.output_name}.{args.save_format}")
    print("="*60 + "\n")
    
    # Load policies if available
    policies = {}
    if args.policy_dir and not args.mock_data:
        policies = load_policies(args.policy_dir, args.policy_names, args.device)
    
    # Collect data
    if args.mock_data or not ISAAC_AVAILABLE or not policies:
        dataset = collect_mock_data(args)
    else:
        dataset = collect_with_isaac(args, policies)
    
    # Save dataset
    save_dataset(dataset, args)
    
    print("\n" + "="*60)
    print("Collection completed successfully!")
    print("="*60)
    
    # Print some statistics
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nDataset Statistics:")
        print(f"  History shape: {sample['history_states'].shape}")
        print(f"  Future states shape: {sample['future_states'].shape}")
        print(f"  Future actions shape: {sample['future_actions'].shape}")
        
        # Success rate
        successes = [traj.success for traj in dataset.trajectories]
        success_rate = sum(successes) / len(successes) if successes else 0
        print(f"  Success rate: {success_rate:.2%}")
    
    return 0


if __name__ == "__main__":
    exit(main())