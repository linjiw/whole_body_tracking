#!/usr/bin/env python3
"""Main training script for BeyondMimic Stage 2 diffusion model.

This script orchestrates the complete training pipeline including:
- Data collection from Stage 1 policies
- Model initialization and configuration
- Training with checkpointing and logging
- Validation and metric tracking
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import torch
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf, DictConfig

# Add project path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking"))

from whole_body_tracking.diffusion.data import TrajectoryDataset
from whole_body_tracking.diffusion.data.data_collection import (
    MotionDataCollector,
    DataCollectionConfig
)
from whole_body_tracking.diffusion.models import StateActionDiffusionModel
from whole_body_tracking.diffusion.training import DDPMTrainer

# Optional imports for WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be limited.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BeyondMimic Stage 2 diffusion model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to existing dataset. If not provided, will collect new data."
    )
    parser.add_argument(
        "--policy_paths",
        type=str,
        nargs="+",
        help="Paths to Stage 1 tracking policies for data collection"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to collect (if collecting new data)"
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default="./data/collected_trajectories.pt",
        help="Path to save collected dataset"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (can use configs/*.yaml)"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to model configuration YAML file (deprecated, use --config)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for transformer"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate"
    )
    
    # Validation arguments
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1000,
        help="Steps between validation"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Steps between logging"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5000,
        help="Steps between checkpoint saves"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="beyondmimic-stage2",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable WandB logging"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def load_config(args) -> DictConfig:
    """Load and merge configurations from YAML and command line.
    
    Priority order (highest to lowest):
    1. Command line arguments
    2. Config file specified by --config
    3. Default values
    
    Args:
        args: Command line arguments
        
    Returns:
        Merged configuration
    """
    # Start with empty config
    config = OmegaConf.create({})
    
    # Load base config if specified
    if args.config:
        config_path = Path(args.config)
        
        # Check in configs directory if not absolute path
        if not config_path.is_absolute() and not config_path.exists():
            config_path = Path(__file__).parent / "configs" / args.config
            if not config_path.suffix:
                config_path = config_path.with_suffix('.yaml')
        
        if config_path.exists():
            loaded_config = OmegaConf.load(config_path)
            config = OmegaConf.merge(config, loaded_config)
            print(f"Loaded configuration from {config_path}")
    
    # Override with command line arguments
    cli_config = OmegaConf.create({
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'num_timesteps': args.num_timesteps,
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'warmup_steps': args.warmup_steps,
            'gradient_clip': args.gradient_clip,
            'ema_decay': args.ema_decay,
        },
        'validation': {
            'val_split': args.val_split,
            'val_interval': args.val_interval,
        },
        'logging': {
            'log_interval': args.log_interval,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project,
        },
        'checkpoint': {
            'checkpoint_dir': args.checkpoint_dir,
            'save_interval': args.save_interval,
        },
        'device': args.device,
    })
    
    # Merge with CLI overrides (CLI takes precedence)
    config = OmegaConf.merge(config, cli_config)
    
    return config


def load_model_config(config_path: Optional[str]) -> Dict:
    """Load model configuration from YAML file (deprecated)."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def collect_or_load_data(args) -> tuple:
    """Collect new data or load existing dataset.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading existing dataset from {args.data_path}")
        dataset = TrajectoryDataset.load(args.data_path)
    else:
        print("Collecting new trajectories...")
        
        # Create data collection config
        collection_cfg = DataCollectionConfig(
            episodes_per_motion=args.num_episodes // 10,  # Distribute across motions
            max_episode_length=1000,
            action_delay_prob=0.5,  # 50% chance of delay
            action_delay_range=(0, 100),  # 0-100ms delay
            state_noise_scale=0.005,
            action_noise_scale=0.01
        )
        
        # Initialize collector
        # Note: In real implementation, we would load actual policies here
        collector = MotionDataCollector(
            env=None,  # Would be actual environment
            policies=None,  # Would be loaded Stage 1 policies
            cfg=collection_cfg
        )
        
        # Collect trajectories
        dataset = collector.collect_trajectories(
            num_episodes=args.num_episodes,
            add_noise=True
        )
        
        # Save dataset
        save_dir = Path(args.save_dataset).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        dataset.save(args.save_dataset)
        print(f"Saved dataset to {args.save_dataset}")
    
    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    # Random split
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and val datasets
    train_trajectories = [dataset.trajectories[i] for i in train_indices]
    val_trajectories = [dataset.trajectories[i] for i in val_indices]
    
    train_dataset = TrajectoryDataset(train_trajectories)
    val_dataset = TrajectoryDataset(val_trajectories) if val_trajectories else None
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")
    
    return train_dataset, val_dataset


def create_model(args, train_dataset) -> StateActionDiffusionModel:
    """Create and initialize the diffusion model."""
    # Load config if provided
    config = load_model_config(args.model_config)
    
    # Override with command line arguments
    model_kwargs = {
        "state_dim": config.get("state_dim", 165),
        "action_dim": config.get("action_dim", 69),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "history_length": config.get("history_length", 4),
        "future_length_states": config.get("future_length_states", 32),
        "future_length_actions": config.get("future_length_actions", 16),
        "num_timesteps": args.num_timesteps,
        "state_schedule_type": config.get("state_schedule_type", "cosine"),
        "action_schedule_type": config.get("action_schedule_type", "linear"),
        "dropout": config.get("dropout", 0.1)
    }
    
    model = StateActionDiffusionModel(**model_kwargs)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Heads: {args.num_heads}")
    print(f"  Diffusion steps: {args.num_timesteps}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up run name
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_run_name = f"diffusion_{timestamp}"
    
    print("\n" + "="*60)
    print("BeyondMimic Stage 2 Diffusion Training")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.use_wandb and WANDB_AVAILABLE:
        print(f"WandB project: {args.wandb_project}")
        print(f"WandB run: {args.wandb_run_name}")
    print("="*60 + "\n")
    
    # Load or collect data
    train_dataset, val_dataset = collect_or_load_data(args)
    
    # Create model
    model = create_model(args, train_dataset)
    
    # Create trainer
    trainer = DDPMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        ema_decay=args.ema_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb and WANDB_AVAILABLE,
        wandb_project=args.wandb_project,
        device=args.device
    )
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        trainer.train()
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. Training can be resumed with --resume flag")
        
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())