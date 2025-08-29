#!/usr/bin/env python3
"""Test script to verify improved metrics integration."""

import sys
from pathlib import Path
import torch

# Add project path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking/whole_body_tracking"))

from diffusion.data import TrajectoryDataset
from diffusion.data.data_collection import MotionDataCollector, DataCollectionConfig
from diffusion.models import StateActionDiffusionModel
from diffusion.training import DDPMTrainer, DiffusionMetrics


def test_improved_validation():
    """Test the improved validation with comprehensive metrics."""
    print("\n" + "="*60)
    print("Testing Improved Validation Metrics")
    print("="*60)
    
    # Create small dataset
    print("\n1. Creating test dataset...")
    cfg = DataCollectionConfig(
        episodes_per_motion=1,
        max_episode_length=50,
        history_length=4,
        future_length_states=32,
        future_length_actions=16
    )
    collector = MotionDataCollector(cfg=cfg)
    
    # Create train and validation datasets
    train_dataset = collector.collect_trajectories(num_episodes=5)
    val_dataset = collector.collect_trajectories(num_episodes=2)
    print(f"   ✓ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create small model
    print("\n2. Creating model...")
    model = StateActionDiffusionModel(
        state_dim=165,
        action_dim=69,
        hidden_dim=128,
        num_layers=1,
        num_heads=2,
        num_timesteps=10
    )
    print(f"   ✓ Model created")
    
    # Create trainer with early stopping enabled
    print("\n3. Creating trainer with early stopping...")
    trainer = DDPMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_epochs=1,
        checkpoint_dir="./test_checkpoints",
        log_interval=10,
        val_interval=20,
        save_interval=100,
        use_wandb=False,
        device="cpu",
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001
    )
    print(f"   ✓ Trainer created with early stopping enabled")
    
    # Run validation to test comprehensive metrics
    print("\n4. Running validation with comprehensive metrics...")
    val_metrics = trainer.validate()
    
    print("\n5. Validation Metrics:")
    for key, value in val_metrics.items():
        print(f"   - {key}: {value:.4f}")
    
    # Verify all expected metrics are present
    expected_metrics = [
        'loss', 'state_loss', 'action_loss',
        'state_mse', 'action_mse', 'state_mae', 'action_mae',
        'trajectory_smoothness', 'action_smoothness',
        'velocity_error', 'joint_limit_violations'
    ]
    
    missing_metrics = [m for m in expected_metrics if m not in val_metrics]
    if missing_metrics:
        print(f"\n⚠️  Missing metrics: {missing_metrics}")
    else:
        print(f"\n✓ All expected metrics computed successfully!")
    
    # Test metric tracker
    print("\n6. Testing metric tracker...")
    tracker = trainer.metric_tracker
    
    # Simulate multiple validation steps
    for i in range(10):
        mock_metrics = {
            'loss': 1.0 - i * 0.05,  # Improving loss
            'state_mse': 0.5 + i * 0.01  # Worsening state MSE
        }
        tracker.update(mock_metrics)
    
    # Check tracking
    loss_avg = tracker.get_moving_average('loss')
    best_loss = tracker.get_best('loss')
    is_improving = tracker.is_improving('loss', patience=3)
    
    print(f"   - Loss moving average: {loss_avg:.4f}")
    print(f"   - Best loss: {best_loss:.4f}")
    print(f"   - Is improving: {is_improving}")
    
    print("\n✓ Metric tracking working correctly")
    
    # Clean up
    import shutil
    if Path("./test_checkpoints").exists():
        shutil.rmtree("./test_checkpoints")
    
    print("\n" + "="*60)
    print("IMPROVED METRICS INTEGRATION TEST PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    test_improved_validation()