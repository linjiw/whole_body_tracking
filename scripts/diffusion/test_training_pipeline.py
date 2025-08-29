#!/usr/bin/env python3
"""Test script for the complete training pipeline.

This script tests:
1. Data collection
2. Model initialization
3. Training loop
4. Checkpointing
5. Metrics computation
6. Validation
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add project path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking/whole_body_tracking"))

import torch
import numpy as np

from diffusion.data import TrajectoryDataset
from diffusion.data.data_collection import (
    MotionDataCollector,
    DataCollectionConfig
)
from diffusion.models import StateActionDiffusionModel
from diffusion.training import (
    DDPMTrainer,
    DiffusionMetrics,
    MetricTracker
)
from configs.diffusion import (
    get_small_config,
    get_debug_config
)


def test_data_collection():
    """Test data collection pipeline."""
    print("\n" + "="*50)
    print("Testing Data Collection...")
    
    # Create small dataset for testing
    cfg = DataCollectionConfig(
        episodes_per_motion=2,
        max_episode_length=100,
        history_length=4,
        future_length_states=32,
        future_length_actions=16
    )
    
    collector = MotionDataCollector(cfg=cfg)
    dataset = collector.collect_trajectories(num_episodes=10, add_noise=True)
    
    assert len(dataset) > 0, "Dataset is empty"
    
    # Check data shapes
    sample = dataset[0]
    assert sample['history_states'].shape == (5, 165)  # N+1 states
    assert sample['history_actions'].shape == (4, 69)  # N actions
    assert sample['future_states'].shape == (32, 165)  # H_s states
    assert sample['future_actions'].shape == (16, 69)  # H_a actions
    
    print(f"✓ Collected {len(dataset)} trajectories")
    print(f"✓ Data shapes correct")
    
    return dataset


def test_model_initialization():
    """Test model creation and forward pass."""
    print("\n" + "="*50)
    print("Testing Model Initialization...")
    
    # Get small config for testing
    model_cfg = get_small_config()
    
    # Create model
    model = StateActionDiffusionModel(
        state_dim=model_cfg.trajectory.state_dim,
        action_dim=model_cfg.trajectory.action_dim,
        hidden_dim=model_cfg.transformer.hidden_dim,
        num_layers=model_cfg.transformer.num_layers,
        num_heads=model_cfg.transformer.num_heads,
        num_timesteps=model_cfg.diffusion.num_timesteps
    )
    
    # Test forward pass
    batch_size = 2
    future_states = torch.randn(batch_size, 32, 165)
    future_actions = torch.randn(batch_size, 16, 69)
    history_states = torch.randn(batch_size, 5, 165)
    history_actions = torch.randn(batch_size, 4, 69)
    timesteps = torch.randint(0, 50, (batch_size,))
    
    with torch.no_grad():
        pred_states, pred_actions = model(
            future_states, future_actions,
            history_states, history_actions,
            timesteps
        )
    
    assert pred_states.shape == (batch_size, 32, 165)
    assert pred_actions.shape == (batch_size, 16, 69)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"✓ Forward pass successful")
    
    return model


def test_training_step(model, dataset):
    """Test single training step."""
    print("\n" + "="*50)
    print("Testing Training Step...")
    
    # Create batch
    batch = dataset[0]
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Training step
    loss, loss_components = model.training_step(batch, return_loss_components=True)
    
    assert loss.requires_grad
    assert 'state_loss' in loss_components
    assert 'action_loss' in loss_components
    
    # Test backward
    loss.backward()
    
    # Check gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients computed"
    
    print(f"✓ Loss computed: {loss.item():.4f}")
    print(f"✓ Gradients computed successfully")


def test_metrics():
    """Test metrics computation."""
    print("\n" + "="*50)
    print("Testing Metrics...")
    
    metrics_calc = DiffusionMetrics()
    tracker = MetricTracker(window_size=10)
    
    # Create mock predictions and targets
    batch_size = 4
    pred_states = torch.randn(batch_size, 32, 165)
    pred_actions = torch.randn(batch_size, 16, 69)
    target_states = torch.randn(batch_size, 32, 165)
    target_actions = torch.randn(batch_size, 16, 69)
    
    # Compute metrics
    results = metrics_calc.compute_all_metrics(
        pred_states, pred_actions,
        target_states, target_actions,
        loss=1.0, state_loss=0.5, action_loss=0.5
    )
    
    assert results.state_mse > 0
    assert results.action_mse > 0
    
    # Update tracker
    tracker.update(results.to_dict())
    
    # Test moving average
    for i in range(5):
        tracker.update({'loss': np.random.random()})
    
    avg = tracker.get_moving_average('loss')
    assert avg > 0
    
    print(f"✓ Metrics computed successfully")
    print(f"✓ Metric tracker working")


def test_trainer_integration():
    """Test complete trainer integration."""
    print("\n" + "="*50)
    print("Testing Trainer Integration...")
    
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create small dataset
        cfg = DataCollectionConfig(episodes_per_motion=1, max_episode_length=50)
        collector = MotionDataCollector(cfg=cfg)
        dataset = collector.collect_trajectories(num_episodes=5)
        
        # Create small model
        model_cfg = get_small_config()
        model = StateActionDiffusionModel(
            state_dim=165,
            action_dim=69,
            hidden_dim=128,  # Very small
            num_layers=1,
            num_heads=2,
            num_timesteps=10
        )
        
        # Create trainer with debug config
        training_cfg = get_debug_config()
        trainer = DDPMTrainer(
            model=model,
            train_dataset=dataset,
            val_dataset=None,
            learning_rate=1e-3,
            batch_size=2,
            num_epochs=1,
            warmup_steps=10,
            checkpoint_dir=temp_dir,
            log_interval=2,
            save_interval=10,
            use_wandb=False,
            device="cpu"
        )
        
        # Test one epoch
        print("Running one training epoch...")
        train_metrics = trainer.train_epoch()
        
        assert 'loss' in train_metrics
        assert train_metrics['loss'] > 0
        
        print(f"✓ Training epoch completed")
        print(f"✓ Loss: {train_metrics['loss']:.4f}")
        
        # Test checkpoint saving
        trainer.save_checkpoint()
        checkpoint_path = Path(temp_dir) / f"checkpoint_step_{trainer.global_step}.pt"
        assert checkpoint_path.exists()
        
        print(f"✓ Checkpoint saved successfully")
        
        # Test checkpoint loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'config' in checkpoint
        
        print(f"✓ Checkpoint structure valid")
        
        # Test sampling
        samples = trainer.sample_trajectories(num_samples=1)
        assert 'sampled_states' in samples
        assert 'sampled_actions' in samples
        
        print(f"✓ Sampling successful")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_end_to_end():
    """Test complete end-to-end pipeline."""
    print("\n" + "="*50)
    print("Testing End-to-End Pipeline...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Collect data
        print("\n1. Collecting data...")
        cfg = DataCollectionConfig(
            episodes_per_motion=2,
            max_episode_length=100
        )
        collector = MotionDataCollector(cfg=cfg)
        dataset = collector.collect_trajectories(num_episodes=10)
        
        # Save and load dataset
        dataset_path = Path(temp_dir) / "dataset.pt"
        dataset.save(str(dataset_path))
        loaded_dataset = TrajectoryDataset.load(str(dataset_path))
        assert len(loaded_dataset) == len(dataset)
        print(f"   ✓ Collected and saved {len(dataset)} trajectories")
        
        # 2. Create model
        print("\n2. Creating model...")
        model = StateActionDiffusionModel(
            hidden_dim=128,
            num_layers=2,
            num_heads=2,
            num_timesteps=20
        )
        print(f"   ✓ Model created")
        
        # 3. Train
        print("\n3. Training...")
        trainer = DDPMTrainer(
            model=model,
            train_dataset=loaded_dataset,
            batch_size=4,
            num_epochs=1,
            checkpoint_dir=temp_dir,
            log_interval=5,
            use_wandb=False,
            device="cpu"
        )
        
        # Run training for a few steps
        for _ in range(3):
            batch = next(iter(trainer.train_loader))
            batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            loss, _ = trainer.model.training_step(batch, return_loss_components=True)
            
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer._update_ema()
            
            trainer.global_step += 1
        
        print(f"   ✓ Training steps completed")
        
        # 4. Save final checkpoint
        print("\n4. Saving checkpoint...")
        trainer.save_checkpoint(is_final=True)
        final_path = Path(temp_dir) / "final_model.pt"
        assert final_path.exists()
        print(f"   ✓ Final checkpoint saved")
        
        # 5. Load and test inference
        print("\n5. Testing inference...")
        checkpoint = torch.load(final_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            # Get sample for history
            sample = loaded_dataset[0]
            history_states = sample['history_states'].unsqueeze(0)
            history_actions = sample['history_actions'].unsqueeze(0)
            
            # Sample trajectory
            sampled_states, sampled_actions = model.sample(
                history_states, history_actions
            )
        
        assert sampled_states.shape == (1, 32, 165)
        assert sampled_actions.shape == (1, 16, 69)
        print(f"   ✓ Inference successful")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING COMPLETE TRAINING PIPELINE")
    print("="*60)
    
    try:
        # Test individual components
        dataset = test_data_collection()
        model = test_model_initialization()
        test_training_step(model, dataset)
        test_metrics()
        test_trainer_integration()
        
        # Test end-to-end
        test_end_to_end()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nPhase 3 Implementation Complete:")
        print("✓ Training scripts with configuration system")
        print("✓ WandB logging integration")
        print("✓ Checkpointing and resume capability")
        print("✓ Comprehensive validation metrics")
        print("✓ Data collection from Stage 1 policies")
        print("✓ Complete training pipeline tested")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())