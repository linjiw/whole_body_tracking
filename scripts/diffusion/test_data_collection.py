"""Test script for data collection pipeline.

This script tests the data collection and dataset functionality
without requiring Isaac Sim, using mock data generation.
"""

import sys
import os

# Add path to directly import diffusion modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/whole_body_tracking/whole_body_tracking"))

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import directly from diffusion submodules to avoid isaaclab dependencies
from diffusion.data.trajectory_dataset import (
    TrajectoryDataset,
    Trajectory,
    StateRepresentation
)
from diffusion.data.data_collection import (
    MotionDataCollector,
    DataCollectionConfig,
    ActionDelayBuffer
)


def test_state_representation():
    """Test StateRepresentation class."""
    print("\n" + "="*50)
    print("Testing StateRepresentation...")
    
    # Create a state representation
    state = StateRepresentation(
        root_pos=torch.randn(3),
        root_vel=torch.randn(3),
        root_rot=torch.randn(3),
        body_positions=torch.randn(23, 3),
        body_velocities=torch.randn(23, 3)
    )
    
    # Test conversion to/from tensor
    tensor = state.to_tensor()
    print(f"State dimension: {state.dim}")
    print(f"Tensor shape: {tensor.shape}")
    
    # Test reconstruction
    state_reconstructed = StateRepresentation.from_tensor(tensor, num_bodies=23)
    
    # Check if reconstruction is correct
    assert torch.allclose(state.root_pos, state_reconstructed.root_pos)
    assert torch.allclose(state.body_positions, state_reconstructed.body_positions)
    print("✓ State representation conversion works correctly")


def test_trajectory():
    """Test Trajectory class."""
    print("\n" + "="*50)
    print("Testing Trajectory...")
    
    # Create a trajectory
    traj = Trajectory(
        history_states=torch.randn(5, 165),  # N+1 states
        history_actions=torch.randn(4, 69),  # N actions
        future_states=torch.randn(32, 165),  # H states
        future_actions=torch.randn(16, 69),  # H actions
        motion_id="test_motion",
        timestep=100,
        success=True
    )
    
    # Test observation history
    hist_states, hist_actions = traj.get_observation_history()
    print(f"History states shape: {hist_states.shape}")
    print(f"History actions shape: {hist_actions.shape}")
    
    # Test future trajectory
    future_states, future_actions = traj.get_future_trajectory()
    print(f"Future states shape: {future_states.shape}")
    print(f"Future actions shape: {future_actions.shape}")
    
    # Test interleaved indices
    indices = traj.get_interleaved_indices()
    print(f"Interleaved indices (first 5): {indices[:5]}")
    print("✓ Trajectory handling works correctly")


def test_data_collection():
    """Test data collection with mock data."""
    print("\n" + "="*50)
    print("Testing Data Collection...")
    
    # Create configuration
    cfg = DataCollectionConfig(
        history_length=4,
        future_length_states=32,
        future_length_actions=16,
        episodes_per_motion=5,
        max_episode_length=200
    )
    
    # Create collector (without env/policies, will use mock data)
    collector = MotionDataCollector(cfg=cfg)
    
    # Collect mock trajectories
    dataset = collector.collect_trajectories(
        num_episodes=3,
        add_noise=True
    )
    
    print(f"Collected {len(dataset)} trajectories")
    print(f"State stats - mean: {dataset.state_mean:.3f}, std: {dataset.state_std:.3f}")
    print(f"Action stats - mean: {dataset.action_mean:.3f}, std: {dataset.action_std:.3f}")
    
    # Test dataset indexing
    sample = dataset[0]
    print(f"\nFirst sample keys: {sample.keys()}")
    print(f"History states shape: {sample['history_states'].shape}")
    print(f"Future actions shape: {sample['future_actions'].shape}")
    print(f"Motion ID: {sample['motion_id']}")
    print("✓ Data collection works correctly")


def test_dataloader():
    """Test PyTorch DataLoader compatibility."""
    print("\n" + "="*50)
    print("Testing DataLoader...")
    
    # Create a small dataset
    cfg = DataCollectionConfig()
    collector = MotionDataCollector(cfg=cfg)
    dataset = collector.collect_trajectories(num_episodes=2, add_noise=False)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    # Test iteration
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  History states shape: {batch['history_states'].shape}")
        print(f"  Future states shape: {batch['future_states'].shape}")
        print(f"  Success rate in batch: {batch['success'].float().mean():.2f}")
        
        if batch_idx >= 1:  # Test just a couple batches
            break
    
    print("✓ DataLoader compatibility works correctly")


def test_save_load():
    """Test dataset save/load functionality."""
    print("\n" + "="*50)
    print("Testing Save/Load...")
    
    # Create and save dataset
    cfg = DataCollectionConfig()
    collector = MotionDataCollector(cfg=cfg)
    dataset1 = collector.collect_trajectories(num_episodes=1)
    
    # Save dataset
    save_path = "/tmp/test_dataset.pt"
    dataset1.save(save_path)
    print(f"Saved dataset to {save_path}")
    
    # Load dataset
    dataset2 = TrajectoryDataset.load(save_path)
    print(f"Loaded dataset with {len(dataset2)} trajectories")
    
    # Verify statistics are preserved
    assert torch.allclose(dataset1.state_mean, dataset2.state_mean)
    assert torch.allclose(dataset1.action_std, dataset2.action_std)
    print("✓ Save/load functionality works correctly")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)


def test_action_delay_buffer():
    """Test action delay buffer functionality."""
    print("\n" + "="*50)
    print("Testing Action Delay Buffer...")
    
    # ActionDelayBuffer is already imported at the top
    buffer = ActionDelayBuffer(max_delay_steps=3)
    
    # Test adding actions with delay
    action1 = torch.ones(69) * 1.0
    action2 = torch.ones(69) * 2.0
    action3 = torch.ones(69) * 3.0
    
    buffer.add_action(action1, delay_steps=0)  # No delay
    delayed1 = buffer.get_delayed_action()
    assert torch.allclose(delayed1, action1)
    
    buffer.reset()
    buffer.add_action(action2, delay_steps=2)  # 2-step delay
    delayed2_1 = buffer.get_delayed_action()
    assert torch.allclose(delayed2_1, torch.zeros(69))  # Should get zeros first
    delayed2_2 = buffer.get_delayed_action()
    assert torch.allclose(delayed2_2, torch.zeros(69))  # Still zeros
    delayed2_3 = buffer.get_delayed_action()
    assert torch.allclose(delayed2_3, action2)  # Now get the actual action
    
    print("✓ Action delay buffer works correctly")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING DIFFUSION DATA COLLECTION PIPELINE")
    print("="*60)
    
    try:
        test_state_representation()
        test_trajectory()
        test_data_collection()
        test_dataloader()
        test_save_load()
        test_action_delay_buffer()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())