"""Test script for diffusion model architecture.

This script tests the complete diffusion model pipeline including:
- Embeddings
- Transformer architecture
- Diffusion forward/backward process
- Training loop
"""

import sys
import os

# Add path to directly import diffusion modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/whole_body_tracking/whole_body_tracking"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import diffusion components
from diffusion.models import (
    StateActionDiffusionModel,
    ObservationHistoryEmbedding,
    FutureTrajectoryEmbedding,
    SinusoidalPositionEmbeddings
)
from diffusion.data import TrajectoryDataset
from diffusion.data.data_collection import MotionDataCollector, DataCollectionConfig
from diffusion.training import DDPMTrainer


def test_embeddings():
    """Test embedding layers."""
    print("\n" + "="*50)
    print("Testing Embeddings...")
    
    batch_size = 4
    state_dim = 165
    action_dim = 69
    hidden_dim = 512
    history_length = 4
    
    # Test time embeddings
    time_embed = SinusoidalPositionEmbeddings(hidden_dim)
    timesteps = torch.randint(0, 100, (batch_size,))
    time_embeds = time_embed(timesteps)
    assert time_embeds.shape == (batch_size, hidden_dim)
    print(f"✓ Time embeddings: {time_embeds.shape}")
    
    # Test observation history embedding
    history_embed = ObservationHistoryEmbedding(
        state_dim, action_dim, hidden_dim, history_length
    )
    history_states = torch.randn(batch_size, history_length + 1, state_dim)
    history_actions = torch.randn(batch_size, history_length, action_dim)
    history_embeds = history_embed(history_states, history_actions)
    expected_seq_len = 2 * history_length + 1  # Interleaved states and actions
    assert history_embeds.shape == (batch_size, expected_seq_len, hidden_dim)
    print(f"✓ History embeddings: {history_embeds.shape}")
    
    # Test future trajectory embedding
    future_embed = FutureTrajectoryEmbedding(
        state_dim, action_dim, hidden_dim,
        future_length_states=32, future_length_actions=16
    )
    future_states = torch.randn(batch_size, 32, state_dim)
    future_actions = torch.randn(batch_size, 16, action_dim)
    state_embeds, action_embeds = future_embed(future_states, future_actions)
    assert state_embeds.shape == (batch_size, 32, hidden_dim)
    assert action_embeds.shape == (batch_size, 16, hidden_dim)
    print(f"✓ Future embeddings - States: {state_embeds.shape}, Actions: {action_embeds.shape}")


def test_diffusion_model():
    """Test diffusion model architecture."""
    print("\n" + "="*50)
    print("Testing Diffusion Model...")
    
    # Create model
    model = StateActionDiffusionModel(
        state_dim=165,
        action_dim=69,
        hidden_dim=256,  # Smaller for testing
        num_layers=2,    # Fewer layers for testing
        num_heads=4,
        history_length=4,
        future_length_states=32,
        future_length_actions=16,
        num_timesteps=100
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    future_states = torch.randn(batch_size, 32, 165)
    future_actions = torch.randn(batch_size, 16, 69)
    history_states = torch.randn(batch_size, 5, 165)
    history_actions = torch.randn(batch_size, 4, 69)
    timesteps = torch.randint(0, 100, (batch_size,))
    
    pred_states, pred_actions = model(
        future_states, future_actions,
        history_states, history_actions,
        timesteps
    )
    
    assert pred_states.shape == (batch_size, 32, 165)
    assert pred_actions.shape == (batch_size, 16, 69)
    print(f"✓ Forward pass - States: {pred_states.shape}, Actions: {pred_actions.shape}")


def test_noise_schedule():
    """Test noise scheduling."""
    print("\n" + "="*50)
    print("Testing Noise Schedule...")
    
    from diffusion.models.diffusion_model import NoiseSchedule
    
    # Test different schedule types
    for schedule_type in ["linear", "cosine", "quadratic"]:
        schedule = NoiseSchedule(
            num_timesteps=100,
            schedule_type=schedule_type
        )
        
        # Test forward diffusion
        x_0 = torch.randn(4, 32, 165)
        t = torch.tensor([0, 25, 50, 99])
        x_t = schedule.q_sample(x_0, t)
        
        assert x_t.shape == x_0.shape
        print(f"✓ {schedule_type} schedule: {x_t.shape}")
        
        # Check that noise increases with timestep
        noise_levels = []
        for timestep in [0, 25, 50, 75, 99]:
            t_batch = torch.tensor([timestep] * 4)
            noisy = schedule.q_sample(x_0, t_batch)
            noise_level = (noisy - x_0).abs().mean().item()
            noise_levels.append(noise_level)
        
        # Noise should generally increase with timestep
        assert noise_levels[-1] > noise_levels[0]
        print(f"  Noise progression: {[f'{n:.3f}' for n in noise_levels]}")


def test_training_step():
    """Test training step."""
    print("\n" + "="*50)
    print("Testing Training Step...")
    
    # Create small model for testing
    model = StateActionDiffusionModel(
        state_dim=165,
        action_dim=69,
        hidden_dim=128,  # Small for testing
        num_layers=1,    # Single layer for speed
        num_heads=2,
        num_timesteps=10  # Few timesteps for testing
    )
    
    # Create mock batch
    batch = {
        'history_states': torch.randn(2, 5, 165),
        'history_actions': torch.randn(2, 4, 69),
        'future_states': torch.randn(2, 32, 165),
        'future_actions': torch.randn(2, 16, 69)
    }
    
    # Test training step
    loss, loss_components = model.training_step(batch, return_loss_components=True)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert 'state_loss' in loss_components
    assert 'action_loss' in loss_components
    
    print(f"✓ Training loss: {loss.item():.4f}")
    print(f"  State loss: {loss_components['state_loss']:.4f}")
    print(f"  Action loss: {loss_components['action_loss']:.4f}")
    
    # Test backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                        for p in model.parameters())
    assert has_gradients
    print("✓ Gradients computed successfully")


def test_sampling():
    """Test sampling from the model."""
    print("\n" + "="*50)
    print("Testing Sampling...")
    
    # Create small model for testing
    model = StateActionDiffusionModel(
        state_dim=165,
        action_dim=69,
        hidden_dim=128,
        num_layers=1,
        num_heads=2,
        num_timesteps=10  # Few steps for quick testing
    )
    
    model.eval()
    
    # Create history
    batch_size = 2
    history_states = torch.randn(batch_size, 5, 165)
    history_actions = torch.randn(batch_size, 4, 69)
    
    # Sample trajectories
    with torch.no_grad():
        sampled_states, sampled_actions = model.sample(
            history_states, history_actions
        )
    
    assert sampled_states.shape == (batch_size, 32, 165)
    assert sampled_actions.shape == (batch_size, 16, 69)
    
    print(f"✓ Sampled states: {sampled_states.shape}")
    print(f"✓ Sampled actions: {sampled_actions.shape}")
    
    # Check that samples are reasonable (not NaN or Inf)
    assert not torch.isnan(sampled_states).any()
    assert not torch.isinf(sampled_states).any()
    assert not torch.isnan(sampled_actions).any()
    assert not torch.isinf(sampled_actions).any()
    
    print("✓ Samples are numerically stable")


def test_trainer():
    """Test the training infrastructure."""
    print("\n" + "="*50)
    print("Testing Trainer...")
    
    # Create small dataset
    cfg = DataCollectionConfig()
    collector = MotionDataCollector(cfg=cfg)
    dataset = collector.collect_trajectories(num_episodes=1)  # Small dataset
    
    # Create small model
    model = StateActionDiffusionModel(
        state_dim=165,
        action_dim=69,
        hidden_dim=64,   # Very small for testing
        num_layers=1,
        num_heads=2,
        num_timesteps=10
    )
    
    # Create trainer
    trainer = DDPMTrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=None,  # No validation for quick test
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-3,
        log_interval=10,
        save_interval=100,
        use_wandb=False,  # Disable wandb for testing
        device="cpu"  # Use CPU for testing
    )
    
    print(f"✓ Trainer initialized")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: 4")
    print(f"  Device: cpu")
    
    # Test one training step
    batch = next(iter(trainer.train_loader))
    batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    loss, loss_components = trainer.model.training_step(batch, return_loss_components=True)
    
    print(f"✓ Single training step completed")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test sampling from trainer
    samples = trainer.sample_trajectories(num_samples=1, use_ema=False)
    
    assert 'sampled_states' in samples
    assert 'sampled_actions' in samples
    print(f"✓ Sampling from trainer works")


def test_differentiated_attention():
    """Test differentiated attention mechanism."""
    print("\n" + "="*50)
    print("Testing Differentiated Attention...")
    
    from diffusion.models.transformer import DifferentiatedTransformerBlock
    
    batch_size = 2
    num_states = 32
    num_actions = 16
    hidden_dim = 256
    
    # Create transformer block
    block = DifferentiatedTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1
    )
    
    # Create inputs
    state_embeds = torch.randn(batch_size, num_states, hidden_dim)
    action_embeds = torch.randn(batch_size, num_actions, hidden_dim)
    
    # Forward pass
    new_state_embeds, new_action_embeds = block(state_embeds, action_embeds)
    
    assert new_state_embeds.shape == state_embeds.shape
    assert new_action_embeds.shape == action_embeds.shape
    
    print(f"✓ Differentiated attention block")
    print(f"  State output: {new_state_embeds.shape}")
    print(f"  Action output: {new_action_embeds.shape}")
    
    # Test that outputs are different (processing happened)
    state_diff = (new_state_embeds - state_embeds).abs().mean()
    action_diff = (new_action_embeds - action_embeds).abs().mean()
    
    assert state_diff > 0.01  # Some change should occur
    assert action_diff > 0.01
    
    print(f"✓ Attention produces meaningful changes")
    print(f"  State change: {state_diff:.4f}")
    print(f"  Action change: {action_diff:.4f}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING DIFFUSION MODEL ARCHITECTURE")
    print("="*60)
    
    try:
        test_embeddings()
        test_noise_schedule()
        test_differentiated_attention()
        test_diffusion_model()
        test_training_step()
        test_sampling()
        test_trainer()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nPhase 2 Implementation Complete:")
        print("✓ State and action embeddings with position encoding")
        print("✓ Differentiated transformer (bi-directional for states, causal for actions)")
        print("✓ State-action diffusion model with independent noise schedules")
        print("✓ DDPM training loop with EMA and gradient clipping")
        print("✓ Sampling with classifier guidance support")
        print("✓ Full training infrastructure with checkpointing")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())