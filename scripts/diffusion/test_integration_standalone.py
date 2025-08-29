#!/usr/bin/env python3
"""Standalone test for Isaac Lab integration components."""

import torch
import sys
from pathlib import Path
import logging

# Add specific modules to path
project_root = Path(__file__).resolve().parents[2]
diffusion_path = project_root / "source/whole_body_tracking/whole_body_tracking/diffusion"
sys.path.insert(0, str(diffusion_path.parent))

# Import only diffusion modules
from diffusion.integration.space_converters import (
    ObservationConverter,
    ActionConverter,
    ObservationSpaceConfig,
)
from diffusion.optimization.performance import (
    PerformanceOptimizer,
    PerformanceMetrics,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_observation_converter():
    """Test observation space conversion."""
    logger.info("\nTesting ObservationConverter...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    num_joints = 19
    
    # Create converter
    config = ObservationSpaceConfig(num_joints=num_joints)
    converter = ObservationConverter(config)
    
    # Create mock observation
    obs_dim = converter.config.total_obs_dim
    policy_obs = torch.randn(batch_size, obs_dim, device=device)
    isaac_obs = {"policy": policy_obs}
    
    # Convert
    diffusion_state = converter.isaac_to_diffusion(isaac_obs)
    
    # Verify
    assert diffusion_state.shape == (batch_size, config.body_pos_state_dim)
    logger.info(f"✓ Observation conversion: {policy_obs.shape} → {diffusion_state.shape}")
    
    return True


def test_action_converter():
    """Test action space conversion."""
    logger.info("\nTesting ActionConverter...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    num_actions = 19
    
    # Create converter
    converter = ActionConverter(num_actions=num_actions)
    
    # Test diffusion to Isaac
    diffusion_action = torch.randn(batch_size, num_actions, device=device)
    isaac_action = converter.diffusion_to_isaac(diffusion_action)
    
    assert isaac_action.shape == (batch_size, num_actions)
    logger.info(f"✓ Action conversion: {diffusion_action.shape} → {isaac_action.shape}")
    
    # Test padding
    small_action = torch.randn(batch_size, 10, device=device)
    padded_action = converter.diffusion_to_isaac(small_action)
    
    assert padded_action.shape == (batch_size, num_actions)
    assert torch.all(padded_action[:, 10:] == 0)
    logger.info(f"✓ Action padding: {small_action.shape} → {padded_action.shape}")
    
    return True


def test_performance_optimizer():
    """Test performance optimization."""
    logger.info("\nTesting PerformanceOptimizer...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create optimizer
    optimizer = PerformanceOptimizer(
        target_latency_ms=20.0,
        device=device,
    )
    
    # Test metrics
    metrics = PerformanceMetrics(
        inference_time_ms=15.0,
        preprocessing_time_ms=2.0,
        postprocessing_time_ms=1.0,
        total_time_ms=18.0,
        memory_usage_mb=256.0,
        batch_size=1,
    )
    
    assert metrics.meets_target
    assert metrics.frequency_hz > 50.0
    logger.info(f"✓ Performance metrics: {metrics.total_time_ms}ms @ {metrics.frequency_hz:.1f}Hz")
    
    # Test model optimization
    model = torch.nn.Sequential(
        torch.nn.Linear(48, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 19),
    )
    
    optimized_model = optimizer.optimize_model(model)
    
    # Check optimizations
    assert not any(p.requires_grad for p in optimized_model.parameters())
    logger.info(f"✓ Model optimization completed")
    
    return True


def test_history_creation():
    """Test history tensor creation."""
    logger.info("\nTesting history tensor creation...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    num_joints = 19
    history_len = 4
    
    # Create converter
    config = ObservationSpaceConfig(num_joints=num_joints)
    converter = ObservationConverter(config)
    
    # Create observation and action buffers
    obs_buffer = []
    action_buffer = []
    
    for _ in range(history_len):
        obs_dim = converter.config.total_obs_dim
        policy_obs = torch.randn(batch_size, obs_dim, device=device)
        obs_buffer.append({"policy": policy_obs})
        
        action = torch.randn(batch_size, num_joints, device=device)
        action_buffer.append(action)
    
    # Create history
    history = converter.create_history_tensor(obs_buffer, action_buffer)
    
    expected_dim = config.body_pos_state_dim + num_joints
    assert history.shape == (batch_size, history_len, expected_dim)
    logger.info(f"✓ History tensor: {history.shape}")
    
    return True


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("ISAAC LAB INTEGRATION COMPONENT TESTS")
    logger.info("="*60)
    
    tests = [
        ("Observation Converter", test_observation_converter),
        ("Action Converter", test_action_converter),
        ("Performance Optimizer", test_performance_optimizer),
        ("History Creation", test_history_creation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                logger.error(f"✗ {name} failed")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {name} failed with error: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.info("✗ SOME TESTS FAILED")
    logger.info("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)