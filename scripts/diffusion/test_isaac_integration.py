#!/usr/bin/env python3
"""
Test script for Isaac Lab integration with diffusion policy.

This script tests the integration components without requiring
a full Isaac Sim launch, useful for debugging.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch

# Add project to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "source/whole_body_tracking"))

from whole_body_tracking.diffusion.integration import (
    ObservationConverter,
    ActionConverter,
    ObservationSpaceConfig,
)
from whole_body_tracking.diffusion.optimization.performance import (
    PerformanceOptimizer,
    PerformanceMetrics,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestObservationConverter(unittest.TestCase):
    """Test observation space conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 4
        self.num_joints = 19
        
        # Create converter
        config = ObservationSpaceConfig(num_joints=self.num_joints)
        self.converter = ObservationConverter(config)
    
    def test_isaac_to_diffusion_conversion(self):
        """Test conversion from Isaac Lab to diffusion format."""
        # Create mock Isaac Lab observation
        obs_dim = self.converter.config.total_obs_dim
        policy_obs = torch.randn(self.batch_size, obs_dim, device=self.device)
        
        isaac_obs = {"policy": policy_obs}
        
        # Convert
        diffusion_state = self.converter.isaac_to_diffusion(isaac_obs)
        
        # Check output
        self.assertEqual(diffusion_state.shape[0], self.batch_size)
        self.assertEqual(diffusion_state.shape[1], self.converter.config.body_pos_state_dim)
        self.assertEqual(diffusion_state.device.type, self.device)
        
        logger.info(f"✓ Isaac to diffusion conversion: {diffusion_state.shape}")
    
    def test_history_tensor_creation(self):
        """Test observation history tensor creation."""
        # Create mock observation buffer
        obs_buffer = []
        action_buffer = []
        
        for _ in range(4):  # History length
            obs_dim = self.converter.config.total_obs_dim
            policy_obs = torch.randn(self.batch_size, obs_dim, device=self.device)
            obs_buffer.append({"policy": policy_obs})
            
            action = torch.randn(self.batch_size, self.num_joints, device=self.device)
            action_buffer.append(action)
        
        # Create history tensor
        history = self.converter.create_history_tensor(obs_buffer, action_buffer)
        
        # Check shape
        expected_dim = self.converter.config.body_pos_state_dim + self.num_joints
        self.assertEqual(history.shape, (self.batch_size, 4, expected_dim))
        
        logger.info(f"✓ History tensor creation: {history.shape}")


class TestActionConverter(unittest.TestCase):
    """Test action space conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 4
        self.num_actions = 19
        
        # Create converter
        self.converter = ActionConverter(num_actions=self.num_actions)
    
    def test_diffusion_to_isaac_conversion(self):
        """Test conversion from diffusion to Isaac Lab format."""
        # Create mock diffusion action
        diffusion_action = torch.randn(
            self.batch_size,
            self.num_actions,
            device=self.device,
        )
        
        # Convert
        isaac_action = self.converter.diffusion_to_isaac(diffusion_action)
        
        # Check output
        self.assertEqual(isaac_action.shape, (self.batch_size, self.num_actions))
        self.assertEqual(isaac_action.device.type, self.device)
        
        logger.info(f"✓ Diffusion to Isaac action conversion: {isaac_action.shape}")
    
    def test_action_padding(self):
        """Test action padding when dimensions don't match."""
        # Create action with fewer dimensions
        small_action = torch.randn(self.batch_size, 10, device=self.device)
        
        # Convert (should pad)
        isaac_action = self.converter.diffusion_to_isaac(small_action)
        
        # Check padding
        self.assertEqual(isaac_action.shape, (self.batch_size, self.num_actions))
        # Check that padding is zeros
        self.assertTrue(torch.all(isaac_action[:, 10:] == 0))
        
        logger.info(f"✓ Action padding: {small_action.shape} → {isaac_action.shape}")


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = PerformanceOptimizer(
            target_latency_ms=20.0,
            device=self.device,
        )
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        metrics = PerformanceMetrics(
            inference_time_ms=15.0,
            preprocessing_time_ms=2.0,
            postprocessing_time_ms=1.0,
            total_time_ms=18.0,
            memory_usage_mb=256.0,
            batch_size=1,
        )
        
        # Check frequency calculation
        self.assertAlmostEqual(metrics.frequency_hz, 1000.0 / 18.0, places=1)
        
        # Check target meeting
        self.assertTrue(metrics.meets_target)
        
        logger.info(f"✓ Performance metrics: {metrics.total_time_ms}ms @ {metrics.frequency_hz:.1f}Hz")
    
    def test_model_optimization(self):
        """Test model optimization."""
        # Create mock model
        model = torch.nn.Sequential(
            torch.nn.Linear(48, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 19),
        )
        
        # Optimize
        optimized_model = self.optimizer.optimize_model(model)
        
        # Check optimizations applied
        self.assertEqual(next(optimized_model.parameters()).device.type, self.device)
        self.assertFalse(any(p.requires_grad for p in optimized_model.parameters()))
        
        logger.info(f"✓ Model optimization completed")
    
    def test_profiling(self):
        """Test inference profiling."""
        # Create simple model
        model = torch.nn.Linear(48, 19).to(self.device)
        model.eval()
        
        # Create input
        inputs = {
            "x": torch.randn(1, 48, device=self.device)
        }
        
        # Mock forward pass for compatibility
        original_forward = model.forward
        model.forward = lambda x: original_forward(x)
        
        # Create wrapper for dict input
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model(x)
        
        wrapped_model = ModelWrapper(model)
        
        # Profile
        metrics = self.optimizer.profile_inference(
            wrapped_model,
            inputs,
            num_warmup=2,
            num_iterations=10,
        )
        
        # Check metrics
        self.assertGreater(metrics.inference_time_ms, 0)
        self.assertGreater(metrics.frequency_hz, 0)
        
        logger.info(f"✓ Profiling: {metrics.inference_time_ms:.2f}ms")


class TestIntegrationMock(unittest.TestCase):
    """Test full integration with mocked Isaac Lab environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_envs = 2
        self.num_joints = 19
    
    @patch('whole_body_tracking.diffusion.integration.isaac_lab_wrapper.ManagerBasedRLEnv')
    def test_wrapper_integration(self, MockEnv):
        """Test IsaacLabDiffusionWrapper with mocked environment."""
        # Create mock environment
        mock_env = MagicMock()
        mock_env.num_envs = self.num_envs
        mock_env.device = self.device
        
        # Mock observations
        obs_dim = ObservationSpaceConfig(num_joints=self.num_joints).total_obs_dim
        mock_obs = {
            "policy": torch.randn(self.num_envs, obs_dim, device=self.device)
        }
        mock_env.get_observations.return_value = (mock_obs, {})
        mock_env.reset.return_value = (mock_obs, {})
        
        # Mock step
        mock_env.step.return_value = (
            mock_obs,  # obs
            torch.zeros(self.num_envs),  # reward
            torch.zeros(self.num_envs, dtype=torch.bool),  # terminated
            torch.zeros(self.num_envs, dtype=torch.bool),  # truncated
            {},  # info
        )
        
        # Create mock diffusion model
        from whole_body_tracking.diffusion.models.diffusion_model import (
            StateActionDiffusionModel
        )
        
        mock_model = MagicMock(spec=StateActionDiffusionModel)
        mock_model.state_dim = 48
        mock_model.action_dim = 19
        mock_model.future_length_states = 16
        mock_model.future_length_actions = 16
        mock_model.num_timesteps = 100
        
        # Test wrapper creation
        from whole_body_tracking.diffusion.integration.isaac_lab_wrapper import (
            IsaacLabDiffusionWrapper
        )
        
        try:
            wrapper = IsaacLabDiffusionWrapper(
                env=mock_env,
                diffusion_model=mock_model,
                device=self.device,
            )
            
            # Test reset
            obs = wrapper.reset()
            self.assertIn("policy", obs)
            
            # Test step
            obs, reward, terminated, truncated, info = wrapper.step()
            self.assertEqual(reward.shape, (self.num_envs,))
            
            logger.info(f"✓ Wrapper integration test passed")
            
        except Exception as e:
            logger.error(f"Wrapper integration failed: {e}")
            raise


def run_all_tests():
    """Run all integration tests."""
    logger.info("="*60)
    logger.info("ISAAC LAB INTEGRATION TESTS")
    logger.info("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestObservationConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestActionConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationMock))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("\n" + "="*60)
    if result.wasSuccessful():
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.info(f"✗ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    logger.info("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)