"""
Unit tests for AdaptiveSampler implementation.

Tests mathematical correctness of the adaptive sampling mechanism described
in the BeyondMimic paper Section III-F.
"""

import numpy as np
import torch
import pytest
import tempfile
import os

# Add the source directory to the path
import sys
sys.path.insert(0, '/home/linji/nfs/whole_body_tracking/source/whole_body_tracking')

from whole_body_tracking.tasks.tracking.mdp.adaptive_sampler import AdaptiveSampler


class TestAdaptiveSampler:
    """Test suite for AdaptiveSampler class."""
    
    @pytest.fixture
    def basic_sampler(self):
        """Create a basic adaptive sampler for testing."""
        return AdaptiveSampler(
            motion_fps=30,
            motion_duration=10.0,  # 10 seconds
            bin_size_seconds=1.0,
            gamma=0.9,
            lambda_uniform=0.1,
            K=5,
            alpha_smooth=0.1,
            device="cpu"
        )
    
    @pytest.fixture
    def short_sampler(self):
        """Create a short motion sampler for detailed testing."""
        return AdaptiveSampler(
            motion_fps=10,
            motion_duration=5.0,  # 5 seconds, 5 bins
            bin_size_seconds=1.0,
            gamma=0.8,
            lambda_uniform=0.2,
            K=3,
            alpha_smooth=0.2,
            device="cpu"
        )
    
    def test_initialization(self, basic_sampler):
        """Test proper initialization of sampler parameters."""
        assert basic_sampler.motion_fps == 30
        assert basic_sampler.motion_duration == 10.0
        assert basic_sampler.bin_size_seconds == 1.0
        assert basic_sampler.gamma == 0.9
        assert basic_sampler.lambda_uniform == 0.1
        assert basic_sampler.K == 5
        assert basic_sampler.alpha_smooth == 0.1
        
        # Check computed values
        assert basic_sampler.num_bins == 10  # 10 seconds / 1 second per bin
        assert basic_sampler.frames_per_bin == 30  # 30 fps * 1 second
        assert basic_sampler.total_frames == 300  # 30 fps * 10 seconds
        
        # Check tensor initialization
        assert basic_sampler.episode_counts.shape == (10,)
        assert basic_sampler.failure_counts.shape == (10,)
        assert basic_sampler.smoothed_failure_rates.shape == (10,)
        assert basic_sampler.sampling_probabilities.shape == (10,)
        
        # Check initial uniform probabilities
        expected_prob = 1.0 / 10
        assert torch.allclose(basic_sampler.sampling_probabilities, torch.full((10,), expected_prob))
    
    def test_frame_to_bin_conversion(self, basic_sampler):
        """Test conversion from frame indices to bin indices."""
        # Test edge cases and boundaries
        frame_indices = torch.tensor([0, 29, 30, 59, 60, 299])
        expected_bins = torch.tensor([0, 0, 1, 1, 2, 9])
        
        actual_bins = basic_sampler.frame_to_bin(frame_indices)
        assert torch.equal(actual_bins, expected_bins)
        
        # Test out-of-bounds frames (should be clamped)
        frame_indices = torch.tensor([300, 400, -1])
        expected_bins = torch.tensor([9, 9, 0])  # Clamped to valid range
        
        actual_bins = basic_sampler.frame_to_bin(frame_indices)
        assert torch.equal(actual_bins, expected_bins)
    
    def test_bin_to_frame_range(self, basic_sampler):
        """Test conversion from bin indices to frame ranges."""
        # Test first bin
        start, end = basic_sampler.bin_to_frame_range(0)
        assert start == 0
        assert end == 30
        
        # Test middle bin
        start, end = basic_sampler.bin_to_frame_range(5)
        assert start == 150
        assert end == 180
        
        # Test last bin
        start, end = basic_sampler.bin_to_frame_range(9)
        assert start == 270
        assert end == 300  # Should be clamped to total_frames
    
    def test_failure_statistics_update(self, short_sampler):
        """Test updating failure statistics with known data."""
        # Initial state: no episodes, no failures
        assert torch.allclose(short_sampler.episode_counts, torch.zeros(5))
        assert torch.allclose(short_sampler.failure_counts, torch.zeros(5))
        
        # Add some episodes to bins 0, 1, 2
        starting_frames = torch.tensor([5, 15, 25, 15, 35])  # Bins 0, 1, 2, 1, 3
        failures = torch.tensor([True, False, True, True, False])
        
        short_sampler.update_failure_statistics(starting_frames, failures)
        
        # Check episode counts
        expected_episodes = torch.tensor([1, 2, 1, 1, 0], dtype=torch.float32)
        assert torch.equal(short_sampler.episode_counts, expected_episodes)
        
        # Check failure counts
        expected_failures = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
        assert torch.equal(short_sampler.failure_counts, expected_failures)
        
        # Check smoothed failure rates (should be updated with EMA)
        expected_raw_rates = torch.tensor([1.0, 0.5, 1.0, 0.0, 0.0])
        # With alpha=0.2: new_rate = 0.2 * raw + 0.8 * old_rate (old_rate was 0)
        expected_smoothed = 0.2 * expected_raw_rates
        assert torch.allclose(short_sampler.smoothed_failure_rates, expected_smoothed, atol=1e-6)
    
    def test_convolution_calculation(self, short_sampler):
        """Test non-causal convolution implementation."""
        # Set up known failure rates
        short_sampler.smoothed_failure_rates = torch.tensor([0.8, 0.6, 0.4, 0.2, 0.0])
        
        convolved = short_sampler.compute_convolved_failure_rates()
        
        # For bin 0 with K=3, γ=0.8:
        # weights: [1.0, 0.8, 0.64] for bins [0, 1, 2]
        # weighted_sum = 1.0*0.8 + 0.8*0.6 + 0.64*0.4 = 0.8 + 0.48 + 0.256 = 1.536
        # normalization = 1.0 + 0.8 + 0.64 = 2.44
        # result = 1.536 / 2.44 ≈ 0.6295
        
        expected_bin0 = (1.0*0.8 + 0.8*0.6 + 0.64*0.4) / (1.0 + 0.8 + 0.64)
        assert torch.allclose(convolved[0], torch.tensor(expected_bin0), atol=1e-4)
        
        # Check that convolution preserves non-negativity
        assert torch.all(convolved >= 0)
    
    def test_probability_calculation(self, short_sampler):
        """Test sampling probability calculation with uniform mixing."""
        # Set up known convolved failure rates
        short_sampler.smoothed_failure_rates = torch.tensor([1.0, 0.5, 0.0, 0.25, 0.75])
        short_sampler.update_sampling_probabilities()
        
        # Check that probabilities sum to 1
        assert torch.allclose(short_sampler.sampling_probabilities.sum(), torch.tensor(1.0), atol=1e-6)
        
        # Check that all probabilities are positive
        assert torch.all(short_sampler.sampling_probabilities > 0)
        
        # Check uniform mixing effect
        # With λ=0.2, each bin should have at least 0.2/5 = 0.04 probability
        min_expected_prob = short_sampler.lambda_uniform / short_sampler.num_bins
        assert torch.all(short_sampler.sampling_probabilities >= min_expected_prob * 0.9)  # Allow small numerical error
    
    def test_sampling_functionality(self, basic_sampler):
        """Test the sampling methods produce valid outputs."""
        batch_size = 100
        
        # Test frame sampling
        frames, bins = basic_sampler.sample_starting_frames(batch_size)
        
        assert frames.shape == (batch_size,)
        assert bins.shape == (batch_size,)
        assert torch.all(frames >= 0)
        assert torch.all(frames < basic_sampler.total_frames)
        assert torch.all(bins >= 0)
        assert torch.all(bins < basic_sampler.num_bins)
        
        # Test phase sampling
        phases, bins2 = basic_sampler.sample_starting_phases(batch_size)
        
        assert phases.shape == (batch_size,)
        assert bins2.shape == (batch_size,)
        assert torch.all(phases >= 0)
        assert torch.all(phases <= 1)
        assert torch.equal(bins, bins2)  # Should be the same bins
    
    def test_empty_batch_handling(self, basic_sampler):
        """Test handling of empty batches."""
        frames, bins = basic_sampler.sample_starting_frames(0)
        assert frames.shape == (0,)
        assert bins.shape == (0,)
        
        phases, bins2 = basic_sampler.sample_starting_phases(0)
        assert phases.shape == (0,)
        assert bins2.shape == (0,)
        
        # Test updating with empty data
        empty_frames = torch.empty(0, dtype=torch.long)
        empty_failures = torch.empty(0, dtype=torch.bool)
        basic_sampler.update_failure_statistics(empty_frames, empty_failures)
        # Should not crash and maintain initial state
    
    def test_statistics_tracking(self, short_sampler):
        """Test statistics tracking and reporting."""
        # Add some episodes
        starting_frames = torch.tensor([5, 15, 25])
        failures = torch.tensor([True, False, True])
        
        short_sampler.update_failure_statistics(starting_frames, failures)
        
        stats = short_sampler.get_statistics()
        
        assert stats["total_episodes"] == 3
        assert stats["total_failures"] == 2
        assert stats["overall_failure_rate"] == 2/3
        assert stats["update_counter"] == 1
        assert "sampling_entropy" in stats
        assert "bin_episode_counts" in stats
        assert "sampling_probabilities" in stats
    
    def test_adaptive_vs_uniform_distribution(self, short_sampler):
        """Test that adaptive sampling creates non-uniform distribution."""
        # Create a scenario where bin 0 has many failures
        for _ in range(10):
            starting_frames = torch.tensor([5])  # Always bin 0
            failures = torch.tensor([True])  # Always fail
            short_sampler.update_failure_statistics(starting_frames, failures)
        
        # Create a scenario where bin 4 has no failures
        for _ in range(10):
            starting_frames = torch.tensor([45])  # Always bin 4
            failures = torch.tensor([False])  # Never fail
            short_sampler.update_failure_statistics(starting_frames, failures)
        
        short_sampler.update_sampling_probabilities()
        
        # Bin 0 should have higher probability than bin 4
        assert short_sampler.sampling_probabilities[0] > short_sampler.sampling_probabilities[4]
        
        # Distribution should not be uniform (entropy < max entropy)
        stats = short_sampler.get_statistics()
        assert stats["normalized_entropy"] < 0.99  # Should be less than maximum entropy
    
    def test_state_save_load(self, basic_sampler):
        """Test saving and loading sampler state."""
        # Add some data
        starting_frames = torch.tensor([10, 50, 100])
        failures = torch.tensor([True, False, True])
        basic_sampler.update_failure_statistics(starting_frames, failures)
        basic_sampler.update_sampling_probabilities()
        
        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            basic_sampler.save_state(tmp.name)
            
            # Create new sampler and load state
            new_sampler = AdaptiveSampler(
                motion_fps=30,
                motion_duration=10.0,
                bin_size_seconds=1.0,
                gamma=0.9,
                lambda_uniform=0.1,
                K=5,
                alpha_smooth=0.1,
                device="cpu"
            )
            
            new_sampler.load_state(tmp.name)
            
            # Verify state was loaded correctly
            assert torch.equal(basic_sampler.episode_counts, new_sampler.episode_counts)
            assert torch.equal(basic_sampler.failure_counts, new_sampler.failure_counts)
            assert torch.equal(basic_sampler.smoothed_failure_rates, new_sampler.smoothed_failure_rates)
            assert torch.equal(basic_sampler.sampling_probabilities, new_sampler.sampling_probabilities)
            assert basic_sampler.total_episodes == new_sampler.total_episodes
            assert basic_sampler.total_failures == new_sampler.total_failures
            
        # Clean up
        os.unlink(tmp.name)
    
    def test_reset_functionality(self, short_sampler):
        """Test reset functionality."""
        # Add some data
        starting_frames = torch.tensor([5, 15])
        failures = torch.tensor([True, False])
        short_sampler.update_failure_statistics(starting_frames, failures)
        
        # Reset
        short_sampler.reset_statistics()
        
        # Verify reset state
        assert torch.allclose(short_sampler.episode_counts, torch.zeros(5))
        assert torch.allclose(short_sampler.failure_counts, torch.zeros(5))
        assert torch.allclose(short_sampler.smoothed_failure_rates, torch.zeros(5))
        assert torch.allclose(short_sampler.sampling_probabilities, torch.full((5,), 0.2))
        assert short_sampler.total_episodes == 0
        assert short_sampler.total_failures == 0
    
    def test_numerical_stability(self, basic_sampler):
        """Test numerical stability with edge cases."""
        # Test with all failures
        starting_frames = torch.arange(10)  # One episode per bin
        failures = torch.ones(10, dtype=torch.bool)
        basic_sampler.update_failure_statistics(starting_frames, failures)
        basic_sampler.update_sampling_probabilities()
        
        # Should not produce NaN or inf
        assert torch.all(torch.isfinite(basic_sampler.sampling_probabilities))
        assert torch.allclose(basic_sampler.sampling_probabilities.sum(), torch.tensor(1.0))
        
        # Test with no failures
        basic_sampler.reset_statistics()
        failures = torch.zeros(10, dtype=torch.bool)
        basic_sampler.update_failure_statistics(starting_frames, failures)
        basic_sampler.update_sampling_probabilities()
        
        # Should not produce NaN or inf
        assert torch.all(torch.isfinite(basic_sampler.sampling_probabilities))
        assert torch.allclose(basic_sampler.sampling_probabilities.sum(), torch.tensor(1.0))


if __name__ == "__main__":
    # Run tests if called directly
    import pytest
    pytest.main([__file__, "-v"])