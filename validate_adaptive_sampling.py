#!/usr/bin/env python3
"""
Quick validation script for adaptive sampling implementation.

This script performs validation checks to ensure our adaptive sampling implementation
is working correctly before running full training experiments.
"""

import sys
import os

# Set up environment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.insert(0, 'source/whole_body_tracking')

def validate_implementation():
    """Validate that the adaptive sampling implementation is correct."""
    print("🔍 Validating Adaptive Sampling Implementation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Import and basic functionality
    total_tests += 1
    try:
        from whole_body_tracking.tasks.tracking.mdp.adaptive_sampler import AdaptiveSampler
        
        # Create test sampler
        sampler = AdaptiveSampler(
            motion_fps=30,
            motion_duration=10.0,
            bin_size_seconds=1.0,
            device='cpu'
        )
        
        # Test basic operations
        frames, bins = sampler.sample_starting_frames(10)
        phases, bins2 = sampler.sample_starting_phases(10)
        
        print("✅ Test 1: Basic AdaptiveSampler functionality")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 1: Basic functionality failed: {e}")
    
    # Test 2: Configuration integration
    total_tests += 1
    try:
        from whole_body_tracking.tasks.tracking.mdp.commands import AdaptiveSamplingCfg, MotionCommandCfg
        
        # Test configuration
        adaptive_cfg = AdaptiveSamplingCfg()
        motion_cfg = MotionCommandCfg()
        
        assert adaptive_cfg.enabled == True
        assert hasattr(motion_cfg, 'adaptive_sampling')
        
        print("✅ Test 2: Configuration integration")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 2: Configuration integration failed: {e}")
    
    # Test 3: Mathematical correctness
    total_tests += 1
    try:
        import torch
        
        sampler = AdaptiveSampler(motion_fps=10, motion_duration=5.0, device='cpu')
        
        # Add some failure data
        starting_frames = torch.tensor([5, 15, 25, 35, 45])  # One per bin
        failures = torch.tensor([True, True, False, False, False])  # First two bins fail
        
        sampler.update_failure_statistics(starting_frames, failures)
        sampler.update_sampling_probabilities()
        
        # Check that failed bins have higher probability
        probs = sampler.sampling_probabilities
        assert probs[0] > probs[2]  # Bin 0 (failed) > Bin 2 (succeeded)
        assert probs[1] > probs[3]  # Bin 1 (failed) > Bin 3 (succeeded)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        
        print("✅ Test 3: Mathematical correctness")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 3: Mathematical correctness failed: {e}")
    
    # Test 4: Integration components
    total_tests += 1
    try:
        from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner
        
        # Check that runner has adaptive sampling methods
        assert hasattr(MotionOnPolicyRunner, '_update_adaptive_sampling_stats')
        
        print("✅ Test 4: Runner integration")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 4: Runner integration failed: {e}")
    
    # Test 5: Stress test with large motion
    total_tests += 1
    try:
        # Test with a large motion (simulate LAFAN1 sequence)
        large_sampler = AdaptiveSampler(
            motion_fps=30,
            motion_duration=60.0,  # 1 minute motion
            bin_size_seconds=1.0,
            device='cpu'
        )
        
        # Simulate many episodes
        for _ in range(100):
            frames = torch.randint(0, large_sampler.total_frames, (20,))
            failures = torch.rand(20) < 0.3  # 30% failure rate
            large_sampler.update_failure_statistics(frames, failures)
        
        large_sampler.update_sampling_probabilities()
        
        # Check numerical stability
        assert torch.all(torch.isfinite(large_sampler.sampling_probabilities))
        assert torch.allclose(large_sampler.sampling_probabilities.sum(), torch.tensor(1.0))
        
        print("✅ Test 5: Large motion stress test")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 5: Large motion stress test failed: {e}")
    
    # Test 6: Paper algorithm validation
    total_tests += 1
    try:
        # Validate specific paper algorithms
        sampler = AdaptiveSampler(
            motion_fps=10,
            motion_duration=5.0,
            gamma=0.9,
            lambda_uniform=0.1,
            K=3,
            device='cpu'
        )
        
        # Set known failure rates
        sampler.smoothed_failure_rates = torch.tensor([0.8, 0.6, 0.4, 0.2, 0.0])
        
        # Test convolution (Equation 3)
        convolved = sampler.compute_convolved_failure_rates()
        
        # Bin 0 should have highest convolved rate
        assert convolved[0] == torch.max(convolved)
        
        # Test probability mixing
        sampler.update_sampling_probabilities()
        probs = sampler.sampling_probabilities
        
        # Each bin should have at least uniform component
        min_prob = sampler.lambda_uniform / sampler.num_bins
        assert torch.all(probs >= min_prob * 0.9)  # Allow small numerical error
        
        print("✅ Test 6: Paper algorithm validation")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Test 6: Paper algorithm validation failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🏁 Validation Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All validation tests passed!")
        print("✨ Adaptive sampling implementation is ready for training!")
        return True
    else:
        print("⚠️  Some validation tests failed.")
        print("🔧 Please check the implementation before proceeding.")
        return False

def test_performance_characteristics():
    """Test performance characteristics of the adaptive sampling."""
    print("\n🚀 Testing Performance Characteristics")
    print("=" * 50)
    
    try:
        import time
        import torch
        from whole_body_tracking.tasks.tracking.mdp.adaptive_sampler import AdaptiveSampler
        
        # Test sampling performance
        sampler = AdaptiveSampler(motion_fps=30, motion_duration=60.0, device='cpu')
        
        # Warm up
        for _ in range(10):
            sampler.sample_starting_frames(100)
        
        # Time sampling
        start_time = time.time()
        for _ in range(1000):
            sampler.sample_starting_frames(100)
        sampling_time = time.time() - start_time
        
        print(f"✅ Sampling performance: {sampling_time:.3f}s for 100k samples")
        print(f"   Rate: {100000/sampling_time:.0f} samples/second")
        
        # Test update performance
        start_time = time.time()
        for _ in range(1000):
            frames = torch.randint(0, sampler.total_frames, (10,))
            failures = torch.rand(10) < 0.3
            sampler.update_failure_statistics(frames, failures)
        update_time = time.time() - start_time
        
        print(f"✅ Update performance: {update_time:.3f}s for 10k updates")
        print(f"   Rate: {10000/update_time:.0f} updates/second")
        
        # Test memory usage
        import psutil
        import gc
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many samplers
        samplers = []
        for _ in range(100):
            samplers.append(AdaptiveSampler(motion_fps=30, motion_duration=10.0, device='cpu'))
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_sampler = (memory_after - memory_before) / 100
        
        print(f"✅ Memory usage: ~{memory_per_sampler:.2f}MB per sampler")
        
        # Cleanup
        del samplers
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main validation entry point."""
    print("🧪 Adaptive Sampling Validation Suite")
    print("🎯 Validating implementation before training experiments")
    print("")
    
    # Run validation
    validation_passed = validate_implementation()
    
    if validation_passed:
        performance_passed = test_performance_characteristics()
        
        if performance_passed:
            print("\n🚀 Ready for Training!")
            print("=" * 50)
            print("✅ Implementation validation: PASSED")
            print("✅ Performance validation: PASSED")
            print("")
            print("🎯 Next steps:")
            print("   1. Run ablation study: python scripts/ablation_study_adaptive_sampling.py")
            print("   2. Test on dance1_subject1: should succeed with adaptive sampling")
            print("   3. Compare against uniform sampling baseline")
            print("")
            print("💡 Expected benefits:")
            print("   - Enable training on long LAFAN1 sequences")
            print("   - 2x speedup on complex motions")
            print("   - Better learning of difficult segments")
            
            return 0
        else:
            print("\n⚠️  Performance validation failed")
            return 1
    else:
        print("\n❌ Implementation validation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())