# Phase 1: Adaptive Sampling Mechanism - Detailed Implementation Plan

## Executive Summary

This document provides a comprehensive, low-level implementation plan for the **Adaptive Sampling Mechanism** as described in the BeyondMimic paper (Section III-F). This is the critical missing feature that enables efficient training on long, multi-motion sequences by sampling difficult motion segments more frequently based on empirical failure statistics.

## 🎯 Problem Analysis

### Current Limitation
**Current Code (Line 196-197 in `mdp/commands.py`):**
```python
def _resample_command(self, env_ids: Sequence[int]):
    phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()
```

**Issues:**
- **Uniform sampling** across entire motion timeline
- **Inefficient training** on long sequences (wastes time on easy segments)
- **Poor learning** of difficult motion segments (cartwheels, balancing, jumps)
- **Training failures** on long LAFAN1 sequences as confirmed by ablation study

### Paper's Solution (Section III-F)
**Adaptive Sampling Algorithm:**
1. **Bin Division**: Divide motion timeline into 1-second bins
2. **Failure Tracking**: Monitor episode failures per bin
3. **Probability Calculation**: Use exponential smoothing + non-causal convolution
4. **Weighted Sampling**: Sample from bins based on failure rates
5. **Uniform Mixing**: Prevent catastrophic forgetting with uniform component

## 🏗️ Implementation Architecture

### Component Hierarchy
```
AdaptiveSampler (New Class)
├── BinManager (Statistics tracking)
├── FailureTracker (Episode outcome monitoring)  
├── ProbabilityCalculator (Sampling distribution)
└── SamplingMixer (Uniform + adaptive mixing)

MotionCommand (Modified)
├── adaptive_sampler: AdaptiveSampler
└── _resample_command() -> Uses adaptive_sampler

MotionOnPolicyRunner (Modified)  
├── Tracks episode outcomes per environment
└── Updates sampler statistics
```

### Data Flow
```
Episode Start -> Track starting bin
Episode End -> Record outcome (success/failure)
Statistics Update -> Update bin failure rates
Sampling -> Calculate weighted probabilities
Reset -> Sample from adaptive distribution
```

## 📊 Detailed Component Specifications

### 1. AdaptiveSampler Class

**Location**: `whole_body_tracking/tasks/tracking/mdp/adaptive_sampler.py` (new file)

**Core Responsibilities:**
- Maintain failure statistics per motion bin
- Calculate adaptive sampling probabilities  
- Provide weighted sampling interface
- Handle exponential smoothing and convolution

**Mathematical Implementation:**
```python
class AdaptiveSampler:
    def __init__(self, motion_fps: int, motion_duration: float, 
                 bin_size_seconds: float = 1.0, gamma: float = 0.9, 
                 lambda_uniform: float = 0.1, K: int = 5):
        self.motion_fps = motion_fps
        self.motion_duration = motion_duration
        self.bin_size_seconds = bin_size_seconds
        self.gamma = gamma  # Decay rate for convolution kernel
        self.lambda_uniform = lambda_uniform  # Uniform mixing ratio
        self.K = K  # Convolution kernel size
        
        # Calculate number of bins (S in paper)
        self.num_bins = int(np.ceil(motion_duration / bin_size_seconds))
        self.frames_per_bin = int(motion_fps * bin_size_seconds)
        
        # Statistics tracking
        self.episode_counts = torch.zeros(self.num_bins)  # Ns in paper
        self.failure_counts = torch.zeros(self.num_bins)  # Fs in paper
        self.smoothed_failure_rates = torch.zeros(self.num_bins)  # r̄s in paper
        self.sampling_probabilities = torch.ones(self.num_bins) / self.num_bins
```

### 2. Statistics Tracking Methods

**Failure Rate Calculation:**
```python
def update_failure_statistics(self, starting_bins: torch.Tensor, 
                            failures: torch.Tensor):
    """
    Update failure statistics for completed episodes
    
    Args:
        starting_bins: (N,) tensor of starting bin indices for episodes
        failures: (N,) tensor of boolean failure indicators
    """
    for bin_idx in range(self.num_bins):
        bin_mask = starting_bins == bin_idx
        if bin_mask.any():
            self.episode_counts[bin_idx] += bin_mask.sum()
            self.failure_counts[bin_idx] += (bin_mask & failures).sum()
    
    # Exponential moving average smoothing
    raw_failure_rates = self.failure_counts / torch.clamp(self.episode_counts, min=1)
    alpha_smooth = 0.1  # EMA smoothing factor
    self.smoothed_failure_rates = (alpha_smooth * raw_failure_rates + 
                                  (1 - alpha_smooth) * self.smoothed_failure_rates)
```

**Non-Causal Convolution (Equation 3):**
```python
def compute_convolved_failure_rates(self):
    """
    Apply non-causal convolution with exponentially decaying kernel
    k(u) = γ^u to assign greater weight to recent past failures
    """
    convolved_rates = torch.zeros_like(self.smoothed_failure_rates)
    
    for s in range(self.num_bins):
        weighted_sum = 0.0
        normalization = 0.0
        
        for u in range(self.K):
            if s + u < self.num_bins:
                weight = self.gamma ** u
                weighted_sum += weight * self.smoothed_failure_rates[s + u]
                normalization += weight
        
        convolved_rates[s] = weighted_sum / max(normalization, 1e-8)
    
    return convolved_rates
```

**Sampling Probability Calculation:**
```python
def update_sampling_probabilities(self):
    """
    Calculate final sampling probabilities with uniform mixing
    p's = λ(1/B) + (1-λ)ps where ps from Equation 3
    """
    convolved_rates = self.compute_convolved_failure_rates()
    
    # Normalize to get pure adaptive probabilities
    adaptive_probs = convolved_rates / torch.clamp(convolved_rates.sum(), min=1e-8)
    
    # Mix with uniform distribution (Equation 3 continuation)
    uniform_probs = torch.ones(self.num_bins) / self.num_bins
    self.sampling_probabilities = (self.lambda_uniform * uniform_probs + 
                                  (1 - self.lambda_uniform) * adaptive_probs)
```

### 3. Sampling Interface

**Weighted Sampling Method:**
```python
def sample_starting_phases(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample starting phases using adaptive probabilities
    
    Returns:
        phases: (batch_size,) tensor of starting phases [0, 1]
    """
    # Sample bins according to learned probabilities
    bin_indices = torch.multinomial(self.sampling_probabilities, 
                                  batch_size, replacement=True)
    
    # Sample uniformly within selected bins
    bin_starts = bin_indices.float() * self.frames_per_bin
    bin_ends = torch.clamp((bin_indices + 1).float() * self.frames_per_bin, 
                          max=self.motion_fps * self.motion_duration)
    
    # Uniform sampling within bin
    frame_indices = torch.rand(batch_size, device=device) * (bin_ends - bin_starts) + bin_starts
    
    # Convert to phase [0, 1]
    phases = frame_indices / (self.motion_fps * self.motion_duration)
    
    return phases, bin_indices
```

## 🔗 Integration Points

### 1. MotionCommand Modifications

**File**: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`

**Changes to `__init__` method:**
```python
class MotionCommand(CommandTerm):
    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # ... existing code ...
        
        # Initialize adaptive sampler
        motion_duration = self.motion.time_step_total / self.motion.fps
        self.adaptive_sampler = AdaptiveSampler(
            motion_fps=self.motion.fps,
            motion_duration=motion_duration,
            bin_size_seconds=cfg.adaptive_sampling.bin_size_seconds,
            gamma=cfg.adaptive_sampling.gamma,
            lambda_uniform=cfg.adaptive_sampling.lambda_uniform,
            K=cfg.adaptive_sampling.K
        )
        
        # Track starting bins for episodes
        self.episode_starting_bins = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
```

**Modified `_resample_command` method:**
```python
def _resample_command(self, env_ids: Sequence[int]):
    if self.cfg.adaptive_sampling.enabled:
        # Use adaptive sampling
        phases, starting_bins = self.adaptive_sampler.sample_starting_phases(
            len(env_ids), self.device
        )
        self.episode_starting_bins[env_ids] = starting_bins
    else:
        # Fallback to uniform sampling
        phases = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        starting_bins = (phases * self.adaptive_sampler.num_bins).long()
        self.episode_starting_bins[env_ids] = starting_bins
    
    self.time_steps[env_ids] = (phases * (self.motion.time_step_total - 1)).long()
    
    # ... rest of existing perturbation code ...
```

### 2. Runner Integration  

**File**: `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`

**Add episode tracking:**
```python
class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, 
                 device="cpu", registry_name: str = None):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        
        # Track episode outcomes for adaptive sampling
        self.episode_starting_bins = None
        self.episode_rewards = None
        
    def collect_rollout(self):
        """Override to track episode outcomes"""
        # Store starting bins at episode start
        if hasattr(self.env.unwrapped.command_manager, 'motion'):
            motion_cmd = self.env.unwrapped.command_manager.get_term('motion')
            self.episode_starting_bins = motion_cmd.episode_starting_bins.clone()
        
        # Run normal rollout collection
        super().collect_rollout()
        
        # Update adaptive sampling statistics on episode completion
        self._update_adaptive_sampling_stats()
    
    def _update_adaptive_sampling_stats(self):
        """Update failure statistics based on episode outcomes"""
        if self.episode_starting_bins is None:
            return
            
        # Detect failures (episodes that terminated early)
        episode_dones = self.rollout_storage.masks[:-1] == 0  # True where episode ended
        episode_lengths = episode_dones.float().sum(dim=0)  # Length of each episode
        
        # Define failure criteria (can be customized)
        max_episode_length = self.rollout_storage.num_transitions_per_env
        failures = episode_lengths < (max_episode_length * 0.8)  # Failed if < 80% of max length
        
        # Update sampler statistics
        motion_cmd = self.env.unwrapped.command_manager.get_term('motion')
        motion_cmd.adaptive_sampler.update_failure_statistics(
            self.episode_starting_bins, failures
        )
        motion_cmd.adaptive_sampler.update_sampling_probabilities()
```

### 3. Configuration Extensions

**File**: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`

**Add configuration class:**
```python
@configclass
class AdaptiveSamplingCfg:
    """Configuration for adaptive sampling mechanism"""
    enabled: bool = True
    bin_size_seconds: float = 1.0  # Size of each bin in seconds
    gamma: float = 0.9  # Decay rate for convolution kernel  
    lambda_uniform: float = 0.1  # Mixing ratio with uniform distribution
    K: int = 5  # Convolution kernel size
    update_frequency: int = 100  # Update probabilities every N episodes

@configclass
class MotionCommandCfg(CommandTermCfg):
    # ... existing fields ...
    adaptive_sampling: AdaptiveSamplingCfg = AdaptiveSamplingCfg()
```

## 📈 Performance & Monitoring

### 1. Logging & Visualization

**WandB Integration:**
```python
def log_adaptive_sampling_metrics(self):
    """Log sampling statistics to WandB"""
    if wandb.run is not None:
        # Log failure rates per bin
        for i, rate in enumerate(self.adaptive_sampler.smoothed_failure_rates):
            wandb.log({f"adaptive_sampling/failure_rate_bin_{i}": rate})
        
        # Log sampling probabilities
        for i, prob in enumerate(self.adaptive_sampler.sampling_probabilities):
            wandb.log({f"adaptive_sampling/sampling_prob_bin_{i}": prob})
        
        # Log entropy of sampling distribution
        entropy = -torch.sum(self.adaptive_sampler.sampling_probabilities * 
                           torch.log(self.adaptive_sampler.sampling_probabilities + 1e-8))
        wandb.log({"adaptive_sampling/sampling_entropy": entropy})
```

### 2. Ablation Study Support

**Toggle Mechanism:**
```python
# In training script args
parser.add_argument("--adaptive_sampling", action="store_true", 
                   help="Enable adaptive sampling mechanism")
parser.add_argument("--adaptive_sampling_start_iter", type=int, default=1000,
                   help="Start adaptive sampling after N iterations")
```

## 🧪 Testing Strategy

### 1. Unit Tests

**Test File**: `tests/test_adaptive_sampler.py`

**Key Test Cases:**
- Probability calculation correctness
- Failure rate smoothing behavior  
- Convolution kernel implementation
- Edge cases (all successes, all failures)
- Bin boundary handling

### 2. Integration Tests

**Validation Approach:**
1. **Short Motion Test**: Verify on Cristiano Ronaldo motion (known to work)
2. **Long Motion Test**: Test on dance1_subject1 (known to fail without adaptive sampling)
3. **Convergence Speed**: Compare iterations to convergence vs. uniform sampling
4. **Distribution Analysis**: Visualize sampling patterns over training

### 3. Ablation Study

**Comparison Matrix:**
```
Motion           | Uniform Sampling | Adaptive Sampling | Speedup
Cristiano        | 3k iterations   | 1.5k iterations   | 2x
dance1_subject1  | Failed          | 8k iterations     | ∞ (enables)
dance2_subject1  | Failed          | 9k iterations     | ∞ (enables)
```

## 🔧 Implementation Schedule

### Phase 1A: Core Implementation (Week 1)
- [ ] Create `AdaptiveSampler` class with mathematical components
- [ ] Implement failure tracking and probability calculation
- [ ] Add configuration framework

### Phase 1B: Integration (Week 2)  
- [ ] Modify `MotionCommand` to use adaptive sampler
- [ ] Update `MotionOnPolicyRunner` for episode tracking
- [ ] Add logging and monitoring

### Phase 1C: Testing & Validation (Week 3)
- [ ] Unit tests for mathematical correctness
- [ ] Integration tests on known motions
- [ ] Ablation study replication
- [ ] Performance optimization

## 🚀 Expected Outcomes

### Immediate Benefits
- **Enable long motion training**: LAFAN1 sequences that currently fail
- **Faster convergence**: 2x speedup on complex motions
- **Robust learning**: Better mastery of difficult segments

### Long-term Impact
- **Foundation for multi-motion**: Essential for Phase 2 (multiple motions)  
- **Curriculum learning**: Natural progression from easy to hard segments
- **Real-world deployment**: Better handling of challenging scenarios

## 📝 Implementation Notes

### Critical Considerations
1. **Thread Safety**: Ensure statistics updates are atomic in multi-env setting
2. **Memory Efficiency**: Use sparse representations for large motion sequences
3. **Hyperparameter Sensitivity**: Start with paper values, tune per motion type
4. **Backward Compatibility**: Maintain ability to disable adaptive sampling

### Debugging Support
1. **Visualization Tools**: Plot failure rates and sampling probabilities over time
2. **Manual Override**: Allow manual bin probability specification for testing
3. **Detailed Logging**: Track episode outcomes and sampling decisions

This implementation plan provides the complete foundation for adaptive sampling that will unlock BeyondMimic's full potential for learning long, complex motion sequences efficiently.