# BeyondMimic Stage 2: Guided Diffusion Implementation Design Plan (Refined)

## Executive Summary

This document outlines a comprehensive implementation plan for Stage 2 of BeyondMimic: the guided diffusion component that enables versatile test-time control through state-action co-diffusion. The implementation leverages the Diffuse-CLoC framework with differentiated attention mechanisms, independent noise schedules, and careful engineering for real-world deployment. This refined version incorporates critical insights from deep analysis of Section IV and comparison with alternative approaches.

## 1. System Architecture Overview

### 1.1 Core Components

```
Stage 1 (Existing)          Stage 2 (To Implement)
┌──────────────┐            ┌─────────────────────┐
│Motion Tracking│            │  Data Collection    │
│   Policies   ├───────────►│     Pipeline        │
└──────────────┘            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │ State-Action Dataset│
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  Diffusion Model    │
                            │    Training         │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │ Guided Inference    │
                            │   with Tasks        │
                            └─────────────────────┘
```

### 1.2 Key Innovations to Implement

1. **Joint State-Action Diffusion**: Model p(τ) where τ = [a_t, s_{t+1}, ..., a_{t+H}]
2. **Differentiated Attention (Diffuse-CLoC)**: 
   - State tokens: Bi-directional attention for holistic planning
   - Action tokens: Causal attention to prevent future contamination
3. **Independent Noise Schedules**: k = (k_s, k_a) with different schedules for states and actions
4. **Classifier Guidance**: Gradient ∇_τ G_c(τ) steering at each denoising step
5. **Action Delay Compensation**: 0-100ms randomization during training for 20ms inference latency

## 2. Data Collection Pipeline

### 2.1 Module: `diffusion/data_collection.py`

```python
class MotionDataCollector:
    """Collects state-action trajectories from trained tracking policies."""
    
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        policies: Dict[str, OnPolicyRunner],
        cfg: DataCollectionCfg
    ):
        self.env = env
        self.policies = policies
        self.cfg = cfg
        self.buffer = TrajectoryBuffer()
        
    def collect_trajectories(
        self,
        num_episodes: int,
        add_noise: bool = True
    ) -> TrajectoryDataset:
        """
        Collects trajectories with domain randomization.
        
        Key features (from paper Section VI.A):
        - Action delay randomization (0-100ms) for inference latency
        - State noise injection for robustness
        - Random policy switching between motion skills
        - Include failure recovery sequences for OOD handling
        - Domain randomization matching Stage 1 training
        """
        # Implementation following PDP [34] and Diffuse-CLoC [3]
        pass
```

### 2.2 Data Format Specification

```python
@dataclass
class Trajectory:
    """Single trajectory with history and future."""
    # History: O_t = [s_{t-N}, a_{t-N}, ..., s_t]
    history_states: torch.Tensor  # (N+1, state_dim)
    history_actions: torch.Tensor  # (N, action_dim)
    
    # Future: τ_t = [a_t, s_{t+1}, ..., s_{t+H}, a_{t+H}]
    future_states: torch.Tensor  # (H, state_dim)
    future_actions: torch.Tensor  # (H+1, action_dim)
    
    # Metadata
    motion_id: str
    timestep: int
    success: bool
```

### 2.3 State Representation (Body-Pos)

Based on paper Section VI.D (lines 785-800):

```python
@dataclass
class StateRepresentation:
    """Body-position state representation."""
    # Global states (relative to current frame)
    root_pos: torch.Tensor      # (3,) - relative position
    root_vel: torch.Tensor      # (3,) - linear velocity  
    root_rot: torch.Tensor      # (3,) - rotation vector
    
    # Local states (in character frame)
    body_positions: torch.Tensor  # (num_bodies, 3)
    body_velocities: torch.Tensor # (num_bodies, 3)
```

## 3. Diffusion Model Architecture

### 3.1 Module: `diffusion/models/state_action_diffusion.py`

```python
class StateActionDiffusionModel(nn.Module):
    """
    Transformer-based diffusion model with differentiated attention.
    Architecture based on paper Section VI.B (lines 747-752).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        horizon: int = 16,
        history_len: int = 4
    ):
        super().__init__()
        
        # Separate embeddings for states and actions
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        
        # Transformer with differentiated attention
        self.transformer = DifferentiatedTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            use_causal_action_mask=True
        )
        
        # Output heads
        self.state_decoder = StateDecoder(hidden_dim, state_dim)
        self.action_decoder = ActionDecoder(hidden_dim, action_dim)
        
        # Time embedding (sinusoidal)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
```

### 3.2 Differentiated Attention Mechanism (Critical Innovation)

```python
class DifferentiatedTransformer(nn.Module):
    """
    Implements differentiated attention from Diffuse-CLoC [3].
    This is CRITICAL for sim-to-real success.
    
    Key insight: States need holistic planning (bi-directional),
    while actions need reactive control (causal only).
    """
    
    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiated processing prevents action contamination from
        potentially noisy future state predictions while allowing
        states to plan holistically across the trajectory.
        """
        # Critical: Different masks for states vs actions
        state_mask = None  # Bi-directional for planning
        action_mask = self.create_causal_mask(action_tokens.shape[1])  # Causal only
        
        # Cross-attention between state and action streams
        for layer in self.layers:
            # States can see all other states + current/past actions
            state_tokens = layer(
                state_tokens, 
                mask=state_mask,
                cross_attn_kv=action_tokens
            )
            # Actions only see current/past states + current/past actions
            action_tokens = layer(
                action_tokens,
                mask=action_mask, 
                cross_attn_kv=state_tokens
            )
            
        return state_tokens, action_tokens
```

### 3.3 DDPM Training Loop with Critical Details

```python
class DDPMTrainer:
    """
    Implements DDPM training with independent noise schedules.
    Critical details from paper Section VI.B:
    - Independent schedules for states vs actions
    - Loss masking to limit action horizon to 8 steps
    - Attention dropout (0.3) for robustness
    """
    
    def __init__(
        self,
        model: StateActionDiffusionModel,
        cfg: DDPMConfig
    ):
        self.model = model
        self.cfg = cfg
        
        # CRITICAL: Independent noise schedules (paper uses k=(ks,ka))
        self.state_noise_schedule = LinearNoiseSchedule(
            start=1e-4, end=0.02, steps=1000  # States need more noise
        )
        self.action_noise_schedule = LinearNoiseSchedule(
            start=1e-4, end=0.01, steps=1000  # Actions need less noise
        )
        
        # Loss mask for action horizon limiting (H=16 but action loss only for 8)
        self.action_loss_mask = torch.ones(16)
        self.action_loss_mask[8:] = 0.0  # Mask out actions beyond step 8
        
    def training_step(
        self,
        batch: Trajectory
    ) -> torch.Tensor:
        """
        Training with MSE loss: L = MSE(x_0,θ(τ^k_t, O_t, k), τ_t)
        Key: Independent noise levels for states and actions
        """
        batch_size = batch.future_trajectory.shape[0]
        
        # Sample INDEPENDENT timesteps for states and actions
        k_state = torch.randint(0, 1000, (batch_size,))
        k_action = torch.randint(0, 1000, (batch_size,))
        k = (k_state, k_action)  # Tuple of noise levels
        
        # Add noise with DIFFERENT schedules
        noised_states = self.add_state_noise(
            batch.future_states, k_state
        )
        noised_actions = self.add_action_noise(
            batch.future_actions, k_action  
        )
        noised_trajectory = self.interleave(noised_states, noised_actions)
        
        # Predict clean trajectory
        pred_trajectory = self.model(
            noised_trajectory,
            batch.history,
            k
        )
        
        # Separate states and actions for loss computation
        pred_states, pred_actions = self.deinterleave(pred_trajectory)
        gt_states, gt_actions = batch.future_states, batch.future_actions
        
        # Compute losses with ACTION MASKING
        state_loss = F.mse_loss(pred_states, gt_states)
        action_loss = F.mse_loss(
            pred_actions[:, :8],  # Only first 8 action predictions
            gt_actions[:, :8]
        )
        
        # Weighted combination (states and actions may have different weights)
        loss = state_loss + self.cfg.action_weight * action_loss
        return loss
```

## 4. Guided Inference Implementation

### 4.1 Module: `diffusion/guidance/classifier_guidance.py`

```python
class GuidedDiffusionInference:
    """
    Implements classifier guidance for test-time control.
    Based on paper Section IV.B (lines 471-495).
    """
    
    def __init__(
        self,
        model: StateActionDiffusionModel,
        guidance_scale: float = 1.0
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        
    def guided_sample(
        self,
        history: torch.Tensor,
        cost_function: Callable,
        num_denoising_steps: int = 20
    ) -> torch.Tensor:
        """
        DDPM sampling with guidance gradient.
        Implements Equation 5 (lines 455-465) with guidance.
        """
        # Initialize with noise
        trajectory = torch.randn(self.trajectory_shape)
        
        for k in reversed(range(num_denoising_steps)):
            # Predict noise
            noise_pred = self.model.predict_noise(
                trajectory, 
                history, 
                k
            )
            
            # Compute guidance gradient
            with torch.enable_grad():
                trajectory.requires_grad_(True)
                cost = cost_function(trajectory)
                guidance_grad = torch.autograd.grad(
                    cost, 
                    trajectory
                )[0]
            
            # DDPM update with guidance
            trajectory = self.ddpm_step(
                trajectory,
                noise_pred,
                k,
                guidance_grad * self.guidance_scale
            )
            
        return trajectory
```

## 5. Downstream Task Implementations

### 5.1 Module: `diffusion/tasks/cost_functions.py`

```python
class TaskCostFunctions:
    """
    Implements cost functions for downstream tasks.
    Based on paper Section IV.C (lines 497-566).
    """
    
    @staticmethod
    def joystick_steering(
        trajectory: torch.Tensor,
        goal_velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Equation 7 (lines 501-510).
        G_js(τ) = 1/2 Σ ||V_xy,t'(τ_t') - g_v||^2
        """
        planar_velocities = extract_planar_velocities(trajectory)
        return 0.5 * torch.sum(
            (planar_velocities - goal_velocity) ** 2
        )
    
    @staticmethod
    def waypoint_navigation(
        trajectory: torch.Tensor,
        goal_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Equation 8 (lines 519-533).
        Proximity reward with velocity penalty near goal.
        """
        positions = extract_positions(trajectory)
        velocities = extract_velocities(trajectory)
        
        distances = torch.norm(positions - goal_position, dim=-1)
        proximity_weight = 1 - torch.exp(-2 * distances)
        velocity_weight = torch.exp(-2 * distances)
        
        position_cost = proximity_weight * distances ** 2
        velocity_cost = velocity_weight * torch.norm(velocities, dim=-1) ** 2
        
        return torch.sum(position_cost + velocity_cost)
    
    @staticmethod
    def obstacle_avoidance(
        trajectory: torch.Tensor,
        sdf_field: SDFField,
        body_radii: torch.Tensor
    ) -> torch.Tensor:
        """
        Equation 9-10 (lines 542-566).
        Barrier function for collision avoidance.
        """
        body_positions = extract_body_positions(trajectory)
        costs = []
        
        for b, radius in enumerate(body_radii):
            sdf_values = sdf_field.query(body_positions[:, b])
            costs.append(
                barrier_function(
                    sdf_values - radius,
                    delta=0.1
                )
            )
            
        return torch.sum(torch.stack(costs))

def barrier_function(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Relaxed barrier function (Equation 10).
    """
    mask = x >= delta
    safe_cost = -torch.log(torch.clamp(x, min=1e-6))
    unsafe_cost = -torch.log(delta) + 0.5 * ((x - 2*delta) / delta) ** 2 - 0.5
    
    return torch.where(mask, safe_cost, unsafe_cost)
```

### 5.2 Module: `diffusion/tasks/sdf_field.py`

```python
class SDFField:
    """Signed Distance Field for obstacle representation."""
    
    def __init__(
        self,
        obstacles: List[Obstacle],
        resolution: float = 0.01
    ):
        self.obstacles = obstacles
        self.resolution = resolution
        self._build_field()
        
    def query(
        self, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns SDF values and gradients at positions.
        """
        # Trilinear interpolation in SDF grid
        sdf_values = self.interpolate(positions)
        
        # Compute gradients via automatic differentiation
        positions.requires_grad_(True)
        sdf_values = self.interpolate(positions)
        gradients = torch.autograd.grad(
            sdf_values.sum(), 
            positions,
            create_graph=True
        )[0]
        
        return sdf_values, gradients
```

## 6. Integration with Current Codebase

### 6.1 Directory Structure

```
whole_body_tracking/
├── source/whole_body_tracking/whole_body_tracking/
│   ├── tasks/tracking/           # [Existing]
│   ├── diffusion/                # [New]
│   │   ├── __init__.py
│   │   ├── data_collection.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── state_action_diffusion.py
│   │   │   ├── transformer.py
│   │   │   └── embeddings.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── ddpm_trainer.py
│   │   │   └── noise_schedules.py
│   │   ├── guidance/
│   │   │   ├── __init__.py
│   │   │   └── classifier_guidance.py
│   │   └── tasks/
│   │       ├── __init__.py
│   │       ├── cost_functions.py
│   │       └── sdf_field.py
│   └── configs/
│       └── diffusion/
│           ├── __init__.py
│           ├── model_cfg.py
│           └── training_cfg.py
```

### 6.2 Training Script: `scripts/diffusion/train.py`

```python
import torch
import wandb
from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.diffusion.data_collection import MotionDataCollector
from whole_body_tracking.diffusion.models import StateActionDiffusionModel
from whole_body_tracking.diffusion.training import DDPMTrainer

def main(args):
    # Load tracking policies
    tracking_policies = load_tracking_policies(args.policy_paths)
    
    # Initialize environment
    env = gym.make(args.env_name)
    
    # Collect data
    collector = MotionDataCollector(env, tracking_policies, cfg)
    dataset = collector.collect_trajectories(
        num_episodes=args.num_episodes,
        add_noise=True
    )
    
    # Initialize model
    model = StateActionDiffusionModel(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **model_cfg
    )
    
    # Train
    trainer = DDPMTrainer(model, training_cfg)
    trainer.fit(
        dataset,
        num_epochs=args.num_epochs,
        logger=wandb
    )
```

### 6.3 Deployment Script: `scripts/diffusion/deploy.py`

```python
def deploy_guided_policy(args):
    # Load model
    model = StateActionDiffusionModel.load(args.model_path)
    inference = GuidedDiffusionInference(model, args.guidance_scale)
    
    # Initialize task
    if args.task == "joystick":
        cost_fn = lambda τ: TaskCostFunctions.joystick_steering(
            τ, args.goal_velocity
        )
    elif args.task == "waypoint":
        cost_fn = lambda τ: TaskCostFunctions.waypoint_navigation(
            τ, args.goal_position
        )
    elif args.task == "obstacle":
        sdf = SDFField(args.obstacles)
        cost_fn = lambda τ: TaskCostFunctions.obstacle_avoidance(
            τ, sdf, args.body_radii
        )
    
    # Run inference loop
    history_buffer = HistoryBuffer(window_size=4)
    
    while not env.done:
        # Get current observation
        obs = env.get_observation()
        history_buffer.append(obs)
        
        # Generate trajectory with guidance
        trajectory = inference.guided_sample(
            history=history_buffer.get_history(),
            cost_function=cost_fn,
            num_denoising_steps=20
        )
        
        # Execute first action
        action = trajectory[0]  # First action in sequence
        env.step(action)
```

## 7. Dependencies and Requirements

### 7.1 New Python Dependencies

```python
# Add to setup.py
DIFFUSION_REQUIRES = [
    "diffusers>=0.21.0",      # Diffusion utilities
    "transformers>=4.30.0",    # Transformer models
    "einops>=0.6.0",          # Tensor operations
    "timm>=0.9.0",            # EMA and training utilities
    "scipy>=1.10.0",          # SDF interpolation
]
```

### 7.2 Hardware Requirements

- **Training**: NVIDIA GPU with ≥24GB VRAM (A5000, RTX 4090, or better)
- **Inference**: NVIDIA GPU with ≥8GB VRAM for real-time deployment
- **CPU**: Multicore processor for data collection parallelization

## 8. Implementation Timeline

### Phase 1: Foundation (2 weeks)
- [ ] Set up diffusion module structure
- [ ] Implement data collection pipeline
- [ ] Create trajectory dataset format
- [ ] Test data collection with existing policies

### Phase 2: Model Development (3 weeks)
- [ ] Implement state-action diffusion model
- [ ] Create differentiated transformer
- [ ] Implement DDPM training loop
- [ ] Add noise schedules and embeddings

### Phase 3: Training Infrastructure (2 weeks)
- [ ] Create training scripts
- [ ] Integrate WandB logging
- [ ] Implement checkpointing
- [ ] Add validation metrics

### Phase 4: Guidance Implementation (2 weeks)
- [ ] Implement classifier guidance
- [ ] Create cost functions for tasks
- [ ] Build SDF field implementation
- [ ] Test guidance gradients

### Phase 5: Integration & Testing (3 weeks)
- [ ] Integrate with Isaac Lab environment
- [ ] Create deployment scripts
- [ ] Benchmark performance
- [ ] Debug and optimize

### Phase 6: Validation (2 weeks)
- [ ] Compare with paper results
- [ ] Ablation studies
- [ ] Documentation
- [ ] Release preparation

## 9. Critical Implementation Details (Refined from Paper Analysis)

### 9.1 Why Body-Pos State Representation Works
The paper's ablation (Section VI.D) shows Body-Pos significantly outperforms Joint-Pos (100% vs 72% success on Walk-Perturb). Key insight: Joint-space representations accumulate errors through the kinematic chain, while Body-Pos provides direct spatial grounding for guidance. This is CRITICAL for sim-to-real success.

### 9.2 Differentiated Attention is Essential
The Diffuse-CLoC approach with differentiated attention is not optional - it's critical. States need bi-directional attention to plan holistically, while actions MUST use causal attention to avoid contamination from noisy future predictions. Without this, guidance fails catastrophically (0% success on joystick control with Joint-Rot).

### 9.3 Action Delay Compensation Strategy
The ~20ms inference latency requires careful handling:
- Train with 0-100ms random action delays
- Run inference asynchronously in separate thread
- Use TensorRT optimization for GPU inference
- Consider action buffering for smooth execution

### 9.4 Loss Masking Rationale
Although H=16, action predictions are masked to 8 steps. This prevents the model from trying to predict actions too far into the future where uncertainty is high, improving stability and reducing compounding errors.

## 10. Key Challenges and Solutions (Updated)

### 10.1 Challenge: Avoiding Planning-Control Gap
**Problem**: Hierarchical approaches suffer from OOD motions that tracking can't follow
**Solution**: Joint state-action diffusion ensures generated trajectories are physically realizable

### 10.2 Challenge: Skill Transition Limitations
**Problem**: Model struggles to transition between distant skills on the manifold
**Solution**: Collect transition sequences during data generation; consider curriculum learning

### 10.3 Challenge: Out-of-Distribution Behavior
**Finding**: Diffusion models exhibit "inert" OOD behavior (robot stays still when confused)
**Benefit**: Safer than erratic RL policies; allows human intervention

### 10.4 Challenge: Gradient Computation at Scale
**Problem**: Computing guidance gradients for all bodies is expensive
**Solution**: Use CppAD for C++ autodiff; consider approximations for less critical bodies

## 11. Evaluation Metrics

### 11.1 Training Metrics
- Reconstruction loss (MSE)
- State prediction accuracy
- Action prediction accuracy
- FID score for motion quality

### 11.2 Deployment Metrics
- Task success rate
- Fall rate  
- Trajectory smoothness
- Guidance effectiveness

### 11.3 Comparison Baselines
- Hierarchical approach (Kin+PHC)
- Pure action diffusion
- Modular planning + control

## 12. Future Extensions

1. **Multi-task Pretraining**: Train on diverse motion datasets
2. **Online Adaptation**: Fine-tune during deployment
3. **Compositional Tasks**: Combine multiple cost functions
4. **Vision Integration**: Add visual observations
5. **Language Conditioning**: Natural language task specification

## 13. Key Refinements Summary

This refined design incorporates critical insights from deep paper analysis:

1. **Differentiated Attention is Non-Negotiable**: The Diffuse-CLoC architecture with separate attention patterns for states (bi-directional) and actions (causal) is essential for success, not an optional enhancement.

2. **Independent Noise Schedules**: States and actions require different noise levels during training (k_s ≠ k_a), reflecting their different characteristics and uncertainties.

3. **Body-Pos Representation is Critical**: The 100% vs 72% success rate difference shows this choice is fundamental to sim-to-real transfer, not a minor implementation detail.

4. **Action Horizon Masking**: Limiting action loss to 8 steps while predicting 16 prevents instability from high-uncertainty long-horizon predictions.

5. **Action Delay Randomization**: The 0-100ms randomization during training is specifically designed to handle the ~20ms inference latency in deployment.

6. **Safe OOD Behavior**: The "inert" failure mode of diffusion models (staying still when confused) is actually a safety feature for real-world deployment.

## Conclusion

This refined design plan provides a comprehensive roadmap for implementing Stage 2 of BeyondMimic with critical details that make the difference between success and failure in real-world deployment. The key innovation lies not just in the joint state-action diffusion, but in the careful engineering of differentiated attention, independent noise schedules, and proper state representation. By incorporating these refined insights and building upon the existing motion tracking infrastructure, this implementation will complete the BeyondMimic framework and enable the versatile humanoid control capabilities demonstrated in the paper.