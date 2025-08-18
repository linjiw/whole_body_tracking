# BeyondMimic: Detailed Code Structure Analysis

## Core Components Deep Dive

### 1. Motion Command System (`mdp/commands.py`)

#### MotionLoader Class
```
Purpose: Loads and manages reference motion data
Key Properties:
- fps: Frame rate of motion data
- joint_pos/vel: Reference joint trajectories
- body_pos/quat_w: World frame body poses
- body_lin/ang_vel_w: World frame body velocities
- time_step_total: Total frames in motion
```

#### MotionCommand Class
```
Core Functionality:
- Tracks current time step in reference motion
- Computes relative transformations for anchor-based tracking
- Provides error metrics for all tracking dimensions
- Handles motion resampling and initial state randomization

Key Methods:
- _resample_command(): Random phase selection + pose/velocity perturbation
- _update_command(): Advances time, computes relative transformations
- _update_metrics(): Computes tracking errors for logging

Key Properties:
- command: Combined joint pos/vel for observation
- anchor_*: Reference anchor body state
- robot_anchor_*: Current robot anchor state
- body_*_relative_w: Target body poses relative to current anchor
```

### 2. Reward Functions (`mdp/rewards.py`)

#### Tracking Rewards (Exponential Form)
```python
# General form: exp(-error² / σ²)

motion_global_anchor_position_error_exp(σ=0.3)
- Tracks anchor body position in world frame
- Critical for global positioning

motion_global_anchor_orientation_error_exp(σ=0.4)
- Tracks anchor body orientation
- Uses quaternion error magnitude

motion_relative_body_position_error_exp(σ=0.3)
- Tracks all bodies relative to anchor
- Position-invariant tracking

motion_relative_body_orientation_error_exp(σ=0.4)
- Tracks body orientations relative to anchor
- Preserves motion coordination

motion_global_body_linear_velocity_error_exp(σ=1.0)
- Matches linear velocities
- Important for dynamic motions

motion_global_body_angular_velocity_error_exp(σ=3.14)
- Matches angular velocities
- Critical for spins and rotations
```

### 3. Observation Functions (`mdp/observations.py`)

#### Robot State Observations
```python
robot_anchor_ori_w()
- Returns first 2 columns of rotation matrix (6D representation)
- More stable than quaternions for learning

robot_body_pos_b()
- Body positions in anchor frame
- Uses subtract_frame_transforms for efficiency

motion_anchor_pos_b()
- Target anchor position in current robot anchor frame
- Key for position error feedback

motion_anchor_ori_b()
- Target anchor orientation relative to current
- 6D rotation representation
```

### 4. Environment Configuration (`tracking_env_cfg.py`)

#### Scene Configuration
```python
MySceneCfg:
- terrain: Flat plane with configurable friction
- robot: Loaded from URDF (G1 or SMPL)
- contact_forces: Contact sensor for all bodies
- lights: Proper lighting for rendering
```

#### MDP Configuration
```python
CommandsCfg:
- motion: Reference motion configuration
- pose_range: Initial pose randomization bounds
- velocity_range: Push force bounds
- joint_position_range: Joint noise bounds

ObservationsCfg:
- PolicyCfg: Noisy observations for robustness
- PrivilegedCfg: Clean observations for critic

RewardsCfg:
- Weighted combination of tracking rewards
- Regularization terms for smooth motion

TerminationsCfg:
- Safety conditions for early termination
- Prevents unstable states
```

### 5. Training Infrastructure

#### PPO Agent Configuration (`rsl_rl_ppo_cfg.py`)
```python
G1FlatPPORunnerCfg:
- Network: 3-layer MLP with [512, 256, 128] units
- Rollout: 24 steps × 4096 envs = 98,304 samples/iteration
- Optimization: 5 epochs, 4 mini-batches
- Learning: Adaptive schedule based on KL divergence
```

#### Custom OnPolicyRunner (`my_on_policy_runner.py`)
```
Extensions to base RSL-RL runner:
- WandB integration for logging
- Motion registry support
- Custom metrics tracking
- Video recording support
```

#### Training Script (`train.py`)
```python
Key Features:
- Hydra configuration system
- WandB artifact loading
- Automatic motion file download
- GPU optimization settings
- Video recording during training
```

### 6. Robot Models (`robots/`)

#### G1 Configuration (`g1.py`)
```python
Actuator Parameters:
- Motor types: 5020 (small), 7520 (medium), 4010 (wrist)
- Armature: Motor inertia values
- Natural frequency: 10Hz for all joints
- Damping ratio: 2.0 (critically damped)

Action Scaling:
- Formula: 0.25 × effort_limit / stiffness
- Ensures stable PD control
- Prevents actuator saturation
```

#### SMPL Configuration (`smpl.py`)
```
Human model for motion retargeting:
- Standard SMPL joint hierarchy
- Virtual actuators for all joints
- Used for motion preprocessing
```

### 7. Utility Scripts

#### Motion Preprocessing (`csv_to_npz.py`)
```python
Workflow:
1. Load CSV with joint angles
2. Create SMPL/G1 robot in Isaac Sim
3. Compute forward kinematics
4. Extract body poses and velocities
5. Save as NPZ with all coordinates
6. Upload to WandB registry
```

#### Motion Replay (`replay_npz.py`)
```python
Purpose: Visualize reference motions
Features:
- Load from WandB registry
- Play back at original FPS
- Verify retargeting quality
```

## Data Flow

### Training Pipeline
```
1. Reference Motion (CSV)
   ↓ csv_to_npz.py
2. Processed Motion (NPZ in WandB)
   ↓ train.py loads
3. MotionCommand generates targets
   ↓ 
4. Environment step:
   - Apply actions to robot
   - Compute observations
   - Calculate rewards
   - Check terminations
   ↓
5. PPO update:
   - Collect rollouts
   - Compute advantages
   - Update policy/value networks
   ↓
6. Save checkpoints to WandB
```

### Inference Pipeline
```
1. Load checkpoint from WandB
   ↓
2. Load reference motion
   ↓
3. For each step:
   - Get observations
   - Query policy network
   - Apply actions
   - Visualize result
```

## Key Design Patterns

### 1. Modular MDP Components
- Each MDP function is atomic and reusable
- Clean separation of concerns
- Easy to extend with new rewards/observations

### 2. Configuration-Driven Design
- All hyperparameters in dataclasses
- Inheritance for variants
- Hydra for experiment management

### 3. Efficient Batching
- All operations vectorized for GPU
- Single kernel launches for 4096 envs
- Minimal CPU-GPU transfers

### 4. Robust Coordinate Transformations
- Quaternions for rotations
- 6D representation for learning
- Careful frame management

### 5. Registry-Based Asset Management
- WandB for version control
- Automatic downloading
- Reproducible experiments