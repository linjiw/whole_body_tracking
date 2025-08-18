# BeyondMimic: Comprehensive Algorithm Analysis

## Project Overview

BeyondMimic is a state-of-the-art humanoid motion tracking framework built on Isaac Lab (v2.1.0) that enables sim-to-real deployment of humanoid robots to replicate complex motions. The key innovation is training highly dynamic motion tracking policies that can generalize across the LAFAN1 dataset without parameter tuning.

## Core Architecture

### 1. Motion Tracking Pipeline

The system follows a Markov Decision Process (MDP) formulation with the following components:

#### Motion Loader & Command System
- **Motion Storage**: Reference motions are stored in WandB registry as NPZ files containing:
  - Joint positions and velocities
  - Body positions, orientations in world frame
  - Body linear and angular velocities
  - FPS information for time synchronization

- **Motion Command Generation**: The `MotionCommand` class manages:
  - Time step tracking for motion playback
  - Anchor body alignment (typically torso/pelvis)
  - Relative body transformations
  - Real-time error metrics computation

#### Key Innovation: Anchor-Based Tracking
The system uses an "anchor body" approach where:
1. One body (typically torso) serves as the reference frame
2. All other body positions/orientations are computed relative to this anchor
3. The anchor alignment allows for position-invariant tracking
4. Yaw-only orientation alignment preserves motion dynamics

### 2. Reward Function Design

The reward structure follows DeepMimic with enhancements:

#### Primary Tracking Rewards (Exponential form: exp(-error/σ²))
1. **Global Anchor Position** (σ=0.3): Tracks anchor body position in world frame
2. **Global Anchor Orientation** (σ=0.4): Tracks anchor body orientation
3. **Relative Body Position** (σ=0.3): Tracks all body positions relative to anchor
4. **Relative Body Orientation** (σ=0.4): Tracks all body orientations relative to anchor
5. **Global Body Linear Velocity** (σ=1.0): Matches body velocities
6. **Global Body Angular Velocity** (σ=3.14): Matches angular velocities

#### Regularization Terms
- **Action Rate L2** (weight=-0.1): Penalizes rapid action changes
- **Joint Limits** (weight=-10.0): Strong penalty for exceeding soft joint limits
- **Undesired Contacts** (weight=-0.1): Penalizes contacts on non-foot/hand bodies

### 3. Observation Space

The observation space is carefully designed with two levels:

#### Policy Observations (with noise injection)
- Motion command: target joint positions and velocities
- Anchor body relative position/orientation
- Base linear/angular velocities
- Joint positions relative to default
- Joint velocities
- Previous actions

#### Privileged Observations (critic only, no noise)
- All policy observations without noise
- Full body positions and orientations
- Used for value function estimation during training

### 4. Domain Randomization

The system employs extensive domain randomization for sim-to-real transfer:

#### Startup Randomization
- **Physics Materials**: Friction (0.3-1.6 static, 0.3-1.2 dynamic), restitution (0.0-0.5)
- **Joint Default Positions**: ±0.01 rad perturbation
- **Center of Mass**: ±2.5cm in x, ±5cm in y/z for torso

#### Episode Randomization
- **Initial State**: Random phase sampling from reference motion
- **Pose Perturbation**: Position (±5cm x/y, ±1cm z), orientation (±0.1 rad roll/pitch, ±0.2 rad yaw)
- **Velocity Perturbation**: Linear (±0.5 m/s), angular (±0.52 rad/s roll/pitch, ±0.78 rad/s yaw)
- **Joint Position**: ±0.1 rad from reference

#### Runtime Randomization
- **Push Forces**: Applied every 1-3 seconds with velocity perturbations

### 5. Training Algorithm: PPO Configuration

The training uses Proximal Policy Optimization with specific hyperparameters:

#### Network Architecture
- **Actor**: [512, 256, 128] hidden units with ELU activation
- **Critic**: [512, 256, 128] hidden units with ELU activation
- **Initialization**: 1.0 noise std for exploration

#### PPO Parameters
- **Rollout**: 24 steps per environment, 4096 environments
- **Learning**: 5 epochs, 4 mini-batches
- **Learning Rate**: 1e-3 with adaptive schedule
- **Discount**: γ=0.99, λ=0.95 (GAE)
- **Clipping**: ε=0.2 for policy, value loss clipping enabled
- **Entropy**: 0.005 coefficient for exploration
- **KL Target**: 0.01 for adaptive learning rate

#### Training Details
- **Empirical Normalization**: Enabled for observation standardization
- **Gradient Clipping**: max_grad_norm=1.0
- **Simulation**: 200Hz physics (dt=0.005s), 50Hz control (decimation=4)
- **Episode Length**: 10 seconds

### 6. Robot Configuration (Unitree G1)

The G1 robot configuration includes:

#### Actuator Model
- **Implicit Actuators**: PD control with motor dynamics
- **Motor Inertia (Armature)**: Varies by motor type (5020, 7520, 4010)
- **Natural Frequency**: 10Hz for all joints
- **Damping Ratio**: 2.0 (critically damped)
- **Stiffness**: K = armature × (2π×10)²
- **Damping**: D = 2 × damping_ratio × armature × (2π×10)

#### Action Scaling
- Computed as: 0.25 × effort_limit / stiffness
- Ensures stable control within actuator limits

#### Joint Configuration
- **Legs**: Hip (yaw/roll/pitch), knee joints with high torque motors
- **Feet**: Ankle (pitch/roll) with medium torque
- **Waist**: 3-DOF for torso mobility
- **Arms**: 7-DOF each (shoulder 3-DOF, elbow 1-DOF, wrist 3-DOF)

### 7. Key Algorithmic Insights

#### Adaptive Sampling (Under Development)
- Dynamically adjusts motion phase sampling based on tracking difficulty
- Addresses numerical stability issues in current implementation

#### Motion Retargeting Requirements
- Motions must be in generalized coordinates
- Requires forward kinematics preprocessing
- Stores full kinematic chain information (pos, vel, acc)

#### Termination Conditions
- Bad anchor position (z > 0.25m deviation)
- Bad anchor orientation (> 0.8 rad error)
- Bad end-effector positions (feet/hands z > 0.25m deviation)
- Episode timeout (10 seconds)

### 8. Implementation Details

#### WandB Integration
- Motion registry for version control
- Training metrics logging
- Model checkpoint management
- Automatic artifact tracking

#### Coordinate Frames
- World frame: Global reference
- Body frame: Local to each rigid body
- Anchor frame: Relative to anchor body
- All transformations use quaternions for stability

#### Performance Optimizations
- Batched tensor operations for 4096 parallel environments
- GPU-accelerated physics simulation
- Efficient quaternion operations
- Vectorized reward computation

## Training Workflow

1. **Motion Preparation**
   - Retarget motion to robot morphology
   - Convert CSV to NPZ with forward kinematics
   - Upload to WandB registry

2. **Environment Setup**
   - Load motion from registry
   - Initialize 4096 parallel environments
   - Configure domain randomization

3. **Policy Training**
   - PPO with adaptive learning rate
   - Empirical observation normalization
   - Checkpoint every 500 iterations
   - WandB logging for monitoring

4. **Evaluation**
   - Load checkpoint from WandB
   - Visualize tracking performance
   - Compute tracking metrics

## Future Directions

- Adaptive sampling for improved sample efficiency
- Deployment pipeline for real hardware
- Extended motion datasets
- Multi-motion policies
- Guided diffusion controllers for test-time control