# BeyondMimic Stage 2: Comprehensive Understanding of Guided Diffusion for Versatile Humanoid Control

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [The Big Picture: Why Stage 2?](#the-big-picture-why-stage-2)
3. [Core Innovation: Joint State-Action Diffusion](#core-innovation-joint-state-action-diffusion)
4. [Phase 1: Data Collection - Building the Foundation](#phase-1-data-collection---building-the-foundation)
5. [Phase 2: Training the Diffusion Model](#phase-2-training-the-diffusion-model)
6. [Phase 3: Guided Inference - The Magic of Test-Time Control](#phase-3-guided-inference---the-magic-of-test-time-control)
7. [Technical Deep Dive: Critical Design Choices](#technical-deep-dive-critical-design-choices)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Key Insights and Rationale](#key-insights-and-rationale)

---

## Executive Summary

Stage 2 of BeyondMimic represents a paradigm shift from motion imitation to intelligent motion synthesis. While Stage 1 taught robots to faithfully copy human motions, Stage 2 enables them to creatively adapt these learned skills to solve new tasks without any retraining. This is achieved through a revolutionary approach: **joint state-action diffusion with classifier guidance**.

The key insight: By modeling both future states (where the robot will be) and actions (how to get there) together in a single diffusion model, we can use simple cost functions at test time to steer the robot toward any goal - whether that's reaching a waypoint, avoiding obstacles, or following joystick commands.

---

## The Big Picture: Why Stage 2?

### The Limitation of Stage 1

Stage 1 (motion tracking) produces impressive results - robots can perform backflips, dance, and sprint. However, these policies are **reactive mimics**. They can only reproduce the exact motions they were trained on. If you want the robot to:
- Navigate to a specific location
- Avoid an unexpected obstacle
- Follow real-time joystick commands
- Combine learned skills in novel ways

You would need to retrain the entire policy with new reward functions and reference motions. This is impractical and doesn't scale.

### The Promise of Stage 2

Stage 2 transforms these rigid motion trackers into a **versatile, intelligent controller** that can:
1. **Reuse learned skills**: Leverage all motions from Stage 1
2. **Zero-shot adaptation**: Solve new tasks without retraining
3. **Real-time control**: Respond to dynamic goals and obstacles
4. **Creative synthesis**: Combine and blend motions to achieve objectives

### The Fundamental Shift

```
Stage 1: "Copy this exact motion"
Stage 2: "Use whatever motions you know to achieve this goal"
```

This is the difference between a robot that can only replay recordings versus one that understands movement primitives and can compose them intelligently.

---

## Core Innovation: Joint State-Action Diffusion

### The Problem with Existing Approaches

Before BeyondMimic, there were two main approaches, each with critical flaws:

#### 1. Hierarchical Approach (Kinematics + Tracking)
```
High-level Planner → Kinematic Motion → Tracking Controller → Robot Actions
```
**Problem**: The "planning-control gap" - planners generate physically impossible motions that tracking controllers can't execute, leading to falls and failures.

#### 2. Pure Action Diffusion
```
Current State → Diffusion Model → Actions
```
**Problem**: Can't be steered at test time because you can't compare task goals (in state space) with action sequences (in joint space).

### The BeyondMimic Solution: Co-Diffusion

BeyondMimic's breakthrough is modeling the **joint distribution** p(states, actions):

```
τ = [action_t, state_t+1, action_t+1, ..., state_t+H, action_t+H]
```

This means the model predicts both:
- **Future states**: Where the robot will be (positions, velocities)
- **Actions**: The motor commands to get there

### Why This Works

By predicting states and actions together:
1. **Actions are grounded in physics**: The model learns which actions lead to which states
2. **States can be guided**: We can apply cost functions to the predicted states
3. **Guidance propagates to actions**: Steering the states automatically steers the actions
4. **No planning-control gap**: Actions and states are jointly consistent

---

## Phase 1: Data Collection - Building the Foundation

### What Data Do We Need?

We need trajectories that demonstrate the relationship between:
- **States**: Robot configurations (joint positions, velocities, body positions)
- **Actions**: Motor commands that transition between states
- **History**: Recent states/actions that provide context

### Where Does This Data Come From?

**Source**: The Stage 1 motion tracking policies!

These policies already know how to:
- Track diverse human motions (walking, jumping, dancing)
- Maintain balance and stability
- Recover from perturbations
- Generate physically valid actions

### How Do We Collect It?

#### Step 1: Load Trained Tracking Policies
```python
# Load all the Stage 1 policies
policies = {
    "walking": load_policy("walk_policy.onnx"),
    "jumping": load_policy("jump_policy.onnx"),
    "dancing": load_policy("dance_policy.onnx"),
    # ... more policies
}
```

#### Step 2: Roll Out Policies with Perturbations
```python
for episode in range(num_episodes):
    # Initialize environment with random motion
    env.reset(motion=random.choice(motions))
    
    for timestep in range(episode_length):
        # Get current observation
        obs = env.get_observation()
        
        # Get action from policy
        action = policy(obs)
        
        # CRITICAL: Add noise for robustness
        action += noise * random.normal()
        
        # CRITICAL: Add action delay (0-100ms) 
        # This prepares for 20ms inference latency
        if random() < 0.5:
            action = delay_buffer.get_delayed_action()
        
        # Step environment
        next_state = env.step(action)
        
        # Save trajectory chunk
        save_trajectory(
            history=[s_{t-4}, a_{t-4}, ..., s_t],
            future=[a_t, s_{t+1}, ..., a_{t+16}, s_{t+16}]
        )
```

#### Step 3: Critical Data Augmentation

**Why Add Noise?**
- Without noise, the diffusion model only learns "perfect" trajectories
- Real deployment has sensor noise, model errors, external disturbances
- Noise teaches the model to generate corrective actions

**Why Add Action Delays?**
- Diffusion inference takes ~20ms in real-time
- Training with 0-100ms random delays teaches the model to handle this latency
- Without this, the robot fails due to control delay

### Data Format and Storage

Each trajectory chunk contains:
```python
{
    "observation_history": {
        "states": [s_{t-4}, s_{t-3}, s_{t-2}, s_{t-1}, s_t],  # Past 4 + current
        "actions": [a_{t-4}, a_{t-3}, a_{t-2}, a_{t-1}],      # Past 4 actions
    },
    "future_trajectory": {
        "states": [s_{t+1}, s_{t+2}, ..., s_{t+16}],          # Next 16 states
        "actions": [a_t, a_{t+1}, ..., a_{t+15}],              # Next 16 actions
    },
    "metadata": {
        "motion_type": "walking",
        "success": true,
        "timestep": 1024
    }
}
```

**State Representation (Critical Choice)**:
Based on ablations, we use **Body-Pos** representation:
```python
state = {
    # Global (relative to current frame)
    "root_position": [x, y, z],
    "root_velocity": [vx, vy, vz], 
    "root_rotation": [rx, ry, rz],  # Rotation vector
    
    # Local (in character frame)
    "body_positions": [[x,y,z] for each body],  # All 23 bodies
    "body_velocities": [[vx,vy,vz] for each body]
}
```

**Why Body-Pos Instead of Joint Angles?**
- Joint angles accumulate errors through kinematic chain
- Body positions provide direct spatial grounding for guidance
- Ablation shows 100% vs 72% success rate difference!

---

## Phase 2: Training the Diffusion Model

### What Are We Training?

A **denoising diffusion model** that learns to predict clean trajectories from noisy ones.

### The Training Process

#### 1. Forward Process (Adding Noise)
```python
# Start with clean trajectory
clean_trajectory = τ_0

# Gradually add noise over K steps
for k in range(K):
    noisy_trajectory = add_gaussian_noise(clean_trajectory, noise_level=k)
```

#### 2. Reverse Process (Learning to Denoise)
```python
# Model learns to predict clean from noisy
predicted_clean = model(noisy_trajectory, observation_history, noise_level)

# Loss function
loss = MSE(predicted_clean, clean_trajectory)
```

### The Architecture: Why Transformer?

The model uses a **Transformer decoder** architecture because:
1. **Long-range dependencies**: Can model relationships across the entire trajectory
2. **Attention mechanism**: Can focus on relevant parts of the history
3. **Scalability**: Handles variable-length sequences efficiently

### Critical Innovation: Differentiated Attention

This is **THE KEY** innovation from Diffuse-CLoC that makes everything work:

```python
# State tokens: Bi-directional attention (can see all states)
state_attention_mask = None  # Full attention

# Action tokens: Causal attention (can only see past)
action_attention_mask = causal_mask()  # Lower triangular
```

**Why Different Attention Patterns?**

Think of it as a team with two types of roles:

**States = Strategic Managers** 👔
- Need to see the whole picture (past, present, future)
- Make high-level plans about "where to be"
- Can brainstorm about uncertain futures
- Bi-directional attention enables holistic planning

**Actions = Front-line Workers** 👷
- Focus on immediate execution of "how to get there"
- Follow current instructions based on current/past information
- Ignore noisy future speculation that would cause confusion
- Causal attention ensures precise, reliable execution

This separation is crucial: managers (states) can afford to consider noisy future scenarios for planning, but workers (actions) need clear, reliable information for execution. Future state predictions are inherently noisy and would corrupt actions if directly attended to.

### Independent Noise Schedules

Another critical detail:
```python
# States and actions get different noise levels
noise_level = (k_state, k_action)

# States can handle more noise (planning is forgiving)
state_noise_schedule = LinearSchedule(start=1e-4, end=0.02)

# Actions need less noise (control is sensitive)
action_noise_schedule = LinearSchedule(start=1e-4, end=0.01)
```

### Action Horizon Masking

Although we predict 16 steps ahead, we only compute loss on the first 8 actions:
```python
action_loss = MSE(predicted_actions[:8], true_actions[:8])
# Ignore actions [8:16] in loss computation
```

**Why?**
- Long-term action prediction has high uncertainty
- Focusing on near-term actions improves stability
- We still predict far to maintain trajectory consistency

---

## Phase 3: Guided Inference - The Magic of Test-Time Control

### The Core Concept: Classifier Guidance

Once trained, the model can be steered using **gradient-based guidance**:

```python
# Define a cost function for your task
def waypoint_cost(trajectory, goal):
    positions = extract_positions(trajectory)
    return sum((pos - goal)^2 for pos in positions)

# During inference, compute gradient
gradient = compute_gradient(waypoint_cost, trajectory)

# Modify the denoising step
guided_trajectory = original_trajectory - guidance_scale * gradient 
```

### How Guidance Works

#### Step 1: Initialize with Noise
```python
trajectory = random_noise()  # Start with pure randomness
```

#### Step 2: Iterative Denoising with Guidance
```python
for k in reversed(range(num_denoising_steps)):  # Takes ~20ms total
    # Predict clean trajectory
    clean_pred = model(trajectory, history, k)
    
    # Compute task cost
    cost = task_cost_function(clean_pred)
    
    # Compute gradient (this is the "steering signal")
    gradient = autograd.grad(cost, trajectory)
    
    # Update trajectory (denoising + guidance)
    trajectory = denoise_step(trajectory) - guidance_scale * gradient
```

#### Step 3: Execute First Action
```python
# Only execute the immediate action
action = trajectory[0]  # First action in sequence
robot.execute(action)

# Replan at next timestep (receding horizon control) 
```

### Example Task Cost Functions

#### Waypoint Navigation
```python
def waypoint_cost(trajectory, goal_position):
    positions = extract_root_positions(trajectory)
    distances = [||pos - goal||^2 for pos in positions]
    
    # Weight near-term positions more
    weights = exponential_decay(len(positions))
    return sum(distances * weights)
```

#### Obstacle Avoidance
```python
def obstacle_cost(trajectory, obstacles):
    body_positions = extract_all_body_positions(trajectory)
    costs = []
    
    for body_pos in body_positions:
        distance_to_obstacle = compute_SDF(body_pos, obstacles)
        # Barrier function: infinite cost inside, smooth decay outside
        costs.append(barrier_function(distance_to_obstacle))
    
    return sum(costs)
```

#### Joystick Control
```python
def joystick_cost(trajectory, desired_velocity):
    velocities = extract_root_velocities(trajectory)
    return sum((vel - desired_velocity)^2 for vel in velocities)
```

### Why This Works: The State-Action Bridge

The magic is that:
1. **Cost functions operate on states** (positions, velocities)
2. **States and actions are jointly modeled**
3. **Gradients on states propagate to actions**
4. **Result: Task-aware action generation**

Without joint modeling, you couldn't connect task objectives to motor commands!

**The Central Magic Summarized**: Cost functions are defined in the intuitive state space (where you want the robot to be), but because states and actions are modeled jointly in a single diffusion process, the guidance gradient applied to states automatically flows through the model to directly influence the action space, producing physically-valid, goal-directed motor commands. This is the bridge that makes everything work.

---

## Technical Deep Dive: Critical Design Choices

### 1. Why Diffusion Instead of Other Generative Models?

**Advantages over VAEs**:
- Better mode coverage (can represent multi-modal distributions)
- No posterior collapse
- Higher quality generation

**Advantages over GANs**:
- Stable training (no adversarial dynamics)
- Better mode coverage (no mode collapse)
- Principled likelihood estimation

**Advantages over Autoregressive Models**:
- Parallel generation
- Bidirectional context
- Natural guidance mechanism

### 2. Why Not Separate Planning and Control?

**The Planning-Control Gap Problem**:
```
Kinematic Planner: "Jump 2 meters high"
        ↓
Tracking Controller: "Physically impossible!"
        ↓
Result: Robot falls
```

Joint modeling ensures physical feasibility by construction.

### 3. Why Transformer Architecture?

**Spatial Reasoning**: Can model relationships between all body parts simultaneously

**Temporal Reasoning**: Can model long-range dependencies in motion sequences

**Attention Flexibility**: Can implement differentiated attention patterns

**Scalability**: Efficiently handles variable-length sequences

### 4. The Critical Role of Rolling Inference

Instead of generating new trajectories from scratch each time:
```python
# FIFO buffer maintains trajectory - creating a "commitment gradient"
buffer = [clean_t-2, clean_t-1, noisy_t, noisy_t+1, ..., noisy_t+15]
#         ← committed → ← adjusting → ← planning → ← exploring →

# Each step:
1. Pop oldest (clean_t-2)        # Remove executed action
2. Push newest (noisy_t+16)      # Add new future to explore
3. Denoise buffer                # Refine the trajectory
4. Execute first action          # Act on the refined plan
```

**Dual Benefits**:

**Benefit 1 - Consistency (Physical Momentum)**:
- Maintains commitment to recent decisions
- Prevents sudden mode switches (e.g., walk→jump→crawl)
- Creates natural, smooth motion like real physics

**Benefit 2 - Speed (Warm-Start Optimization)**:
- Reuses previous denoising computation
- 25% faster by starting from partial solution instead of pure noise
- Previous solution provides excellent initialization for gradient descent

**Why This Matters**: Physical systems have inertia - a running robot can't instantly reverse direction. Rolling inference respects this physical reality while also providing computational benefits. It's not just an optimization trick; it's fundamental to generating realistic motion.

### 5. Why These Specific Hyperparameters?

**History Length N=4 (~0.13s)**:
- Enough context for motion phase
- Not so long that it biases against transitions

**Prediction Horizon H=32 (~1s)**:
- Long enough for obstacle avoidance planning
- Short enough to maintain accuracy

**Action Horizon = 16 steps**:
- Balance between planning and control accuracy
- Loss only on first 8 for stability

**Denoising Steps = 20**:
- Quality vs speed tradeoff
- Sufficient for clean generation
- Fast enough for real-time (20ms)

---

## Implementation Roadmap

### Phase 1: Data Infrastructure (Week 1-2)
```
Goal: Collect 100+ hours of trajectory data
```
1. Set up data collection pipeline
2. Implement trajectory chunking
3. Add noise and delay augmentation
4. Validate data quality

### Phase 2: Model Architecture (Week 3-4)
```
Goal: Implement and validate model architecture
```
1. Build transformer with differentiated attention
2. Implement independent noise schedules
3. Add action horizon masking
4. Verify gradient flow

### Phase 3: Training Pipeline (Week 5-6)
```
Goal: Train robust diffusion model
```
1. Implement DDPM training loop
2. Set up distributed training
3. Monitor convergence metrics
4. Validate on held-out data

### Phase 4: Guidance System (Week 7-8)
```
Goal: Enable task-specific control
```
1. Implement classifier guidance
2. Create task cost functions
3. Build SDF for obstacles
4. Test gradient computation

### Phase 5: Integration (Week 9-10)
```
Goal: Deploy in Isaac Sim (Critical for safe testing)
```
1. **Isaac Sim Testing First** (Essential for safety):
   - Test all tasks without risking $30k+ robot hardware
   - Validate trajectories are physically valid
   - Debug with perfect state information
   - Run 1000s of test scenarios in parallel
2. **Real-time Optimization**:
   - Export to ONNX format for deployment
   - TensorRT acceleration (20ms inference target)
3. **System Integration**:
   - ROS bridge for robot communication
   - Asynchronous inference pipeline
4. **Safety Validation**:
   - Joint limit checking
   - Collision detection
   - Emergency stop protocols

### Phase 6: Validation (Week 11-12)
```
Goal: Comprehensive testing
```
1. Benchmark against baselines
2. Ablation studies
3. Failure analysis
4. Documentation

---

## Key Insights and Rationale

### Insight 1: The Power of Joint Modeling

**Traditional Approach**: Separate what (planning) from how (control)

**BeyondMimic**: Unify what and how in a single model

**Rationale**: Physical feasibility emerges from joint training rather than being enforced through constraints.

### Insight 2: Guidance as Universal Control

**Traditional**: Train new policy for each task

**BeyondMimic**: One policy + different cost functions = infinite tasks

**Rationale**: The motion manifold learned during training contains the building blocks for any reasonable task.

### Insight 3: History as Commitment

**Problem**: Pure replanning causes oscillation

**Solution**: Rolling inference maintains trajectory commitment

**Rationale**: Physical systems have momentum; decisions should persist.

### Insight 4: Differentiated Processing

**Key Innovation**: States and actions need different treatment

**States**: Global planning → bi-directional attention
**Actions**: Local control → causal attention

**Rationale**: Planning and control have fundamentally different information requirements.

### Insight 5: Robustness Through Noise

**Critical**: Training with noise and delays

**Without noise**: Model fails on first imperfection
**With noise**: Model learns recovery and correction

**Rationale**: Real world is noisy; training must reflect this.

### The Ultimate Achievement

Stage 2 transforms BeyondMimic from a **motion imitator** to a **motion intelligence**:

- **Stage 1**: "I can copy these 50 motions perfectly"
- **Stage 2**: "I understand movement and can solve any task you give me"

This is not just incremental improvement - it's a fundamental leap in capability, enabling robots to exhibit creative, adaptive, and intelligent motion behavior that goes far beyond simple mimicry.

---

## Conclusion

Stage 2 represents the culmination of several breakthrough insights:
1. Joint state-action modeling bridges planning and control
2. Diffusion provides a principled way to generate and guide trajectories
3. Differentiated attention respects the different natures of planning and control
4. Classifier guidance enables zero-shot task adaptation
5. Proper data collection and augmentation ensure real-world robustness

The result is a system that can take the motion skills learned in Stage 1 and creatively compose them to solve arbitrary tasks in real-time, without any retraining. This is the key to truly versatile humanoid control.

The implementation requires careful attention to details - from data collection protocols to architecture choices to training procedures. But when done correctly, it unlocks capabilities that were previously impossible: robots that don't just copy, but truly understand and creatively use human motion.