# BeyondMimic: Mathematical Formulation

## 1. Problem Formulation

### Markov Decision Process (MDP)

The motion tracking problem is formulated as an MDP: (S, A, P, R, γ)

- **State Space S**: Robot proprioceptive state + reference motion state
- **Action Space A**: Target joint positions
- **Transition P**: Physics simulation dynamics
- **Reward R**: Tracking rewards + regularization
- **Discount γ**: 0.99

## 2. State Representation

### Robot State
```
s_robot = {q, q̇, p_anchor, R_anchor, v_anchor, ω_anchor}
```
- q ∈ ℝ^n: Joint positions
- q̇ ∈ ℝ^n: Joint velocities  
- p_anchor ∈ ℝ³: Anchor body position
- R_anchor ∈ SO(3): Anchor body rotation
- v_anchor ∈ ℝ³: Anchor linear velocity
- ω_anchor ∈ ℝ³: Anchor angular velocity

### Reference Motion State
```
s_ref = {q*, q̇*, p*_i, R*_i, v*_i, ω*_i} for i ∈ bodies
```
- Superscript * denotes reference values
- Subscript i indexes body parts

## 3. Coordinate Transformations

### Anchor-Relative Transformation

For each body i, compute relative pose to anchor:

```
Position: p̃_i = R^T_anchor(p_i - p_anchor)
Orientation: R̃_i = R^T_anchor R_i
```

This makes tracking position-invariant and rotation-invariant (yaw only).

### Yaw-Only Alignment

Extract yaw component for orientation alignment:
```
R_yaw = Rz(atan2(R[1,0], R[0,0]))
```
Where Rz is rotation around z-axis.

## 4. Observation Space

### Policy Observations (with noise η)

```
o_policy = [
    q* - q_default,           # Target joint positions (relative)
    q̇*,                       # Target joint velocities
    p̃*_anchor,                # Target anchor position in robot frame
    vec(R̃*_anchor)[:2],       # Target anchor orientation (6D repr)
    v_robot,                   # Robot base linear velocity
    ω_robot,                   # Robot base angular velocity
    q - q_default,             # Current joint positions (relative)
    q̇,                        # Current joint velocities
    a_prev                     # Previous actions
] + η
```

Noise model: η ~ Uniform(η_min, η_max) with bounds specified per observation type.

### 6D Rotation Representation

For rotation matrix R ∈ SO(3), use first two columns:
```
vec(R)[:2] = [R[:,0], R[:,1]] ∈ ℝ^6
```
Third column can be recovered: R[:,2] = R[:,0] × R[:,1]

## 5. Action Space

Actions are target joint positions:
```
a = q_target ∈ [-π, π]^n
```

Applied through PD controller:
```
τ = K_p(a × scale - q) - K_d q̇
```

Where scale is computed as:
```
scale = 0.25 × τ_max / K_p
```

## 6. Reward Function

### Total Reward
```
R = Σ_i w_i r_i
```

### Tracking Rewards (Exponential Form)

All tracking rewards use exponential kernel: r = exp(-e²/σ²)

#### Anchor Position
```
e_pos = ||p*_anchor - p_anchor||₂
r_pos = exp(-e²_pos / σ²_pos), σ_pos = 0.3
```

#### Anchor Orientation
```
e_ori = arccos((tr(R*_anchor R^T_anchor) - 1) / 2)
r_ori = exp(-e²_ori / σ²_ori), σ_ori = 0.4
```

#### Body Positions (Relative)
```
e_body_pos = 1/N Σ_i ||p̃*_i - p̃_i||²₂
r_body_pos = exp(-e_body_pos / σ²_body_pos), σ_body_pos = 0.3
```

#### Body Orientations (Relative)
```
e_body_ori = 1/N Σ_i quat_error(R̃*_i, R̃_i)²
r_body_ori = exp(-e_body_ori / σ²_body_ori), σ_body_ori = 0.4
```

#### Velocities
```
e_lin_vel = 1/N Σ_i ||v*_i - v_i||²₂
r_lin_vel = exp(-e_lin_vel / σ²_lin_vel), σ_lin_vel = 1.0

e_ang_vel = 1/N Σ_i ||ω*_i - ω_i||²₂
r_ang_vel = exp(-e_ang_vel / σ²_ang_vel), σ_ang_vel = π
```

### Regularization Terms

#### Action Rate
```
r_action_rate = -||a_t - a_{t-1}||²₂
```

#### Joint Limits
```
r_limits = -Σ_j max(0, |q_j| - q_limit_j)²
```

#### Undesired Contacts
```
r_contact = -Σ_b∉{feet,hands} 1[F_contact_b > threshold]
```

## 7. Domain Randomization

### Initial State Randomization

Sample initial phase φ ~ U(0, 1), then:
```
t_0 = φ × T_motion
```

Add pose perturbations:
```
p_0 = p*(t_0) + δp, δp ~ U(p_min, p_max)
R_0 = R*(t_0) × R_δ, R_δ = Euler(δr, δp, δy)
```

### Physics Randomization

Static friction: μ_s ~ U(0.3, 1.6)
Dynamic friction: μ_d ~ U(0.3, 1.2)
Restitution: e ~ U(0.0, 0.5)

### Push Forces

Every Δt ~ U(1, 3) seconds, apply:
```
v_push ~ U(v_min, v_max)
ω_push ~ U(ω_min, ω_max)
```

## 8. PPO Optimization

### Objective Function

```
L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is importance ratio
- Â_t is GAE advantage estimate
- ε = 0.2 is clipping parameter

### Generalized Advantage Estimation (GAE)

```
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

With γ = 0.99, λ = 0.95

### Value Function Loss

```
L^VF(θ) = max((V_θ(s_t) - V_target)², 
              (clip(V_θ(s_t), V_old - ε, V_old + ε) - V_target)²)
```

### Entropy Bonus

```
H(π) = -E_π[log π(a|s)]
```

### Total Loss

```
L_total = L^CLIP - c₁L^VF + c₂H(π)
```

With c₁ = 1.0 (value coefficient), c₂ = 0.005 (entropy coefficient)

## 9. Adaptive Learning Rate

Based on KL divergence:
```
If KL[π_old || π] > 2 × KL_target: lr *= 0.5
If KL[π_old || π] < 0.5 × KL_target: lr *= 1.5
```

With KL_target = 0.01

## 10. Termination Conditions

Episode terminates if:

1. **Bad Anchor Height**: |z_anchor - z*_anchor| > 0.25
2. **Bad Anchor Orientation**: quat_error(R_anchor, R*_anchor) > 0.8
3. **Bad End-Effector Height**: |z_ee - z*_ee| > 0.25 for ee ∈ {feet, hands}
4. **Timeout**: t > T_max = 10 seconds

## 11. Actuator Model

### PD Controller with Motor Dynamics

```
τ = K_p(q_target - q) - K_d q̇
```

With gains computed from motor parameters:
```
K_p = I_motor × (2πf)²
K_d = 2 × ζ × I_motor × (2πf)
```

Where:
- I_motor: Motor armature (inertia)
- f = 10 Hz: Natural frequency
- ζ = 2.0: Damping ratio

### Torque Limits

```
τ_applied = clip(τ, -τ_max, τ_max)
```

## 12. Simulation Parameters

- Physics timestep: dt_physics = 5ms (200 Hz)
- Control timestep: dt_control = 20ms (50 Hz)
- Decimation: 4 (control every 4 physics steps)
- Episode length: 10 seconds
- Rollout length: 24 control steps
- Batch size: 4096 environments