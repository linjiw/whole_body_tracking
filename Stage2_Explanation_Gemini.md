# BeyondMimic Stage 2: From Imitation to Intelligence with Guided Diffusion

## 1. The Big Picture: Why Do We Need Stage 2?

Stage 1 of BeyondMimic is a remarkable achievement: it creates highly robust and dynamic "tracking policies" that can make a humanoid robot faithfully imitate a reference human motion. However, these policies are like perfect mimics—they can only replay what they've seen. They cannot adapt to new situations or goals.

**The core limitation of Stage 1 is its lack of versatility.** If you want the robot to walk to a specific point, avoid an obstacle that wasn't in the training data, or follow a joystick, you would need to create a new reference motion and retrain a new policy. This is not a scalable or intelligent solution.

**Stage 2 solves this by transforming the library of "mimics" from Stage 1 into a single, powerful, and versatile "generative policy."** This new policy doesn't just copy motions; it *understands* the underlying principles of movement it has learned. It can then synthesize entirely new, physically realistic motions to solve tasks it has never seen before, all without any retraining.

**The paradigm shift is:**
*   **Stage 1:** "Copy this exact motion."
*   **Stage 2:** "Use the skills you've learned from all motions to achieve this new goal."

---

## 2. The Core Breakthrough: Joint State-Action Diffusion

To achieve this versatility, Stage 2 relies on a key insight from the `Diffuse-CLoC` paper: **modeling the joint distribution of future states and actions in a single diffusion model.**

Let's break down why this is so important by looking at the alternatives:

1.  **Hierarchical Approach (Planner + Tracker):**
    *   **How it works:** A high-level planner (like a kinematics diffusion model) generates a sequence of desired body poses (states). A low-level tracking controller (like a Stage 1 policy) then tries to execute those poses.
    *   **The Flaw (The "Planning-Control Gap"):** The planner doesn't understand physics. It can easily generate a trajectory that is physically impossible for the robot to follow (e.g., floating, unnatural acceleration). The tracking controller, when given this out-of-distribution command, fails.

2.  **Action-Only Diffusion (like `PDP`):**
    *   **How it works:** The model learns to generate a sequence of actions based on the current state. This ensures physical realism because it's only predicting actions.
    *   **The Flaw (No Steerability):** You can't easily guide this model at test time. A goal is usually defined in *state space* (e.g., "be at position X"). There's no direct way to translate that state-based goal into a gradient for an *action sequence*.

### The BeyondMimic/Diffuse-CLoC Solution:

Stage 2 models a future trajectory `τ` that contains **both** states and actions, interleaved:
`τ = [action_t, state_t+1, action_t+1, state_t+2, ...]`

This is the magic bullet. By predicting both what the robot will do (`actions`) and where it will be (`states`) together, we get the best of both worlds:
*   **Physical Realism:** The model learns the intricate physical relationship between an action and its resulting state. The generated trajectories are therefore physically plausible.
*   **Steerability:** Because the model predicts future states, we can apply a cost function directly to those states. We can calculate the gradient of a task cost (e.g., "distance to waypoint") with respect to the *predicted states* and use it to guide the entire generation process.
*   **Bridging the Gap:** This guidance on the states naturally propagates to the actions within the joint model, ensuring the generated motor commands are the correct ones to achieve the guided states. The planning-control gap is eliminated.

---

## 3. The "How-To": A Three-Phase Process

### Phase 1: Data Collection (Building the Expert Dataset)

**What data do we collect?**
We need a large dataset of `(History, Future Trajectory)` pairs.
*   **History `O_t`**: The robot's state and action for the last `N` timesteps (e.g., `N=4`). This provides context.
*   **Future Trajectory `τ_t`**: The robot's subsequent actions and resulting states for the next `H` timesteps (e.g., `H=16` or `H=32`).

**Where does the data come from?**
From the **expert Stage 1 tracking policies**. We run these policies in simulation across all the diverse reference motions (walking, running, jumping, etc.). This gives us a rich dataset of physically-grounded, high-quality, and diverse motions.

**How do we collect it?**
We roll out the policies and, for every timestep, we save the history and the future trajectory. Two critical augmentations are applied during this process:

1.  **Action Noise Injection:** We add small random noise to the actions from the expert policy.
    *   **Why?** This forces the policy to make small corrections. The diffusion model then learns not just the "perfect" motion but also how to recover from small perturbations, which is essential for real-world robustness.
2.  **Action Delay Randomization:** We randomly delay the application of the action by 0-100ms.
    *   **Why?** The diffusion model takes time to run during inference (approx. 20ms). If the model is only trained on instantaneous actions, this real-world latency will cause it to fail. By training on delayed actions, the model learns to be robust to this inference latency.

**What is the data format? (State Representation)**
This is a **critical design choice**. The papers show a massive performance difference between two types of state representations. Stage 2 uses **Body-Pos State**:
*   **Global:** Root position, velocity, and rotation.
*   **Local:** The Cartesian positions and velocities of all major body parts, relative to the robot's root frame.

*   **Why not joint angles?** While seemingly simpler, joint angles suffer from "kinematic chain error accumulation." A small error in a hip joint angle can lead to a huge error at the foot. Body-Pos provides direct spatial coordinates, which is much more stable and provides a better signal for guidance. The ablation in the paper shows this choice is responsible for a jump from 72% to 100% success rate on one task.

### Phase 2: Training the Diffusion Model

**What are we training?**
A Transformer-based model that learns to **denoise** trajectories. The training process is:
1.  Take a clean expert trajectory `τ_0` from our dataset.
2.  Add a random amount of Gaussian noise to get a corrupted version `τ_k`.
3.  Feed `τ_k`, the history `O_t`, and the noise level `k` into the model.
4.  The model's task is to predict the original, clean trajectory `τ_0`.
5.  The loss is simply the Mean Squared Error (MSE) between the model's prediction and the actual clean trajectory.

**What's special about the model architecture?**
The `Diffuse-CLoC` paper introduces a crucial innovation: **Differentiated Attention**.

*   **State Tokens get Bi-directional (Full) Attention:** A state at time `t+5` can see all other states, both past (`t+1`) and future (`t+10`).
    *   **Why?** This enables **planning**. To decide where to be *now*, it's helpful to know where you need to be in the future. If the robot needs to jump over an obstacle in 10 steps, it needs to start adjusting its posture now.
*   **Action Tokens get Causal Attention:** An action at time `t+5` can only see past and present states and actions (up to `t+5`). It cannot see the future.
    *   **Why?** This enables **reactive control** and **prevents contamination**. Actions are the immediate motor commands. They should be based on the current plan, not on noisy, uncertain future *predictions*. Allowing actions to see noisy future state predictions would make them unstable.

This separation is not a minor detail; it's fundamental to making the system work.

**Other Key Training Details:**
*   **Independent Noise Schedules:** States and actions are different modalities. States (plans) can tolerate more noise than actions (sensitive motor commands). Therefore, they are noised according to different schedules.
*   **Action Horizon Masking:** The model might predict a long state horizon (e.g., `H=32` for planning) but a shorter action horizon (e.g., 16). Furthermore, the loss is often only calculated on the *first half* of the predicted actions (e.g., first 8).
    *   **Why?** Predicting actions far into the future is extremely difficult and uncertain. Focusing the training loss on near-term, higher-certainty actions results in a more stable and effective policy.

### Phase 3: Guided Inference (Zero-Shot Control)

This is where the magic happens. We have a trained model that can generate realistic motions. Now we want to steer it to solve a new task.

**How does it work?**
The core is **Classifier Guidance**. At each step of the denoising process (from random noise to a clean trajectory), we do the following:
1.  Let the model make its prediction for the clean trajectory.
2.  Take this predicted trajectory and evaluate it against a simple, differentiable **cost function** that we define for our task. For example: `cost = distance_from_predicted_position_to_waypoint`.
3.  Calculate the **gradient** of this cost with respect to the predicted trajectory. This gradient is our "steering signal." It points in the direction that will most effectively reduce the cost.
4.  Nudge the denoising process in the direction of that gradient.
5.  Repeat for all denoising steps.

The final result is a clean, physically-plausible trajectory that has been "steered" to minimize our task cost.

**Example Cost Functions:**
*   **Waypoint Navigation:** `cost = || predicted_root_position - goal_position ||²`
*   **Joystick Control:** `cost = || predicted_root_velocity - joystick_velocity ||²`
*   **Obstacle Avoidance:** `cost = barrier_function(SDF_distance_to_obstacle)`

Finally, we use a **receding horizon** approach. The model generates a trajectory of `H` steps, but we only execute the very first action `a_t`. Then, we discard the rest, get the new state of the robot, and re-plan from there. This makes the system constantly responsive to the latest sensor information.

---

## 4. Summary of Rationale: What, Why, How

| Feature / Question | What is it? | Why is it done? | How is it done? |
| :--- | :--- | :--- | :--- |
| **Core Idea** | Joint State-Action Diffusion | To bridge the planning-control gap and enable state-based guidance for physically-realizable actions. | Model a single trajectory `τ` containing both future states and actions. |
| **Data Source** | Stage 1 Tracking Policies | To get a large, diverse dataset of expert, physically-valid motions. | Roll out the trained RL policies in simulation. |
| **Data Augmentation** | Action noise & delays | To make the model robust to real-world sensor noise and inference latency. | Add random noise to actions and randomly delay their execution during data collection. |
| **State Representation** | Body-Pos (Cartesian) | To provide direct spatial grounding for guidance and avoid kinematic error accumulation. | Represent state as root pose + Cartesian positions/velocities of body parts. |
| **Model Architecture** | Transformer with Differentiated Attention | To allow states to plan holistically (bi-directional) while actions react causally, preventing instability. | Use a full attention mask for state tokens and a causal (triangular) mask for action tokens. |
| **Control Method** | Classifier Guidance | To enable zero-shot adaptation to new tasks without retraining. | At inference time, use the gradient of a task-specific cost function to steer the denoising process. |
