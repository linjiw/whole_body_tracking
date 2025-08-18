# Analysis of Multi-Motion Training in BeyondMimic

This document provides a comprehensive analysis of the BeyondMimic framework's capabilities for training on single versus multiple motions. It synthesizes information from the "BeyondMimic" paper, the current codebase, and existing analysis documents to clarify the present state and propose a concrete plan for future development.

## 1. Executive Summary: Single vs. Multiple Motions

**Direct Answer:** The "BeyondMimic" paper describes a complete system designed for **multi-motion** tracking and synthesis. However, the current codebase implementation exclusively supports **training one motion at a time**.

- **Paper's Vision:** The paper outlines a two-stage framework:
    1.  **Scalable Motion Tracking:** Train robust policies on diverse, minutes-long motion sequences. A key feature for this is **Adaptive Sampling**, which focuses training on difficult parts of the motion, making it feasible to learn from long, complex sequences.
    2.  **Guided Diffusion:** Distill the skills from one or more trained tracking policies into a single, unified diffusion policy that can be guided at test-time to perform novel tasks (e.g., waypoint navigation, joystick control).

- **Current Code Reality:**
    - The implementation focuses entirely on the **first stage (Motion Tracking)**.
    - The training script (`scripts/rsl_rl/train.py`) is configured to load exactly one motion artifact from a WandB registry for each training run.
    - The crucial **Adaptive Sampling** mechanism described in the paper (Section III-F) is **not implemented**. This makes it inefficient and difficult to train on the long, multi-skill reference motions the paper targets.
    - The **Guided Diffusion** policy (Section IV) is **entirely absent** from the codebase.

In essence, the current system can create excellent "expert" policies for individual motions but does not yet have the capability to train on multiple motions simultaneously or compose them for downstream tasks as envisioned in the paper.

## 2. Detailed Analysis: Paper vs. Code

### 2.1. Scalable Motion Tracking

| Feature | Paper's Description (Section III) | Code Implementation | Gap Analysis |
| :--- | :--- | :--- | :--- |
| **Core Training** | A unified MDP and reward structure to track a reference motion. | **Implemented.** The files in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/` (rewards, observations, etc.) align well with the paper's description. | Low. The foundational tracking is solid. |
| **Motion Input** | Can handle "minutes-long human references containing many distinct motions." | **Partially Implemented.** The `MotionLoader` can technically load a long motion file, but the training loop is not optimized for it. The `train.py` script loads only one motion artifact. | High. The system is used for single, short motions, not the long, diverse sequences mentioned. |
| **Adaptive Sampling** | **Crucial for multi-motion.** Samples difficult segments of a long motion more frequently to improve training efficiency and robustness. Uses failure statistics to create a non-uniform sampling distribution. | **Not Implemented.** The environment resets by sampling uniformly from the motion's timeline. There is no mechanism to track failure rates per segment or adapt the sampling strategy. | **Critical.** This is the key missing feature that prevents effective training on long, multi-skill motions. Without it, the agent wastes time on easy segments and fails to master hard ones. |

### 2.2. Trajectory Synthesis via Guided Diffusion

| Feature | Paper's Description (Section IV) | Code Implementation | Gap Analysis |
| :--- | :--- | :--- | :--- |
| **Core Model** | A state-action co-diffusion model (`Diffuse-CLoC`) trained on offline data from the tracking policies. | **Not Implemented.** No diffusion models, denoising networks, or related training logic exist in the codebase. | **Complete.** This entire second stage of the project is not yet started. |
| **Guidance** | Uses classifier guidance with differentiable cost functions (`G_js`, `G_wp`, `G_sdf`) to steer the robot at test time. | **Not Implemented.** No guidance mechanisms or cost functions are present. | **Complete.** |
| **Downstream Tasks** | Enables zero-shot control for joystick steering, waypoint navigation, and obstacle avoidance. | **Not Implemented.** The current policies can only execute the single motion they were trained on. | **Complete.** |

## 3. Design Plan for Full Paper Implementation

To align the codebase with the full vision of the "BeyondMimic" paper, the following two-phase implementation plan is proposed.

### Phase 1: Implement True Multi-Motion Tracking

The goal of this phase is to fully implement the "Scalable Motion Tracking" pipeline from the paper, enabling efficient training on long, diverse motion sequences.

1.  **Implement Adaptive Sampling Mechanism:**
    *   **Location:** Modify the environment reset and data collection logic, likely within `my_on_policy_runner.py` and the `MotionCommand` class in `mdp/commands.py`.
    *   **Logic:**
        1.  Divide the total motion timeline into discrete bins (e.g., 1-second intervals as per the paper).
        2.  In the runner, track the starting bin for each episode and whether it ended in a failure.
        3.  Maintain a persistent, smoothed failure rate for each bin (e.g., using an Exponential Moving Average).
        4.  Implement the non-causal convolution (Eq. 3) to calculate the final sampling probabilities for each bin.
        5.  In the environment reset logic, instead of uniform sampling, draw the starting phase from the weighted distribution calculated in the previous step. Mix with a small amount of uniform sampling to prevent catastrophic forgetting.

2.  **Update Data Pipeline for Multiple Motions:**
    *   **Location:** Modify `scripts/rsl_rl/train.py`.
    *   **Logic:**
        1.  Allow the `--registry_name` argument to accept a list of motion artifacts or a single artifact containing multiple motions.
        2.  Modify the `MotionLoader` and `MotionCommand` classes to load and manage a collection of motions.
        3.  The policy's observation space must be updated to include a one-hot encoding or an embedding of the current motion ID, so the policy knows which motion to perform.

### Phase 2: Implement Guided Diffusion for Skill Synthesis

This phase implements the second half of the paper, moving from mimicking to versatile control.

1.  **Offline Data Collection:**
    *   Create a new script to run the trained multi-motion policy from Phase 1 across all its learned skills.
    *   Log the state-action trajectories `(st, at)` to a large offline dataset (e.g., HDF5 file or Parquet files). This dataset is the training data for the diffusion model.

2.  **Implement the Diffusion Model:**
    *   **Architecture:** Create a new policy class implementing the `Diffuse-CLoC` architecture, which is a Transformer-based network that performs denoising.
    *   **Input/Output:** The model takes a history of state-action pairs `Ot` and a noised future trajectory `τ_k` and predicts the clean trajectory `τ_0`.
    *   **Training:** Implement the denoising training loop that minimizes the MSE loss between the predicted and ground-truth trajectories (Eq. 4).

3.  **Implement Test-Time Guidance:**
    *   **Framework:** Create a system for defining differentiable cost functions (e.g., `G(τ)`). These functions will take a predicted trajectory and return a scalar cost.
    *   **Gradient Computation:** Use a library with automatic differentiation (e.g., PyTorch's `autograd`) to compute the gradient of the cost function with respect to the trajectory `∇τ Gc(τ)`.
    *   **Sampling Loop:** Modify the DDPM sampling loop to incorporate the guidance gradient, pushing the generated trajectory towards lower-cost solutions (i.e., those that better solve the task).

4.  **Create Downstream Task Examples:**
    *   Implement the specific cost functions mentioned in the paper: `G_js` (joystick), `G_wp` (waypoint), and `G_sdf` (obstacle avoidance).
    - Create a new evaluation script (`play_diffusion.py`) that loads the trained diffusion model and allows selecting a guidance function at runtime to demonstrate the zero-shot control capabilities.
