# BeyondMimic Stage 2: Guided Diffusion Implementation Plan

This document outlines a detailed design plan for implementing Stage 2 of the BeyondMimic paper: the guided diffusion policy that enables versatile, real-time control of a humanoid robot.

## 1. Overview & Core Concepts

The goal is to build a single generative policy that can be adapted to various downstream tasks at test time using **classifier guidance**. This involves three main phases:
1.  **Data Collection**: Generate an expert dataset from the existing Stage 1 motion tracking policies.
2.  **Model Training**: Train a state-action diffusion model on this dataset to learn the underlying motion manifold.
3.  **Guided Inference**: Deploy the model with task-specific cost functions that steer the motion generation process in real-time.

---

## 2. Phase 1: Data Collection

We need to create a dataset of expert trajectories from the existing RL-based tracking policies.

### 2.1. Proposed Module: `scripts/diffusion/collect_trajectories.py`

This script will be responsible for generating and saving the dataset.

**Logic:**
1.  Load a pre-trained Stage 1 motion tracking policy (`rsl_rl`).
2.  Initialize the `TrackingEnv` environment.
3.  Roll out the policy for thousands of steps across various reference motions.
4.  During the rollout, apply the same domain randomizations used for training the original policy to ensure the data is robust.
5.  For each step `t`, save a data chunk containing the observation history and the future trajectory. Based on the paper (N=4, H=16), this chunk will be:
    *   **Observation History `O_t`**: A sequence of the last `N=4` state-action pairs: `[s_{t-4}, a_{t-4}, ..., s_t]`.
    *   **Future Trajectory `τ_t`**: The sequence of the next `H=16` predicted states and actions: `[a_t, s_{t+1}, ..., a_{t+H}, a_{t+H}]`.
6.  Save these `(O_t, τ_t)` pairs to a file (e.g., using `.npz` or `.h5` format) for efficient loading.

---

## 3. Phase 2: Diffusion Model and Training

This phase involves creating the model architecture and the training pipeline.

### 3.1. Proposed Module: `source/whole_body_tracking/diffusion/model.py`

This module will define the core neural network architecture.

**`StateActionDiffusionModel(nn.Module)` Class:**
*   **Architecture**: A Transformer Decoder, as specified in the paper.
*   **Inputs**:
    1.  Noisy trajectory `τ_k` (shape: `[batch_size, horizon, state_dim + action_dim]`).
    2.  Observation history `O_t` (shape: `[batch_size, history_len, state_dim + action_dim]`).
    3.  Noise level `k`.
*   **Components**:
    1.  **Embedding Layers**: Separate `nn.Linear` layers to project states, actions, and noise level `k` into the 512-dimensional embedding space.
    2.  **Positional Encoding**: Standard sinusoidal positional encoding for the trajectory sequence.
    3.  **Transformer Decoder**: A `torch.nn.TransformerDecoder` with 6 layers and 4 attention heads. The observation history `O_t` will be encoded and used as the `memory` input to the decoder, conditioning the generation process.
    4.  **Output Head**: An `nn.Linear` layer to project the transformer's output back to the original state and action dimensions.
*   **Forward Pass `forward(self, noisy_trajectory, history, noise_level)`**:
    1.  Embed all inputs.
    2.  Concatenate the history embeddings to form the `memory` tensor.
    3.  Pass the embedded noisy trajectory and memory through the transformer decoder.
    4.  Return the predicted clean trajectory `τ_0`.

### 3.2. Proposed Module: `scripts/diffusion/train_diffusion.py`

This script will handle the model training.

**Logic:**
1.  **Dataset Loader**: Create a PyTorch `Dataset` and `DataLoader` to efficiently load the `(O_t, τ_t)` pairs from the saved file.
2.  **DDPM Scheduler**: Implement a simple scheduler to handle the diffusion process, as described in the paper (DDPM, Ho et al., 2020). This involves:
    *   `add_noise()`: The forward process that adds Gaussian noise to a clean trajectory `τ_0` to get `τ_k`.
    *   `step()`: The reverse process update rule (Eq. 5 in the paper) for denoising.
3.  **Training Loop**:
    *   Instantiate the `StateActionDiffusionModel`.
    *   For each batch of data:
        *   Sample a random noise level `k`.
        *   Generate the noisy trajectory `τ_k` using `add_noise()`.
        *   Pass `τ_k`, `O_t`, and `k` to the model to get the predicted clean trajectory `τ_0_pred`.
        *   Calculate the **MSE loss** between `τ_0_pred` and the ground truth `τ_0`.
        *   Backpropagate and update weights.
    *   Periodically save model checkpoints.

---

## 4. Phase 3: Guided Inference and Downstream Tasks

This is the core of Stage 2, enabling real-time, task-driven control.

### 4.1. Proposed Module: `source/whole_body_tracking/diffusion/inference.py`

This module will manage the guided sampling process.

**`GuidedInference` Class:**
*   **`__init__(self, model, guidance_scale)`**: Takes the trained diffusion model and a `guidance_scale` parameter.
*   **`sample(self, history, cost_function)` Method**:
    1.  Initialize a random trajectory `τ_T` from pure Gaussian noise.
    2.  Loop from `k=T` down to `1`:
        *   Enable gradients for the trajectory tensor: `τ_k.requires_grad_()`.
        *   **Calculate Cost**: Compute the task-specific cost `G_c = cost_function(τ_k)`.
        *   **Compute Guidance Gradient**: Use `torch.autograd.grad(G_c, τ_k)[0]` to get the gradient `∇τ Gc(τ)`.
        *   **Denoise with Guidance**:
            *   Get the model's prediction for the clean trajectory `τ_0_pred`.
            *   Use the DDPM `step()` function, but modify the predicted noise by subtracting the guidance gradient, effectively steering the result: `new_mean = original_mean - guidance_scale * guidance_gradient`.
    3.  Return the final, clean trajectory `τ_0`.

### 4.2. Proposed Module: `source/whole_body_tracking/diffusion/tasks.py`

This module will contain the differentiable cost functions for the downstream tasks. Each function will take a trajectory tensor `τ` and return a scalar cost.

**Functions:**
*   **`joystick_cost(trajectory, goal_velocity)`**:
    *   Extracts the predicted root velocities `V_xy` from the state sequence in the trajectory.
    *   Returns the mean squared error between the predicted velocities and the `goal_velocity`.
*   **`waypoint_cost(trajectory, goal_position)`**:
    *   Implements the cost function from Eq. 8 in the paper.
    *   Calculates the distance `d` to the goal.
    *   Computes a weighted sum of the squared distance to the goal and the squared velocity, where the weights change based on `d`.
*   **`obstacle_avoidance_cost(trajectory, sdf)`**:
    *   Requires a Signed Distance Field (SDF) of the environment.
    *   Extracts the Cartesian positions of robot bodies from the trajectory.
    *   Queries the `sdf` to get the distance to the nearest obstacle for each body.
    *   Applies the relaxed barrier function `B(x, δ)` from Eq. 10 in the paper to penalize proximity to obstacles.

### 4.3. Proposed Module: `scripts/diffusion/play_guided.py`

This script will be the entry point for running the final guided policy.

**Logic:**
1.  Load the trained `StateActionDiffusionModel`.
2.  Instantiate the `GuidedInference` class.
3.  Initialize the simulation environment.
4.  Use command-line arguments (`--task`, `--goal`) to select the desired task and parameters.
5.  Based on the task, select the corresponding cost function from the `tasks` module.
6.  **Run the Control Loop**:
    *   At each step, get the current observation history `O_t`.
    *   Call `inference.sample(O_t, cost_function)` to generate a future trajectory.
    *   Extract the first action `a_t` from the generated trajectory and apply it to the robot.
    *   Repeat.

---

## 5. Proposed File Structure

```
/Users/linji/projects/whole_body_tracking/
├── scripts/
│   └── diffusion/
│       ├── collect_trajectories.py  # For Phase 1
│       ├── train_diffusion.py       # For Phase 2
│       └── play_guided.py           # For Phase 3
└── source/
    └── whole_body_tracking/
        └── whole_body_tracking/
            └── diffusion/
                ├── model.py         # Transformer architecture
                ├── inference.py     # GuidedInference class
                └── tasks.py         # Cost functions
```

This design provides a complete roadmap for implementing the innovative second stage of the BeyondMimic paper, building directly upon the existing codebase and leveraging modern deep learning techniques in PyTorch.
