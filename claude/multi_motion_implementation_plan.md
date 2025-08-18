# Updated Implementation Plan for Multi-Motion Training

This document provides an updated and justified plan to complete Phase 1 of the BeyondMimic implementation: enabling a single policy to train on multiple, diverse motions.

### 1. Justification from the "BeyondMimic" Paper

The core motivation for multi-motion training comes directly from the paper's stated goals and its critique of prior work. The objective is not just to mimic one motion perfectly, but to create a "versatile" and "generalizable" controller.

Here are the key justifications:

*   **Explicit Goal of Scalability (Section II.A):** The paper clearly distinguishes itself from "single-motion policies" and states its ambition:
    > "To move beyond single-motion policies, recent research explores **scalable motion-tracking frameworks that can learn a diverse set of motions within a single policy.**"

*   **Addressing a Gap in the Field (Section II.A):** The paper identifies the lack of high-quality, multi-motion tracking on real hardware as a key problem to be solved:
    > "To date, **multi-motion tracking on real humanoid hardware** with both high motion quality and support for highly dynamic skills **has not been demonstrated.** In this work, we aim to fill this gap by tracking **diverse, minutes-long human references** containing many distinct motions..."

*   **Foundation for Diffusion Policy (Section I & Fig. 1):** The entire "Guided Diffusion" stage of the project is predicated on having a policy (or set of policies) that has learned a rich library of skills. The diffusion model's purpose is to "distill learned motion primitives into a single policy." To do this, we must first *learn* those primitives from multiple motions. The data for the diffusion model comes from the output of the tracking policies (π1, π2, π3 in Figure 1), implying multiple skills have been learned.

*   **Experimental Setup (Section V.A):** The authors explicitly state they use a large, diverse dataset for their experiments, which would be impractical to train as thousands of separate, single-motion policies.
    > "The LAFAN1 dataset contains **40 several-minute-long references**, each in a broad category with many distinct motions... We randomly select **25 of these references**..."

This evidence makes it clear that implementing a **single policy capable of tracking multiple motions** is the logical and necessary next step to fulfilling the paper's vision. The `AdaptiveSampler` you just implemented is the key that unlocks our ability to do this efficiently.

### 2. Updated & Detailed Plan for Phase 1 Completion

Here is the updated, more detailed plan to implement the multi-motion tracking capability.

#### **Step 1: Generalize Configuration and Data Loading**

*   **Goal:** Allow the training script to accept a list of motions instead of a single one.
*   **File to Modify:** `scripts/rsl_rl/train.py`
*   **Actions:**
    1.  Modify the `--registry_name` argument to accept multiple comma-separated values (e.g., `--registry_name motion1,motion2,motion3`).
    2.  Inside `main()`, loop through the provided registry names, download each artifact from WandB, and collect the paths to the `motion.npz` files into a list.
    3.  Pass this list of file paths to the environment configuration (`env_cfg`).

#### **Step 2: Create a Multi-Motion Library in the Environment**

*   **Goal:** Enable the environment to load and manage a library of multiple motions.
*   **Files to Modify:** `mdp/commands.py`
*   **Actions:**
    1.  Create a new `MotionLibrary` class. This class will be initialized with the list of motion file paths. It should contain a list of `MotionLoader` instances, one for each motion.
    2.  The `MotionLibrary` should also be responsible for creating and managing a corresponding list of `AdaptiveSampler` instances, one for each motion.
    3.  In the `MotionCommand` class's `__init__`, replace the single `self.motion` and `self.adaptive_sampler` with an instance of `self.motion_library`.
    4.  Add a new tensor to `MotionCommand`: `self.motion_ids`, an integer tensor of shape `(num_envs,)` that tracks which motion each environment is currently assigned.

#### **Step 3: Add Motion Conditioning to the Policy**

*   **Goal:** Inform the policy which motion it is supposed to be tracking.
*   **Files to Modify:** `mdp/observations.py`, `tracking_env_cfg.py`
*   **Actions:**
    1.  Create a new observation function in `mdp/observations.py` called `motion_id_one_hot`.
    2.  This function will take the `motion_ids` tensor from the `MotionCommand` term and convert it into a one-hot encoding. For example, if there are 5 motions in the library, this will be a tensor of shape `(num_envs, 5)`.
    3.  Add this new observation term to the `PolicyCfg` in `tracking_env_cfg.py`. This ensures the motion ID is part of the policy's input at every step.

#### **Step 4: Update the Reset and Sampling Logic**

*   **Goal:** At the start of each episode, sample which motion to practice and then use the corresponding adaptive sampler to choose a starting phase.
*   **File to Modify:** `mdp/commands.py` (`MotionCommand._resample_command`)
*   **Actions:**
    1.  When an environment `env_id` needs to be reset, first sample a motion ID from the motion library. This can be a simple uniform random choice initially (e.g., `new_motion_id = torch.randint(0, num_motions, (1,))`).
    2.  Store this choice in `self.motion_ids[env_id] = new_motion_id`.
    3.  Use the selected `new_motion_id` to retrieve the correct `AdaptiveSampler` from the `self.motion_library`.
    4.  Call the `sample_starting_phases` method on that specific sampler to get the starting phase for the chosen motion.
    5.  The rest of the reset logic (setting joint positions, etc.) must now use the data from the selected motion.

After implementing these four steps, the project will be able to train a single, powerful policy on a diverse library of motions, fully realizing the "Scalable Motion Tracking" contribution of the paper and completing Phase 1 of our plan.