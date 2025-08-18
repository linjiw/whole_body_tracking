# BeyondMimic Motion Trakcing Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repository implements the motion tracking training component of BeyondMimic, featuring:

## ✅ Phase 1 Features (Implemented)

### 🎯 Adaptive Sampling Mechanism (BeyondMimic Section III-F)
- **Non-causal convolution** with exponentially decaying kernel for failure rate estimation
- **Exponential moving average** smoothing of per-bin failure statistics  
- **Weighted sampling** based on convolved failure rates to focus training on challenging motion phases
- **Mathematically principled** approach following Equation 6 from the paper
- **Zero hyperparameter tuning** - works out-of-the-box for all LAFAN1 motions

### 🤖 Multi-Motion Training System
- **Single policy** learns multiple diverse motions simultaneously
- **Motion library management** with per-motion adaptive samplers
- **Policy conditioning** through motion ID one-hot encoding in observations
- **Automatic scaling** of neural networks based on motion count
- **WandB integration** for seamless motion artifact management

### 📁 Motion Set Configuration System
- **YAML-based configurations** for elegant experiment management
- **Category-organized motion sets** (locomotion, dynamic skills, dance, recovery)
- **Flexible training workflows** supporting both single and multi-motion training
- **Complete LAFAN1 dataset** integration with all 40 processed motions

**Status**: After adaptive sampling implementation, you can train any sim-to-real-ready motion in the LAFAN1 dataset without tuning parameters.

## 🚧 Upcoming Features

- [ ] Steerable test-time control with guided diffusion-based controllers
- [ ] Real-world deployment pipeline (separate repository)

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- **Dataset Setup Complete** ✅
  
    - **Downloaded**: Unitree-retargeted LAFAN1 Dataset from [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - **Location**: `/home/linji/nfs/LAFAN1_Retargeting_Dataset/g1/` (40 motion files for G1 robot)
    - **Categories**: dance, walk, run, sprint, fight, jumps, fallAndGetUp
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)

- **WandB Configuration** ✅
  
    - **Account**: `16726` (authenticated)
    - **Project**: `csv_to_npz`
    - **Registry Type**: `motions`
    - **Note**: Set `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to resolve protobuf compatibility

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics:

```bash
# Use isaac_lab_0817 conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaac_lab_0817
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Process motion file (example with actual path)
python scripts/csv_to_npz.py --input_file /home/linji/nfs/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv --input_fps 30 --output_name dance1_subject1 --headless
```

This will automatically upload the processed motion file to the WandB project `csv_to_npz` with artifact name matching the `--output_name` parameter.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
# Replay processed motion from WandB
python scripts/replay_npz.py --registry_name=16726/csv_to_npz/dance1_subject1:latest --headless
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

#### Motion Set Approach (Recommended)

Use pre-configured motion sets for elegant experiment management:

```bash
# Basic locomotion training (12 motions: walking, running, sprinting)
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_set locomotion_basic --headless --logger wandb \
--log_project_name tracking_training --run_name locomotion_basic_training

# Dynamic skills training (15 motions: dance, jumps, fights, recovery)
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_set dynamic_skills --headless --logger wandb \
--log_project_name tracking_training --run_name dynamic_skills_training

# Complete LAFAN1 dataset (40 motions: all categories)
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_set lafan1_full --headless --logger wandb \
--log_project_name tracking_training --run_name lafan1_full_training

# Single motion training
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_set single_motion_test --headless --logger wandb \
--log_project_name tracking_training --run_name single_motion_training
```

**Available Motion Sets:**
- `locomotion_basic` - Walking, running, sprinting (12 motions)
- `dynamic_skills` - Dance, jumps, fights, recovery (15 motions)  
- `walking_only` - All walking variations (12 motions)
- `dance_performance` - All dance motions (8 motions)
- `recovery_training` - Fall and recovery motions (6 motions)
- `lafan1_full` - Complete dataset (40 motions)
- `single_motion_test` - Single motion testing (1 motion)

See `configs/motion_sets/README.md` for complete documentation.

#### Direct Registry Approach (Legacy)

Train with specific motion registry names:

```bash
# Single motion training
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name "16726/csv_to_npz/walk3_subject2:latest" \
--headless --logger wandb --log_project_name tracking_training --run_name walk3_subject2_tracking

# Multi-motion training
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name "16726/csv_to_npz/dance1_subject1:latest,16726/csv_to_npz/walk1_subject1:latest" \
--headless --logger wandb --log_project_name tracking_training --run_name multi_motion_training
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}

# Example with actual run path format
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path=16726/tracking_training/{run_id}
```

The WandB run path can be located in the run overview. It follows the format `16726/tracking_training/{8-character-run-id}`. Note that run_name is different from run_path.

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. Includes pose and velocity error calculation, initial state randomization, and **multi-motion training support**.

    - **`adaptive_sampler.py`** ⭐ **[NEW - Phase 1]**
      Implements the **adaptive sampling mechanism** from BeyondMimic Section III-F. Features non-causal convolution
      with exponentially decaying kernel, exponential moving average failure rate smoothing, and weighted sampling
      based on convolved failure rates following Equation 6 from the paper.

    - **`motion_library.py`** ⭐ **[NEW - Phase 1]**  
      **Multi-motion training system** that manages multiple motion files and their corresponding adaptive samplers.
      Enables single policy learning across diverse motions with automatic motion ID encoding and sampling coordination.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection. **Extended with motion ID encoding** for multi-motion training.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`configs/motion_sets/`** ⭐ **[NEW - Phase 1]**
  **Motion set configuration system** with YAML-based motion set definitions for elegant experiment management.
  Includes category-organized sets for locomotion, dynamic skills, dance, recovery, and complete dataset training.
  See `configs/motion_sets/README.md` for comprehensive documentation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.
  **Enhanced training script** now supports both motion set configurations and direct registry access.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.

## Implementation Details & Paper Correspondence

### Adaptive Sampling (BeyondMimic Section III-F)

Our implementation directly follows the mathematical formulation from the paper:

**Equation 6**: Convolved failure rate computation
```
f̃ₛ = (1/Zₛ) Σᵤ₌₀^{K-1} γᵘ · fₛ₊ᵤ
```

**Key Implementation Features:**
- **Non-causal convolution** with exponentially decaying kernel (γ=0.9)
- **Exponential moving average** for failure rate smoothing (λ=0.1) 
- **Bin-based tracking** with temporal locality (K=5 lookahead)
- **Weighted sampling** using softmax temperature (α=0.1)

The adaptive sampler automatically identifies challenging motion phases and increases sampling frequency, eliminating the need for manual curriculum design or hyperparameter tuning across different motion types.

### Multi-Motion Training

Extends the single-motion framework to support simultaneous learning of diverse behaviors:

- **Policy conditioning** through motion ID one-hot encoding in observations
- **Per-motion adaptive sampling** maintaining individual failure statistics
- **Automatic scaling** of neural network input dimensions based on motion count
- **Motion library management** with seamless WandB integration

This enables training versatile policies that can perform complex behavior switching and generalization across motion categories.
