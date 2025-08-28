# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BeyondMimic is a humanoid motion tracking framework built on top of Isaac Lab (v2.1.0) for training highly dynamic motion tracking policies. The project enables sim-to-real deployment of humanoid robots to replicate complex motions from datasets like LAFAN1.

## Common Development Commands

### Installation
```bash
# Install Isaac Lab v2.1.0 first (see Isaac Lab installation guide)
# Then install this extension:
python -m pip install -e source/whole_body_tracking

# Pull robot description files from GCS
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

### Motion Processing
```bash
# Convert retargeted motion CSV to NPZ format and upload to WandB registry
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless

# Test motion replay in Isaac Sim
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}

# Upload existing NPZ motion to WandB registry
python scripts/upload_npz.py --input_file {motion_file}.npz --output_name {motion_name}
```

### Training
```bash
# Train a tracking policy
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project_name} --run_name {run_name}

# Resume training from checkpoint
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --resume --load_run {run_folder} --checkpoint {checkpoint_file}
```

### Evaluation
```bash
# Play trained policy
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}

# Play with specific checkpoint
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
  --load_run {run_folder} --checkpoint {checkpoint_file}
```

### Code Quality
```bash
# Format code with black
black --line-length 120 --preview .

# Sort imports
isort --profile black --filter-files .

# Run flake8 linting
flake8 .

# Run all pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Core Components

**Motion Tracking MDP** (`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/`)
- `commands.py`: Core motion tracking logic including `MotionLoader` for loading NPZ motions and `MotionCommand` for computing tracking errors, adaptive sampling, and reference motion state management
- `rewards.py`: DeepMimic reward functions (pose, velocity tracking) and smoothing terms for joint accelerations and actions
- `observations.py`: Observation space definition including proprioceptive data, reference motion states, and tracking errors
- `terminations.py`: Early termination based on contact violations, joint limits, and episode timeouts
- `events.py`: Domain randomization for robustness including mass, CoM, joint properties, and external forces

**Environment Configuration** (`source/whole_body_tracking/whole_body_tracking/tasks/tracking/`)
- `tracking_env_cfg.py`: Base environment configuration with MDP component definitions
- `config/g1/flat_env_cfg.py`: G1 robot specific configuration for flat terrain
- `config/humanoid/flat_env_cfg.py`: SMPL humanoid configuration

**Robot Models** (`source/whole_body_tracking/whole_body_tracking/robots/`)
- `g1.py`: Unitree G1 robot configuration with custom actuator model, armature values, and action scaling
- `smpl.py`: SMPL humanoid model configuration for motion retargeting
- `actuator.py`: Custom actuator implementations including `ImplicitActuator` with PD control

**Training Infrastructure** (`scripts/rsl_rl/`)
- `train.py`: Main training script with PPO algorithm and WandB integration
- `play.py`: Policy evaluation and visualization with Isaac Sim GUI
- `cli_args.py`: Command-line argument parsing for training configuration

**Utilities** (`source/whole_body_tracking/whole_body_tracking/utils/`)
- `my_on_policy_runner.py`: Custom PPO runner with enhanced logging and adaptive sampling support
- `exporter.py`: ONNX export functionality for sim-to-real deployment

### Key Design Patterns

1. **WandB Registry Integration**: All reference motions are stored as WandB artifacts for version control and automatic downloading during training
2. **Modular MDP Design**: Each MDP component (rewards, observations, terminations) is implemented as atomic functions that can be composed in configuration
3. **Config-Driven Architecture**: Extensive use of dataclasses for configuration with inheritance from Isaac Lab base configs
4. **Adaptive Sampling**: Intelligent sampling of reference motion frames based on tracking difficulty to improve sample efficiency
5. **Multi-Robot Support**: Architecture supports different robot models (G1, SMPL) through configuration switching

### Motion Data Flow

1. **Preprocessing**: CSV motion files → `csv_to_npz.py` → NPZ format with full kinematic data
2. **Registry**: NPZ files → WandB artifact registry → Automatic download during training
3. **Training**: `MotionLoader` loads NPZ → `MotionCommand` computes tracking targets → MDP rewards/observations
4. **Deployment**: Trained policy → ONNX export → Real robot controller

### Environment Tasks

- `Tracking-Flat-G1-v0`: G1 robot tracking on flat terrain
- `Tracking-Flat-Humanoid-v0`: SMPL humanoid tracking on flat terrain

## Important Notes

- Requires NVIDIA Isaac Sim 4.5.0 and Isaac Lab 2.1.0
- Python 3.10+ required
- GPU with CUDA support necessary for simulation
- WandB account required for motion registry and training logs
- Motion files must be retargeted to robot morphology before processing
- Set WANDB_ENTITY environment variable to your WandB organization name
- Default training uses 4096 parallel environments
- PPO hyperparameters optimized for motion tracking (30k iterations default)