# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BeyondMimic is a versatile humanoid control framework for dynamic motion tracking using reinforcement learning. This repository implements the motion tracking training component, focused on training policies to follow reference motion data on humanoid robots (primarily Unitree G1).

## Development Environment Setup

### Conda Environment
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_0817
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### Installation
```bash
# Install the package in development mode
python -m pip install -e source/whole_body_tracking

# Download robot assets
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

## Common Commands

### Motion Processing
```bash
# Convert CSV motion data to NPZ format (configured for entity 16726)
python scripts/csv_to_npz.py --input_file /path/to/motion.csv --input_fps 30 --output_name motion_name --headless

# Process with frame range for testing (faster)
python scripts/csv_to_npz.py --input_file /path/to/motion.csv --input_fps 30 --output_name motion_name --headless --frame_range 1 100

# Batch process all motions in LAFAN1 dataset (40 files, ~3-5 min each)
./scripts/batch_process_csv.sh

# Replay motion from WandB (direct artifact access)
python scripts/replay_npz.py --registry_name=16726/csv_to_npz/motion_name:latest --headless
```

### Training
```bash
# Train motion tracking policy (using registry collection)
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name 16726-org/wandb-registry-Motions/{motion_name} \
--headless --logger wandb --log_project_name tracking_training --run_name {motion_name}_tracking

# Example with walk3_subject2 motion
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name 16726-org/wandb-registry-Motions/walk3_subject2 \
--headless --logger wandb --log_project_name tracking_training --run_name walk3_subject2_tracking

# Example with dance motion
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name 16726-org/wandb-registry-Motions/dance1_subject1 \
--headless --logger wandb --log_project_name tracking_training --run_name dance1_subject1_tracking

# Evaluate trained policy
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path=16726/{project_name}/{run_id}
```

### Motion Access
```bash
# Direct artifact access (for replay)
python scripts/replay_npz.py --registry_name=16726/csv_to_npz/dance1_subject1:latest --headless

# Registry collection access (for training)
# Automatically uses: 16726-org/wandb-registry-Motions/motion_name
```

### Code Quality
```bash
# Run pre-commit hooks
pre-commit run --all-files

# Format code with black
black --line-length 120 --preview .

# Sort imports
isort --profile black --filter-files .

# Type checking
pyright
```

## Architecture

### Core Structure
- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/`**: Main tracking task implementation
  - `mdp/`: MDP components (commands, rewards, observations, terminations, events)
  - `config/`: Robot-specific configurations (G1, humanoid)
  - `tracking_env_cfg.py`: Environment configuration
- **`source/whole_body_tracking/whole_body_tracking/robots/`**: Robot-specific implementations
- **`scripts/`**: Utility scripts for data processing and training

### Key MDP Components
- **`commands.py`**: Reference motion processing, pose/velocity error computation, adaptive sampling
- **`rewards.py`**: DeepMimic reward functions and smoothing terms
- **`observations.py`**: Motion tracking observations and data collection
- **`terminations.py`**: Early termination conditions and timeouts
- **`events.py`**: Domain randomization

### Motion Data Pipeline
1. CSV motions (LAFAN1 dataset) → `csv_to_npz.py` → NPZ format with kinematic data
2. NPZ files uploaded to WandB registry for centralized motion management
3. Training loads motions from WandB registry by name

## Configuration

### WandB Setup
- **Organization**: `16726`
- **Project**: `csv_to_npz`
- **Registry Type**: `motions`
- Set environment variable: `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

### Dataset Locations
- **LAFAN1 Dataset**: `/home/linji/nfs/LAFAN1_Retargeting_Dataset/g1/`
- **Motion Categories**: dance, walk, run, sprint, fight, jumps, fallAndGetUp

## Configuration Updates Made

### Scripts Updated (Latest)
- **`scripts/csv_to_npz.py`**: 
  - Configured WandB entity as "16726"
  - Enhanced metadata tracking (fps, duration, frames, robot type)
  - Improved artifact naming and registry linking
  - Added proper cleanup and error handling

- **`scripts/batch_process_csv.sh`**: 
  - Added proper environment variable setup (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`)
  - Enhanced progress tracking with counters
  - Added detailed logging and error reporting
  - Implemented rate limiting to avoid API overload
  - Added final summary with success/failure statistics

### Testing Results
- ✅ Conda environment `isaac_lab_0817` properly activated
- ✅ WandB authentication verified (user: linjiw, entity: 16726)
- ✅ Scripts successfully initialize Isaac Sim and process motions
- ✅ WandB artifacts are properly created and linked to registry
- ⏱️ Processing time: ~3-5 minutes per motion (due to Isaac Sim initialization)
- 📊 Dataset: 40 motion files available in `/home/linji/nfs/LAFAN1_Retargeting_Dataset/g1/`

## Important Notes

- Always use the `isaac_lab_0817` conda environment with `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- WandB entity "16726" is pre-configured in all scripts
- Motion processing takes time due to Isaac Sim initialization (~3-5 min per file)
- Use `--frame_range 1 100` for quick testing with subset of motion frames
- Artifact format: `16726/csv_to_npz/{motion_name}:latest` (direct access)
- Registry format: `16726-org/wandb-registry-Motions/{motion_name}` (for training)
- Batch processing includes automatic progress tracking and error reporting
- Pre-commit hooks enforce code quality standards