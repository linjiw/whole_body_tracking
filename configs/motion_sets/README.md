# Motion Set Configurations

This directory contains YAML configuration files for organizing and managing multi-motion training experiments with the BeyondMimic system. Each motion set provides a curated collection of motions for specific training scenarios.

## Available Motion Sets

### 📊 Complete Dataset
- **`lafan1_full.yaml`** - Complete LAFAN1 dataset with all 40 motions
- **`multi_motion_test.yaml`** - Complete LAFAN1 dataset (same as lafan1_full)

### 🚶 Locomotion-Focused Sets
- **`locomotion_basic.yaml`** - 12 locomotion motions (walking, running, sprinting)
- **`walking_only.yaml`** - 12 walking motions for focused gait training

### 🤸 Dynamic Skills Sets
- **`dynamic_skills.yaml`** - 15 dynamic motions (dance, jumps, fights, recovery)
- **`dance_performance.yaml`** - 8 dance motions for expressive movement
- **`recovery_training.yaml`** - 6 fall/recovery motions for robustness

### 🧪 Testing Sets
- **`single_motion_test.yaml`** - Single motion for compatibility testing

## Motion Categories Overview

| Category | Count | Motion Types | Example Sets |
|----------|-------|--------------|--------------|
| **Dance** | 8 | dance1_subject1-3, dance2_subject1-5 | `dance_performance`, `dynamic_skills` |
| **Walking** | 12 | walk1-4 across multiple subjects | `walking_only`, `locomotion_basic` |
| **Running** | 4 | run1-2 across subjects | `locomotion_basic` |
| **Sprint** | 2 | sprint1 subjects 2,4 | `locomotion_basic` |
| **Fight/Sports** | 5 | fight1 + fightAndSports1 | `dynamic_skills` |
| **Jumps** | 3 | jumps1 subjects 1,2,5 | `dynamic_skills` |
| **Fall/Recovery** | 6 | fallAndGetUp1-3 across subjects | `recovery_training`, `dynamic_skills` |

## Usage Examples

### Basic Usage
```bash
# Train with a specific motion set
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --motion_set locomotion_basic --headless

# Train with complete dataset
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --motion_set lafan1_full --headless

# Train with dynamic skills
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --motion_set dynamic_skills --headless
```

### Experiment Recommendations
```bash
# Start with locomotion basics for stable training
--motion_set locomotion_basic

# Progress to dynamic skills for challenging movements
--motion_set dynamic_skills  

# Use complete dataset for final comprehensive training
--motion_set lafan1_full

# Focus on specific skills
--motion_set walking_only     # For gait optimization
--motion_set dance_performance # For coordination and expression
--motion_set recovery_training # For robustness and failure recovery
```

## YAML File Format

Each motion set configuration follows this structure:

```yaml
name: "Motion Set Name"
description: "Detailed description of the motion set purpose and contents."
motions:
  # Category comments for organization
  - "16726/csv_to_npz/motion_name:latest"
  - "16726/csv_to_npz/another_motion:latest"
  # ... additional motions
```

## Training Behavior

- **Single Motion** (1 motion): Falls back to single-motion training mode
- **Multi-Motion** (2+ motions): Enables multi-motion training with:
  - Motion ID one-hot encoding in observations
  - Per-motion adaptive sampling
  - Motion library management
  - Policy conditioning on motion type

## Neural Network Scaling

The policy and critic networks automatically scale based on motion count:

| Motion Count | Policy Input | Critic Input | Motion ID Encoding |
|--------------|--------------|--------------|-------------------|
| 1 motion | 161 features | 287 features | (1,) |
| 2 motions | 162 features | 288 features | (2,) |
| 12 motions | 172 features | 298 features | (12,) |
| 40 motions | 200 features | 326 features | (40,) |

## Creating New Motion Sets

1. Create a new YAML file in this directory
2. Follow the naming convention: `{purpose}_{scope}.yaml`
3. Include descriptive name and purpose documentation
4. Organize motions by category with comments
5. Use registry format: `"16726/csv_to_npz/{motion_name}:latest"`

## WandB Registry Integration

All motions are automatically downloaded from WandB registry:
- **Entity**: `16726`
- **Project**: `csv_to_npz`
- **Type**: `motions`
- **Format**: `{motion_name}:latest` (or specify version like `:v1`)

The system handles artifact downloading, caching, and version management automatically.