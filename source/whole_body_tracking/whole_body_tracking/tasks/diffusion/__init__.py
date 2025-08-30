# Copyright (c) 2024, Whole Body Tracking Project  
# SPDX-License-Identifier: BSD-3-Clause

"""
Diffusion-based task environments for BeyondMimic Stage 2.

This module provides environments and configurations for collecting trajectory
data from trained tracking policies to train diffusion models.
"""

import gymnasium as gym
from . import agents
from .data_collection_env_cfg import G1DiffusionDataCollectionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Diffusion-DataCollection-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    kwargs={
        "env_cfg_entry_point": G1DiffusionDataCollectionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DiffusionDataCollectionPPORunnerCfg",
    },
    disable_env_checker=True,
)