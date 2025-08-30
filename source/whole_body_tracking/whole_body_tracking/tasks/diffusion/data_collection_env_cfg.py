"""
Environment configuration for Stage 2 diffusion data collection.
Extends the tracking environment to support trajectory data collection.
"""

from dataclasses import MISSING
from isaaclab.utils import configclass

# Import base tracking environment
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

import whole_body_tracking.tasks.diffusion.mdp as diffusion_mdp


@configclass 
class DiffusionDataCollectionEnvCfg(TrackingEnvCfg):
    """
    Configuration for diffusion data collection environment.
    
    Extends the base tracking environment to:
    1. Support multiple trained policies
    2. Add trajectory-specific observations
    3. Include action delay randomization
    4. Enable domain randomization for robustness
    """
    
    def __post_init__(self):
        """Post initialization to customize for data collection."""
        super().__post_init__()
        
        # Adjust environment settings for data collection
        self.decimation = 4  # 20Hz control frequency (50ms timesteps)
        self.episode_length_s = 10.0  # Longer episodes for more trajectory data
        
        # CRITICAL: Relax termination conditions for data collection
        # We want to collect trajectory data even when tracking isn't perfect
        
        # Increase anchor position/orientation termination thresholds
        if hasattr(self.terminations, 'anchor_pos'):
            self.terminations.anchor_pos.params["asset_cfg"].body_names = []  # Disable anchor pos termination
        if hasattr(self.terminations, 'anchor_ori'):
            self.terminations.anchor_ori.params["asset_cfg"].body_names = []  # Disable anchor ori termination
        if hasattr(self.terminations, 'ee_body_pos'):
            self.terminations.ee_body_pos.params["asset_cfg"].body_names = []  # Disable ee termination
            
        # Increase domain randomization for robust data collection
        # This matches the paper's emphasis on domain randomization during data collection
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.5)  # Less extreme
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)  # Less extreme
        
        # Reduce perturbations for initial data collection
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), 
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5), 
            "yaw": (-0.8, 0.8),
        }
        self.events.push_robot.interval_range_s = (2.0, 5.0)  # Less frequent perturbations


@configclass
class G1DiffusionDataCollectionEnvCfg(DiffusionDataCollectionEnvCfg, G1FlatEnvCfg):
    """G1 robot configuration for diffusion data collection."""
    
    def __post_init__(self):
        """Post initialization."""
        # Call parent post_init methods
        DiffusionDataCollectionEnvCfg.__post_init__(self)
        G1FlatEnvCfg.__post_init__(self)
        
        # G1-specific adjustments for data collection
        # Reduce number of environments for data collection (more manageable)
        self.scene.num_envs = 1024
        
        # Enable additional domain randomization for G1
        self.events.base_com.params["com_range"] = {
            "x": (-0.05, 0.05), 
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1)
        }