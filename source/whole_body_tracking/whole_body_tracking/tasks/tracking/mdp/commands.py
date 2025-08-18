from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .adaptive_sampler import AdaptiveSampler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Determine if this is single-motion or multi-motion training
        if cfg.motion_files:
            # Multi-motion training
            from .motion_library import MotionLibrary
            self.motion_library = MotionLibrary(
                motion_files=cfg.motion_files,
                body_indexes=self.body_indexes,
                adaptive_sampling_cfg=cfg.adaptive_sampling,
                device=self.device
            )
            self.is_multi_motion = True
            print(f"Multi-motion training initialized with {self.motion_library.num_motions} motions")
        elif cfg.motion_file:
            # Single-motion training (backward compatibility)
            self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
            motion_duration = self.motion.time_step_total / self.motion.fps
            self.adaptive_sampler = AdaptiveSampler(
                motion_fps=int(self.motion.fps),
                motion_duration=motion_duration,
                bin_size_seconds=cfg.adaptive_sampling.bin_size_seconds,
                gamma=cfg.adaptive_sampling.gamma,
                lambda_uniform=cfg.adaptive_sampling.lambda_uniform,
                K=cfg.adaptive_sampling.K,
                alpha_smooth=cfg.adaptive_sampling.alpha_smooth,
                device=self.device
            )
            self.is_multi_motion = False
            print("Single-motion training initialized")
        else:
            raise ValueError("Either motion_file or motion_files must be specified")

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        
        # Track starting frames and motion assignments for episodes
        self.episode_starting_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        if self.is_multi_motion:
            # Track which motion each environment is assigned to
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            # For single-motion, all environments use motion 0
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        if self.is_multi_motion:
            return self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'joint_pos')
        else:
            return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        if self.is_multi_motion:
            return self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'joint_vel')
        else:
            return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            body_pos = self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_pos_w')
            return body_pos + self._env.scene.env_origins[:, None, :]
        else:
            return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            return self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_quat_w')
        else:
            return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            return self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_lin_vel_w')
        else:
            return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            return self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_ang_vel_w')
        else:
            return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            body_pos = self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_pos_w')
            return body_pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins
        else:
            return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            body_quat = self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_quat_w')
            return body_quat[:, self.motion_anchor_body_index]
        else:
            return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            body_lin_vel = self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_lin_vel_w')
            return body_lin_vel[:, self.motion_anchor_body_index]
        else:
            return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        if self.is_multi_motion:
            body_ang_vel = self.motion_library.get_motion_data(self.motion_ids, self.time_steps, 'body_ang_vel_w')
            return body_ang_vel[:, self.motion_anchor_body_index]
        else:
            return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
            
        if self.is_multi_motion:
            # Multi-motion sampling
            if self.cfg.adaptive_sampling.enabled:
                # Sample motions and starting phases using adaptive sampling
                motion_ids, phases, starting_bins = self.motion_library.sample_motion_and_phase(len(env_ids))
                
                # Convert phases to frame indices for each motion
                starting_frames = torch.zeros_like(phases, dtype=torch.long)
                for i, (motion_id, phase) in enumerate(zip(motion_ids, phases)):
                    motion = self.motion_library.motions[motion_id]
                    starting_frames[i] = (phase * (motion.time_step_total - 1)).long()
                
                # Update environment assignments
                self.motion_ids[env_ids] = motion_ids
                self.episode_starting_frames[env_ids] = starting_frames
                self.time_steps[env_ids] = starting_frames
            else:
                # Uniform multi-motion sampling
                motion_ids = torch.randint(0, self.motion_library.num_motions, (len(env_ids),), device=self.device)
                phases = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
                
                starting_frames = torch.zeros_like(phases, dtype=torch.long)
                for i, (motion_id, phase) in enumerate(zip(motion_ids, phases)):
                    motion = self.motion_library.motions[motion_id]
                    starting_frames[i] = (phase * (motion.time_step_total - 1)).long()
                
                self.motion_ids[env_ids] = motion_ids
                self.episode_starting_frames[env_ids] = starting_frames
                self.time_steps[env_ids] = starting_frames
        else:
            # Single-motion sampling (backward compatibility)
            if self.cfg.adaptive_sampling.enabled:
                phases, starting_bins = self.adaptive_sampler.sample_starting_phases(len(env_ids))
                starting_frames = (phases * (self.motion.time_step_total - 1)).long()
            else:
                phases = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
                starting_frames = (phases * (self.motion.time_step_total - 1)).long()
            
            self.episode_starting_frames[env_ids] = starting_frames
            self.time_steps[env_ids] = starting_frames

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        
        if self.is_multi_motion:
            # Multi-motion: check end condition per motion
            env_ids_to_reset = []
            for env_id in range(self.num_envs):
                motion_id = self.motion_ids[env_id]
                motion = self.motion_library.motions[motion_id]
                if self.time_steps[env_id] >= motion.time_step_total:
                    env_ids_to_reset.append(env_id)
            
            if env_ids_to_reset:
                env_ids = torch.tensor(env_ids_to_reset, device=self.device, dtype=torch.long)
                self._resample_command(env_ids)
        else:
            # Single-motion: original logic
            env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
            self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
    
    def update_adaptive_sampling_stats(self, env_ids: Sequence[int], failures: torch.Tensor):
        """
        Update adaptive sampling statistics based on episode outcomes.
        
        Args:
            env_ids: Environment IDs that completed episodes
            failures: Boolean tensor indicating which episodes failed
        """
        if not self.cfg.adaptive_sampling.enabled or len(env_ids) == 0:
            return
            
        # Get starting frames and motion IDs for completed episodes
        starting_frames = self.episode_starting_frames[env_ids]
        
        if self.is_multi_motion:
            # Multi-motion: update motion library statistics
            motion_ids = self.motion_ids[env_ids]
            self.motion_library.update_failure_statistics(motion_ids, starting_frames, failures)
            
            # Update sampling probabilities periodically
            if len(env_ids) > 0 and self.motion_library.motion_episode_counts.sum() % self.cfg.adaptive_sampling.update_frequency == 0:
                self.motion_library.update_sampling_probabilities()
            
            # Log to WandB periodically
            if self.motion_library.motion_episode_counts.sum() % self.cfg.adaptive_sampling.log_frequency == 0:
                self.motion_library.log_to_wandb()
        else:
            # Single-motion: update adaptive sampler statistics
            self.adaptive_sampler.update_failure_statistics(starting_frames, failures)
            
            # Update sampling probabilities periodically
            if self.adaptive_sampler.update_counter % self.cfg.adaptive_sampling.update_frequency == 0:
                self.adaptive_sampler.update_sampling_probabilities()
            
            # Log to WandB periodically
            if self.adaptive_sampler.total_episodes % self.cfg.adaptive_sampling.log_frequency == 0:
                self.adaptive_sampler.log_to_wandb()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class AdaptiveSamplingCfg:
    """Configuration for adaptive sampling mechanism."""
    
    enabled: bool = True
    """Enable adaptive sampling mechanism. If False, falls back to uniform sampling."""
    
    bin_size_seconds: float = 1.0
    """Size of each time bin in seconds for failure tracking."""
    
    gamma: float = 0.9
    """Decay rate for convolution kernel (γ in paper Equation 3)."""
    
    lambda_uniform: float = 0.1
    """Mixing ratio with uniform distribution to prevent catastrophic forgetting."""
    
    K: int = 5
    """Convolution kernel size for failure rate smoothing."""
    
    alpha_smooth: float = 0.1
    """Exponential moving average smoothing factor for failure rates."""
    
    update_frequency: int = 100
    """Update sampling probabilities every N episodes."""
    
    log_frequency: int = 1000
    """Log adaptive sampling metrics to WandB every N episodes."""
    
    failure_threshold: float = 0.8
    """Episodes shorter than this fraction of max length are considered failures."""


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = ""  # Optional: for single-motion training (backward compatibility)
    motion_files: list[str] = []  # Optional: for multi-motion training
    anchor_body: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    
    adaptive_sampling: AdaptiveSamplingCfg = AdaptiveSamplingCfg()
    """Configuration for adaptive sampling mechanism."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
