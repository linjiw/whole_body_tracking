import os
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        
        # Track episode outcomes for adaptive sampling
        self.episode_start_steps = torch.zeros(self.env.num_envs, dtype=torch.long, device=device)
        self.total_env_steps = 0
    
    def collect_rollout(self):
        """Override to track episode outcomes for adaptive sampling."""
        # Call parent implementation
        super().collect_rollout()
        
        # Update adaptive sampling statistics based on episode outcomes
        self._update_adaptive_sampling_stats()
    
    def _update_adaptive_sampling_stats(self):
        """Update adaptive sampling statistics based on episode outcomes."""
        try:
            # Check if the environment has a motion command manager
            if not hasattr(self.env.unwrapped, 'command_manager'):
                return
                
            command_manager = self.env.unwrapped.command_manager
            if not hasattr(command_manager, 'get_term') or 'motion' not in command_manager._terms:
                return
                
            motion_cmd = command_manager.get_term('motion')
            if not hasattr(motion_cmd, 'adaptive_sampler') or not motion_cmd.cfg.adaptive_sampling.enabled:
                return
            
            # Detect episodes that completed during this rollout
            # Get masks from rollout storage - shape: (num_steps + 1, num_envs)
            masks = self.rollout_storage.masks
            
            # Find environment steps where episodes ended (mask transitioned from 1 to 0)
            episode_ended = (masks[:-1] == 1) & (masks[1:] == 0)  # Shape: (num_steps, num_envs)
            
            # Get environment IDs that had episodes end
            ended_envs_per_step = []
            for step in range(episode_ended.shape[0]):
                ended_env_ids = torch.where(episode_ended[step])[0]
                if len(ended_env_ids) > 0:
                    ended_envs_per_step.append((step, ended_env_ids))
            
            # Process each batch of completed episodes
            for step, env_ids in ended_envs_per_step:
                if len(env_ids) == 0:
                    continue
                
                # Calculate episode lengths for these environments
                current_step = self.total_env_steps + step + 1
                episode_lengths = current_step - self.episode_start_steps[env_ids]
                
                # Define failure criteria (episodes shorter than threshold are failures)
                max_episode_length = self.rollout_storage.num_transitions_per_env
                failure_threshold = motion_cmd.cfg.adaptive_sampling.failure_threshold
                failures = episode_lengths < (max_episode_length * failure_threshold)
                
                # Update adaptive sampling statistics
                motion_cmd.update_adaptive_sampling_stats(env_ids, failures)
                
                # Reset episode start times for environments that ended
                self.episode_start_steps[env_ids] = current_step
            
            # Update total environment steps
            self.total_env_steps += self.rollout_storage.num_transitions_per_env
                
        except Exception as e:
            # Don't let adaptive sampling errors crash training
            print(f"Warning: Error updating adaptive sampling stats: {e}")
            pass

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                # Handle both single and multi-motion artifact names
                artifact_names = self.registry_name.split(',')
                for artifact_name in artifact_names:
                    artifact_name = artifact_name.strip()
                    if artifact_name:  # Skip empty strings
                        try:
                            wandb.run.use_artifact(artifact_name)
                            print(f"Successfully linked artifact: {artifact_name}")
                        except Exception as e:
                            print(f"Warning: Could not link artifact {artifact_name}: {e}")
                            # Continue training despite artifact linking failure
                self.registry_name = None
