"""
RSL-RL PPO configuration for diffusion data collection.
This is mainly used for environment registration - actual training uses pre-trained policies.
"""

from isaaclab_rl.rsl_rl.ppo import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class G1DiffusionDataCollectionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for G1 diffusion data collection runner."""
    
    num_steps_per_env = 24
    max_iterations = 1  # Not used for data collection
    save_interval = 1000
    experiment_name = "g1_diffusion_data_collection"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )