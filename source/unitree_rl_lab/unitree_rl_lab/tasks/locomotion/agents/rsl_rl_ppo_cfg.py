# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoActorCriticVAECfg(RslRlPpoActorCriticCfg):
    """Actor-critic configuration with a VAE-compressed height map."""

    class_name: str = "ActorCriticVAE"
    height_map_obs_group: str = "critic"
    height_map_dim: int = 187  # 1.6 x 1.0 m grid at 0.1 m resolution -> 17 x 11
    height_map_start_index: int | None = None  # None selects the final height_map_dim values
    latent_dim: int = 16
    vae_hidden_dims: list[int] = [128, 64]
    vae_beta: float = 1.0e-3 # total_loss = reconstruction_loss + beta * kl_loss
    vae_loss_coef: float = 1.0


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    max_iterations = 50000
    save_interval = 500
    experiment_name = ""  # same as task name
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
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


@configclass
class Go2VAERunnerCfg(BasePPORunnerCfg):
    """Go2 PPO runner using the height-map VAE actor."""

    policy = RslRlPpoActorCriticVAECfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=16,
    )
