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
    height_map_obs_group: str = "policy"
    height_map_dim: int = 187  # 1.6 x 1.0 m grid at 0.1 m resolution -> 17 x 11
    height_map_start_index: int | None = None  # None selects the final height_map_dim values
    latent_dim: int = 16
    vae_hidden_dims: list[int] = [128, 64]
    vae_beta: float = 1.0e-3 # total_loss = reconstruction_loss + beta * kl_loss
    vae_loss_coef: float = 1.0


@configclass
class AttentionVisualizationCfg:
    """Runtime options for attention coloring in the play script."""

    enabled: bool = True
    update_interval: int = 5
    # Only the strongest points are highlighted; all remaining height-scan
    # markers keep the blue base color.
    top_k: int = 30
    aggregation: str = "mean"
    normalization: str = "percentile"
    percentile_low: float = 5.0
    percentile_high: float = 95.0
    show_invalid_points: bool = False


@configclass
class RslRlPpoAttentionActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic configuration for the Go2 map-attention policy."""

    class_name: str = "AttentionMapActorCritic"
    proprioception_obs_group: str = "proprioception"
    map_scans_obs_group: str = "map_scans"
    embedding_dim: int = 64
    num_heads: int = 16
    map_shape: tuple[int, int] = (26, 16)
    return_attention_weights: bool = False
    attention_visualization: AttentionVisualizationCfg = AttentionVisualizationCfg()


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


@configclass
class Go2AttentionRunnerCfg(BasePPORunnerCfg):
    """Go2 PPO runner using the height-map cross-attention actor."""

    obs_groups = {"policy": ["proprioception", "map_scans"], "critic": ["critic"]}
    policy = RslRlPpoAttentionActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        embedding_dim=64,
        num_heads=16,
        map_shape=(26, 16),
    )
