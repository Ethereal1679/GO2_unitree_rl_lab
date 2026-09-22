# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class HeightMapVAE(nn.Module):
    """Variational auto-encoder for a flattened height map."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64),
        activation: str = "elu",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, 2 * latent_dim, list(hidden_dims), activation)
        self.decoder = MLP(latent_dim, input_dim, list(reversed(hidden_dims)), activation)

    def encode(self, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and log-variance of q(z | height_map)."""
        if height_map.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected a height map with {self.input_dim} values, got {height_map.shape[-1]}."
            )
        mean, log_var = self.encoder(height_map).chunk(2, dim=-1)
        # Bounding log-variance avoids numerical overflow early in PPO training.
        return mean, log_var.clamp(min=-20.0, max=10.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct a flattened height map from a latent vector."""
        return self.decoder(latent)

    def forward(
        self, height_map: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(height_map)
        latent = self.reparameterize(mean, log_var) if sample else mean
        reconstruction = self.decode(latent)
        return reconstruction, mean, log_var, latent

    def loss(
        self, height_map: torch.Tensor, beta: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute beta-VAE reconstruction and KL losses."""
        reconstruction, mean, log_var, _ = self(height_map, sample=True)
        reconstruction_loss = F.mse_loss(reconstruction, height_map)
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + log_var - mean.square() - log_var.exp(), dim=-1))
        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss, {
            "vae_reconstruction": reconstruction_loss,
            "vae_kl": kl_loss,
        }


class HeightMapActor(nn.Module):
    """Actor that replaces the raw height map with its VAE latent mean."""

    def __init__(
        self,
        policy_obs_dim: int,
        height_map_dim: int,
        latent_dim: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int],
        vae_hidden_dims: Sequence[int],
        activation: str,
    ):
        super().__init__()
        self.policy_obs_dim = policy_obs_dim
        self.height_map_dim = height_map_dim
        # Isaac Lab's generic ONNX exporter obtains the observation dimension
        # through ``actor[0].in_features``.  HeightMapActor is not a Sequential,
        # so expose the raw actor-input dimension for exporter compatibility.
        self.in_features = policy_obs_dim + height_map_dim
        self.vae = HeightMapVAE(height_map_dim, latent_dim, vae_hidden_dims, activation)
        self.policy = MLP(policy_obs_dim + latent_dim, num_actions, list(actor_hidden_dims), activation)

    def __getitem__(self, index: int):
        """Provide the input-size interface expected by Isaac Lab exporters."""
        if index != 0:
            raise IndexError(f"HeightMapActor only exposes exporter index 0, got {index}.")
        return self

    def split_input(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_dim = self.policy_obs_dim + self.height_map_dim
        if actor_input.shape[-1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} actor inputs, got {actor_input.shape[-1]}.")
        policy_obs = actor_input[..., : self.policy_obs_dim]
        height_map = actor_input[..., self.policy_obs_dim :]
        return policy_obs, height_map

    def get_latent(self, height_map: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean, log_var = self.vae.encode(height_map)
        return self.vae.reparameterize(mean, log_var) if sample else mean

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        policy_obs, height_map = self.split_input(actor_input)
        # The mean is deterministic, which keeps PPO action likelihood ratios stable.
        latent = self.get_latent(height_map, sample=False)
        return self.policy(torch.cat((policy_obs, latent), dim=-1))


class ActorCriticVAE(nn.Module):
    """Actor-critic with a beta-VAE height-map encoder in the actor.

    The raw actor input is ``[policy observations without the height map, height map]``.
    ``HeightMapActor`` replaces the height map with its latent mean before running
    the action MLP. When ``height_map_obs_group`` is ``"policy"``, the height map
    is removed from the policy observation before it is appended to the actor input,
    avoiding a duplicate copy of the scan.
    The critic continues to receive the configured critic observation groups.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        height_map_obs_group: str = "policy",
        height_map_dim: int | None = 187,
        height_map_start_index: int | None = None,
        latent_dim: int = 16,
        vae_hidden_dims=(128, 64),
        vae_beta: float = 1.0e-3,
        vae_loss_coef: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.height_map_obs_group = height_map_obs_group
        if vae_beta < 0.0:
            raise ValueError(f"vae_beta must be non-negative, got {vae_beta}.")
        if vae_loss_coef < 0.0:
            raise ValueError(f"vae_loss_coef must be non-negative, got {vae_loss_coef}.")
        self.vae_beta = vae_beta
        self.vae_loss_coef = vae_loss_coef

        if height_map_obs_group not in obs:
            raise KeyError(
                f"Height-map observation group '{height_map_obs_group}' was not found. "
                f"Available groups: {list(obs.keys())}."
            )
        source_dim = obs[height_map_obs_group].shape[-1]
        self.height_map_dim = source_dim if height_map_dim is None else height_map_dim
        if self.height_map_dim <= 0 or self.height_map_dim > source_dim:
            raise ValueError(
                f"height_map_dim must be in [1, {source_dim}] for group '{height_map_obs_group}', "
                f"got {self.height_map_dim}."
            )
        self.height_map_start_index = (
            source_dim - self.height_map_dim if height_map_start_index is None else height_map_start_index
        )
        if self.height_map_start_index < 0:
            self.height_map_start_index += source_dim
        height_map_end_index = self.height_map_start_index + self.height_map_dim
        if self.height_map_start_index < 0 or height_map_end_index > source_dim:
            raise ValueError(
                f"Height-map slice [{self.height_map_start_index}:{height_map_end_index}] is outside "
                f"observation group '{height_map_obs_group}' with dimension {source_dim}."
            )
        self.height_map_end_index = height_map_end_index

        num_policy_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticVAE only supports 1D observations."
            num_policy_obs += obs[obs_group].shape[-1]

        # If the height map comes from a policy observation group, remove it from
        # the ordinary actor features and append it back as the VAE input below.
        self.height_map_in_policy = height_map_obs_group in obs_groups["policy"]
        self.height_map_policy_start_index = None
        if self.height_map_in_policy:
            group_offset = 0
            for obs_group in obs_groups["policy"]:
                if obs_group == height_map_obs_group:
                    self.height_map_policy_start_index = group_offset + self.height_map_start_index
                    break
                group_offset += obs[obs_group].shape[-1]
            assert self.height_map_policy_start_index is not None
            num_actor_policy_obs = num_policy_obs - self.height_map_dim
        else:
            num_actor_policy_obs = num_policy_obs

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticVAE only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # The exported actor receives policy observations (without the raw height
        # map) followed by the raw height map for the VAE encoder.
        num_raw_actor_obs = num_actor_policy_obs + self.height_map_dim
        self.actor = HeightMapActor(
            num_actor_policy_obs,
            self.height_map_dim,
            latent_dim,
            num_actions,
            actor_hidden_dims,
            vae_hidden_dims,
            activation,
        )
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_raw_actor_obs) if actor_obs_normalization else nn.Identity()
        )
        print(f"Height-map VAE: {self.actor.vae}")
        print(f"Actor MLP ({num_actor_policy_obs} policy + {latent_dim} latent inputs): {self.actor.policy}")

        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'.")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_input):
        mean = self.actor(actor_input)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_input)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_height_map(self, obs) -> torch.Tensor:
        source = obs[self.height_map_obs_group]
        return source[..., self.height_map_start_index : self.height_map_end_index]

    def get_height_map_latent(self, obs, sample: bool = False) -> torch.Tensor:
        """Expose the encoded height-map latent for debugging or downstream use."""
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        _, height_map = self.actor.split_input(actor_input)
        return self.actor.get_latent(height_map, sample=sample)

    def get_actor_obs(self, obs):
        policy_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)
        if self.height_map_in_policy:
            start = self.height_map_policy_start_index
            end = start + self.height_map_dim
            policy_obs = torch.cat((policy_obs[..., :start], policy_obs[..., end:]), dim=-1)
        return torch.cat((policy_obs, self.get_height_map(obs)), dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def compute_auxiliary_loss(self, obs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted VAE loss consumed by PPO and its logging terms."""
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        _, height_map = self.actor.split_input(actor_input)
        vae_loss, loss_terms = self.actor.vae.loss(height_map, beta=self.vae_beta)
        weighted_vae_loss = self.vae_loss_coef * vae_loss
        loss_terms["vae_total"] = weighted_vae_loss
        return weighted_vae_loss, loss_terms

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


# Allows direct imports from this file to keep using the conventional class name.
ActorCritic = ActorCriticVAE
