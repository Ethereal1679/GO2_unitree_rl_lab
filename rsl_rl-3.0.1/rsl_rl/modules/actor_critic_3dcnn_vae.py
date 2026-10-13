# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic with a height-map VAE and a temporal 3D-CNN predictor.

The temporal branch consumes four maps ordered from oldest to newest:
``[t - 3 * interval, t - 2 * interval, t - interval, t]``.  It predicts the
VAE latent of the next map.  During PPO collection the window is maintained per
environment; the window is copied into the rollout observation so that shuffled
PPO minibatches recompute exactly the same action distribution as collection.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


def _activation(name: str) -> nn.Module:
    """Build an activation matching the names used by RSL-RL configs."""

    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'.")


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
        if not hidden_dims or any(int(size) <= 0 for size in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes.")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = MLP(self.input_dim, 2 * self.latent_dim, list(hidden_dims), activation)
        self.decoder = MLP(self.latent_dim, self.input_dim, list(reversed(hidden_dims)), activation)

    def encode(self, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and log-variance of ``q(z | height_map)``."""

        if height_map.ndim != 2 or height_map.shape[-1] != self.input_dim:
            raise ValueError("Expected a rank-2 height map with the configured input dimension.")
        mean, log_var = self.encoder(height_map).chunk(2, dim=-1)
        # Bounding log-variance avoids numerical overflow early in PPO training.
        return mean, log_var.clamp(min=-20.0, max=10.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("Expected a rank-2 latent with the configured latent dimension.")
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


class TemporalHeightMapPredictor(nn.Module):
    """3D-CNN that predicts the next height-map VAE latent.

    Input is ``(batch, temporal_frames, height_map_dim)``.  The temporal axis
    is kept intact by the spatial convolutional blocks and then collapsed by a
    learned kernel spanning all frames.  This makes the output explicitly
    sensitive to temporal order rather than treating the frames as a set.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        latent_dim: int,
        temporal_frames: int = 4,
        cnn_channels: Sequence[int] = (16, 32, 64),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if len(map_shape) != 2 or any(int(size) <= 0 for size in map_shape):
            raise ValueError(f"map_shape must contain two positive dimensions, got {map_shape}.")
        if temporal_frames < 2:
            raise ValueError(f"temporal_frames must be at least 2, got {temporal_frames}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if not cnn_channels or any(int(size) <= 0 for size in cnn_channels):
            raise ValueError("cnn_channels must contain positive channel sizes.")

        self.map_height = int(map_shape[0])
        self.map_width = int(map_shape[1])
        self.map_shape = (self.map_height, self.map_width)
        self.map_dim = self.map_height * self.map_width
        self.latent_dim = int(latent_dim)
        self.temporal_frames = int(temporal_frames)

        layers: list[nn.Module] = []
        in_channels = 1
        for block_index, channel_count in enumerate(cnn_channels):
            channel_count = int(channel_count)
            # Downsample only the spatial dimensions.  The temporal resolution
            # remains available to the final 3D convolution.
            stride = (1, 1, 1) if block_index == 0 else (1, 2, 2)
            layers.extend(
                [
                    nn.Conv3d(in_channels, channel_count, kernel_size=3, stride=stride, padding=1),
                    _activation(activation),
                ]
            )
            in_channels = channel_count
        layers.extend(
            [
                nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=(self.temporal_frames, 1, 1),
                ),
                _activation(activation),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, self.latent_dim),
            ]
        )
        self.network = nn.Sequential(*layers)

    def _check_sequence(self, height_maps: torch.Tensor) -> torch.Tensor:
        """Convert supported sequence layouts to ``(N, 1, T, H, W)``."""

        if height_maps.ndim == 3:
            # (N, T, flattened_map)
            if height_maps.shape[1] != self.temporal_frames or height_maps.shape[2] != self.map_dim:
                raise ValueError("Flattened temporal height maps have an incompatible shape.")
            return height_maps.reshape(-1, 1, self.temporal_frames, self.map_height, self.map_width)
        if height_maps.ndim == 4:
            # (N, T, H, W)
            if (
                height_maps.shape[1] != self.temporal_frames
                or height_maps.shape[2] != self.map_height
                or height_maps.shape[3] != self.map_width
            ):
                raise ValueError("Spatial temporal height maps have an incompatible shape.")
            return height_maps.unsqueeze(1)
        if height_maps.ndim == 5:
            # Accept either (N, 1, T, H, W) or (N, T, 1, H, W).
            if (
                height_maps.shape[1] == 1
                and height_maps.shape[2] == self.temporal_frames
                and height_maps.shape[3] == self.map_height
                and height_maps.shape[4] == self.map_width
            ):
                return height_maps
            if (
                height_maps.shape[1] == self.temporal_frames
                and height_maps.shape[2] == 1
                and height_maps.shape[3] == self.map_height
                and height_maps.shape[4] == self.map_width
            ):
                return height_maps.transpose(1, 2)
        raise ValueError("Temporal height maps must have a supported rank and configured dimensions.")

    def forward(self, height_maps: torch.Tensor) -> torch.Tensor:
        return self.network(self._check_sequence(height_maps))


class HeightMapActor(nn.Module):
    """Actor wrapper combining current and predicted-future VAE latents."""

    def __init__(
        self,
        policy_obs_dim: int,
        height_map_dim: int,
        latent_dim: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int],
        vae_hidden_dims: Sequence[int],
        activation: str,
        map_shape: tuple[int, int],
        temporal_frames: int,
        temporal_cnn_channels: Sequence[int],
    ) -> None:
        super().__init__()
        self.policy_obs_dim = int(policy_obs_dim)
        self.height_map_dim = int(height_map_dim)
        self.latent_dim = int(latent_dim)
        self.temporal_frames = int(temporal_frames)
        # Isaac Lab's generic ONNX exporter obtains the observation dimension
        # through ``actor[0].in_features``.  Keep this as the raw current-frame
        # input; temporal deployment can provide the optional sequence input.
        self.in_features = self.policy_obs_dim + self.height_map_dim
        self.vae = HeightMapVAE(self.height_map_dim, latent_dim, vae_hidden_dims, activation)
        self.temporal_predictor = TemporalHeightMapPredictor(
            map_shape=map_shape,
            latent_dim=latent_dim,
            temporal_frames=temporal_frames,
            cnn_channels=temporal_cnn_channels,
            activation=activation,
        )
        self.policy = MLP(
            self.policy_obs_dim + 2 * latent_dim,
            num_actions,
            list(actor_hidden_dims),
            activation,
        )

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(f"HeightMapActor only exposes exporter index 0, got {index}.")
        return self

    def split_input(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_dim = self.policy_obs_dim + self.height_map_dim
        if actor_input.ndim != 2 or actor_input.shape[-1] != expected_dim:
            raise ValueError("Actor inputs must be rank 2 with the configured feature dimension.")
        return actor_input[..., : self.policy_obs_dim], actor_input[..., self.policy_obs_dim :]

    def get_latent(self, height_map: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean, log_var = self.vae.encode(height_map)
        return self.vae.reparameterize(mean, log_var) if sample else mean

    def predict_future_latent(self, temporal_height_maps: torch.Tensor) -> torch.Tensor:
        return self.temporal_predictor(temporal_height_maps)

    def forward(
        self,
        actor_input: torch.Tensor,
        temporal_height_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_obs, height_map = self.split_input(actor_input)
        # The mean is deterministic, which keeps PPO action likelihood ratios stable.
        current_latent = self.get_latent(height_map, sample=False)
        if temporal_height_maps is None:
            # This fallback keeps direct actor/exporter calls well-defined.  The
            # actor-critic supplies real rollout windows during training.
            temporal_height_maps = height_map.unsqueeze(1).expand(-1, self.temporal_frames, -1)
        predicted_latent = self.predict_future_latent(temporal_height_maps)
        return self.policy(torch.cat((policy_obs, current_latent, predicted_latent), dim=-1))


class ActorCritic3DCNNVAE(nn.Module):
    """PPO actor-critic with VAE and supervised temporal 3D-CNN prediction."""

    is_recurrent = False

    # These keys are appended to rollout observations by PPO.  Keeping them in
    # the TensorDict makes temporal inputs survive random minibatch shuffling.
    temporal_observation_key = "_3dcnn_temporal_height_maps"
    next_height_map_key = "_3dcnn_next_height_map"
    next_height_map_valid_key = "_3dcnn_next_height_map_valid"

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
        temporal_history_frames: int = 3,
        temporal_frame_interval: int = 1,
        temporal_map_shape: tuple[int, int] | None = (17, 11),
        temporal_cnn_channels=(16, 32, 64),
        temporal_prediction_loss_coef: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic3DCNNVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if vae_beta < 0.0:
            raise ValueError(f"vae_beta must be non-negative, got {vae_beta}.")
        if vae_loss_coef < 0.0:
            raise ValueError(f"vae_loss_coef must be non-negative, got {vae_loss_coef}.")
        if temporal_prediction_loss_coef < 0.0:
            raise ValueError(
                f"temporal_prediction_loss_coef must be non-negative, got {temporal_prediction_loss_coef}."
            )
        if temporal_history_frames != 3:
            raise ValueError(
                "The temporal predictor is configured for three history frames plus the current frame; "
                f"got temporal_history_frames={temporal_history_frames}."
            )
        if temporal_frame_interval <= 0:
            raise ValueError(f"temporal_frame_interval must be positive, got {temporal_frame_interval}.")

        self.obs_groups = obs_groups
        self.height_map_obs_group = height_map_obs_group
        self.vae_beta = float(vae_beta)
        self.vae_loss_coef = float(vae_loss_coef)
        self.temporal_prediction_loss_coef = float(temporal_prediction_loss_coef)
        self.temporal_history_frames = int(temporal_history_frames)
        self.temporal_frame_interval = int(temporal_frame_interval)
        self.temporal_frames = self.temporal_history_frames + 1

        if height_map_obs_group not in obs:
            raise KeyError(
                f"Height-map observation group '{height_map_obs_group}' was not found. "
                f"Available groups: {list(obs.keys())}."
            )
        source_dim = obs[height_map_obs_group].shape[-1]
        self.height_map_dim = source_dim if height_map_dim is None else int(height_map_dim)
        if self.height_map_dim <= 0 or self.height_map_dim > source_dim:
            raise ValueError(
                f"height_map_dim must be in [1, {source_dim}] for group '{height_map_obs_group}', "
                f"got {self.height_map_dim}."
            )
        self.height_map_start_index = (
            source_dim - self.height_map_dim if height_map_start_index is None else int(height_map_start_index)
        )
        if self.height_map_start_index < 0:
            self.height_map_start_index += source_dim
        self.height_map_end_index = self.height_map_start_index + self.height_map_dim
        if self.height_map_start_index < 0 or self.height_map_end_index > source_dim:
            raise ValueError(
                f"Height-map slice [{self.height_map_start_index}:{self.height_map_end_index}] is outside "
                f"observation group '{height_map_obs_group}' with dimension {source_dim}."
            )

        if temporal_map_shape is None:
            temporal_map_shape = self._infer_map_shape(self.height_map_dim)
        if (
            len(temporal_map_shape) != 2
            or int(temporal_map_shape[0]) * int(temporal_map_shape[1]) != self.height_map_dim
        ):
            raise ValueError(
                f"temporal_map_shape={temporal_map_shape} must multiply to height_map_dim={self.height_map_dim}."
            )
        self.temporal_map_shape = (int(temporal_map_shape[0]), int(temporal_map_shape[1]))
        self.temporal_observation_shape = (self.temporal_frames, self.height_map_dim)

        num_policy_obs = 0
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError("ActorCritic3DCNNVAE only supports 1D observations.")
            num_policy_obs += obs[obs_group].shape[-1]

        # If the height map comes from a policy observation group, remove it from
        # ordinary actor features and append it back as the VAE input below.
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
            if len(obs[obs_group].shape) != 2:
                raise ValueError("ActorCritic3DCNNVAE only supports 1D observations.")
            num_critic_obs += obs[obs_group].shape[-1]

        self.actor = HeightMapActor(
            num_actor_policy_obs,
            self.height_map_dim,
            latent_dim,
            num_actions,
            actor_hidden_dims,
            vae_hidden_dims,
            activation,
            map_shape=self.temporal_map_shape,
            temporal_frames=self.temporal_frames,
            temporal_cnn_channels=temporal_cnn_channels,
        )
        num_raw_actor_obs = num_actor_policy_obs + self.height_map_dim
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_raw_actor_obs) if actor_obs_normalization else nn.Identity()
        )
        print(f"Height-map VAE: {self.actor.vae}")
        print(f"Temporal 3D-CNN predictor: {self.actor.temporal_predictor}")
        print(
            f"Actor MLP ({num_actor_policy_obs} policy + {2 * latent_dim} current/future latent inputs): "
            f"{self.actor.policy}"
        )

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
        self._height_map_history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._last_temporal_height_maps: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _infer_map_shape(map_dim: int) -> tuple[int, int]:
        """Infer a compact factorization when a map shape is not configured."""

        for height in range(int(map_dim**0.5), 0, -1):
            if map_dim % height == 0:
                return height, map_dim // height
        raise ValueError(f"Unable to infer a 2D map shape from dimension {map_dim}.")

    def reset(self, dones=None):
        if dones is None or self._height_map_history is None:
            return
        done_mask = dones.reshape(-1).bool()
        if done_mask.shape[0] != self._height_map_history.shape[1]:
            self._height_map_history = None
            self._history_valid = None
            self._last_temporal_height_maps = None
            return
        self._history_valid[done_mask] = False
        self._height_map_history[:, done_mask] = 0.0

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _history_window(self, obs) -> torch.Tensor:
        """Update the online per-environment history and return oldest-to-newest maps."""

        current = self.get_height_map(obs).detach()
        batch_size = current.shape[0]
        buffer_length = self.temporal_history_frames * self.temporal_frame_interval + 1
        if (
            self._height_map_history is None
            or self._height_map_history.shape != (buffer_length, batch_size, self.height_map_dim)
            or self._height_map_history.device != current.device
        ):
            self._height_map_history = current.unsqueeze(0).repeat(buffer_length, 1, 1)
            self._history_valid = torch.ones(batch_size, dtype=torch.bool, device=current.device)
        else:
            valid_mask = self._history_valid
            invalid_mask = ~valid_mask
            if torch.any(invalid_mask):
                self._height_map_history[:, invalid_mask] = current[invalid_mask].unsqueeze(0)
                valid_mask[invalid_mask] = True
            if torch.any(valid_mask):
                self._height_map_history[1:] = self._height_map_history[:-1].clone()
                self._height_map_history[0] = current

        offsets = [
            index * self.temporal_frame_interval
            for index in range(self.temporal_history_frames, -1, -1)
        ]
        window = torch.stack([self._height_map_history[offset] for offset in offsets], dim=1)
        self._last_temporal_height_maps = window.detach()
        return window

    def get_temporal_height_maps(self) -> torch.Tensor | None:
        """Return the most recent online window for rollout storage."""

        return self._last_temporal_height_maps

    def _stored_temporal_height_maps(self, obs, temporal_height_maps=None) -> torch.Tensor | None:
        if temporal_height_maps is not None:
            return temporal_height_maps
        if hasattr(obs, "keys") and self.temporal_observation_key in obs.keys():
            return obs[self.temporal_observation_key]
        return None

    def _normalize_temporal_height_maps(self, temporal_height_maps: torch.Tensor) -> torch.Tensor:
        """Apply the current-map normalization statistics to every temporal frame."""

        if not self.actor_obs_normalization:
            return temporal_height_maps
        mean = self.actor_obs_normalizer._mean[..., -self.height_map_dim :]
        std = self.actor_obs_normalizer._std[..., -self.height_map_dim :]
        return (temporal_height_maps - mean.unsqueeze(1)) / (std.unsqueeze(1) + self.actor_obs_normalizer.eps)

    def _normalize_height_map(self, height_map: torch.Tensor) -> torch.Tensor:
        if not self.actor_obs_normalization:
            return height_map
        mean = self.actor_obs_normalizer._mean[..., -self.height_map_dim :]
        std = self.actor_obs_normalizer._std[..., -self.height_map_dim :]
        return (height_map - mean) / (std + self.actor_obs_normalizer.eps)

    def _denormalize_height_map(self, height_map: torch.Tensor) -> torch.Tensor:
        if not self.actor_obs_normalization:
            return height_map
        mean = self.actor_obs_normalizer._mean[..., -self.height_map_dim :]
        std = self.actor_obs_normalizer._std[..., -self.height_map_dim :]
        return height_map * (std + self.actor_obs_normalizer.eps) + mean

    @torch.no_grad()
    def get_temporal_height_map_reconstruction(self, obs=None, temporal_height_maps=None):
        """Return temporal inputs, the current-map VAE reconstruction, and the predicted next map."""

        if temporal_height_maps is None:
            temporal_height_maps = self._stored_temporal_height_maps(obs) if obs is not None else None
        if temporal_height_maps is None:
            temporal_height_maps = self._last_temporal_height_maps
        if temporal_height_maps is None:
            if obs is None:
                raise ValueError("Either obs or temporal_height_maps must be provided before the first policy call.")
            temporal_height_maps = self._history_window(obs)

        if temporal_height_maps.ndim != 3 or tuple(temporal_height_maps.shape[1:]) != self.temporal_observation_shape:
            raise ValueError(
                f"Expected temporal maps shaped (batch, {self.temporal_frames}, {self.height_map_dim}), "
                f"got {tuple(temporal_height_maps.shape)}."
            )
        batch_size = temporal_height_maps.shape[0]
        normalized = self._normalize_temporal_height_maps(temporal_height_maps)
        normalized_current = normalized[:, -1]
        mean, _ = self.actor.vae.encode(normalized_current)
        reconstructed = self.actor.vae.decode(mean)
        predicted_latent = self.actor.predict_future_latent(normalized)
        predicted = self.actor.vae.decode(predicted_latent)

        original = temporal_height_maps.reshape(batch_size, self.temporal_frames, *self.temporal_map_shape)
        reconstructed = self._denormalize_height_map(reconstructed).reshape(batch_size, *self.temporal_map_shape)
        predicted = self._denormalize_height_map(predicted).reshape(batch_size, *self.temporal_map_shape)
        return original, reconstructed, predicted

    def update_distribution(self, actor_input, temporal_height_maps=None):
        mean = self.actor(actor_input, temporal_height_maps=temporal_height_maps)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, temporal_height_maps=None, **kwargs):
        # Collection has no stored window, so update the online history.  PPO
        # minibatches provide the stored window through the observation key.
        temporal_height_maps = self._stored_temporal_height_maps(obs, temporal_height_maps)
        if temporal_height_maps is None:
            temporal_height_maps = self._history_window(obs)
        else:
            self._last_temporal_height_maps = temporal_height_maps.detach()
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(
            actor_input,
            temporal_height_maps=self._normalize_temporal_height_maps(temporal_height_maps),
        )
        return self.distribution.sample()

    def act_inference(self, obs):
        temporal_height_maps = self._stored_temporal_height_maps(obs)
        if temporal_height_maps is None:
            temporal_height_maps = self._history_window(obs)
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(
            actor_input,
            temporal_height_maps=self._normalize_temporal_height_maps(temporal_height_maps),
        )

    def forward(self, obs):
        return self.act_inference(obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_height_map(self, obs) -> torch.Tensor:
        source = obs[self.height_map_obs_group]
        return source[..., self.height_map_start_index : self.height_map_end_index]

    def get_height_map_latent(self, obs, sample: bool = False) -> torch.Tensor:
        """Expose the current-frame VAE latent for debugging or downstream use."""

        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        _, height_map = self.actor.split_input(actor_input)
        return self.actor.get_latent(height_map, sample=sample)

    def get_predicted_height_map_latent(self, obs=None, temporal_height_maps=None) -> torch.Tensor:
        """Return the 3D-CNN prediction for inspection."""

        if temporal_height_maps is None:
            temporal_height_maps = self._stored_temporal_height_maps(obs) if obs is not None else None
        if temporal_height_maps is None:
            temporal_height_maps = self._last_temporal_height_maps
        if temporal_height_maps is None:
            if obs is None:
                raise ValueError("Either obs or temporal_height_maps must be provided.")
            temporal_height_maps = self._history_window(obs)
        return self.actor.predict_future_latent(self._normalize_temporal_height_maps(temporal_height_maps))

    def get_actor_obs(self, obs):
        policy_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)
        if self.height_map_in_policy:
            start = self.height_map_policy_start_index
            end = start + self.height_map_dim
            policy_obs = torch.cat((policy_obs[..., :start], policy_obs[..., end:]), dim=-1)
        return torch.cat((policy_obs, self.get_height_map(obs)), dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def get_storage_observation_template(self, obs):
        """Add zero-filled temporal fields before rollout storage is allocated."""

        storage_obs = obs.clone()
        height_map = self.get_height_map(obs)
        storage_obs.set(
            self.temporal_observation_key,
            height_map.new_zeros((*height_map.shape[:-1], *self.temporal_observation_shape)),
        )
        storage_obs.set(self.next_height_map_key, height_map.new_zeros(height_map.shape))
        storage_obs.set(self.next_height_map_valid_key, height_map.new_zeros((*height_map.shape[:-1], 1)))
        return storage_obs

    def prepare_observation_for_storage(self, obs):
        """Copy the window used for the just-sampled action into the transition."""

        storage_obs = obs.clone()
        temporal_height_maps = self.get_temporal_height_maps()
        if temporal_height_maps is None:
            temporal_height_maps = self._history_window(obs)
        storage_obs.set(self.temporal_observation_key, temporal_height_maps.detach())
        # The next-map fields are overwritten by ``set_next_observation_for_storage``.
        height_map = self.get_height_map(obs)
        storage_obs.set(self.next_height_map_key, height_map.new_zeros(height_map.shape))
        storage_obs.set(self.next_height_map_valid_key, height_map.new_zeros((*height_map.shape[:-1], 1)))
        return storage_obs

    def set_next_observation_for_storage(self, transition_obs, next_obs, dones):
        """Attach the rollout next-state map and validity mask to a transition."""

        next_height_map = self.get_height_map(next_obs).detach()
        valid = (~dones.reshape(-1).bool()).to(next_height_map.dtype).unsqueeze(-1)
        transition_obs.set(self.next_height_map_key, next_height_map)
        transition_obs.set(self.next_height_map_valid_key, valid)
        return transition_obs

    def compute_auxiliary_loss(self, obs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return VAE loss plus supervised future-latent prediction loss."""

        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        _, height_map = self.actor.split_input(actor_input)
        vae_loss, loss_terms = self.actor.vae.loss(height_map, beta=self.vae_beta)
        weighted_vae_loss = self.vae_loss_coef * vae_loss
        loss_terms["vae_total"] = weighted_vae_loss

        temporal_height_maps = self._stored_temporal_height_maps(obs)
        next_height_map = (
            obs[self.next_height_map_key]
            if hasattr(obs, "keys") and self.next_height_map_key in obs.keys()
            else None
        )
        valid = (
            obs[self.next_height_map_valid_key]
            if hasattr(obs, "keys") and self.next_height_map_valid_key in obs.keys()
            else None
        )
        if temporal_height_maps is not None and next_height_map is not None:
            predicted_latent = self.actor.predict_future_latent(
                self._normalize_temporal_height_maps(temporal_height_maps)
            )
            with torch.no_grad():
                target_latent, _ = self.actor.vae.encode(self._normalize_height_map(next_height_map))
            prediction_error = (predicted_latent - target_latent).square().mean(dim=-1)
            if valid is not None:
                valid_mask = valid.reshape(-1) > 0.5
                if torch.any(valid_mask):
                    prediction_loss = prediction_error[valid_mask].mean()
                else:
                    prediction_loss = prediction_error.sum() * 0.0
            else:
                prediction_loss = prediction_error.mean()
        else:
            prediction_loss = height_map.sum() * 0.0

        weighted_prediction_loss = self.temporal_prediction_loss_coef * prediction_loss
        loss_terms["temporal_prediction"] = prediction_loss
        loss_terms["temporal_prediction_total"] = weighted_prediction_loss
        return weighted_vae_loss + weighted_prediction_loss, loss_terms

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


# The target file historically exposed these names.  Keep them available while
# giving the implementation an explicit descriptive class name for new configs.
ActorCriticVAE = ActorCritic3DCNNVAE
ActorCritic = ActorCritic3DCNNVAE
