"""Actor-critic with a convolutional height-map autoencoder.

The actor observation is expected to end with a flattened map containing
``map_channels`` channels in channel-major order.  For the default G1 setup
the tail is ``[height_scan(16x11), height_gradient(16x11)]``.  The encoder
turns that tail into a latent vector used by the actor MLP; the decoder is
trained with PPO through :meth:`compute_auxiliary_loss` to reconstruct the
same two-channel map.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


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
    raise ValueError(f"Unsupported CNN activation '{name}'.")


class HeightMapCNN(nn.Module):
    """Convolutional encoder and transposed-convolutional decoder."""

    def __init__(
        self,
        map_shape: tuple[int, int] = (16, 11),
        map_channels: int = 2,
        latent_dim: int = 32,
        cnn_channels: Sequence[int] = (32, 64, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if len(map_shape) != 2 or any(int(size) <= 0 for size in map_shape):
            raise ValueError(f"map_shape must contain two positive dimensions, got {map_shape}.")
        if map_channels <= 0 or latent_dim <= 0:
            raise ValueError("map_channels and latent_dim must be positive.")
        if len(cnn_channels) != 3 or any(int(size) <= 0 for size in cnn_channels):
            raise ValueError("cnn_channels must contain three positive channel sizes.")

        self.map_shape = (int(map_shape[0]), int(map_shape[1]))
        self.map_channels = int(map_channels)
        self.latent_dim = int(latent_dim)
        self.map_dim = self.map_channels * self.map_shape[0] * self.map_shape[1]
        c1, c2, c3 = (int(size) for size in cnn_channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.map_channels, c1, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            _activation(activation),
        )
        with torch.no_grad():
            feature_shape = self.encoder_conv(torch.zeros(1, self.map_channels, *self.map_shape)).shape[1:]
        self._feature_shape = tuple(int(size) for size in feature_shape)
        feature_dim = int(torch.tensor(self._feature_shape).prod().item())
        self.encoder_fc = nn.Linear(feature_dim, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, feature_dim)
        self.decoder_deconv = nn.Sequential(
            # For 4x3 -> 8x6, both dimensions need one output pixel.
            nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1),
            _activation(activation),
            # For 8x6 -> 16x11, the width needs no output padding.
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            _activation(activation),
            nn.Conv2d(c1, self.map_channels, kernel_size=3, padding=1),
        )

    def _check_map(self, height_map: torch.Tensor) -> torch.Tensor:
        if height_map.ndim == 2:
            expected = self.map_dim
            if height_map.shape[-1] != expected:
                raise ValueError(f"Expected flattened map with {expected} values, got {height_map.shape[-1]}.")
            return height_map.reshape(-1, self.map_channels, *self.map_shape)
        expected_shape = (self.map_channels, *self.map_shape)
        if tuple(height_map.shape[-3:]) != expected_shape:
            raise ValueError(f"Expected map shape (..., {expected_shape}), got {tuple(height_map.shape)}.")
        return height_map.reshape(-1, *expected_shape)

    def encode(self, height_map: torch.Tensor) -> torch.Tensor:
        map_tensor = self._check_map(height_map)
        features = self.encoder_conv(map_tensor)
        return self.encoder_fc(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent shape (N, {self.latent_dim}), got {tuple(latent.shape)}.")
        features = self.decoder_fc(latent).reshape(-1, *self._feature_shape)
        reconstruction = self.decoder_deconv(features)
        # Keep the module valid for custom map dimensions whose stride-2
        # geometry does not invert exactly.
        if reconstruction.shape[-2:] != self.map_shape:
            reconstruction = F.interpolate(reconstruction, size=self.map_shape, mode="bilinear", align_corners=False)
        return reconstruction

    def forward(self, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        map_tensor = self._check_map(height_map)
        latent = self.encode(map_tensor)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class _GradCNNActor(nn.Module):
    """Actor wrapper retaining the exporter-compatible ``in_features`` field."""

    def __init__(self, map_encoder: HeightMapCNN, policy: nn.Module, policy_obs_dim: int):
        super().__init__()
        self.map_encoder = map_encoder
        self.policy = policy
        self.policy_obs_dim = int(policy_obs_dim)
        self.in_features = self.policy_obs_dim + map_encoder.map_dim

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self

    def split_input(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.in_features
        if actor_input.ndim != 2 or actor_input.shape[-1] != expected:
            raise ValueError(f"Expected actor input shape (N, {expected}), got {tuple(actor_input.shape)}.")
        return actor_input[..., : self.policy_obs_dim], actor_input[..., self.policy_obs_dim :]

    def encode_map(self, height_map: torch.Tensor) -> torch.Tensor:
        return self.map_encoder.encode(height_map)

    def decode_map(self, latent: torch.Tensor) -> torch.Tensor:
        return self.map_encoder.decode(latent)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        policy_obs, height_map = self.split_input(actor_input)
        latent = self.encode_map(height_map)
        return self.policy(torch.cat((policy_obs, latent), dim=-1))


class ActorCriticGradCNN(nn.Module):
    """PPO actor-critic using a two-channel height-map convolutional latent."""

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
        map_shape: tuple[int, int] = (16, 11),
        map_channels: int = 2,
        latent_dim: int = 32,
        cnn_channels=(32, 64, 128),
        reconstruction_loss_coef: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "ActorCriticGradCNN.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        if reconstruction_loss_coef < 0.0:
            raise ValueError("reconstruction_loss_coef must be non-negative.")
        self.obs_groups = obs_groups
        self.reconstruction_loss_coef = float(reconstruction_loss_coef)
        self.map_shape = (int(map_shape[0]), int(map_shape[1]))
        self.map_channels = int(map_channels)
        self.map_dim = self.map_channels * self.map_shape[0] * self.map_shape[1]

        actor_obs = self._flatten_groups(obs, obs_groups["policy"])
        critic_obs = self._flatten_groups(obs, obs_groups["critic"])
        if actor_obs.shape[-1] <= self.map_dim:
            raise ValueError(
                f"Actor observation ({actor_obs.shape[-1]}) must contain proprioception plus map tail ({self.map_dim})."
            )
        self.actor_obs_dim = int(actor_obs.shape[-1])
        self.policy_obs_dim = self.actor_obs_dim - self.map_dim
        self.critic_obs_dim = int(critic_obs.shape[-1])

        map_encoder = HeightMapCNN(
            map_shape=self.map_shape,
            map_channels=self.map_channels,
            latent_dim=latent_dim,
            cnn_channels=cnn_channels,
            activation=activation,
        )
        actor_policy = MLP(self.policy_obs_dim + latent_dim, num_actions, list(actor_hidden_dims), activation)
        self.actor = _GradCNNActor(map_encoder, actor_policy, self.policy_obs_dim)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(self.actor_obs_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic = MLP(self.critic_obs_dim, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.critic_obs_dim) if critic_obs_normalization else nn.Identity()
        )
        print(f"Height-map CNN autoencoder: {self.actor.map_encoder}")
        print(f"CNN actor: {self.actor.policy}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}.")
        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def encoder(self) -> nn.Module:
        """Convolutional feature extractor used before the latent projection."""

        return self.actor.map_encoder.encoder_conv

    @property
    def decoder(self) -> nn.Module:
        """Transposed-convolutional feature decoder."""

        return self.actor.map_encoder.decoder_deconv

    @staticmethod
    def _flatten_group(group) -> torch.Tensor:
        if isinstance(group, torch.Tensor):
            return group.flatten(start_dim=1)
        return torch.cat([value.flatten(start_dim=1) for value in group.values()], dim=-1)

    @classmethod
    def _flatten_groups(cls, obs, groups) -> torch.Tensor:
        return torch.cat([cls._flatten_group(obs[name]) for name in groups], dim=-1)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_obs(self, obs):
        actor_obs = self._flatten_groups(obs, self.obs_groups["policy"])
        if actor_obs.shape[-1] != self.actor_obs_dim:
            raise ValueError(f"Expected {self.actor_obs_dim} actor observations, got {actor_obs.shape[-1]}.")
        # The map is deliberately the final observation tail so term ordering
        # determines the channel order: raw height first, temporal gradient second.
        return actor_obs

    def get_critic_obs(self, obs):
        critic_obs = self._flatten_groups(obs, self.obs_groups["critic"])
        if critic_obs.shape[-1] != self.critic_obs_dim:
            raise ValueError(f"Expected {self.critic_obs_dim} critic observations, got {critic_obs.shape[-1]}.")
        return critic_obs

    def _actor_input(self, obs) -> torch.Tensor:
        return self.actor_obs_normalizer(self.get_actor_obs(obs))

    def _map_from_actor_input(self, actor_input: torch.Tensor) -> torch.Tensor:
        return actor_input[..., -self.map_dim :]

    def update_distribution(self, obs):
        actor_input = self._actor_input(obs)
        mean = self.actor(actor_input)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self._actor_input(obs))

    def forward(self, obs):
        return self.act_inference(obs)

    def evaluate(self, obs, **kwargs):
        return self.critic(self.critic_obs_normalizer(self.get_critic_obs(obs)))

    def get_height_map(self, obs) -> torch.Tensor:
        """Return the actor map as ``(N, channels, height, width)``."""

        actor_input = self._actor_input(obs)
        return self._map_from_actor_input(actor_input).reshape(-1, self.map_channels, *self.map_shape)

    def get_height_map_latent(self, obs) -> torch.Tensor:
        return self.actor.map_encoder.encode(self._map_from_actor_input(self._actor_input(obs)))

    def compute_auxiliary_loss(self, obs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the decoder reconstruction loss consumed by PPO."""

        map_input = self._map_from_actor_input(self._actor_input(obs))
        reconstruction, _ = self.actor.map_encoder(map_input)
        target = map_input.reshape(-1, self.map_channels, *self.map_shape)
        reconstruction_loss = F.mse_loss(reconstruction, target)
        total_loss = self.reconstruction_loss_coef * reconstruction_loss
        return total_loss, {
            "cnn_reconstruction": reconstruction_loss,
            "cnn_reconstruction_total": total_loss,
        }

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


# Keep the explicit class available under a short conventional alias for
# callers that select policies by a class name in a YAML/config file.
ActorCritic = ActorCriticGradCNN
