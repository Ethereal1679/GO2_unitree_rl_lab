"""Actor-critic with a convolutional height-map variational autoencoder.

The actor observation ends with a flattened two-channel map containing the
current height scan and its spatial edge map. The edge map is computed by
the observation term before it reaches this module.
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


class HeightMapCNNVAE(nn.Module):
    """Convolutional variational encoder and transposed-convolutional decoder."""

    def __init__(
        self,
        map_shape: tuple[int, int] = (17, 11),
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

        self.map_height = int(map_shape[0])
        self.map_width = int(map_shape[1])
        self.map_shape = (self.map_height, self.map_width)
        self.map_channels = int(map_channels)
        self.latent_dim = int(latent_dim)
        self.map_dim = self.map_channels * self.map_height * self.map_width
        c1, c2, c3 = (int(size) for size in cnn_channels)
        first_height = (self.map_height + 1) // 2
        first_width = (self.map_width + 1) // 2
        second_height = (first_height + 1) // 2
        second_width = (first_width + 1) // 2
        first_output_padding = (
            first_height - (2 * second_height - 1),
            first_width - (2 * second_width - 1),
        )
        second_output_padding = (
            self.map_height - (2 * first_height - 1),
            self.map_width - (2 * first_width - 1),
        )
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.map_channels, c1, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            _activation(activation),
        )
        with torch.no_grad():
            feature_shape = self.encoder_conv(
                torch.zeros(1, self.map_channels, self.map_height, self.map_width)
            ).shape
        self.feature_channels = int(feature_shape[1])
        self.feature_height = int(feature_shape[2])
        self.feature_width = int(feature_shape[3])
        feature_dim = self.feature_channels * self.feature_height * self.feature_width
        self.encoder_mean = nn.Linear(feature_dim, self.latent_dim)
        self.encoder_log_var = nn.Linear(feature_dim, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, feature_dim)
        self.decoder_deconv = nn.Sequential(
            # Mirror both stride-2 encoder stages exactly. For a 17 x 11 map
            # this reconstructs 5 x 3 -> 9 x 6 -> 17 x 11.
            nn.ConvTranspose2d(
                c3,
                c2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=first_output_padding,
            ),
            _activation(activation),
            nn.ConvTranspose2d(
                c2,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=second_output_padding,
            ),
            _activation(activation),
            nn.Conv2d(c1, self.map_channels, kernel_size=3, padding=1),
        )

    def _check_map(self, height_map: torch.Tensor) -> torch.Tensor:
        # TorchScript cannot infer tuple sizes created from ``tensor.shape``
        # slices or starred tuple expansion. Keep every dimension explicit.
        if height_map.dim() == 2:
            if height_map.size(1) != self.map_dim:
                raise ValueError("Flattened height-map dimension does not match map_channels * height * width.")
            return height_map.reshape(-1, self.map_channels, self.map_height, self.map_width)
        if height_map.dim() != 4:
            raise ValueError("Height map must be a rank-2 flattened tensor or a rank-4 NCHW tensor.")
        if (
            height_map.size(1) != self.map_channels
            or height_map.size(2) != self.map_height
            or height_map.size(3) != self.map_width
        ):
            raise ValueError("Height-map NCHW dimensions do not match the configured map shape.")
        return height_map.reshape(-1, self.map_channels, self.map_height, self.map_width)

    def encode(self, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and log-variance of ``q(z | height_map)``."""

        map_tensor = self._check_map(height_map)
        features = self.encoder_conv(map_tensor)
        flattened_features = features.flatten(start_dim=1)
        mean = self.encoder_mean(flattened_features)
        # Bounding log-variance prevents numerical overflow early in PPO training.
        log_var = self.encoder_log_var(flattened_features).clamp(min=-20.0, max=10.0)
        return mean, log_var

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Draw a differentiable sample from the encoded latent distribution."""

        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2 or latent.size(1) != self.latent_dim:
            raise ValueError("Latent tensor does not match the configured latent dimension.")
        features = self.decoder_fc(latent).reshape(
            -1, self.feature_channels, self.feature_height, self.feature_width
        )
        reconstruction = self.decoder_deconv(features)
        # Keep the module valid for custom map dimensions whose stride-2
        # geometry does not invert exactly.
        if reconstruction.size(2) != self.map_height or reconstruction.size(3) != self.map_width:
            reconstruction = F.interpolate(
                reconstruction,
                size=[self.map_height, self.map_width],
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def forward(
        self, height_map: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_tensor = self._check_map(height_map)
        mean, log_var = self.encode(map_tensor)
        latent = self.reparameterize(mean, log_var) if sample else mean
        reconstruction = self.decode(latent)
        return reconstruction, mean, log_var, latent

    def loss(
        self, height_map: torch.Tensor, beta: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the beta-VAE reconstruction and KL losses."""

        map_tensor = self._check_map(height_map)
        reconstruction, mean, log_var, _ = self(map_tensor, sample=True)
        reconstruction_loss = F.mse_loss(reconstruction, map_tensor)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1.0 + log_var - mean.square() - log_var.exp(), dim=-1)
        )
        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss, {
            "cnn_vae_reconstruction": reconstruction_loss,
            "cnn_vae_kl": kl_loss,
        }


class _GradCNNActor(nn.Module):
    """Actor wrapper retaining the exporter-compatible ``in_features`` field."""

    def __init__(
        self,
        map_encoder: HeightMapCNNVAE,
        policy: nn.Module,
        policy_obs_dim: int,
    ):
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
        if actor_input.dim() != 2 or actor_input.size(1) != expected:
            raise ValueError("Actor input does not match proprioception plus flattened height-map dimensions.")
        return actor_input[..., : self.policy_obs_dim], actor_input[..., self.policy_obs_dim :]

    def encode_map(self, height_map: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean, log_var = self.map_encoder.encode(height_map)
        return self.map_encoder.reparameterize(mean, log_var) if sample else mean

    def decode_map(self, latent: torch.Tensor) -> torch.Tensor:
        return self.map_encoder.decode(latent)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        policy_obs, height_map = self.split_input(actor_input)
        # Deterministic latents keep PPO likelihood ratios and exported policies stable.
        latent = self.encode_map(height_map, sample=False)
        return self.policy(torch.cat((policy_obs, latent), dim=-1))


class ActorCriticGradCNN(nn.Module):
    """PPO actor-critic using a two-channel height-map CNN-VAE latent."""

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
        map_shape: tuple[int, int] = (17, 11),
        map_channels: int = 2,
        latent_dim: int = 32,
        cnn_channels=(32, 64, 128),
        reconstruction_loss_coef: float = 1.0,
        vae_beta: float = 1.0e-3,
        action_scale: float | Sequence[float] = 0.25,
        default_joint_pos: Sequence[float] | None = None,
        joint_pos_lower_limits: Sequence[float] | None = None,
        joint_pos_upper_limits: Sequence[float] | None = None,
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
        if vae_beta < 0.0:
            raise ValueError("vae_beta must be non-negative.")
        if map_channels != 2:
            raise ValueError("ActorCriticGradCNN requires two map channels: height and spatial edge map.")
        self.obs_groups = obs_groups
        self.reconstruction_loss_coef = float(reconstruction_loss_coef)
        self.vae_beta = float(vae_beta)
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

        map_encoder = HeightMapCNNVAE(
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
        print(f"Height-map CNN-VAE: {self.actor.map_encoder}")
        print(f"CNN-VAE actor: {self.actor.policy}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}.")

        joint_config = (default_joint_pos, joint_pos_lower_limits, joint_pos_upper_limits)
        if any(value is not None for value in joint_config):
            if any(value is None for value in joint_config):
                raise ValueError("Default joint positions and both joint limit arrays must be provided together.")
            self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32), persistent=False)
            self.register_buffer("default_joint_pos", torch.as_tensor(default_joint_pos, dtype=torch.float32), persistent=False)
            self.register_buffer("joint_pos_lower_limits",torch.as_tensor(joint_pos_lower_limits, dtype=torch.float32),persistent=False,)
            self.register_buffer("joint_pos_upper_limits",torch.as_tensor(joint_pos_upper_limits, dtype=torch.float32),persistent=False,)
        else:
            self.action_scale = None
            self.default_joint_pos = None
            self.joint_pos_lower_limits = None
            self.joint_pos_upper_limits = None
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
        # The final observation term contains the raw height map followed by its gradient.
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
        # 根据关节位置限制裁剪高斯策略的动作均值
        if self.action_scale is not None and self.default_joint_pos is not None and self.joint_pos_lower_limits is not None and self.joint_pos_upper_limits is not None:
            lim_max = (self.joint_pos_upper_limits - self.default_joint_pos) / self.action_scale
            lim_min = (self.joint_pos_lower_limits - self.default_joint_pos) / self.action_scale
            mean = torch.clamp(mean, min=lim_min, max=lim_max,) 

        # if not torch.isfinite(mean).all():
        #     raise FloatingPointError("Actor mean contains NaN/Inf")
        # if not torch.isfinite(std).all() or (std <= 0).any():
        #     raise FloatingPointError(f"Invalid action std: {std}")
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

    def get_height_map_reconstruction(
        self, obs, env_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the CNN input map and its decoder reconstruction.

        When ``env_id`` is provided, only that environment is decoded. This
        keeps live play-mode visualization inexpensive with many environments.
        """

        actor_input = self._actor_input(obs)
        if env_id is not None:
            if env_id < 0 or env_id >= actor_input.shape[0]:
                raise ValueError(f"env_id {env_id} is outside batch size {actor_input.shape[0]}.")
            actor_input = actor_input[env_id : env_id + 1]
        map_input = self._map_from_actor_input(actor_input)
        original = map_input.reshape(-1, self.map_channels, *self.map_shape)
        reconstruction, _, _, _ = self.actor.map_encoder(map_input, sample=False)
        return original, reconstruction

    def get_height_map_latent(self, obs, sample: bool = False) -> torch.Tensor:
        """Return the latent mean, or a sample when requested."""

        map_input = self._map_from_actor_input(self._actor_input(obs))
        return self.actor.encode_map(map_input, sample=sample)

    def compute_auxiliary_loss(self, obs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted beta-VAE loss consumed by PPO."""

        map_input = self._map_from_actor_input(self._actor_input(obs))
        vae_loss, loss_terms = self.actor.map_encoder.loss(map_input, beta=self.vae_beta)
        weighted_vae_loss = self.reconstruction_loss_coef * vae_loss
        loss_terms["cnn_vae_total"] = weighted_vae_loss
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


# Keep the explicit class available under a short conventional alias for
# callers that select policies by a class name in a YAML/config file.
ActorCritic = ActorCriticGradCNN

# Preserve the old helper name for callers that import it directly.
HeightMapCNN = HeightMapCNNVAE
