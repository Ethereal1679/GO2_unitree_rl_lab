"""RSL-RL Actor-Critic wrapper for the Go2 attention policy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP

from .attention_policy import AttentionMapEncoder


class _FlatAttentionActor(nn.Module):
    """Flat-input actor wrapper used by PPO and Isaac Lab exporters."""

    def __init__(self, encoder: AttentionMapEncoder, mlp: nn.Module, in_features: int):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.in_features = in_features

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        proprioception, map_scans = self.encoder.split_observation(observation)
        map_encoding = self.encoder.encode(map_scans, proprioception, role="actor").flatten(start_dim=1)
        return self.mlp(torch.cat((map_encoding, proprioception), dim=-1))

    def forward_with_attention(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        map_encoding, attention, proprioception = self.encoder.forward_observation(
            observation, role="actor", return_attention=True
        )
        return {
            "actions": self.mlp(torch.cat((map_encoding, proprioception), dim=-1)),
            "attention_weights": attention,
        }


class AttentionMapActorCritic(nn.Module):
    """PPO actor-critic whose actor and critic both encode the map tail."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        embedding_dim: int = 64,
        num_heads: int = 16,
        map_shape: tuple[int, int] = (26, 16),
        return_attention_weights: bool = False,
        attention_visualization=None,
        use_actor_att_map_for_critic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "AttentionMapActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        self.obs_groups = obs_groups
        actor_obs = self._flatten_groups(obs, obs_groups["policy"])
        critic_obs = self._flatten_groups(obs, obs_groups["critic"])
        map_dim = map_shape[0] * map_shape[1] * 3
        actor_proprioception_dim = actor_obs.shape[-1] - map_dim
        critic_proprioception_dim = critic_obs.shape[-1] - map_dim
        if actor_proprioception_dim <= 0 or critic_proprioception_dim <= 0:
            raise ValueError(
                f"map tail ({map_dim}) must leave proprioception in both observations: "
                f"actor={actor_obs.shape[-1]}, critic={critic_obs.shape[-1]}"
            )
        self.map_shape = map_shape
        self.map_dim = map_dim
        self.actor_proprioception_dim = actor_proprioception_dim
        self.critic_proprioception_dim = critic_proprioception_dim
        encoder = AttentionMapEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            proprioception_dim=actor_proprioception_dim,
            critic_proprioception_dim=critic_proprioception_dim,
            map_shape=map_shape,
        )
        actor_mlp = MLP(embedding_dim + actor_proprioception_dim, num_actions, list(actor_hidden_dims), activation)
        self.actor_obs_dim = actor_obs.shape[-1]
        self.actor = _FlatAttentionActor(encoder, actor_mlp, self.actor_obs_dim)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_obs.shape[-1]) if actor_obs_normalization else nn.Identity()
        )
        self.critic = MLP(embedding_dim + critic_proprioception_dim, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_obs.shape[-1]) if critic_obs_normalization else nn.Identity()
        )
        self.use_actor_att_map_for_critic = use_actor_att_map_for_critic
        self._cached_actor_obs = None
        self._cached_actor_map_encoding = None
        self.register_buffer("last_attention_weights", torch.empty(0), persistent=False)
        print(f"Attention encoder: {self.encoder}")
        print(f"Attention actor: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def encoder(self) -> AttentionMapEncoder:
        return self.actor.encoder

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

    def forward(self, obs, return_attention: bool = False):
        """Run the actor and optionally return the map attention weights."""

        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        map_encoding, attention, proprioception = self.encoder.forward_observation(
            actor_obs, role="actor", return_attention=return_attention
        )
        actions = self.actor.mlp(torch.cat((map_encoding, proprioception), dim=-1))
        if return_attention:
            self.last_attention_weights = attention.detach()
            return {"actions": actions, "attention_weights": attention}
        return actions

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
        return self._flatten_groups(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs):
        return self._flatten_groups(obs, self.obs_groups["critic"])

    def update_distribution(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        map_encoding, _, proprioception = self.encoder.forward_observation(actor_obs, role="actor")
        if self.use_actor_att_map_for_critic:
            self._cached_actor_obs = obs
            self._cached_actor_map_encoding = map_encoding
        mean = self.actor.mlp(torch.cat((map_encoding, proprioception), dim=-1))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs, return_attention: bool = False):
        return self.forward(obs, return_attention=return_attention)

    @property
    def attention_weights(self):
        """Most recent inference attention weights, or an empty tensor when disabled."""

        return self.last_attention_weights

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if self.use_actor_att_map_for_critic:
            if obs is self._cached_actor_obs:
                map_encoding = self._cached_actor_map_encoding.detach()
            else:
                actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
                map_encoding, _, _ = self.encoder.forward_observation(actor_obs, role="actor")
                map_encoding = map_encoding.detach()
            # A cached encoding is valid for exactly one act/evaluate pair.
            # Clearing it prevents a mutable observation container from
            # accidentally reusing a previous step's map during bootstrapping.
            self._cached_actor_obs = None
            self._cached_actor_map_encoding = None
            proprioception, _ = self.encoder.split_observation(critic_obs)
            return self.critic(torch.cat((map_encoding, proprioception), dim=-1))
        map_encoding, _, proprioception = self.encoder.forward_observation(critic_obs, role="critic")
        return self.critic(torch.cat((map_encoding, proprioception), dim=-1))

    def get_actor_map_scans(self, obs) -> torch.Tensor:
        """Return the actor map tail for visualization."""

        flat_obs = self.get_actor_obs(obs)
        return flat_obs[:, -self.map_dim :].reshape(-1, *self.map_shape, 3)

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
