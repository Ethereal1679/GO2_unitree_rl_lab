"""RSL-RL Actor-Critic wrapper for the Go2 attention policy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP

from .attention_policy import AttentionMapActor


class AttentionMapActorCritic(nn.Module):
    """PPO-compatible actor-critic with attention-based map encoding.

    The environment must provide top-level ``proprioception`` and ``map_scans``
    observation groups. The critic continues to use the configured flattened
    critic observation groups.
    """

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
        proprioception_obs_group: str = "proprioception",
        map_scans_obs_group: str = "map_scans",
        embedding_dim: int = 64,
        num_heads: int = 16,
        map_shape: tuple[int, int] = (26, 16),
        return_attention_weights: bool = False,
        attention_visualization=None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "AttentionMapActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        self.obs_groups = obs_groups
        self.proprioception_obs_group = proprioception_obs_group
        self.map_scans_obs_group = map_scans_obs_group

        proprioception = obs[proprioception_obs_group]
        map_scans = obs[map_scans_obs_group]
        if tuple(proprioception.shape[1:]) != (48,):
            raise ValueError(f"Expected proprioception shape [B, 48], got {tuple(proprioception.shape)}")
        if map_scans.ndim != 4 or map_scans.shape[-1] != 3:
            raise ValueError(f"Expected map_scans shape [B, H, W, 3], got {tuple(map_scans.shape)}")
        map_shape = tuple(map_scans.shape[1:3])
        expected_map_shape = (*map_shape, 3)
        if tuple(map_scans.shape[1:]) != expected_map_shape:
            raise ValueError(f"Expected map_scans shape [B, *{expected_map_shape}], got {tuple(map_scans.shape)}")

        num_critic_obs = self._get_flat_group_dim(obs, obs_groups["critic"])
        self.actor = AttentionMapActor(
            num_actions=num_actions,
            proprioception_dim=48,
            map_shape=map_shape,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dims=actor_hidden_dims,
            return_attention_weights=return_attention_weights,
        )
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(self.actor.in_features) if actor_obs_normalization else nn.Identity()
        )
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )
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

    @staticmethod
    def _get_flat_group_dim(obs, groups) -> int:
        total = 0
        for group_name in groups:
            group = obs[group_name]
            if isinstance(group, torch.Tensor):
                total += group[0].numel()
            else:
                for value in group.values():
                    total += value[0].numel() if value.ndim > 1 else value.shape[-1]
        return total

    @staticmethod
    def _flatten_group(group) -> torch.Tensor:
        if isinstance(group, torch.Tensor):
            return group
        return torch.cat([value.flatten(start_dim=1) for value in group.values()], dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self, obs, return_attention: bool = False):
        """Run the actor and optionally return the map attention weights."""

        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if return_attention:
            return self.actor.forward_with_attention(actor_obs)
        return self.actor(actor_obs)

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
        proprioception = obs[self.proprioception_obs_group]
        map_scans = obs[self.map_scans_obs_group]
        return torch.cat((proprioception, map_scans.flatten(start_dim=1)), dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([self._flatten_group(obs[group_name]) for group_name in self.obs_groups["critic"]], dim=-1)

    def update_distribution(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        mean = self.actor(actor_obs)
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

        return self.actor.last_attention_weights

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

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
