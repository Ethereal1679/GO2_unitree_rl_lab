"""Reusable attention networks for the Unitree Go2 policy."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class AttentionMapEncoder(nn.Module):
    """Encode map XYZ points stored at the tail of a flat observation."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 16,
        proprioception_dim: int = 48,
        critic_proprioception_dim: int | None = None,
        map_shape: tuple[int, int] = (26, 16),
        map_point_dim: int = 3,
    ) -> None:
        super().__init__()
        map_feature_channels = embedding_dim - map_point_dim
        if map_feature_channels <= 0:
            raise ValueError("embedding_dim must be greater than map_point_dim")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.proprioception_dim = proprioception_dim
        self.actor_proprioception_dim = proprioception_dim
        self.critic_proprioception_dim = critic_proprioception_dim or proprioception_dim
        self.map_shape = map_shape
        self.map_point_dim = map_point_dim
        self.num_map_points = map_shape[0] * map_shape[1]

        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, map_feature_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(map_feature_channels),
        )
        self.actor_proprioception_encoder = nn.Sequential(
            nn.Linear(self.actor_proprioception_dim, embedding_dim), nn.ELU()
        )
        self.critic_proprioception_encoder = (
            self.actor_proprioception_encoder
            if self.critic_proprioception_dim == self.actor_proprioception_dim
            else nn.Sequential(nn.Linear(self.critic_proprioception_dim, embedding_dim), nn.ELU())
        )
        self.proprioception_encoder = self.actor_proprioception_encoder
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def _validate(self, map_scans: Tensor, proprioception: Tensor, valid_mask: Tensor | None) -> None:
        expected_map = (*self.map_shape, self.map_point_dim)
        if map_scans.ndim != 4 or tuple(map_scans.shape[1:]) != expected_map:
            raise ValueError(f"map_scans must have shape [B, *{expected_map}], got {tuple(map_scans.shape)}")
        if proprioception.ndim != 2:
            raise ValueError(f"proprioception must have shape [B, D], got {tuple(proprioception.shape)}")
        if map_scans.shape[0] != proprioception.shape[0]:
            raise ValueError("map_scans and proprioception must have the same batch size")
        if valid_mask is not None and tuple(valid_mask.shape) != (map_scans.shape[0], self.num_map_points):
            raise ValueError(
                f"valid_mask must have shape {(map_scans.shape[0], self.num_map_points)}, got {tuple(valid_mask.shape)}"
            )

    def split_observation(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Split ``[proprioception, map_xyz]`` by the map tail dimension."""

        if observation.ndim != 2:
            raise ValueError("observation must have shape [B, D]")
        map_dim = self.num_map_points * self.map_point_dim
        if observation.shape[-1] <= map_dim:
            raise ValueError("observation must contain proprioception and map points")
        proprioception = observation[:, :-map_dim]
        map_scans = observation[:, -map_dim:].reshape(
            observation.shape[0], self.map_shape[0], self.map_shape[1], self.map_point_dim
        )
        return proprioception, map_scans

    def _query(self, proprioception: Tensor, role: str) -> Tensor:
        if role == "actor":
            return self.actor_proprioception_encoder(proprioception)
        if role == "critic":
            return self.critic_proprioception_encoder(proprioception)
        raise ValueError("role must be 'actor' or 'critic'")

    def forward_with_intermediates(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        role: str = "actor",
    ) -> dict[str, Tensor]:
        self._validate(map_scans, proprioception, valid_mask)
        height = map_scans[..., 2].unsqueeze(1)
        cnn_features_nchw = self.map_cnn(height)
        cnn_features = cnn_features_nchw.permute(0, 2, 3, 1)
        point_features = torch.cat((cnn_features, map_scans), dim=-1).reshape(
            map_scans.shape[0], self.num_map_points, self.embedding_dim
        )
        query = self._query(proprioception, role).unsqueeze(1)
        key_padding_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=map_scans.device, dtype=torch.bool)
            if (~valid_mask).all(dim=1).any():
                raise ValueError("each batch item must contain at least one valid map point")
            key_padding_mask = ~valid_mask
        # import ipdb; ipdb.set_trace()
        map_encoding, attention_weights = self.cross_attention(
            query=query,
            key=point_features,
            value=point_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return {
            "height": height,
            "cnn_features_nchw": cnn_features_nchw,
            "cnn_features": cnn_features,
            "point_features": point_features,
            "query": query,
            "map_encoding": map_encoding,
            "attention_weights": attention_weights,
        }

    def forward_observation(
        self, observation: Tensor, role: str = "actor", return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Encode a flat observation whose final fields are map XYZ points."""

        proprioception, map_scans = self.split_observation(observation)
        expected_dim = self.actor_proprioception_dim if role == "actor" else self.critic_proprioception_dim
        if proprioception.shape[-1] != expected_dim:
            raise ValueError("proprioception dimension does not match the configured role")
        if return_attention:
            output = self.forward_with_intermediates(map_scans, proprioception, role=role)
            return output["map_encoding"].flatten(start_dim=1), output["attention_weights"], proprioception
        return self.encode(map_scans, proprioception, role=role).flatten(start_dim=1), None, proprioception

    # play ffp
    def forward(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        role: str = "actor",
    ) -> tuple[Tensor, Tensor]:
        height = map_scans[..., 2].unsqueeze(1)
        cnn_features = self.map_cnn(height).permute(0, 2, 3, 1)
        point_features = torch.cat((cnn_features, map_scans), dim=-1).reshape(
            map_scans.shape[0], self.num_map_points, self.embedding_dim
        )
        query = self._query(proprioception, role).unsqueeze(1)
        key_padding_mask = None if valid_mask is None else ~valid_mask.to(dtype=torch.bool)
        # import ipdb; ipdb.set_trace()
        map_encoding, attention_weights = self.cross_attention(
            query,
            point_features,
            point_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        assert attention_weights is not None
        return map_encoding, attention_weights

    # train ffp, donot return attention weights
    def encode(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        role: str = "actor",
    ) -> Tensor:
        """Encode the map without materializing per-head attention weights."""

        height = map_scans[..., 2].unsqueeze(1) # torch.Size([2048, 1, 16, 11])
        cnn_features = self.map_cnn(height).permute(0, 2, 3, 1) # torch.Size([2048, 16, 11, 61])
        point_features = torch.cat((cnn_features, map_scans), dim=-1).reshape(
            map_scans.shape[0], self.num_map_points, self.embedding_dim
        ) # torch.Size([2048, 176, 64])
        query = self._query(proprioception, role).unsqueeze(1) # torch.Size([2048, 1, 64])
        key_padding_mask = None if valid_mask is None else ~valid_mask.to(dtype=torch.bool)
        # import ipdb; ipdb.set_trace()
        map_encoding, _ = self.cross_attention( # torch.Size([2048, 1, 64])
            query,
            point_features,
            point_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return map_encoding


class Go2AttentionPolicy(nn.Module):
    """Standalone deterministic Go2 attention policy."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 16,
        proprioception_dim: int = 48,
        action_dim: int = 12,
        hidden_dims: Sequence[int] = (256, 128),
        map_shape: tuple[int, int] = (26, 16),
    ) -> None:
        super().__init__()
        self.map_encoder = AttentionMapEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            proprioception_dim=proprioception_dim,
            map_shape=map_shape,
        )
        layers: list[nn.Module] = []
        input_dim = embedding_dim + proprioception_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.extend((nn.Linear(input_dim, action_dim), nn.Tanh()))
        self.policy_mlp = nn.Sequential(*layers)

    def forward(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        output = self.forward_with_intermediates(map_scans, proprioception, valid_mask)
        if return_attention:
            return {
                "actions": output["actions"],
                "attention_weights": output["attention_weights"],
            }
        return {
            "actions": output["actions"],
            "map_encoding": output["map_encoding"],
            "attention_weights": output["attention_weights"],
        }

    def forward_with_intermediates(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        output = self.map_encoder.forward_with_intermediates(map_scans, proprioception, valid_mask)
        map_encoding_flat = output["map_encoding"].flatten(start_dim=1)
        policy_input = torch.cat((map_encoding_flat, proprioception), dim=-1)
        return {
            **output,
            "map_encoding_flat": map_encoding_flat,
            "policy_input": policy_input,
            "actions": self.policy_mlp(policy_input),
        }


class AttentionMapActor(nn.Module):
    """Flat-input actor used by RSL-RL and Isaac Lab exporters."""

    def __init__(
        self,
        num_actions: int,
        proprioception_dim: int = 48,
        map_shape: tuple[int, int] = (26, 16),
        embedding_dim: int = 64,
        num_heads: int = 16,
        hidden_dims: Sequence[int] = (256, 128),
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()
        self.proprioception_dim = proprioception_dim
        self.map_shape = map_shape
        self.map_height = map_shape[0]
        self.map_width = map_shape[1]
        self.map_dim = map_shape[0] * map_shape[1] * 3
        self.in_features = proprioception_dim + self.map_dim
        self.return_attention_weights = return_attention_weights
        self.register_buffer("last_attention_weights", torch.empty(0), persistent=False)
        self.map_encoder = AttentionMapEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            proprioception_dim=proprioception_dim,
            map_shape=map_shape,
        )
        layers: list[nn.Module] = []
        input_dim = embedding_dim + proprioception_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.extend((nn.Linear(input_dim, num_actions), nn.Tanh()))
        self.policy_mlp = nn.Sequential(*layers)

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self

    def forward(self, actor_input: Tensor) -> Tensor:
        proprioception, map_scans = self.map_encoder.split_observation(actor_input)
        if self.return_attention_weights:
            map_encoding, attention_weights = self.map_encoder(map_scans, proprioception)
            self.last_attention_weights = attention_weights.detach()
        else:
            map_encoding = self.map_encoder.encode(map_scans, proprioception)
        policy_input = torch.cat((map_encoding.flatten(start_dim=1), proprioception), dim=-1)
        return self.policy_mlp(policy_input)

    def forward_with_attention(self, actor_input: Tensor) -> dict[str, Tensor]:
        """Run the actor while returning its per-head cross-attention weights."""

        proprioception, map_scans = self.map_encoder.split_observation(actor_input)
        map_encoding, attention_weights = self.map_encoder(map_scans, proprioception)
        self.last_attention_weights = attention_weights.detach()
        policy_input = torch.cat((map_encoding.flatten(start_dim=1), proprioception), dim=-1)
        return {
            "actions": self.policy_mlp(policy_input),
            "attention_weights": attention_weights,
        }
