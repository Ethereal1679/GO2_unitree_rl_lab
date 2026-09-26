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
        
        # process image
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
        # Keep these checks TorchScript-compatible.  In particular,
        # ``tuple(tensor.shape)`` and dynamic tuple expansion cannot be
        # statically inferred by ``torch.jit.script`` during policy export.
        if map_scans.ndim != 4:
            raise ValueError("map_scans must have rank 4: [B, map_h, map_w, point_dim]")
        if (
            map_scans.shape[1] != self.map_shape[0]
            or map_scans.shape[2] != self.map_shape[1]
            or map_scans.shape[3] != self.map_point_dim
        ):
            raise ValueError("map_scans has an unexpected spatial or point dimension")
        if proprioception.ndim != 2:
            raise ValueError("proprioception must have rank 2: [B, D]")
        if map_scans.shape[0] != proprioception.shape[0]:
            raise ValueError("map_scans and proprioception must have the same batch size")
        if valid_mask is not None:
            if (
                valid_mask.ndim != 2
                or valid_mask.shape[0] != map_scans.shape[0]
                or valid_mask.shape[1] != self.num_map_points
            ):
                raise ValueError("valid_mask must have shape [B, num_map_points]")

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

    def _prepare_attention(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None,
        role: str,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Build attention inputs shared by the weighted and fast paths."""
        self._validate(map_scans, proprioception, valid_mask)
        height = map_scans[..., 2].unsqueeze(1)
        cnn_features = self.map_cnn(height).permute(0, 2, 3, 1)
        point_features = torch.cat((cnn_features, map_scans), dim=-1).reshape(
            map_scans.shape[0], self.num_map_points, self.embedding_dim
        )
        query = self._query(proprioception, role).unsqueeze(1)

        if valid_mask is None:
            key_padding_mask = None
        else:
            valid_mask = valid_mask.to(device=map_scans.device, dtype=torch.bool)
            if (~valid_mask).all(dim=1).any():
                raise ValueError("each batch item must contain at least one valid map point")
            key_padding_mask = ~valid_mask
        return query, point_features, key_padding_mask

    def _attend(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None,
        role: str,
        return_attention: bool,
    ) -> tuple[Tensor, Tensor | None]:
        query, point_features, key_padding_mask = self._prepare_attention(
            map_scans, proprioception, valid_mask, role
        )
        map_encoding, attention_weights = self.cross_attention(
            query,
            point_features,
            point_features,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            # The visualizer consumes one map per attention head.  PyTorch's
            # default averages the head dimension and changes the result from
            # [B, heads, 1, points] to [B, 1, points].
            average_attn_weights=False,
        )
        return map_encoding, attention_weights

    def forward_with_intermediates(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        role: str = "actor",
    ) -> dict[str, Tensor]:
        """Compatibility/debug path returning the complete attention pipeline."""

        query, point_features, key_padding_mask = self._prepare_attention(
            map_scans, proprioception, valid_mask, role
        )
        map_encoding, attention_weights = self.cross_attention(
            query,
            point_features,
            point_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        assert attention_weights is not None
        height = map_scans[..., 2].unsqueeze(1)
        cnn_features_nchw = point_features[..., : self.embedding_dim - self.map_point_dim]
        cnn_features = cnn_features_nchw.reshape(
            map_scans.shape[0], self.map_shape[0], self.map_shape[1], -1
        )
        return {
            "height": height,
            "cnn_features_nchw": cnn_features.permute(0, 3, 1, 2),
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
            map_encoding, attention = self(map_scans, proprioception, role=role)
            return map_encoding.flatten(start_dim=1), attention, proprioception
        return self.encode(map_scans, proprioception, role=role).flatten(start_dim=1), None, proprioception

    def forward(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        role: str = "actor",
    ) -> tuple[Tensor, Tensor]:
        map_encoding, attention_weights = self._attend(
            map_scans, proprioception, valid_mask, role, return_attention=True
        )
        assert attention_weights is not None
        return map_encoding, attention_weights

    def encode(
        self,
        map_scans: Tensor,
        proprioception: Tensor,
        valid_mask: Tensor | None = None,
        role: str = "actor",
    ) -> Tensor:
        """Encode the map without materializing per-head attention weights."""
        map_encoding, _ = self._attend(
            map_scans, proprioception, valid_mask, role, return_attention=False
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
        if return_attention:
            map_encoding, attention = self.map_encoder(map_scans, proprioception, valid_mask)
        else:
            map_encoding = self.map_encoder.encode(map_scans, proprioception, valid_mask)
            attention = None
        policy_input = torch.cat((map_encoding.flatten(start_dim=1), proprioception), dim=-1)
        actions = self.policy_mlp(policy_input)
        if return_attention:
            assert attention is not None
            return {"actions": actions, "attention_weights": attention}
        return {"actions": actions, "map_encoding": map_encoding}


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
