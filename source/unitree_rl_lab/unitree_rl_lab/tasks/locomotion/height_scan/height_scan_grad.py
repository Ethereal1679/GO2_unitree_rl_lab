"""Temporal differences between actor-frame height scans."""

from __future__ import annotations

from typing import Any

import torch

from .height_scan_points import ray_hits_in_base_frame


class HeightScanGradient:
    """Compute differences between consecutive actor-frame height maps.

    Both frames use exactly the same ``ray_hits_b`` preprocessing and ray order
    as the actor's raw height-map channel. Corresponding Z values are subtracted
    directly, so this channel is the temporal difference of its actor-frame
    source map. Independent raw-channel observation noise is not added here.

    ``update`` returns ``(gradient, valid_mask)`` tensors shaped ``(N, R)``.
    The signed gradient is ``current_height_b - previous_height_b`` in metres.
    It is intentionally not divided by the simulation time step. The first
    sample after construction or an episode reset is a valid zero field and
    becomes the new history for that environment.
    """

    def __init__(self):
        self._previous_heights_b: torch.Tensor | None = None
        self._previous_timestamps: torch.Tensor | None = None
        self._has_history: torch.Tensor | None = None
        self._height_difference: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget all scan history, for example when changing environments."""

        self._previous_heights_b = None
        self._previous_timestamps = None
        self._has_history = None
        self._height_difference = None
        self._valid_mask = None

    def update(
        self,
        sensor: Any,
        asset: Any,
        grid_shape: tuple[int, int] = (17, 11),
        offset: float = 0.5,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update from actor-frame RayCaster points and return difference and mask."""
        # 统一获得
        hits_b = ray_hits_in_base_frame(sensor, asset, grid_shape, invalid_height=-1.2, offset=offset)
        heights_b = hits_b[..., 2].detach()

        timestamps = getattr(sensor, "_timestamp_last_update", None)
        if timestamps is not None:
            timestamps = timestamps.detach().to(device=heights_b.device, dtype=heights_b.dtype)
            if timestamps.shape != (heights_b.shape[0],):
                timestamps = None

        shape = heights_b.shape
        if (
            self._height_difference is None
            or self._height_difference.shape != shape
            or self._height_difference.device != heights_b.device
        ):
            self.reset()
            self._height_difference = torch.zeros(shape, device=heights_b.device, dtype=heights_b.dtype)
            self._valid_mask = torch.zeros(shape, device=heights_b.device, dtype=torch.bool)
            self._has_history = torch.zeros(heights_b.shape[0], device=heights_b.device, dtype=torch.bool)
        assert self._has_history is not None
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=heights_b.device, dtype=torch.bool).reshape(-1)
            if reset_mask.shape != (heights_b.shape[0],):
                raise ValueError(
                    f"reset_mask must have shape ({heights_b.shape[0]},), got {tuple(reset_mask.shape)}"
                )
            # Never compare a reset environment's new terrain against the last
            # scan from its previous episode. The new scan is initialized below
            # as a zero-difference first frame and then stored as fresh history.
            self._has_history[reset_mask] = False

        if timestamps is None or self._previous_timestamps is None:
            fresh = torch.ones(heights_b.shape[0], device=heights_b.device, dtype=torch.bool)
        else:
            time_delta = timestamps - self._previous_timestamps
            fresh = time_delta.abs() > 1.0e-6
        if reset_mask is not None:
            fresh |= reset_mask

        assert self._height_difference is not None and self._valid_mask is not None
        new_ids = torch.nonzero(fresh, as_tuple=False).squeeze(-1)
        if new_ids.numel() == 0:
            return self._height_difference, self._valid_mask

        history_ids = torch.empty(0, device=heights_b.device, dtype=torch.long)
        if self._previous_heights_b is not None:
            history_ids = new_ids[self._has_history[new_ids]]

        history_for_new = torch.zeros(heights_b.shape[0], device=heights_b.device, dtype=torch.bool)
        history_for_new[history_ids] = True
        first_mask = ~history_for_new[new_ids]
        first_ids = new_ids[first_mask]
        if history_ids.numel() > 0:
            difference = heights_b[history_ids] - self._previous_heights_b[history_ids]
            self._height_difference[history_ids] = difference
            self._valid_mask[history_ids] = True

        if first_ids.numel() > 0:
            self._height_difference[first_ids] = 0.0
            self._valid_mask[first_ids] = torch.isfinite(heights_b[first_ids])

        if self._previous_heights_b is None:
            self._previous_heights_b = heights_b.clone()
            self._previous_timestamps = timestamps.clone() if timestamps is not None else None
        else:
            self._previous_heights_b[new_ids] = heights_b[new_ids]
            if timestamps is not None:
                if self._previous_timestamps is None:
                    self._previous_timestamps = timestamps.clone()
                else:
                    self._previous_timestamps[new_ids] = timestamps[new_ids]
        self._has_history[new_ids] = True

        return self._height_difference, self._valid_mask


def height_scan_gradient(
    env: Any,
    sensor_cfg: Any,
    asset_cfg: Any,
    grid_shape: tuple[int, int] = (17, 11),
    offset: float = 0.5,
) -> torch.Tensor:
    """Return the actor-frame temporal height difference as a flat observation.

    The state is kept on the environment, so this function can be used as an
    Isaac Lab observation term without losing the previous scan between calls.
    Invalid ray heights use the same ``-1.2`` value as the raw actor channel.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    state = getattr(env, "_height_scan_gradient_states", None)
    if state is None:
        state = {}
        setattr(env, "_height_scan_gradient_states", state)
    key = (
        sensor_cfg.name,
        asset_cfg.name,
        int(grid_shape[0]),
        int(grid_shape[1]),
        float(offset),
    )
    calculator = state.get(key)
    if calculator is None:
        calculator = HeightScanGradient()
        state[key] = calculator
    reset_mask = getattr(env, "episode_length_buf", None)
    if reset_mask is not None:
        reset_mask = reset_mask == 0
    gradient, valid_mask = calculator.update(
        sensor, asset, grid_shape=grid_shape, offset=offset, reset_mask=reset_mask
    )
    return torch.where(valid_mask, gradient, torch.zeros_like(gradient)).reshape(gradient.shape[0], -1)
