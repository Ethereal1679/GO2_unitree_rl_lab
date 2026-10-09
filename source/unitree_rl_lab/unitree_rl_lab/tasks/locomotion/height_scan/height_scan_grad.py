"""Temporal height-scan differences expressed in the current yaw frame."""

from __future__ import annotations

from typing import Any

import torch


def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw from Isaac Lab's scalar-first (w, x, y, z) quaternions."""

    w, x, y, z = quat_w.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _points_in_yaw_frame(
    points_w: torch.Tensor, frame_pos_w: torch.Tensor, frame_yaw: torch.Tensor
) -> torch.Tensor:
    """Express world points in a translated, yaw-only frame."""

    relative = points_w - frame_pos_w.unsqueeze(1)
    cos_yaw = torch.cos(frame_yaw).unsqueeze(1)
    sin_yaw = torch.sin(frame_yaw).unsqueeze(1)
    x = cos_yaw * relative[..., 0] + sin_yaw * relative[..., 1]
    y = -sin_yaw * relative[..., 0] + cos_yaw * relative[..., 1]
    return torch.stack((x, y, relative[..., 2]), dim=-1)


def _interpolate_previous_heights(
    current_points: torch.Tensor,
    previous_points: torch.Tensor,
    max_distance: float,
    num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate the previous scan onto current XY sample locations."""

    current_valid = torch.isfinite(current_points).all(dim=-1)
    previous_valid = torch.isfinite(previous_points).all(dim=-1)
    distances = torch.cdist(current_points[..., :2], previous_points[..., :2])
    distances = distances.nan_to_num(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    distances.masked_fill_(~previous_valid.unsqueeze(1), float("inf"))
    distances.masked_fill_(~current_valid.unsqueeze(-1), float("inf"))

    count = min(max(1, int(num_neighbors)), previous_points.shape[1])
    nearest_distances, nearest_indices = torch.topk(distances, k=count, dim=-1, largest=False)
    usable = torch.isfinite(nearest_distances) & (nearest_distances <= max_distance)
    weights = torch.where(
        usable,
        nearest_distances.square().clamp_min(1.0e-12).reciprocal(),
        torch.zeros_like(nearest_distances),
    )
    previous_heights = previous_points[..., 2].unsqueeze(1).expand(-1, current_points.shape[1], -1)
    neighbor_heights = torch.gather(previous_heights, dim=-1, index=nearest_indices)
    neighbor_heights = torch.where(usable, neighbor_heights, torch.zeros_like(neighbor_heights))
    weight_sum = weights.sum(dim=-1)
    inverse_distance = (weights * neighbor_heights).sum(dim=-1) / weight_sum.clamp_min(1.0e-12)

    # A local plane fit reproduces gradual terrain slopes through yaw/translation
    # resampling; inverse-distance weighting alone can create spurious changes.
    previous_xy = previous_points[..., :2].unsqueeze(1).expand(-1, current_points.shape[1], -1, -1)
    neighbor_xy = torch.gather(
        previous_xy,
        dim=2,
        index=nearest_indices.unsqueeze(-1).expand(-1, -1, -1, 2),
    )
    neighbor_xy = neighbor_xy.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    current_xy = current_points[..., :2].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    offsets = neighbor_xy - current_xy[..., None, :]
    design = torch.cat((offsets, torch.ones_like(offsets[..., :1])), dim=-1)
    normalized_weights = weights / weight_sum.unsqueeze(-1).clamp_min(1.0e-12)
    weighted_design = design * normalized_weights.sqrt().unsqueeze(-1)
    normal_matrix = weighted_design.transpose(-1, -2) @ weighted_design
    rhs = weighted_design.transpose(-1, -2) @ (neighbor_heights * normalized_weights.sqrt()).unsqueeze(-1)
    coefficients = (torch.linalg.pinv(normal_matrix) @ rhs).squeeze(-1)
    plane_height = coefficients[..., 2]
    plane_valid = usable.sum(dim=-1) >= 3
    exact_match = nearest_distances[..., 0] <= 1.0e-6
    interpolated = torch.where(plane_valid, plane_height, inverse_distance)
    interpolated = torch.where(exact_match, neighbor_heights[..., 0], interpolated)
    matched = current_valid & (weight_sum > 0.0) & torch.isfinite(interpolated)
    return interpolated, matched


class HeightScanGradient:
    """Compute per-ray height change rates after aligning consecutive scans.

    Previous and current hits are both transformed into the current sensor's
    translated yaw frame. The previous height field is then interpolated by XY
    position, rather than paired by ray index, so robot translation and yaw do
    not create a false temporal difference on static terrain.

    ``update`` returns ``(gradient, valid_mask)`` tensors shaped ``(N, R)``.
    The signed gradient is either ``current_height - previous_height`` in m,
    or that difference divided by ``dt`` in m/s when ``normalize_by_dt=True``.
    The first sample after construction or reset is a valid zero field.
    """

    def __init__(
        self,
        max_correspondence_distance: float = 0.2,
        num_neighbors: int = 4,
        normalize_by_dt: bool = True,
    ):
        if max_correspondence_distance <= 0.0:
            raise ValueError("max_correspondence_distance must be positive.")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be at least 1.")
        self.max_correspondence_distance = float(max_correspondence_distance)
        self.num_neighbors = int(num_neighbors)
        self.normalize_by_dt = bool(normalize_by_dt)
        self._previous_hits_w: torch.Tensor | None = None
        self._previous_timestamps: torch.Tensor | None = None
        self._has_history: torch.Tensor | None = None
        self._height_difference: torch.Tensor | None = None
        self._height_rate: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget all scan history, for example when changing environments."""

        self._previous_hits_w = None
        self._previous_timestamps = None
        self._has_history = None
        self._height_difference = None
        self._height_rate = None
        self._valid_mask = None

    def update(
        self,
        sensor: Any,
        dt: float | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update from a RayCaster and return the latest rate field and mask."""

        data = sensor.data
        hits_w = data.ray_hits_w.detach()
        pos_w = data.pos_w.detach()
        quat_w = data.quat_w.detach()
        if hits_w.ndim != 3 or hits_w.shape[-1] != 3:
            raise ValueError(f"Expected ray hits shaped (N, R, 3), got {tuple(hits_w.shape)}")
        if pos_w.shape != (hits_w.shape[0], 3) or quat_w.shape != (hits_w.shape[0], 4):
            raise ValueError("RayCaster poses do not match the number of ray-hit batches.")

        timestamps = getattr(sensor, "_timestamp_last_update", None)
        if timestamps is not None:
            timestamps = timestamps.detach().to(device=hits_w.device, dtype=hits_w.dtype)
            if timestamps.shape != (hits_w.shape[0],):
                timestamps = None

        shape = hits_w.shape[:2]
        if self._height_rate is None or self._height_rate.shape != shape or self._height_rate.device != hits_w.device:
            self.reset()
            self._height_difference = torch.zeros(shape, device=hits_w.device, dtype=hits_w.dtype)
            self._height_rate = torch.zeros(shape, device=hits_w.device, dtype=hits_w.dtype)
            self._valid_mask = torch.zeros(shape, device=hits_w.device, dtype=torch.bool)
            self._has_history = torch.zeros(hits_w.shape[0], device=hits_w.device, dtype=torch.bool)
        assert self._has_history is not None
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=hits_w.device, dtype=torch.bool).reshape(-1)
            if reset_mask.shape != (hits_w.shape[0],):
                raise ValueError(f"reset_mask must have shape ({hits_w.shape[0]},), got {tuple(reset_mask.shape)}")
            self._has_history[reset_mask] = False

        if timestamps is None or self._previous_timestamps is None:
            fresh = torch.ones(hits_w.shape[0], device=hits_w.device, dtype=torch.bool)
        else:
            time_delta = timestamps - self._previous_timestamps
            fresh = time_delta.abs() > 1.0e-6
        if reset_mask is not None:
            fresh |= reset_mask

        assert self._height_rate is not None and self._valid_mask is not None
        new_ids = torch.nonzero(fresh, as_tuple=False).squeeze(-1)
        if new_ids.numel() == 0:
            return self._height_rate, self._valid_mask

        history_ids = torch.empty(0, device=hits_w.device, dtype=torch.long)
        elapsed = torch.empty(0, device=hits_w.device, dtype=hits_w.dtype)
        if self._previous_hits_w is not None:
            if timestamps is None or self._previous_timestamps is None:
                fallback_dt = dt if dt is not None else getattr(sensor.cfg, "update_period", 0.0)
                if fallback_dt is None or float(fallback_dt) <= 0.0:
                    raise ValueError("A positive dt or a timestamped RayCaster is required for height rates.")
                history_ids = new_ids[self._has_history[new_ids]]
                elapsed = torch.full(
                    (history_ids.numel(),), float(fallback_dt), device=hits_w.device, dtype=hits_w.dtype
                )
            else:
                time_delta = timestamps - self._previous_timestamps
                history_mask = fresh & self._has_history & (time_delta > 1.0e-6)
                history_ids = torch.nonzero(history_mask, as_tuple=False).squeeze(-1)
                elapsed = time_delta[history_ids]

        history_for_new = torch.zeros(hits_w.shape[0], device=hits_w.device, dtype=torch.bool)
        history_for_new[history_ids] = True
        first_mask = ~history_for_new[new_ids]
        first_ids = new_ids[first_mask]
        if history_ids.numel() > 0:
            current_yaw = _yaw_from_quat(quat_w[history_ids])
            current_points = _points_in_yaw_frame(hits_w[history_ids], pos_w[history_ids], current_yaw)
            previous_points = _points_in_yaw_frame(
                self._previous_hits_w[history_ids], pos_w[history_ids], current_yaw
            )
            previous_heights, matched = _interpolate_previous_heights(
                current_points,
                previous_points,
                self.max_correspondence_distance,
                self.num_neighbors,
            )
            valid = matched & (elapsed.unsqueeze(-1) > 0.0)
            difference = current_points[..., 2] - previous_heights
            rate = difference / elapsed.unsqueeze(-1).clamp_min(1.0e-6)
            assert self._height_difference is not None
            self._height_difference[history_ids] = torch.where(valid, difference, torch.zeros_like(difference))
            self._height_rate[history_ids] = torch.where(valid, rate, torch.zeros_like(rate))
            self._valid_mask[history_ids] = valid

        if first_ids.numel() > 0:
            assert self._height_difference is not None
            self._height_difference[first_ids] = 0.0
            self._height_rate[first_ids] = 0.0
            self._valid_mask[first_ids] = torch.isfinite(hits_w[first_ids]).all(dim=-1)

        if self._previous_hits_w is None:
            self._previous_hits_w = hits_w.clone()
            self._previous_timestamps = timestamps.clone() if timestamps is not None else None
        else:
            self._previous_hits_w[new_ids] = hits_w[new_ids]
            if timestamps is not None:
                if self._previous_timestamps is None:
                    self._previous_timestamps = timestamps.clone()
                else:
                    self._previous_timestamps[new_ids] = timestamps[new_ids]
        self._has_history[new_ids] = True

        assert self._height_difference is not None
        gradient = self._height_rate if self.normalize_by_dt else self._height_difference
        return gradient, self._valid_mask


def height_scan_gradient(
    env: Any,
    sensor_cfg: Any,
    normalize_by_dt: bool = True,
    max_correspondence_distance: float = 0.2,
    num_neighbors: int = 4,
) -> torch.Tensor:
    """Return the aligned temporal height gradient as a flattened observation.

    The state is kept on the environment, so this function can be used as an
    Isaac Lab observation term without losing the previous scan between calls.
    Invalid rays are returned as zero and can be masked by a separate sensor
    validity observation when needed.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    state = getattr(env, "_height_scan_gradient_states", None)
    if state is None:
        state = {}
        setattr(env, "_height_scan_gradient_states", state)
    key = (sensor_cfg.name, bool(normalize_by_dt), float(max_correspondence_distance), int(num_neighbors))
    calculator = state.get(key)
    if calculator is None:
        calculator = HeightScanGradient(
            max_correspondence_distance=max_correspondence_distance,
            num_neighbors=num_neighbors,
            normalize_by_dt=normalize_by_dt,
        )
        state[key] = calculator
    reset_mask = getattr(env, "episode_length_buf", None)
    if reset_mask is not None:
        reset_mask = reset_mask == 0
    gradient, valid_mask = calculator.update(
        sensor, dt=getattr(env, "step_dt", None), reset_mask=reset_mask
    )
    return torch.where(valid_mask, gradient, torch.zeros_like(gradient)).reshape(gradient.shape[0], -1)
