"""Temporal scan differences and single-frame spatial terrain edges."""

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
    """Compute normalized per-ray height differences between consecutive scans.

    Previous and current hits are both transformed into the current sensor's
    translated yaw frame. The previous height field is then interpolated by XY
    position, rather than paired by ray index, so robot translation and yaw do
    not create a false temporal difference on static terrain.

    ``update`` returns ``(gradient, valid_mask)`` tensors shaped ``(N, R)``.
    Each valid height difference is divided by the largest absolute valid
    difference in its environment, keeping the output approximately in
    ``[-1, 1]``. Timestamps are used only to detect new sensor scans.
    The first sample after construction or reset is a valid zero field.
    """

    def __init__(
        self,
        max_correspondence_distance: float = 0.2,
        num_neighbors: int = 4,
        # normalize_by_dt: bool = True,
    ):
        if max_correspondence_distance <= 0.0:
            raise ValueError("max_correspondence_distance must be positive.")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be at least 1.")
        self.max_correspondence_distance = float(max_correspondence_distance)
        self.num_neighbors = int(num_neighbors)
        # Kept for compatibility with older configs; normalization is no
        # longer time-based and this flag is intentionally ignored.
        # del normalize_by_dt
        self._previous_hits_w: torch.Tensor | None = None
        self._previous_timestamps: torch.Tensor | None = None
        self._has_history: torch.Tensor | None = None
        self._normalized_difference: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget all scan history, for example when changing environments."""

        self._previous_hits_w = None
        self._previous_timestamps = None
        self._has_history = None
        self._normalized_difference = None
        self._valid_mask = None

    def update(
        self,
        sensor: Any,
        dt: float | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update from a RayCaster and return normalized differences and validity."""

        # ``dt`` used to scale the output. It is accepted but intentionally
        # unused because the output is now normalized per environment.
        del dt

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
        if (
            self._normalized_difference is None
            or self._normalized_difference.shape != shape
            or self._normalized_difference.device != hits_w.device
        ):
            self.reset()
            self._normalized_difference = torch.zeros(shape, device=hits_w.device, dtype=hits_w.dtype)
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

        assert self._normalized_difference is not None and self._valid_mask is not None
        new_ids = torch.nonzero(fresh, as_tuple=False).squeeze(-1)
        if new_ids.numel() == 0:
            return self._normalized_difference, self._valid_mask

        history_ids = torch.empty(0, device=hits_w.device, dtype=torch.long)
        if self._previous_hits_w is not None:
            history_ids = new_ids[self._has_history[new_ids]]

        history_for_new = torch.zeros(hits_w.shape[0], device=hits_w.device, dtype=torch.bool)
        history_for_new[history_ids] = True
        first_mask = ~history_for_new[new_ids]
        first_ids = new_ids[first_mask]
        if history_ids.numel() > 0:
            current_yaw = _yaw_from_quat(quat_w[history_ids])
            current_points = _points_in_yaw_frame(hits_w[history_ids], pos_w[history_ids], current_yaw)
            previous_points = _points_in_yaw_frame(self._previous_hits_w[history_ids], pos_w[history_ids], current_yaw)
            previous_heights, matched = _interpolate_previous_heights(
                current_points,
                previous_points,
                self.max_correspondence_distance,
                self.num_neighbors,
            )
            valid = matched
            difference = current_points[..., 2] - previous_heights
            valid_difference = torch.where(valid, difference, torch.zeros_like(difference))
            max_difference = valid_difference.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
            self._normalized_difference[history_ids] = valid_difference / max_difference
            self._valid_mask[history_ids] = valid

        if first_ids.numel() > 0:
            self._normalized_difference[first_ids] = 0.0
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

        return self._normalized_difference, self._valid_mask



def height_scan_temporal_gradient(
    env: Any,
    sensor_cfg: Any,
    # normalize_by_dt: bool = True,
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
    key = (sensor_cfg.name, float(max_correspondence_distance), int(num_neighbors))
    calculator = state.get(key)
    if calculator is None:
        calculator = HeightScanGradient(
            max_correspondence_distance=max_correspondence_distance,
            num_neighbors=num_neighbors,
            # normalize_by_dt=normalize_by_dt,
        )
        state[key] = calculator
    reset_mask = getattr(env, "episode_length_buf", None)
    if reset_mask is not None:
        reset_mask = reset_mask == 0
    # 时间差分：将上一帧按空间位置对齐到当前帧，再比较对应位置的高度。
    gradient, valid_mask = calculator.update(sensor, dt=getattr(env, "step_dt", None), reset_mask=reset_mask)
    return torch.where(valid_mask, gradient, torch.zeros_like(gradient)).reshape(gradient.shape[0], -1)



def height_map_xy_differences(
    height_map: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute signed neighboring height differences within one frame.

    ``height_map`` has shape ``(N, nx, ny)``: axis 1 increases along x and
    axis 2 along y. Return ``(dx, dy)`` with the same shape. These are height
    differences, not derivatives divided by grid spacing or elapsed time.
    A difference is zero if either of its two samples is invalid.
    """

    if height_map.ndim != 3 or height_map.shape[1] < 1 or height_map.shape[2] < 1:
        raise ValueError(f"Expected a height map shaped (N, nx, ny), got {tuple(height_map.shape)}.")
    valid = torch.isfinite(height_map)
    if valid_mask is not None:
        if valid_mask.shape != height_map.shape:
            raise ValueError("valid_mask must have the same shape as height_map.")
        valid = valid & valid_mask.to(device=height_map.device, dtype=torch.bool)
    # 先屏蔽无效高度；只有相邻两个点都有效时才计算差分，避免漏射点产生假边缘。
    heights = torch.where(valid, height_map, torch.zeros_like(height_map))
    dx = torch.zeros_like(heights)
    dy = torch.zeros_like(heights)

    # x 方向：dx[i, j] = h[i + 1, j] - h[i, j]。
    # 最后一行使用向后差分；不把地图两端连起来，也不补零制造边界跳变。
    if heights.shape[1] > 1:
        x_pairs = valid[:, 1:, :] & valid[:, :-1, :]
        x_difference = heights[:, 1:, :] - heights[:, :-1, :]
        x_difference = torch.where(x_pairs, x_difference, torch.zeros_like(x_difference))
        dx[:, :-1, :] = x_difference
        dx[:, -1, :] = x_difference[:, -1, :]

    # y 方向：dy[i, j] = h[i, j + 1] - h[i, j]；最后一列同样用向后差分。
    # 某方向只有一个采样点时，没有相邻点，该方向的差分保持为零。
    if heights.shape[2] > 1:
        y_pairs = valid[:, :, 1:] & valid[:, :, :-1]
        y_difference = heights[:, :, 1:] - heights[:, :, :-1]
        y_difference = torch.where(y_pairs, y_difference, torch.zeros_like(y_difference))
        dy[:, :, :-1] = y_difference
        dy[:, :, -1] = y_difference[:, :, -1]

    return dx, dy


def height_scan_spacial_gradient(
    env: Any,
    sensor_cfg: Any,
    grid_shape: tuple[int, int] = (17, 11),
    component: str = "magnitude",
    normalize: bool = True,
    # scale: float = 10.0,
) -> torch.Tensor:
    """Return single-frame spatial height differences as a flat observation.

    ``grid_shape`` is ``(nx, ny)`` for an Isaac Lab GridPattern RayCaster.
    ``component`` selects ``"x"``, ``"y"`` or ``"magnitude"`` (the default
    edge strength ``sqrt(dx**2 + dy**2)``). With normalization, all components
    use the maximum edge strength per environment as a shared denominator.
    The default magnitude output is in ``[0, 10]`` and has shape
    ``(N, nx * ny)``. No previous scan, sensor pose or time step is needed.
    """

    if len(grid_shape) != 2 or any(size < 1 or int(size) != size for size in grid_shape):
        raise ValueError(f"grid_shape must contain two positive integers, got {grid_shape}.")
    if component not in ("x", "y", "magnitude"):
        raise ValueError(f"component must be 'x', 'y' or 'magnitude', got {component!r}.")
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_w = sensor.data.ray_hits_w.detach()
    nx, ny = (int(size) for size in grid_shape)
    if hits_w.ndim != 3 or hits_w.shape[1:] != (nx * ny, 3):
        raise ValueError(f"Expected ray hits shaped (N, {nx * ny}, 3), got {tuple(hits_w.shape)}.")

    pattern_cfg = getattr(getattr(sensor, "cfg", None), "pattern_cfg", None)
    ordering = getattr(pattern_cfg, "ordering", "yx")
    if ordering not in ("yx", "xy"):
        raise ValueError(f"Expected grid ordering 'yx' or 'xy', got {ordering!r}.")
    # 直接读取本帧命中点的世界高度 z，不使用前一帧，也不引入观测中的随机高度噪声。
    # 减去同一个机器人高度或扫描 offset 不会改变相邻高度差，因此无需额外转换 z。
    # Isaac Lab ordering='yx' 时，reshape(nx, ny) 的行对应 x，列对应 y。
    raw_shape = (nx, ny) if ordering == "yx" else (ny, nx)
    heights = hits_w[..., 2].reshape(hits_w.shape[0], *raw_shape)
    valid = torch.isfinite(hits_w).all(dim=-1).reshape(hits_w.shape[0], *raw_shape)
    if ordering == "xy":
        heights = heights.transpose(1, 2)
        valid = valid.transpose(1, 2)
    dx, dy = height_map_xy_differences(heights, valid)

    # 两个方向合成边缘强度：平地为 0，台阶/坑边缘较大，斜坡也会有响应。
    magnitude = torch.hypot(dx, dy)
    gradient = {"x": dx, "y": dy, "magnitude": magnitude}[component]
    if normalize:
        # 各环境独立归一化，不在不同机器人之间取最大值；全零地图仍然输出全零。
        max_magnitude = magnitude.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        gradient = gradient / max_magnitude
    if ordering == "xy":
        # 恢复传感器原来的展平顺序，确保与高程观测中的同一个射线索引对应。
        gradient = gradient.transpose(1, 2)
    return (gradient).reshape(hits_w.shape[0], -1)


# 兼容原来的时间差分名称，防止现有配置在导入时找不到函数。
height_scan_gradient = height_scan_temporal_gradient
# ``spacial`` 保留为现有配置使用的名称，同时提供正确拼写给新配置。
height_scan_spatial_gradient = height_scan_spacial_gradient
