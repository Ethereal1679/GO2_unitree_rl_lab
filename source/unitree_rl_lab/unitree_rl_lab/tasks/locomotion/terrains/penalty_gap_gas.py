"""Visualization-only gap volumes and penetration helpers.

The gas volumes in this module are intentionally represented by Isaac Lab
visualization markers. They have no collision properties, so enabling them
cannot change contacts, ray-casts, or the training dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase


@dataclass(frozen=True)
class GapGasBoxes:
    """Axis-aligned transparent volumes that occupy terrain gaps.

    ``centers`` and ``sizes`` are expressed in world coordinates.  The arrays
    have shape ``(N, 3)`` and can be passed directly to
    :class:`isaaclab.markers.VisualizationMarkers`.
    """

    centers: np.ndarray
    sizes: np.ndarray

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[1] != 3:
            raise ValueError(f"centers must have shape (N, 3), got {self.centers.shape}")
        if self.sizes.shape != self.centers.shape:
            raise ValueError(f"sizes must match centers, got {self.sizes.shape} and {self.centers.shape}")


def _resolve_depth(depth: float | tuple[float, float], difficulty: float) -> float:
    if isinstance(depth, (tuple, list)):
        if len(depth) != 2:
            raise ValueError(f"depth range must contain exactly two values, got {depth}")
        return float(depth[0]) + float(difficulty) * (float(depth[1]) - float(depth[0]))
    return float(depth)


def _height_field_inner_size(cfg: Any) -> tuple[int, int, float, float]:
    """Match ``height_field_to_mesh``'s border and interior dimensions."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
    border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
    inner_width = width_pixels - 2 * border_pixels
    inner_length = length_pixels - 2 * border_pixels
    if inner_width <= 0 or inner_length <= 0:
        raise ValueError("terrain border leaves no room for a gap overlay")
    return (
        inner_width,
        inner_length,
        border_pixels * cfg.horizontal_scale,
        border_pixels * cfg.horizontal_scale,
    )


def _empty_boxes() -> GapGasBoxes:
    return GapGasBoxes(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32))


def _append_square_annulus(
    centers: list[tuple[float, float, float]],
    sizes: list[tuple[float, float, float]],
    inner_half: float,
    outer_half: float,
    depth: float,
    center_xy: tuple[float, float],
    center_z: float,
) -> None:
    """Append four cuboids forming a square annulus."""
    ring_width = outer_half - inner_half
    if ring_width <= 0.0 or depth <= 0.0:
        return
    center_x, center_y = center_xy
    z = center_z - 0.5 * depth
    y_offset = inner_half + 0.5 * ring_width
    for y in (center_y - y_offset, center_y + y_offset):
        centers.append((center_x, y, z))
        sizes.append((2.0 * outer_half, ring_width, depth))
    x_offset = inner_half + 0.5 * ring_width
    for x in (center_x - x_offset, center_x + x_offset):
        centers.append((x, center_y, z))
        sizes.append((ring_width, 2.0 * inner_half, depth))


def concentric_gap_gas_boxes(
    difficulty: float,
    cfg: Any,
    tile_center_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GapGasBoxes:
    """Create green overlay volumes for every ring in ``hf_gaps``.

    ``tile_center_w`` is the world-space center of the terrain-generator tile.
    The overlay follows the same border, gap width, and ground width logic as
    ``concentric_gap_terrain``.
    """
    inner_width, inner_length, _, _ = _height_field_inner_size(cfg)
    horizontal_scale = float(cfg.horizontal_scale)
    gap_width = int(
        (cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0]))
        / horizontal_scale
    )
    ground_width = int(
        (cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0]))
        / horizontal_scale
    )
    gap_width = max(gap_width, 1)
    ground_width = max(ground_width, 1)
    platform_width = int(cfg.platform_width / horizontal_scale)
    gap_depth = abs(_resolve_depth(cfg.gap_depth, difficulty))
    if gap_depth <= 0.0:
        return _empty_boxes()

    current_width = min(inner_width, inner_length)
    current_half = 0.5 * current_width * horizontal_scale
    platform_width = max(platform_width, 1) * horizontal_scale
    centers: list[tuple[float, float, float]] = []
    sizes: list[tuple[float, float, float]] = []
    is_gap = True
    while current_width > int(platform_width / horizontal_scale):
        thickness_pixels = gap_width if is_gap else ground_width
        thickness = min(thickness_pixels * horizontal_scale, 2.0 * current_half)
        if is_gap:
            _append_square_annulus(
                centers,
                sizes,
                max(0.0, current_half - thickness),
                current_half,
                gap_depth,
                (tile_center_w[0], tile_center_w[1]),
                tile_center_w[2],
            )
        current_half -= thickness
        current_width -= thickness_pixels * 2
        is_gap = not is_gap

    if not centers:
        return _empty_boxes()
    return GapGasBoxes(np.asarray(centers, dtype=np.float32), np.asarray(sizes, dtype=np.float32))


def gap_mask_from_height_field(height_field: np.ndarray, holes_depth: float, vertical_scale: float) -> np.ndarray:
    """Return the exact hole mask from a stepping-stones height field."""
    depth_units = int(round(float(holes_depth) / float(vertical_scale)))
    return np.asarray(height_field) == depth_units


def stepping_stones_gap_gas_boxes(
    gap_mask: np.ndarray,
    horizontal_scale: float,
    depth: float,
    tile_center_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    top_z: float = 0.0,
) -> GapGasBoxes:
    """Create boxes for every hole cell in an exact stepping-stones mask.

    Pass the raw height-field mask used to create the terrain.  The mask must
    be ``True`` only in holes; this preserves the random stone layout exactly.
    Adjacent cells in each row are merged into one cuboid to keep marker counts
    manageable.
    """
    mask = np.asarray(gap_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"gap_mask must have shape (W, L), got {mask.shape}")
    depth = abs(float(depth))
    if depth <= 0.0:
        return _empty_boxes()
    width, length = mask.shape
    centers: list[tuple[float, float, float]] = []
    sizes: list[tuple[float, float, float]] = []
    x0 = tile_center_w[0] - 0.5 * width * horizontal_scale
    y0 = tile_center_w[1] - 0.5 * length * horizontal_scale
    z = top_z + tile_center_w[2] - 0.5 * depth
    for ix in range(width):
        iy = 0
        while iy < length:
            if not mask[ix, iy]:
                iy += 1
                continue
            end = iy + 1
            while end < length and mask[ix, end]:
                end += 1
            centers.append(
                (x0 + (ix + 0.5) * horizontal_scale, y0 + (iy + end) * 0.5 * horizontal_scale, z)
            )
            sizes.append(((horizontal_scale), (end - iy) * horizontal_scale, depth))
            iy = end
    if not centers:
        return _empty_boxes()
    return GapGasBoxes(np.asarray(centers, dtype=np.float32), np.asarray(sizes, dtype=np.float32))


def stepping_stones_gap_gas_boxes_from_height_field(
    height_field: np.ndarray,
    cfg: Any,
    difficulty: float,
    tile_center_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    top_z: float = 0.0,
) -> GapGasBoxes:
    """Build an exact stepping-stones overlay from the generated height field."""
    depth = _resolve_depth(cfg.holes_depth, difficulty)
    mask = gap_mask_from_height_field(height_field, depth, cfg.vertical_scale)
    return stepping_stones_gap_gas_boxes(
        mask,
        cfg.horizontal_scale,
        depth,
        tile_center_w=tile_center_w,
        top_z=top_z,
    )


def make_gap_gas_marker_cfg(prim_path: str = "/Visuals/GapGas", opacity: float = 0.22) -> Any:
    """Build a non-colliding green transparent marker configuration."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg

    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "gap_gas": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    opacity=float(opacity),
                ),
            )
        },
    )


class GapGasVisualizer:
    """Optional visualizer for gap volumes.

    The marker is created only when ``enabled=True``.  It is a display-only
    PointInstancer and therefore does not participate in physics collisions.
    """

    def __init__(self, enabled: bool = False, prim_path: str = "/Visuals/GapGas", opacity: float = 0.22):
        self.enabled = bool(enabled)
        self._prim_path = prim_path
        self._opacity = float(opacity)
        self._marker = None
        if self.enabled:
            from isaaclab.markers import VisualizationMarkers

            self._marker = VisualizationMarkers(make_gap_gas_marker_cfg(prim_path, opacity))
            self._marker.set_visibility(True)

    @property
    def marker(self) -> Any:
        return self._marker

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and self._marker is None:
            from isaaclab.markers import VisualizationMarkers

            self._marker = VisualizationMarkers(make_gap_gas_marker_cfg(self._prim_path, self._opacity))
        self.enabled = enabled
        if self._marker is not None:
            self._marker.set_visibility(enabled)

    def visualize(self, boxes: GapGasBoxes) -> None:
        if not self.enabled or self._marker is None or boxes.centers.shape[0] == 0:
            return
        self._marker.visualize(translations=boxes.centers, scales=boxes.sizes)

# 区域下扎深度
def gap_penetration_depth(
    points_w: Any,
    gas_centers_w: Any,
    gas_sizes: Any,
) -> Any:
    """Return per-point penetration depth for a future reward term.

    ``points_w`` is shaped ``(N, P, 3)``.  The returned tensor is ``(N, P)``;
    points outside all gas boxes have zero depth.  This function uses torch
    lazily so importing this visualization module does not require torch.
    """
    import torch

    points = torch.as_tensor(points_w)
    centers = torch.as_tensor(gas_centers_w, device=points.device, dtype=points.dtype)
    sizes = torch.as_tensor(gas_sizes, device=points.device, dtype=points.dtype)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points_w must have shape (N, P, 3), got {tuple(points.shape)}")
    if centers.ndim != 2 or centers.shape[-1] != 3 or sizes.shape != centers.shape:
        raise ValueError("gas_centers_w and gas_sizes must both have shape (G, 3)")
    if centers.shape[0] == 0:
        return torch.zeros(points.shape[:-1], device=points.device, dtype=points.dtype)
    lower = centers - 0.5 * sizes
    upper = centers + 0.5 * sizes
    inside_xy = (
        (points[..., None, 0] >= lower[None, None, :, 0])
        & (points[..., None, 0] <= upper[None, None, :, 0])
        & (points[..., None, 1] >= lower[None, None, :, 1])
        & (points[..., None, 1] <= upper[None, None, :, 1])
    )
    inside_z = (points[..., None, 2] >= lower[None, None, :, 2]) & (
        points[..., None, 2] <= upper[None, None, :, 2]
    )
    depth = (upper[None, None, :, 2] - points[..., None, 2]).clamp_min(0.0)
    return torch.where(inside_xy & inside_z, depth, torch.zeros_like(depth)).amax(dim=-1)


def ray_gap_gas_boxes(
    ray_hits_w: Any,
    cell_size: float = 0.1,
    min_depth: float = 0.05,
    max_boxes: int | None = None,
) -> GapGasBoxes:
    """Create display-only boxes for the currently scanned low gap points."""
    hits = torch.as_tensor(ray_hits_w).detach().cpu()
    if hits.ndim != 3 or hits.shape[-1] != 3:
        raise ValueError(f"ray_hits_w must have shape (N, R, 3), got {tuple(hits.shape)}")
    valid = torch.isfinite(hits).all(dim=-1)
    support = torch.where(valid, hits[..., 2], torch.full_like(hits[..., 2], -torch.inf)).amax(dim=1)
    depth = support[:, None] - hits[..., 2]
    mask = valid & (depth >= min_depth)
    env_ids, ray_ids = torch.nonzero(mask, as_tuple=True)
    if env_ids.numel() == 0:
        return _empty_boxes()
    # Do not use one global cap: it would keep only early envs and make later
    # environments appear to have no gas. A cap, when explicitly requested,
    # is applied only to the flattened debug marker list.
    if max_boxes is not None and env_ids.numel() > max_boxes:
        env_ids, ray_ids = env_ids[:max_boxes], ray_ids[:max_boxes]
    selected = hits[env_ids, ray_ids]
    selected_depth = depth[env_ids, ray_ids].clamp_min(0.0)
    centers = selected.clone()
    centers[:, 2] += 0.5 * selected_depth
    sizes = torch.ones_like(centers) * float(cell_size)
    sizes[:, 2] = selected_depth
    return GapGasBoxes(centers.numpy().astype(np.float32), sizes.numpy().astype(np.float32))


class GapPenetrationPenalty(ManagerTermBase):
    """Penalize body parts that enter a scanned gap for a sustained duration."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.inside_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.inside_time.zero_()
        else:
            self.inside_time[env_ids] = 0.0

    def __call__(
        self,
        env,
        sensor_cfg,
        asset_cfg,
        support_radius: float = 0.35,
        min_depth: float = 0.05,
        depth_scale: float = 1.0,
        duration_scale: float = 0.5,
        max_depth: float = 2.5,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w
        # Use each link's center of mass as the representative body point.
        # This is still a point approximation, not the full collision shape.
        body = asset.data.body_com_pos_w[:, asset_cfg.body_ids]
        valid = torch.isfinite(hits).all(dim=-1)
        # -- scan for the nearest support surface below each body point, ignoring gaps
        distance_sq = torch.sum((body[:, :, None, :2] - hits[:, None, :, :2]) ** 2, dim=-1)
        nearby = valid[:, None, :] & (distance_sq <= support_radius**2)
        support = torch.where(nearby, hits[:, None, :, 2], torch.full_like(hits[:, None, :, 2], -torch.inf)).amax(-1)
        support = torch.where(nearby.any(-1), support, body[..., 2])
        depth = (support - body[..., 2] - min_depth).clamp(0.0, max_depth).amax(1)
        inside = depth > 0.0
        # -- accumulate time spent inside the gap volume, reset when outside
        self.inside_time = torch.where(inside, self.inside_time + env.step_dt, torch.zeros_like(self.inside_time))
        return (depth / max(depth_scale, 1.0e-6)).clamp(0.0, 1.0) * (self.inside_time / max(duration_scale, 1.0e-6)).clamp(0.0, 1.0)
