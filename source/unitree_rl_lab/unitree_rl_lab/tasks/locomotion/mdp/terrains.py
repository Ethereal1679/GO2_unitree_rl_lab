from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen


def _make_square_annulus(
    inner_half_width: float,
    outer_half_width: float,
    height: float,
    center_xy: tuple[float, float],
) -> list[trimesh.Trimesh]:
    """Create four boxes whose top faces form a square annulus at ``z=0``."""
    ring_width = outer_half_width - inner_half_width
    if ring_width <= 0.0:
        return []

    center_x, center_y = center_xy
    center_z = -0.5 * height
    meshes = []

    # North and south strips span the full outer width.
    north_south_dims = (2.0 * outer_half_width, ring_width, height)
    y_offset = inner_half_width + 0.5 * ring_width
    for y in (center_y - y_offset, center_y + y_offset):
        meshes.append(
            trimesh.creation.box(
                north_south_dims,
                trimesh.transformations.translation_matrix((center_x, y, center_z)),
            )
        )

    # East and west strips fill the space between the north/south strips.
    east_west_dims = (ring_width, 2.0 * inner_half_width, height)
    x_offset = inner_half_width + 0.5 * ring_width
    for x in (center_x - x_offset, center_x + x_offset):
        meshes.append(
            trimesh.creation.box(
                east_west_dims,
                trimesh.transformations.translation_matrix((x, center_y, center_z)),
            )
        )

    return meshes


def courage_gaps_terrain(
    difficulty: float, cfg: CourageGapsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate concentric square trenches around a safe center platform.

    The gaps are trenches with a physical floor instead of empty holes. This keeps
    downward height-scanner rays finite even when they fall inside a gap.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_spacing = cfg.gap_spacing_range[0] + difficulty * (
        cfg.gap_spacing_range[1] - cfg.gap_spacing_range[0]
    )
    gap_depth = cfg.gap_depth_range[0] + difficulty * (cfg.gap_depth_range[1] - cfg.gap_depth_range[0])

    if cfg.num_gaps < 1:
        raise ValueError(f"num_gaps must be at least 1, got {cfg.num_gaps}")
    if cfg.platform_width <= 0.0:
        raise ValueError(f"platform_width must be positive, got {cfg.platform_width}")
    if gap_width <= 0.0 or gap_spacing <= 0.0 or gap_depth <= 0.0 or cfg.floor_thickness <= 0.0:
        raise ValueError("gap width, gap spacing, gap depth, and floor thickness must all be positive")
    if not np.isclose(cfg.size[0], cfg.size[1]):
        raise ValueError(f"courage_gaps requires a square terrain, got size={cfg.size}")

    terrain_half_width = 0.5 * cfg.size[0]
    platform_half_width = 0.5 * cfg.platform_width
    last_gap_outer_edge = (
        platform_half_width + cfg.num_gaps * gap_width + (cfg.num_gaps - 1) * gap_spacing
    )
    if last_gap_outer_edge >= terrain_half_width:
        raise ValueError(
            "courage_gaps does not fit inside the terrain: "
            f"last gap ends at {last_gap_outer_edge:.3f} m but terrain half-width is {terrain_half_width:.3f} m"
        )

    center_xy = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    meshes: list[trimesh.Trimesh] = []

    # A floor at z=-gap_depth covers the complete tile, so the gaps never become voids.
    floor_dims = (cfg.size[0], cfg.size[1], cfg.floor_thickness)
    floor_center = (center_xy[0], center_xy[1], -gap_depth - 0.5 * cfg.floor_thickness)
    meshes.append(
        trimesh.creation.box(floor_dims, trimesh.transformations.translation_matrix(floor_center))
    )

    # The center platform top is z=0 and is also the robot spawn origin.
    platform_dims = (cfg.platform_width, cfg.platform_width, gap_depth)
    platform_center = (center_xy[0], center_xy[1], -0.5 * gap_depth)
    meshes.append(
        trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center))
    )

    # Each gap is followed by a landing ring. The final landing ring extends to
    # the tile boundary so the robot can leave the obstacle after the last jump.
    gap_inner_edge = platform_half_width
    for gap_index in range(cfg.num_gaps):
        gap_outer_edge = gap_inner_edge + gap_width
        if gap_index == cfg.num_gaps - 1:
            landing_outer_edge = terrain_half_width
        else:
            landing_outer_edge = gap_outer_edge + gap_spacing
        meshes.extend(_make_square_annulus(gap_outer_edge, landing_outer_edge, gap_depth, center_xy))
        gap_inner_edge = landing_outer_edge

    origin = np.array([center_xy[0], center_xy[1], 0.0])
    return meshes, origin


@configclass
class CourageGapsTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    """Concentric, curriculum-controlled trenches around a center platform."""

    function = courage_gaps_terrain

    platform_width: float = 2.0
    """Width of the square center platform where the robot starts (m)."""

    num_gaps: int = 3
    """Number of concentric gap rings."""

    gap_width_range: tuple[float, float] = (0.12, 0.45)
    """Gap width at difficulty 0 and 1 respectively (m)."""

    gap_spacing_range: tuple[float, float] = (0.80, 0.45)
    """Landing-ring width at difficulty 0 and 1 respectively (m)."""

    gap_depth_range: tuple[float, float] = (0.08, 0.55)
    """Gap depth at difficulty 0 and 1 respectively (m)."""

    floor_thickness: float = 0.10
    """Thickness of the physical floor below all gaps (m)."""
