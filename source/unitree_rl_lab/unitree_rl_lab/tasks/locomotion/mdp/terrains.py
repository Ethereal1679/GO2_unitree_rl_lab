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


def modular_pillar_terrain(
    difficulty: float, cfg: ModularPillarTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one excavated pillar field for a terrain-generator tile.

    The complete terrain tile is one square pit.  Its center platform rises
    from the pit floor to ``z=0``; random pillars also rise from the floor but
    never above the ground plane.  TerrainGenerator arranges these tiles, so
    this function intentionally has no internal zone layout.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    if cfg.pillar_size <= 0.0:
        raise ValueError("pillar_size must be positive")
    if (
        cfg.dense_grid_spacing <= cfg.pillar_size
        or cfg.grid_spacing <= cfg.pillar_size
        or cfg.dense_grid_spacing > cfg.grid_spacing
    ):
        raise ValueError(
            "dense_grid_spacing and grid_spacing must exceed pillar_size, "
            "with dense_grid_spacing <= grid_spacing"
        )
    if not np.isclose(cfg.platform_height, 0.0):
        raise ValueError(
            "platform_height must be 0.0 so the center platform stays level with the ground"
        )
    if (
        cfg.pillar_height_range[0] <= 0.0
        or cfg.pillar_height_range[1] < cfg.pillar_height_range[0]
    ):
        raise ValueError("pillar_height_range must be positive and ordered")
    if cfg.floor_thickness <= 0.0:
        raise ValueError("floor_thickness must be positive")
    if not np.isclose(cfg.size[0], cfg.size[1]):
        raise ValueError(f"modular_pillars requires a square terrain, got size={cfg.size}")

    terrain_width = float(cfg.size[0])
    if cfg.platform_size <= 0.0 or cfg.platform_size >= terrain_width:
        raise ValueError("platform_size must be positive and smaller than the terrain size")
    if cfg.border_width <= 0.0 or 2.0 * cfg.border_width >= terrain_width:
        raise ValueError("border_width must be positive and leave room for the pit")
    if cfg.platform_size >= terrain_width - 2.0 * cfg.border_width:
        raise ValueError("platform_size must fit inside the excavated area")

    # A fixed seed gives reproducible pillar layouts for terrain caching and
    # training restarts. Difficulty controls spacing only: easy tiles are
    # densely packed and hard tiles are sparse. The deepest allowed pillar
    # defines the pit depth, so no pillar extends above the ground plane.
    seed = cfg.random_seed + int(round(difficulty * 100_000.0)) * 7_919
    rng = np.random.default_rng(seed)
    min_height, max_height = cfg.pillar_height_range
    grid_spacing = cfg.dense_grid_spacing + difficulty * (cfg.grid_spacing - cfg.dense_grid_spacing)
    pit_depth = max_height
    meshes: list[trimesh.Trimesh] = []

    def add_box(center: tuple[float, float, float], dimensions: tuple[float, float, float]) -> None:
        meshes.append(
            trimesh.creation.box(
                dimensions,
                trimesh.transformations.translation_matrix(center),
            )
        )

    terrain_center = terrain_width / 2.0
    platform_half = cfg.platform_size / 2.0
    terrain_half = terrain_width / 2.0
    pit_half = terrain_half - cfg.border_width

    # The floor closes every excavation, so rays and falling robots never enter
    # an unbounded void.  It is also the base from which pillars grow.
    add_box(
        (terrain_center, terrain_center, -pit_depth - cfg.floor_thickness / 2.0),
        (terrain_width, terrain_width, cfg.floor_thickness),
    )

    # A level solid rim surrounds the pit.  Neighboring terrain-generator
    # tiles meet on this rim rather than connecting their pillar fields.
    meshes.extend(
        _make_square_annulus(
            pit_half,
            terrain_half,
            pit_depth,
            (terrain_center, terrain_center),
        )
    )

    # This platform grows from the pit floor, with its top exactly level with
    # the ground plane of neighboring terrain tiles.
    add_box(
        (terrain_center, terrain_center, -pit_depth / 2.0),
        (cfg.platform_size, cfg.platform_size, pit_depth),
    )

    # Pillars start on the pit floor.  A single, centered square grid is filled
    # row-by-row, so every adjacent pair uses exactly the curriculum spacing.
    # The grid deliberately extends into the level rim when needed; overlapping
    # that outer platform is preferable to creating irregular pillar spacing.
    pillar_half = cfg.pillar_size / 2.0
    axis_count = int(np.floor((terrain_half - pillar_half) / grid_spacing))
    axis_coordinates = grid_spacing * np.arange(-axis_count, axis_count + 1, dtype=float)

    for local_y in axis_coordinates:
        for local_x in axis_coordinates:
            if (
                abs(local_x) < platform_half + pillar_half
                and abs(local_y) < platform_half + pillar_half
            ):
                continue

            height = float(rng.uniform(min_height, max_height))
            add_box(
                (terrain_center + local_x, terrain_center + local_y, -pit_depth + height / 2.0),
                (cfg.pillar_size, cfg.pillar_size, height),
            )

    # The center platform is the unique spawn platform for this terrain tile.
    origin = np.array([terrain_center, terrain_center, 0.0])
    return meshes, origin


@configclass
class CourageGapsTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    """Concentric, curriculum-controlled trenches around a center platform."""

    function = courage_gaps_terrain

    platform_width: float = 1.5
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


@configclass
class ModularPillarTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    """Configuration for one terrain tile with an excavated pillar field."""

    function = modular_pillar_terrain

    pillar_size: float = 0.4
    """Square pillar side length (m)."""

    grid_spacing: float = 1.0
    """Pillar spacing at difficulty 1, the sparsest curriculum level (m)."""

    dense_grid_spacing: float = 0.6
    """Pillar spacing at difficulty 0, the densest curriculum level (m)."""

    pillar_height_range: tuple[float, float] = (0.5, 2.0)
    """Minimum and maximum pillar heights measured from the pit floor (m)."""

    platform_size: float = 5.0
    """Side length of each level square center platform (m)."""

    border_width: float = 1.0
    """Width of the level outer rim that separates adjacent terrain tiles (m)."""

    platform_height: float = 0.0
    """Must remain zero: the center platform is level with the ground plane."""

    floor_thickness: float = 0.10
    """Thickness of the physical floor beneath each large excavation (m)."""

    random_seed: int = 20260924
    """Seed used for reproducible pillar heights."""


