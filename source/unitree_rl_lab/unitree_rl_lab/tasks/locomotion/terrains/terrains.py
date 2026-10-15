from __future__ import annotations

import math
from dataclasses import MISSING
import numpy as np
import trimesh
from random import randint

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh


# ======= terrain generator for go2_att task =======
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



# == gap地形，来源 https://github.com/SII-FUSC/AME_Locomotion

def _resolve_curriculum_value(value: float | tuple[float, float], difficulty: float) -> float:
    """Resolve a fixed value or a ``(difficulty=0, difficulty=1)`` range."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"curriculum range must contain exactly two values, got {value}")
        return float(value[0]) + float(difficulty) * (float(value[1]) - float(value[0]))
    return float(value)


def _resolve_depth(depth: float | tuple[float, float], difficulty: float) -> float:
    """Resolve a fixed depth or a curriculum range in meters."""
    return _resolve_curriculum_value(depth, difficulty)

@height_field_to_mesh
def concentric_gap_terrain(difficulty: float, cfg: HfConcentricGapTerrainCfg) -> np.ndarray:
    """
    Generate concentric gap terrain with a center platform.
    Gap width and depth can both vary with curriculum difficulty.
    """
    # Gap depth varies with curriculum difficulty. The height field stores
    # negative obstacle heights directly, e.g. (-0.5, -3.0) meters.
    gap_depth = int(round(_resolve_depth(cfg.gap_depth, difficulty) / cfg.vertical_scale))
    # Gap width varies with difficulty
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_width = int(gap_width / cfg.horizontal_scale)
    # Ground width varies with difficulty (narrower for harder terrains)
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0])
    ground_width = int(ground_width / cfg.horizontal_scale)
    # Ground height
    ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
    # Terrain dimensions
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # Platform width
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    is_gap = True
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        if is_gap:
            # Fill gap ring
            hf_raw[start_x:stop_x, start_y:stop_y] = gap_depth
            start_x += gap_width
            stop_x -= gap_width
            start_y += gap_width
            stop_y -= gap_width
        else:
            # Fill ground ring
            hf_raw[start_x:stop_x, start_y:stop_y] = randint(-ground_height_max, ground_height_max)
            start_x += ground_width
            stop_x -= ground_width
            start_y += ground_width
            stop_y -= ground_width
        is_gap = not is_gap
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfConcentricGapTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a concentric gaps height field terrain."""

    function = concentric_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gaps (in m)."""
    ground_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ground (in m)."""
    ground_height_max: float = MISSING
    """The maximum height of the ground (in m)."""
    gap_depth: float | tuple[float, float] = -2.0
    """Gap depth in meters, or its ``(easy, hard)`` curriculum range."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""





def stepping_stones_terrain(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes: support either a fixed depth or an (easy, hard) range.
    holes_depth = int(round(_resolve_depth(cfg.holes_depth, difficulty) / cfg.vertical_scale))
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    if stone_width < 1:
        raise ValueError("stone_width_range must produce at least one grid cell")
    if stone_distance < 0:
        raise ValueError("stone_distance_range must be non-negative")
    if stone_height_max < 0:
        raise ValueError("stone_height_max must be non-negative")
    if platform_width < 1:
        raise ValueError("platform_width must be at least one grid cell")
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)
    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # ensure that stone stops along y-axis
            stop_y = min(length_pixels, start_y + stone_width)
            # randomly sample x-position
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            # fill first stone
            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
            # fill row with stones
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            # update y-position
            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            # ensure that stone stops along x-axis
            stop_x = min(width_pixels, start_x + stone_width)
            # randomly sample y-position
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            # fill first stone
            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)
            # fill column with stones
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            # update x-position
            start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)




def _append_box(
    meshes: list[trimesh.Trimesh],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    bottom_z: float,
    top_z: float,
) -> None:
    """Append one non-degenerate axis-aligned box to ``meshes``."""

    if x1 <= x0 or y1 <= y0 or top_z <= bottom_z:
        return
    meshes.append(
        trimesh.creation.box(
            extents=(x1 - x0, y1 - y0, top_z - bottom_z),
            transform=trimesh.transformations.translation_matrix(
                ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (bottom_z + top_z) * 0.5)
            ),
        )
    )


def _append_mask_rectangles(
    meshes: list[trimesh.Trimesh],
    mask: np.ndarray,
    horizontal_scale: float,
    x_offset: float,
    y_offset: float,
    bottom_z: float,
    top_z: float,
) -> None:
    """Convert only ``True`` cells into maximal, non-overlapping boxes."""

    used = np.zeros(mask.shape, dtype=bool)
    width, length = mask.shape
    for x in range(width):
        y = 0
        while y < length:
            if not mask[x, y] or used[x, y]:
                y += 1
                continue

            y_end = y + 1
            while y_end < length and mask[x, y_end] and not used[x, y_end]:
                y_end += 1
            x_end = x + 1
            while x_end < width:
                row = mask[x_end, y:y_end]
                if not np.all(row) or np.any(used[x_end, y:y_end]):
                    break
                x_end += 1

            used[x:x_end, y:y_end] = True
            _append_box(
                meshes,
                x_offset + x * horizontal_scale,
                x_offset + x_end * horizontal_scale,
                y_offset + y * horizontal_scale,
                y_offset + y_end * horizontal_scale,
                bottom_z,
                top_z,
            )
            y = y_end


def stepping_stones_clean_terrain(difficulty, cfg):
    """Generate independent stepping-stone boxes without height-field triangles.

    The random stone layout still comes from :func:`stepping_stones_terrain`.
    Unlike the old implementation, the floor is created only in hole cells,
    ``border_width`` is honored, and stone bases meet the floor exactly without
    overlapping it.
    """

    hs = float(cfg.horizontal_scale)
    vs = float(cfg.vertical_scale)
    size_x, size_y = (float(cfg.size[0]), float(cfg.size[1]))
    border = float(cfg.border_width)
    if hs <= 0.0 or vs <= 0.0:
        raise ValueError("horizontal_scale and vertical_scale must be positive")
    if border < 0.0 or 2.0 * border >= min(size_x, size_y):
        raise ValueError("border_width must leave a positive interior")

    # Use whole cells inside the configured tile. The remainder is split
    # equally between both sides, so ``border_width`` affects the actual
    # geometry and the final mesh remains exactly ``cfg.size`` wide.
    nx = int((size_x - 2.0 * border) / hs)
    ny = int((size_y - 2.0 * border) / hs)
    if nx < 1 or ny < 1:
        raise ValueError("terrain size and border_width leave no stepping-stone cells")
    interior_x = nx * hs
    interior_y = ny * hs
    x_offset = 0.5 * (size_x - interior_x)
    y_offset = 0.5 * (size_y - interior_y)

    cfg_for_layout = cfg.copy() if hasattr(cfg, "copy") else cfg
    original_size = cfg_for_layout.size
    cfg_for_layout.size = (interior_x, interior_y)
    try:
        hf = stepping_stones_terrain(difficulty, cfg_for_layout)
    finally:
        cfg_for_layout.size = original_size

    floor_raw = int(round(_resolve_depth(cfg.holes_depth, difficulty) / vs))
    floor_z = floor_raw * vs
    if floor_raw >= 0:
        raise ValueError("holes_depth must be negative")
    if np.any(hf < floor_raw):
        raise ValueError("holes_depth must be below every stepping-stone top")

    meshes: list[trimesh.Trimesh] = []
    floor_bottom = floor_z - max(0.05, hs)
    hole_mask = hf == floor_raw
    _append_mask_rectangles(meshes, hole_mask, hs, x_offset, y_offset, floor_bottom, floor_z)

    # Keep the center platform as its own box. Otherwise a randomly sampled
    # stone with the same height (usually 0) could be merged into the platform.
    platform_cells = max(1, int(round(float(cfg.platform_width) / hs)))
    platform_x0 = max(0, (nx - platform_cells) // 2)
    platform_y0 = max(0, (ny - platform_cells) // 2)
    platform_x1 = min(nx, platform_x0 + platform_cells)
    platform_y1 = min(ny, platform_y0 + platform_cells)
    platform_mask = np.zeros_like(hole_mask)
    platform_mask[platform_x0:platform_x1, platform_y0:platform_y1] = True
    _append_box(
        meshes,
        x_offset + platform_x0 * hs,
        x_offset + platform_x1 * hs,
        y_offset + platform_y0 * hs,
        y_offset + platform_y1 * hs,
        floor_z,
        0.0,
    )

    stone_mask = ~hole_mask & ~platform_mask
    for height_value in np.unique(hf[stone_mask]):
        _append_mask_rectangles(
            meshes,
            stone_mask & (hf == height_value),
            hs,
            x_offset,
            y_offset,
            floor_z,
            float(height_value) * vs,
        )

    # Preserve the height-field border convention: top of the border is z=0.
    if x_offset > 0.0:
        _append_box(meshes, 0.0, x_offset, 0.0, size_y, floor_bottom, 0.0)
        _append_box(meshes, x_offset + interior_x, size_x, 0.0, size_y, floor_bottom, 0.0)
    if y_offset > 0.0:
        _append_box(meshes, x_offset, x_offset + interior_x, 0.0, y_offset, floor_bottom, 0.0)
        _append_box(meshes, x_offset, x_offset + interior_x, y_offset + interior_y, size_y, floor_bottom, 0.0)

    return meshes, np.array([size_x * 0.5, size_y * 0.5, 0.0], dtype=np.float64)



@configclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for the box-mesh stepping-stones terrain.

    The class keeps the ``HfTerrainBaseCfg`` name for compatibility with the
    existing terrain configuration, but its ``function`` returns independent
    box meshes and does not use height-field triangulation.
    """

    function = stepping_stones_clean_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: float | tuple[float, float] = -10.0
    """Hole depth in meters, or its ``(easy, hard)`` curriculum range."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@height_field_to_mesh
def i_bridge_terrain(difficulty: float, cfg: HfIBridgeTerrainCfg) -> np.ndarray:
    """Generate a four-way bridge around a square center platform.

    Two perpendicular, configurable-width bridge strips connect the center
    platform to all four edges. Everything outside the bridge is a depressed
    hole, so the robot can start on the center platform and walk in any
    cardinal direction.
    """
    if cfg.horizontal_scale <= 0.0 or cfg.vertical_scale <= 0.0:
        raise ValueError("horizontal_scale and vertical_scale must be positive")

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    bridge_width_m = _resolve_curriculum_value(cfg.bridge_width, difficulty)
    bridge_width = int(round(bridge_width_m / cfg.horizontal_scale))
    platform_width = int(round(cfg.platform_width / cfg.horizontal_scale))
    holes_depth_m = _resolve_curriculum_value(cfg.holes_depth, difficulty)
    holes_depth = int(round(holes_depth_m / cfg.vertical_scale))

    if bridge_width <= 0 or platform_width <= 0:
        raise ValueError("bridge_width and platform_width must be at least one height-field cell")
    if bridge_width > min(width_pixels, length_pixels):
        raise ValueError("bridge_width must fit within the terrain size")
    if platform_width > min(width_pixels, length_pixels):
        raise ValueError("platform_width must fit within the terrain size")
    if holes_depth >= 0:
        raise ValueError("holes_depth must be negative to create depressed ground")

    hf_raw = np.full((width_pixels, length_pixels), holes_depth, dtype=np.int16)

    # The two full-length strips form four connected bridge arms.
    x1 = (width_pixels - bridge_width) // 2
    y1 = (length_pixels - bridge_width) // 2
    hf_raw[x1 : x1 + bridge_width, :] = 0
    hf_raw[:, y1 : y1 + bridge_width] = 0

    # Keep the center platform square and aligned with the center of each axis.
    platform_x1 = (width_pixels - platform_width) // 2
    platform_y1 = (length_pixels - platform_width) // 2
    hf_raw[
        platform_x1 : platform_x1 + platform_width,
        platform_y1 : platform_y1 + platform_width,
    ] = 0

    return hf_raw


@configclass
class HfIBridgeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a four-way narrow bridge height-field terrain."""

    function = i_bridge_terrain

    bridge_width: float | tuple[float, float] = 0.5
    """Bridge width in meters, or its ``(difficulty=0, difficulty=1)`` range."""

    platform_width: float = 1.5
    """Width of the square center platform in meters."""

    holes_depth: float | tuple[float, float] = -2.0
    """Hole depth in meters, or its ``(difficulty=0, difficulty=1)`` range."""
