# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg
from unitree_rl_lab.tasks.locomotion.terrains.terrains import (
    HfConcentricGapTerrainCfg,
    HfSteppingStonesTerrainCfg
)


def _cardinal_target_patches() -> dict[str, FlatPatchSamplingCfg]:
    """Sample targets on the flat midpoint of each terrain border."""
    ranges = {
        "target_pos_x": ((3.2, 3.7), (-0.35, 0.35)),
        "target_neg_x": ((-3.7, -3.2), (-0.35, 0.35)),
        "target_pos_y": ((-0.35, 0.35), (3.2, 3.7)),
        "target_neg_y": ((-0.35, 0.35), (-3.7, -3.2)),
    }
    return {
        name: FlatPatchSamplingCfg(
            num_patches=32,
            patch_radius=0.1,
            x_range=x_range,
            y_range=y_range,
            z_range=(-0.1, 0.1),
            max_height_diff=0.02,
        )
        for name, (x_range, y_range) in ranges.items()
    }


# ======= terrain config =======
COMPLEX_RANDOM_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=30.0,
    num_rows=15,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # "flat": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.1,
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.5
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.5
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.5
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.1, 0.25),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.1, 0.25),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "hf_gaps": HfConcentricGapTerrainCfg(
            proportion=0.2, gap_width_range=(0.1, 0.3), platform_width=2.0,
            gap_depth=(-1.1, -2.0), border_width=1.0,
            ground_width_range=(0.5, 0.5), ground_height_max=0.025,
            flat_patch_sampling=_cardinal_target_patches(),
        ),
        "hf_steppingstones": HfSteppingStonesTerrainCfg(
            proportion=0.2, stone_height_max=0.05, stone_width_range=(0.25, 0.5), stone_distance_range=(0.1, 0.2), platform_width=2.0,
            holes_depth=(-1.1, -2.0), border_width=1.0,
            flat_patch_sampling=_cardinal_target_patches(),
        ),
    },
)
