"""Local height-scan visualization configuration.

Isaac Lab's built-in ray-caster marker uses a red PreviewSurface material.
Attention visualization writes per-instance ``primvars:displayColor`` values,
so the height-scan marker must not keep that fixed material bound.
"""

import isaaclab.sim as sim_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG


# Keep Isaac Lab's marker geometry and PointInstancer behavior, but remove the
# fixed red PreviewSurface.  With no bound visual material, USD's displayColor
# primvar written by HeightScanAttentionVisualizer controls the appearance.
HEIGHT_SCAN_MARKER_CFG = RAY_CASTER_MARKER_CFG.replace(
    prim_path="/Visuals/HeightScanner",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=None,
        ),
    },
)

