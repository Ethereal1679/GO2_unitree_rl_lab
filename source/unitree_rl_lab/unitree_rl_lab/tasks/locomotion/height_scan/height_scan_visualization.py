"""Height-scan marker configuration and temporal-gradient coloring."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pxr import Sdf, Vt

import isaaclab.sim as sim_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG


# Keep Isaac Lab's marker geometry and PointInstancer behavior, but remove the
# fixed PreviewSurface. With no bound visual material, USD's per-instance
# ``displayColor`` primvar controls both attention and gradient coloring.
HEIGHT_SCAN_MARKER_CFG = RAY_CASTER_MARKER_CFG.replace(
    prim_path="/Visuals/HeightScanner",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=None,
        ),
    },
)


class HeightScanGradientVisualizer:
    """Color the RayCaster's existing scan dots by temporal height-rate magnitude."""

    def __init__(
        self,
        env: Any,
        sensor_name: str = "height_scanner",
        max_gradient: float = 1.0,
        low_color: tuple[float, float, float] = (1.0, 0.68, 0.8),
        high_color: tuple[float, float, float] = (0.42, 0.0, 0.06),
    ):
        if max_gradient <= 0.0:
            raise ValueError("max_gradient must be positive.")
        self.env = getattr(env, "unwrapped", env)
        self.sensor = self.env.scene.sensors[sensor_name]
        self.max_gradient = float(max_gradient)
        self.low_color = torch.tensor(low_color, dtype=torch.float32)
        self.high_color = torch.tensor(high_color, dtype=torch.float32)
        self._color_attr = None
        self._opacity_attr = None

    @property
    def marker_visualizer(self):
        return getattr(self.sensor, "ray_visualizer", None)

    def _ensure_primvars(self) -> bool:
        visualizer = self.marker_visualizer
        if visualizer is None:
            return False

        prim = visualizer._instancer_manager.GetPrim()
        color_name = "primvars:displayColor"
        opacity_name = "primvars:displayOpacity"
        self._color_attr = prim.GetAttribute(color_name)
        if not self._color_attr:
            self._color_attr = prim.CreateAttribute(color_name, Sdf.ValueTypeNames.Color3fArray)
        self._color_attr.SetMetadata("interpolation", "instance")
        self._opacity_attr = prim.GetAttribute(opacity_name)
        if not self._opacity_attr:
            self._opacity_attr = prim.CreateAttribute(opacity_name, Sdf.ValueTypeNames.FloatArray)
        self._opacity_attr.SetMetadata("interpolation", "instance")
        return True

    def update(self, gradient: torch.Tensor, valid_mask: torch.Tensor | None = None) -> bool:
        """Apply a signed scalar gradient (or 3D gradient vector) to the current dots."""

        visualizer = self.marker_visualizer
        if visualizer is None:
            return False
        ray_hits_w = self.sensor.data.ray_hits_w
        marker_valid = ~torch.any(torch.isinf(ray_hits_w), dim=-1)
        if gradient.ndim == 3 and gradient.shape[-1] == 3:
            magnitude = torch.linalg.vector_norm(gradient, dim=-1)
        elif gradient.ndim == 2:
            magnitude = gradient.abs()
        else:
            raise ValueError(f"Expected gradient shaped (N, R) or (N, R, 3), got {tuple(gradient.shape)}")
        if magnitude.shape != marker_valid.shape:
            raise ValueError(
                f"Gradient shape {tuple(magnitude.shape)} does not match RayCaster shape {tuple(marker_valid.shape)}"
            )
        if valid_mask is None:
            valid_mask = torch.isfinite(magnitude)
        elif valid_mask.shape != magnitude.shape:
            raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match {tuple(magnitude.shape)}")

        selected_count = int(marker_valid.sum().item())
        if selected_count == 0 or visualizer.count != selected_count:
            return False
        if not self._ensure_primvars():
            return False

        finite = torch.isfinite(magnitude) & valid_mask
        normalized = (magnitude / self.max_gradient).nan_to_num(0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        low_color = self.low_color.to(device=magnitude.device)
        high_color = self.high_color.to(device=magnitude.device)
        colors = low_color + normalized.unsqueeze(-1) * (high_color - low_color)
        colors = colors[marker_valid].detach().cpu().numpy().astype(np.float32, copy=False)
        opacity = finite[marker_valid].float().detach().cpu().numpy().astype(np.float32, copy=False)
        self._color_attr.Set(Vt.Vec3fArray.FromNumpy(colors))
        self._opacity_attr.Set(Vt.FloatArray.FromNumpy(opacity))
        return True
