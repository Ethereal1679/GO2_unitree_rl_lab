"""Attention coloring for the existing Isaac Lab height-scan markers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pxr import Sdf, Vt


class HeightScanAttentionVisualizer:
    """Update the RayCaster's existing PointInstancer colors from policy attention.

    This helper never creates markers and never updates marker positions.  The
    RayCaster remains the owner of the point locations and point ordering.
    """

    def __init__(self, env: Any, cfg: Any, sensor_name: str = "height_scanner", env_id: int = 0):
        self.env = getattr(env, "unwrapped", env)
        self.cfg = cfg
        self.sensor = self.env.scene.sensors[sensor_name]
        self.env_id = int(env_id)
        self._colors: np.ndarray | None = None
        self._opacity: np.ndarray | None = None
        self._color_attr = None
        self._opacity_attr = None

    @property
    def marker_visualizer(self):
        return getattr(self.sensor, "ray_visualizer", None)

    @staticmethod
    def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    @staticmethod
    def _base_color(device: torch.device) -> torch.Tensor:
        """Base color for height-scan markers that are not in the top-k."""

        return torch.tensor((0.0, 0.2, 1.0), device=device, dtype=torch.float32)

    def _top_attention_colors(
        self, attention: torch.Tensor, finite_mask: torch.Tensor
    ) -> torch.Tensor:
        """Color only the strongest attention points red, fading to pale red.

        All finite points start blue.  The strongest point is pure red and the
        remaining selected points progressively fade toward a pale red.  This
        rank-based coloring makes the visualization readable even when the
        attention values have a narrow dynamic range.
        """

        flat_attention = attention.reshape(-1)
        flat_finite = finite_mask.reshape(-1)
        colors = self._base_color(flat_attention.device).expand(flat_attention.numel(), 3).clone()

        top_k = max(0, int(self._cfg_value(self.cfg, "top_k", 30)))
        valid_indices = torch.nonzero(flat_finite, as_tuple=False).squeeze(-1)
        if top_k == 0 or valid_indices.numel() == 0:
            return colors

        count = min(top_k, int(valid_indices.numel()))
        _, order = torch.topk(flat_attention[valid_indices], k=count, largest=True, sorted=True)
        selected_indices = valid_indices[order]

        # Rank 0 is pure red; lower-ranked points fade toward a pale red.
        rank = torch.arange(count, device=flat_attention.device, dtype=torch.float32)
        fade = rank / max(count - 1, 1)
        red = torch.tensor((1.0, 0.0, 0.0), device=flat_attention.device, dtype=torch.float32)
        pale_red = torch.tensor((1.0, 0.82, 0.82), device=flat_attention.device, dtype=torch.float32)
        selected_colors = red.unsqueeze(0) * (1.0 - fade.unsqueeze(1)) + pale_red.unsqueeze(0) * fade.unsqueeze(1)
        colors[selected_indices] = selected_colors
        return colors

    def _normalize(self, attention: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = attention.float()
        finite_mask = torch.isfinite(attention) & valid_mask
        if not torch.any(finite_mask):
            return torch.zeros_like(attention), finite_mask

        values = attention[finite_mask]
        normalization = str(self._cfg_value(self.cfg, "normalization", "percentile")).lower()
        if normalization == "percentile":
            low = float(self._cfg_value(self.cfg, "percentile_low", 5.0))
            high = float(self._cfg_value(self.cfg, "percentile_high", 95.0))
            low = min(max(low, 0.0), 100.0)
            high = min(max(high, low), 100.0)
            bounds = torch.tensor([low / 100.0, high / 100.0], device=values.device, dtype=values.dtype)
            lower, upper = torch.quantile(values, bounds)
        elif normalization in {"minmax", "min-max"}:
            lower, upper = values.min(), values.max()
        else:
            raise ValueError(f"Unsupported attention normalization: {normalization}")

        normalized = (attention - lower) / (upper - lower).clamp_min(torch.finfo(attention.dtype).eps)
        normalized = normalized.nan_to_num(0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        normalized = torch.where(finite_mask, normalized, torch.zeros_like(normalized))
        return normalized, finite_mask

    def _ensure_primvars(self, count: int) -> None:
        visualizer = self.marker_visualizer
        if visualizer is None:
            return

        if self._colors is None or self._colors.shape != (count, 3):
            # Keep the complete height scan blue until attention is available.
            # This also gives non-selected environments the same neutral base.
            self._colors = np.tile((0.0, 0.2, 1.0), (count, 1)).astype(np.float32)
            self._opacity = np.ones((count,), dtype=np.float32)

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

    def _marker_valid_mask(self) -> torch.Tensor:
        # Match RayCaster._debug_vis_callback exactly: it flattens env-major
        # and removes only points containing +/-inf before visualizing them.
        ray_hits_w = self.sensor.data.ray_hits_w
        return ~torch.any(torch.isinf(ray_hits_w), dim=-1)

    def update(self, attention_weights: torch.Tensor, map_scans: torch.Tensor) -> bool:
        """Batch-update colors for the selected environment's existing points."""

        visualizer = self.marker_visualizer
        if visualizer is None or attention_weights is None:
            return False
        if attention_weights.ndim != 4 or map_scans.ndim != 4:
            return False
        if self.env_id >= attention_weights.shape[0] or self.env_id >= map_scans.shape[0]:
            return False

        aggregation = str(self._cfg_value(self.cfg, "aggregation", "mean")).lower()
        if aggregation == "mean":
            attention = attention_weights.mean(dim=1)
        elif aggregation == "max":
            attention = attention_weights.max(dim=1).values
        else:
            raise ValueError(f"Unsupported attention aggregation: {aggregation}")
        attention = attention.squeeze(1)
        height, width = map_scans.shape[1:3]
        if attention.shape[-1] != height * width:
            raise ValueError(
                f"Attention length {attention.shape[-1]} does not match height scan shape {(height, width)}"
            )
        attention = attention.reshape(attention.shape[0], height, width)[self.env_id]
        valid_mask = torch.isfinite(map_scans[self.env_id]).all(dim=-1)
        # Rank raw finite attention values so percentile clipping cannot make
        # many points tie at the same color.
        finite_mask = torch.isfinite(attention) & valid_mask
        colors = self._top_attention_colors(attention, finite_mask)

        marker_valid = self._marker_valid_mask()
        if marker_valid.shape[0] != map_scans.shape[0] or marker_valid.shape[1] != height * width:
            raise ValueError(
                f"RayCaster marker shape {tuple(marker_valid.shape)} does not match map scan shape "
                f"{tuple(map_scans.shape)}"
            )
        expected_count = int(marker_valid.sum().item())
        if expected_count == 0:
            return False
        if visualizer.count != expected_count:
            # The RayCaster post-update callback owns marker positions. Wait
            # until its existing buffer has the matching point count.
            return False

        flat_valid = marker_valid.reshape(-1)
        points_per_env = height * width
        start = self.env_id * points_per_env
        stop = start + points_per_env
        selected_mask = flat_valid[start:stop]
        selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        offset = flat_valid[:start].sum()
        selected_indices = (selected_indices + offset).detach().cpu().numpy()

        self._ensure_primvars(visualizer.count)
        assert self._colors is not None and self._opacity is not None
        selected_colors = colors[selected_mask]
        self._colors[selected_indices] = selected_colors.detach().cpu().numpy().astype(np.float32, copy=False)
        show_invalid = bool(self._cfg_value(self.cfg, "show_invalid_points", False))
        selected_finite_mask = finite_mask.reshape(-1)[selected_mask]
        selected_opacity = torch.ones(
            selected_finite_mask.shape, device=selected_finite_mask.device, dtype=torch.float32
        )
        if not show_invalid:
            selected_opacity = selected_opacity * selected_finite_mask.float()
        self._opacity[selected_indices] = selected_opacity.detach().cpu().numpy().astype(np.float32, copy=False)
        self._color_attr.Set(Vt.Vec3fArray.FromNumpy(self._colors))
        self._opacity_attr.Set(Vt.FloatArray.FromNumpy(self._opacity))
        return True
