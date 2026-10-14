"""Live pixel heatmaps for the GradCNN map input and reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_HEIGHT_STOPS = np.asarray(
    [
        [0.188, 0.071, 0.231],
        [0.275, 0.396, 0.847],
        [0.110, 0.792, 0.847],
        [0.976, 0.831, 0.137],
        [0.843, 0.098, 0.110],
    ],
    dtype=np.float32,
)
_DIFFERENCE_STOPS = np.asarray(
    [
        [0.090, 0.259, 0.620],
        [0.420, 0.682, 0.839],
        [0.965, 0.965, 0.965],
        [0.957, 0.647, 0.510],
        [0.698, 0.094, 0.169],
    ],
    dtype=np.float32,
)
_EDGE_STOPS = np.asarray(
    [
        [0.035, 0.145, 0.360],
        [0.110, 0.792, 0.847],
        [0.976, 0.831, 0.137],
        [0.843, 0.098, 0.110],
    ],
    dtype=np.float32,
)


def _piecewise_rgb(normalized: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Map values in ``[0, 1]`` to an RGB palette."""

    normalized = np.clip(normalized, 0.0, 1.0)
    position = normalized * (len(stops) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    alpha = (position - lower)[..., None]
    return stops[lower] * (1.0 - alpha) + stops[upper] * alpha


def _colorbar_rgba(width: int, height: int, *, diverging: bool, edge: bool = False) -> np.ndarray:
    """Create a horizontal legend strip using the same palette as the map."""

    stops = _EDGE_STOPS if edge else (_DIFFERENCE_STOPS if diverging else _HEIGHT_STOPS)
    values = np.linspace(0.0, 1.0, width, dtype=np.float32)
    colors = _piecewise_rgb(values, stops)
    rgb = np.repeat(colors[None, :, :], height, axis=0)
    alpha = np.ones((height, width, 1), dtype=np.float32)
    return np.round(np.concatenate((rgb, alpha), axis=-1) * 255.0).astype(np.uint8)


def scalar_map_to_rgba(
    values: np.ndarray,
    value_min: float,
    value_max: float,
    *,
    diverging: bool = False,
    edge: bool = False,
    pixel_size: int = 16,
) -> np.ndarray:
    """Convert a scalar grid into an enlarged RGBA pixel heatmap.

    Invalid values are drawn in magenta. ``value_min`` and ``value_max`` are
    supplied by the caller so an original map and its reconstruction can use
    exactly the same color scale.
    """

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D scalar map, got shape {values.shape}.")
    if pixel_size < 1:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}.")
    span = max(float(value_max) - float(value_min), 1.0e-6)
    finite = np.isfinite(values)
    normalized = (np.nan_to_num(values, nan=value_min, posinf=value_max, neginf=value_min) - value_min) / span
    palette = _EDGE_STOPS if edge else (_DIFFERENCE_STOPS if diverging else _HEIGHT_STOPS)
    rgb = _piecewise_rgb(normalized, palette)
    rgb[~finite] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    rgba = np.concatenate(
        (rgb, np.ones((*values.shape, 1), dtype=np.float32)),
        axis=-1,
    )
    rgba = np.round(rgba * 255.0).astype(np.uint8)
    # Enlarge with nearest-neighbor repetition so every map value remains a
    # sharp, individually visible colored pixel in the UI texture.
    return np.repeat(np.repeat(rgba, pixel_size, axis=0), pixel_size, axis=1)


def _finite_range(*maps: np.ndarray) -> tuple[float, float]:
    finite_parts = [array[np.isfinite(array)] for array in maps]
    finite_parts = [part for part in finite_parts if part.size > 0]
    if not finite_parts:
        return -1.0, 1.0
    values = np.concatenate(finite_parts)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max - value_min < 1.0e-6:
        center = 0.5 * (value_min + value_max)
        return center - 0.5, center + 0.5
    return value_min, value_max


class GradCNNMapVisualizer:
    """Display one environment's two input and two reconstructed maps."""

    def __init__(
        self,
        policy: Any,
        env_id: int,
        num_envs: int,
        pixel_size: int = 16,
        channel_names: tuple[str, str] = ("height", "normalized spatial edge"),
    ) -> None:
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"env_id must be in [0, {num_envs - 1}], got {env_id}.")
        if not hasattr(policy, "get_height_map_reconstruction"):
            raise TypeError("Policy does not expose GradCNN map reconstruction.")
        if int(getattr(policy, "map_channels", 0)) != 2:
            raise ValueError("GradCNN map visualization requires exactly two map channels.")

        try:
            import omni.ui as ui
        except ImportError as exc:
            raise RuntimeError("omni.ui is unavailable; run play.py with the Isaac Sim GUI.") from exc

        self._ui = ui
        self.policy = policy
        self.env_id = int(env_id)
        self.pixel_size = int(pixel_size)
        self.map_height = int(policy.map_shape[0])
        self.map_width = int(policy.map_shape[1])
        self.channel_names = channel_names
        self.image_width = self.map_width * self.pixel_size
        self.image_height = self.map_height * self.pixel_size
        self._providers = [ui.ByteImageProvider() for _ in range(4)]
        self._labels = []
        self._legend_width = 2 * self.image_width + 10
        self._legend_height = 12
        self._legend_providers = [ui.ByteImageProvider() for _ in range(2)]
        self._legend_value_labels: list[list[Any]] = []

        initial = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        initial[..., 3] = 255
        for provider in self._providers:
            provider.set_bytes_data(initial.flatten().data, [self.image_width, self.image_height])
        for provider, (diverging, edge) in zip(self._legend_providers, ((False, False), (False, True))):
            legend = _colorbar_rgba(
                self._legend_width,
                self._legend_height,
                diverging=diverging,
                edge=edge,
            )
            provider.set_bytes_data(legend.flatten().data, [self._legend_width, self._legend_height])

        window_width = 2 * self.image_width + 54
        window_height = 2 * self.image_height + 220
        self.window = ui.Window(
            f"GradCNN Maps - Env {self.env_id}",
            width=window_width,
            height=window_height,
            visible=True,
        )
        with self.window.frame:
            with ui.VStack(spacing=6):
                self._status_label = ui.Label(
                    "Waiting for map data...",
                    height=22,
                    alignment=ui.Alignment.CENTER,
                )
                with ui.HStack(spacing=10):
                    self._build_panel(0, f"Original {self.channel_names[0]}")
                    self._build_panel(1, f"Original {self.channel_names[1]}")
                with ui.HStack(spacing=10):
                    self._build_panel(2, f"Reconstructed {self.channel_names[0]}")
                    self._build_panel(3, f"Reconstructed {self.channel_names[1]}")
                with ui.VStack(spacing=4):
                    self._build_legend(0, f"{self.channel_names[0]} (m)")
                    self._build_legend(1, "Normalized spatial edge (x10)")

    def _build_panel(self, index: int, title: str) -> None:
        ui = self._ui
        with ui.VStack(width=self.image_width, spacing=2):
            label = ui.Label(title, height=20, alignment=ui.Alignment.CENTER)
            self._labels.append(label)
            ui.ImageWithProvider(
                self._providers[index],
                width=self.image_width,
                height=self.image_height,
                fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                pixel_aligned=True,
            )

    def _build_legend(self, index: int, title: str) -> None:
        ui = self._ui
        with ui.VStack(width=self._legend_width, spacing=1):
            ui.Label(title, height=18, alignment=ui.Alignment.CENTER)
            ui.ImageWithProvider(
                self._legend_providers[index],
                width=self._legend_width,
                height=self._legend_height,
                fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                pixel_aligned=True,
            )
            with ui.HStack(spacing=0):
                labels = []
                for _ in range(5):
                    labels.append(
                        ui.Label(
                            "",
                            width=self._legend_width / 5,
                            height=18,
                            alignment=ui.Alignment.CENTER,
                        )
                    )
                self._legend_value_labels.append(labels)

    @staticmethod
    def _description(title: str, values: np.ndarray, mse: float | None = None) -> str:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            description = f"{title} | no finite values"
        else:
            description = f"{title} | min {finite.min():+.3f}  max {finite.max():+.3f}"
        if mse is not None:
            description += f"  MSE {mse:.5f}"
        return description

    def update(self, obs: Any) -> bool:
        """Decode the selected environment and update all four heatmaps."""

        if self.window is None or not self.window.visible:
            return False
        with torch.inference_mode():
            original, reconstruction = self.policy.get_height_map_reconstruction(obs, env_id=self.env_id)
        original = original[0].detach().float().cpu().numpy()
        reconstruction = reconstruction[0].detach().float().cpu().numpy()
        if original.shape != (2, self.map_height, self.map_width):
            raise ValueError(
                f"Expected map shape (2, {self.map_height}, {self.map_width}), got {original.shape}."
            )

        raw_height, raw_edge = original
        reconstructed_height, reconstructed_edge = reconstruction
        height_min, height_max = _finite_range(raw_height, reconstructed_height)
        edge_limit = 10.0  # The spatial edge observation is normalized and scaled by 10.

        images = (
            scalar_map_to_rgba(raw_height, height_min, height_max, pixel_size=self.pixel_size),
            scalar_map_to_rgba(raw_edge, 0.0, edge_limit, edge=True, pixel_size=self.pixel_size),
            scalar_map_to_rgba(reconstructed_height, height_min, height_max, pixel_size=self.pixel_size),
            scalar_map_to_rgba(
                reconstructed_edge,
                0.0,
                edge_limit,
                edge=True,
                pixel_size=self.pixel_size,
            ),
        )
        for provider, image in zip(self._providers, images):
            # Rows follow increasing robot-frame x; show forward (+x) at the top.
            image = np.flipud(image) # 翻转保证高程图可视化是正常的
            provider.set_bytes_data(image.flatten().data, [self.image_width, self.image_height])

        height_mse = float(np.mean((reconstructed_height - raw_height) ** 2))
        edge_mse = float(np.mean((reconstructed_edge - raw_edge) ** 2))
        descriptions = (
            self._description(f"Original {self.channel_names[0]}", raw_height),
            self._description(f"Original {self.channel_names[1]}", raw_edge),
            self._description(f"Reconstructed {self.channel_names[0]}", reconstructed_height, height_mse),
            self._description(f"Reconstructed {self.channel_names[1]}", reconstructed_edge, edge_mse),
        )
        for label, description in zip(self._labels, descriptions):
            label.text = description

        legend_values = (
            tuple(np.linspace(height_min, height_max, 5)),
            tuple(np.linspace(0.0, edge_limit, 5)),
        )
        for labels, values in zip(self._legend_value_labels, legend_values):
            for label, value in zip(labels, values):
                label.text = f"{value:+.1f}"

        self._status_label.text = (
            f"Env {self.env_id} | height [{height_min:+.2f}, {height_max:+.2f}] m | "
            f"normalized spatial edge x10 [0, {edge_limit:.0f}]"
        )
        return True

    def close(self) -> None:
        if self.window is not None:
            self.window.visible = False
            self.window = None
        self._providers.clear()
        self._labels.clear()
        self._legend_providers.clear()
        self._legend_value_labels.clear()
