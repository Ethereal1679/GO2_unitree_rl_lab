"""OpenCV visualization for the temporal 3D-CNN/VAE height-map branch."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import numpy as np


def _has_highgui(module) -> bool:
    for line in module.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip() != "NONE"
    return False


def _load_gui_opencv():
    current = sys.modules.get("cv2")
    if current is not None and _has_highgui(current):
        return current

    try:
        package_root = importlib.metadata.distribution("opencv-python").locate_file("cv2").parent
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The imported OpenCV build has no HighGUI support and the GUI-enabled opencv-python package is missing."
        ) from exc

    for name in tuple(sys.modules):
        if name == "cv2" or name.startswith("cv2."):
            del sys.modules[name]
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError(f"Unable to load GUI-enabled OpenCV from {package_root}.") from exc
    if not _has_highgui(module):
        raise RuntimeError(
            "The installed opencv-python build has no HighGUI support; install a GUI-enabled OpenCV package."
        )
    return module


class ThreeDCNNMapVisualizer:
    """Show the three history maps, current map/reconstruction, and prediction."""

    def __init__(self, policy) -> None:
        self.cv2 = _load_gui_opencv()
        gui_backend = next(
            (line.split(":", 1)[1].strip() for line in self.cv2.getBuildInformation().splitlines() if line.strip().startswith("GUI:")),
            "unknown",
        )
        print(f"[INFO] 3D-CNN visualization OpenCV: {self.cv2.__version__} ({gui_backend})")
        self.policy = policy
        self.window_name = "3D-CNN height maps"
        self.map_shape = tuple(int(size) for size in policy.temporal_map_shape)
        self._closed = False
        self._first_frame_shown = False
        self._last_update_error = None
        try:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(self.window_name, 1200, 850)
            self.cv2.moveWindow(self.window_name, 40, 40)
            try:
                self.cv2.setWindowProperty(
                    self.window_name, self.cv2.WND_PROP_TOPMOST, 1.0
                )
            except self.cv2.error:
                pass
            placeholder = np.zeros((850, 1200, 3), dtype=np.uint8)
            self.cv2.putText(
                placeholder,
                "Waiting for the first 3D-CNN inference...",
                (35, 60),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
            self.cv2.imshow(self.window_name, placeholder)
            self.cv2.waitKey(100)
            try:
                self.cv2.setWindowProperty(
                    self.window_name, self.cv2.WND_PROP_TOPMOST, 1.0
                )
                self.cv2.moveWindow(self.window_name, 40, 40)
            except self.cv2.error:
                pass
        except self.cv2.error as exc:
            raise RuntimeError(f"Unable to open the OpenCV visualization window: {exc}") from exc
        print("[INFO] OpenCV window created: 3D-CNN height maps (topmost, at screen position 40,40).")

    def _heatmap(self, values: np.ndarray, size: tuple[int, int], low: float, high: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values)
        normalized = (np.nan_to_num(values, nan=low, posinf=high, neginf=low) - low) / max(high - low, 1e-6)
        color_map = getattr(self.cv2, "COLORMAP_TURBO", self.cv2.COLORMAP_JET)
        image = self.cv2.applyColorMap(np.clip(normalized * 255, 0, 255).astype(np.uint8), color_map)
        image[~finite] = (255, 0, 255)
        image = self.cv2.resize(image, size, interpolation=self.cv2.INTER_NEAREST)
        return image

    def update(self) -> bool:
        if self._closed:
            return False
        try:
            result = self.policy.get_temporal_height_map_reconstruction()
            temporal, reconstructed_current, predicted = (array[60].detach().cpu().numpy() for array in result)
            history_count = int(self.policy.temporal_history_frames)
            history = temporal[-(history_count + 1) : -1]
            if len(history) != history_count:
                raise ValueError(
                    f"Expected {history_count} history maps plus current map, got {len(temporal)} frames."
                )

            selected_maps = [*history, temporal[-1], reconstructed_current, predicted]
            finite_parts = [values[np.isfinite(values)] for values in selected_maps]
            finite_parts = [values for values in finite_parts if values.size]
            if finite_parts:
                low = min(float(values.min()) for values in finite_parts)
                high = max(float(values.max()) for values in finite_parts)
                if high <= low:
                    low, high = low - 0.5, high + 0.5
            else:
                low, high = -1.0, 1.0

            cell_size = (self.map_shape[1] * 14, self.map_shape[0] * 14)
            titles = [f"History t-{history_count - index}" for index in range(history_count)]
            titles.extend(("Current t", "VAE reconstruction", "3D-CNN prediction t+1"))
            panels = [
                self._labelled(self._heatmap(values, cell_size, low, high), title)
                for values, title in zip(selected_maps, titles)
            ]
            rows = [panels[:history_count], panels[history_count:]]
            gap = 12
            panel_height, panel_width = panels[0].shape[:2]
            canvas = np.full(
                (
                    len(rows) * panel_height + (len(rows) + 1) * gap,
                    history_count * panel_width + (history_count + 1) * gap,
                    3,
                ),
                (48, 48, 48),
                dtype=np.uint8,
            )
            for row_index, row in enumerate(rows):
                for column_index, panel in enumerate(row):
                    top = gap + row_index * (panel_height + gap)
                    left = gap + column_index * (panel_width + gap)
                    canvas[top : top + panel_height, left : left + panel_width] = panel
            self.cv2.imshow(self.window_name, canvas)
            key = self.cv2.waitKey(1) & 0xFF
        except self.cv2.error:
            print("[WARNING] OpenCV could not update the 3D-CNN map window; retrying next frame.")
            return False
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self._last_update_error:
                print(f"[WARNING] 3D-CNN map visualization update failed; retrying: {message}", flush=True)
                self._last_update_error = message
            return False
        if not self._first_frame_shown:
            self._first_frame_shown = True
            self._last_update_error = None
            try:
                self.cv2.setWindowProperty(self.window_name, self.cv2.WND_PROP_TOPMOST, 1.0)
                self.cv2.moveWindow(self.window_name, 40, 40)
                visible = self.cv2.getWindowProperty(self.window_name, self.cv2.WND_PROP_VISIBLE)
            except self.cv2.error:
                visible = "unknown"
            print(
                f"[INFO] First 3D-CNN heatmap frame displayed with cv2.imshow(); window_visible={visible}.",
                flush=True,
            )
        if key in (27, ord("q")):
            self.close()
        return not self._closed

    def _labelled(self, image: np.ndarray, title: str) -> np.ndarray:
        output = np.full((image.shape[0] + 26, image.shape[1], 3), (24, 24, 24), dtype=np.uint8)
        output[26:] = image
        self.cv2.putText(
            output, title, (6, 19), self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, self.cv2.LINE_AA
        )
        return output

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.cv2.destroyWindow(self.window_name)
                self.cv2.waitKey(1)
            except self.cv2.error:
                pass
