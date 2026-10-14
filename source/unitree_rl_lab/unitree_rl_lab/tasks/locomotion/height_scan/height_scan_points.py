"""Shared preprocessing for actor-frame height-scan points."""

from __future__ import annotations

from typing import Any

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
    from isaaclab.utils.math import yaw_quat


def ray_hits_in_base_frame(
    sensor: Any,
    asset: Any,
    grid_shape: tuple[int, int],
    invalid_height: float = -1.2,
    offset: float = 0.5,
) -> torch.Tensor:
    """Return yaw-frame scan points carrying Isaac Lab height values in Z.

    This is the canonical preprocessing for the actor's raw height map. The
    spatial edge map uses the same height convention. XY is expressed relative
    to the robot root using yaw only. Z follows Isaac Lab's official
    convention: ``sensor_z - ray_hit_z - offset``.
    """

    ray_hits_w = sensor.data.ray_hits_w
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3:
        raise ValueError(f"Expected ray hits shaped (N, R, 3), got {tuple(ray_hits_w.shape)}")

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    sensor_pos_w = sensor.data.pos_w
    if root_pos_w.shape != (ray_hits_w.shape[0], 3):
        raise ValueError("Robot root positions do not match the RayCaster batch size.")
    if root_quat_w.shape != (ray_hits_w.shape[0], 4):
        raise ValueError("Robot root quaternions do not match the RayCaster batch size.")
    if sensor_pos_w.shape != (ray_hits_w.shape[0], 3):
        raise ValueError("RayCaster positions do not match the ray-hit batch size.")

    # 相对于root的坐标系，只处理yaw
    relative_hits_w = ray_hits_w - root_pos_w.unsqueeze(1)
    root_yaw_quat_w = yaw_quat(root_quat_w)
    expanded_root_yaw_quat_w = root_yaw_quat_w.unsqueeze(1).expand(-1, ray_hits_w.shape[1], -1)
    ray_hits_b = quat_apply_inverse(expanded_root_yaw_quat_w, relative_hits_w)

    # Match isaaclab.envs.mdp.height_scan exactly. The RayCaster's configured
    # +20 m start offset is deliberately absent because data.pos_w is the
    # attached link position, as in the official observation function.
    # height = sensor_pos_w[:, 2].unsqueeze(1) - ray_hits_w[..., 2] - float(offset)
    # 处理nan无效点，设置为invalid_height
    invalid_points = ~torch.isfinite(ray_hits_b).all(dim=-1)
    ray_hits_b = torch.nan_to_num(ray_hits_b, nan=0.0, posinf=0.0, neginf=0.0)
    ray_hits_b[..., 2] = torch.where(
        invalid_points,
        torch.full_like(ray_hits_b[..., 2], float(invalid_height)),
        ray_hits_b[..., 2],
    )

    expected_points = int(grid_shape[0]) * int(grid_shape[1])
    if ray_hits_b.shape[1] != expected_points:
        raise RuntimeError(
            f"Expected {expected_points} map points for grid {grid_shape}, got {ray_hits_b.shape[1]}"
        )
    return ray_hits_b
