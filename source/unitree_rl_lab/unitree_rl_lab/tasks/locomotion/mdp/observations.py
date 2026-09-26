from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp as isaaclab_mdp

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def height_scan_with_delay(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    max_delay_frames: int = 2,
) -> torch.Tensor:
    """Return a height scan with a per-environment randomized observation delay.

    The delay is sampled uniformly from ``[0, max_delay_frames]`` when an
    environment is reset and remains fixed until its next reset.  The history
    is maintained in control/observation steps, so a delay of one means that
    the previous height-scan observation is returned.

    Args:
        env: The manager-based environment.
        sensor_cfg: Scene entity configuration passed to ``mdp.height_scan``.
        max_delay_frames: Maximum integer delay in observation steps.

    Returns:
        Delayed height-scan tensor with the same shape as ``mdp.height_scan``.
    """
    max_delay_frames = int(max_delay_frames)
    if max_delay_frames < 0:
        raise ValueError(f"max_delay_frames must be non-negative, got {max_delay_frames}.")

    current_scan = isaaclab_mdp.height_scan(env, sensor_cfg=sensor_cfg)
    if current_scan.ndim == 1:
        current_scan = current_scan.unsqueeze(0)

    state = getattr(env, "_height_scan_delay_state", None)
    expected_buffer_shape = (max_delay_frames + 1, *current_scan.shape)
    if (
        state is None
        or state["buffer"].shape != expected_buffer_shape
        or state["buffer"].device != current_scan.device
        or state["max_delay_frames"] != max_delay_frames
    ):
        delay_frames = torch.randint(
            0,
            max_delay_frames + 1,
            (current_scan.shape[0],),
            device=current_scan.device,
        )
        state = {
            "buffer": current_scan.unsqueeze(0).repeat(
                max_delay_frames + 1, *([1] * current_scan.ndim)
            ),
            "delay_frames": delay_frames,
            "max_delay_frames": max_delay_frames,
        }
        setattr(env, "_height_scan_delay_state", state)
    else:
        # Each environment can reset independently in a vectorized environment.
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            reset_ids = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
            state["buffer"][:, reset_ids] = current_scan[reset_ids].unsqueeze(0)
            state["delay_frames"][reset_ids] = torch.randint(
                0,
                max_delay_frames + 1,
                (reset_ids.numel(),),
                device=current_scan.device,
            )

        # Newest scan is at index 0; index 1 is one observation step old, etc.
        state["buffer"][1:] = state["buffer"][:-1].clone()
        state["buffer"][0] = current_scan

    env_ids = torch.arange(current_scan.shape[0], device=current_scan.device)
    return state["buffer"][state["delay_frames"], env_ids]


def map_scan_points(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    asset_cfg,
    grid_shape: tuple[int, int] = (16, 11),
    noise: bool = False,
) -> torch.Tensor:
    """Return ray-hit XYZ points in base frame as a flat observation tail.

    When ``noise`` is enabled, Gaussian height noise (3 cm standard deviation)
    and a per-environment height offset (resampled in the range +/-5 cm on
    reset) are applied before flattening.  The offset is kept on ``env`` so it
    remains fixed throughout an episode.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w
    relative_hits_w = ray_hits_w - asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, ray_hits_w.shape[1], -1)
    ray_hits_b = quat_apply_inverse(root_quat_w, relative_hits_w)
    # 命中点现在会被替换成有限坐标，并将高度设为扫描下限
    invalid_points = ~torch.isfinite(ray_hits_b).all(dim=-1)
    ray_hits_b = torch.nan_to_num(ray_hits_b, nan=0.0, posinf=0.0, neginf=0.0)
    ray_hits_b[..., 2] = torch.where(invalid_points, torch.full_like(ray_hits_b[..., 2], -1.2), ray_hits_b[..., 2])
    expected_points = grid_shape[0] * grid_shape[1]
    if ray_hits_b.shape[1] != expected_points:
        raise RuntimeError(f"Expected {expected_points} map points for grid {grid_shape}, got {ray_hits_b.shape[1]}")

    if noise:
        num_envs = ray_hits_b.shape[0]
        offset = getattr(env, "_map_scan_points_offset", None)
        if (
            offset is None
            or offset.shape != (num_envs, 1)
            or offset.device != ray_hits_b.device
        ):
            offset = torch.zeros((num_envs, 1), device=ray_hits_b.device)
            setattr(env, "_map_scan_points_offset", offset)

        # Resample the sensor initialization error independently per reset env.
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                offset[reset_env_ids] = torch.rand(
                    (reset_env_ids.numel(), 1), device=ray_hits_b.device
                ) * 0.1 - 0.05 # 每个环境在 reset 时重采样 ±5 cm 的整体高度偏移；

        # Z 坐标加入标准差 3 cm 的高斯噪声
        ray_hits_b = ray_hits_b.clone()
        ray_hits_b[..., 2] += torch.randn_like(ray_hits_b[..., 2]) * 0.03
        ray_hits_b[..., 2] += offset
        ray_hits_b[..., 2] = torch.clamp(ray_hits_b[..., 2], min=-1.2, max=0.0) # Z 值限制在 [-1.2, 0.0]；
    return ray_hits_b.reshape(ray_hits_b.shape[0], -1)
