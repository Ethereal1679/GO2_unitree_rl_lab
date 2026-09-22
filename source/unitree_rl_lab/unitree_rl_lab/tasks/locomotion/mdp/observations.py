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


def go2_proprioception(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Return the Go2 proprioception vector in the attention-policy order.

    The returned order is command (3), base linear/angular velocity (6),
    projected gravity (3), relative joint position (12), relative joint
    velocity (12), and previous action (12).
    """

    command = isaaclab_mdp.generated_commands(env, command_name=command_name)
    base_lin_vel = isaaclab_mdp.base_lin_vel(env)
    base_ang_vel = isaaclab_mdp.base_ang_vel(env)
    projected_gravity = isaaclab_mdp.projected_gravity(env)
    joint_pos_rel = isaaclab_mdp.joint_pos_rel(env)
    joint_vel_rel = isaaclab_mdp.joint_vel_rel(env)
    last_action = isaaclab_mdp.last_action(env)
    proprioception = torch.cat(
        (
            command,
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos_rel,
            joint_vel_rel,
            last_action,
        ),
        dim=-1,
    )
    if proprioception.shape[-1] != 48:
        raise RuntimeError(f"Go2 proprioception must have 48 values, got {proprioception.shape[-1]}")
    return proprioception


def map_scan_points(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    asset_cfg,
    grid_shape: tuple[int, int] = (26, 16),
) -> torch.Tensor:
    """Return ray-hit points in the robot base frame as ``[B, 26, 16, 3]``."""

    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w
    relative_hits_w = ray_hits_w - asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, ray_hits_w.shape[1], -1)
    ray_hits_b = quat_apply_inverse(root_quat_w, relative_hits_w)
    expected_points = grid_shape[0] * grid_shape[1]
    if ray_hits_b.shape[1] != expected_points:
        raise RuntimeError(
            f"Expected {expected_points} map points for grid {grid_shape}, got {ray_hits_b.shape[1]}"
        )
    return ray_hits_b.reshape(ray_hits_b.shape[0], grid_shape[0], grid_shape[1], 3)
