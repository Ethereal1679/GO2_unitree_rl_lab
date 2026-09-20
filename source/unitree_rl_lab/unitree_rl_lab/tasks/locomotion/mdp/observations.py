from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp as isaaclab_mdp

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
