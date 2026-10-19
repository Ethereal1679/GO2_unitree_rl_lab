from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


def _split_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` by weights with an exact integer sum."""
    weight_sum = sum(weights)
    if total < 0 or not weights or weight_sum <= 0.0 or any(weight < 0.0 for weight in weights):
        raise ValueError("total must be non-negative and weights must have a positive sum")
    raw = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _env_ids(env: Any, env_ids: Sequence[int] | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()


class _ProgressiveState:
    """Small per-environment state kept by the curriculum term."""

    def __init__(
        self,
        env: Any,
        terrain_groups: Sequence[Sequence[str]],
        level_threshold: float,
        check_interval_iterations: int,
        steps_per_iteration: int,
        release_fraction: float,
        min_remaining_fraction: float,
    ):
        terrain = env.scene.terrain
        generator = getattr(terrain.cfg, "terrain_generator", None)
        if generator is None or terrain.terrain_origins is None:
            raise ValueError("progressive terrain allocation requires generator terrain")

        groups = tuple(tuple(group) for group in terrain_groups)
        names = tuple(name for group in groups for name in group)
        configured = tuple(generator.sub_terrains.keys())
        unknown = sorted(set(names) - set(configured))
        if not groups or not all(groups) or unknown:
            raise ValueError(f"Invalid terrain_groups; unknown terrain(s): {unknown}")
        if len(set(names)) != len(names):
            raise ValueError("Each terrain may occur in only one progressive group")
        if not 0.0 <= level_threshold <= 1.0:
            raise ValueError("level_threshold must be in [0, 1]")
        if check_interval_iterations <= 0 or steps_per_iteration <= 0:
            raise ValueError("check interval and steps per iteration must be positive")
        if not 0.0 < release_fraction <= 1.0 or not 0.0 <= min_remaining_fraction <= 1.0:
            raise ValueError("release_fraction and min_remaining_fraction are invalid")

        proportions = {name: float(cfg.proportion) for name, cfg in generator.sub_terrains.items()}
        total_proportion = sum(proportions.values())
        if total_proportion <= 0.0:
            raise ValueError("terrain proportions must have a positive sum")
        columns: dict[str, list[int]] = {name: [] for name in names}
        cumulative = []
        running = 0.0
        for name in configured:
            running += proportions[name] / total_proportion
            cumulative.append(running)
        for column in range(generator.num_cols):
            position = column / generator.num_cols + 0.001
            terrain_index = next((index for index, upper in enumerate(cumulative) if position < upper), -1)
            if terrain_index >= 0 and configured[terrain_index] in columns:
                columns[configured[terrain_index]].append(column)
        missing_columns = [name for name in names if not columns[name]]
        if missing_columns:
            raise ValueError(f"Selected terrain has no generated column: {missing_columns}")

        self.device = env.device
        self.names = names
        self.name_to_index = {name: index for index, name in enumerate(names)}
        self.groups = tuple(tuple(self.name_to_index[name] for name in group) for group in groups)
        self.group_of = {index: group_index for group_index, group in enumerate(self.groups) for index in group}
        self.columns = columns
        self.column_cursor = [0] * len(names)
        self.target_cursor = [0] * len(groups)
        self.assignment = torch.full((env.num_envs,), -1, dtype=torch.long, device=self.device)
        self.current_counts = [0] * len(names)
        self.initial_counts = [0] * len(names)
        self.pending = [0] * len(names)
        self.level_sum = torch.zeros(len(names), dtype=torch.float32, device=self.device)
        self.level_samples = torch.zeros(len(names), dtype=torch.long, device=self.device)
        self.level_threshold = level_threshold
        self.check_interval_steps = check_interval_iterations * steps_per_iteration
        self.release_fraction = release_fraction
        self.min_remaining_fraction = min_remaining_fraction
        self.last_step = 0
        self.next_check_step = self.check_interval_steps
        self.level_snapshot: torch.Tensor | None = None
        self.assignment_snapshot: torch.Tensor | None = None
        self.metric_values: dict[str, float] = {}
        self.initialized = False

    def _set_terrain_types(self, terrain: Any, env_ids: torch.Tensor, name_ids: torch.Tensor) -> None:
        """Assign generated columns and refresh origins for reset environments."""
        self.assignment[env_ids] = name_ids
        for name_id in torch.unique(name_ids).tolist():
            selected = env_ids[name_ids == name_id]
            available = torch.as_tensor(self.columns[self.names[name_id]], device=self.device, dtype=torch.long)
            offsets = (
                torch.arange(selected.numel(), device=self.device) + self.column_cursor[name_id]
            ) % available.numel()
            column_ids = available[offsets]
            self.column_cursor[name_id] = (self.column_cursor[name_id] + selected.numel()) % available.numel()
            terrain.terrain_types[selected] = column_ids
            terrain.env_origins[selected] = terrain.terrain_origins[terrain.terrain_levels[selected], column_ids]

    def _snapshot(self, terrain: Any, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.level_snapshot = terrain.terrain_levels.clone().float()
            self.assignment_snapshot = self.assignment.clone()
        else:
            assert self.level_snapshot is not None and self.assignment_snapshot is not None
            self.level_snapshot[env_ids] = terrain.terrain_levels[env_ids].float()
            self.assignment_snapshot[env_ids] = self.assignment[env_ids]

    def _accumulate(self, steps: int) -> None:
        assert self.level_snapshot is not None and self.assignment_snapshot is not None
        valid = self.assignment_snapshot >= 0
        names = self.assignment_snapshot[valid]
        self.level_sum.scatter_add_(0, names, self.level_snapshot[valid] * steps)
        self.level_samples.scatter_add_(0, names, torch.ones_like(names) * steps)

    def _check_window(self, terrain: Any) -> None:
        means = torch.full_like(self.level_sum, torch.nan)
        sampled = self.level_samples > 0
        means[sampled] = self.level_sum[sampled] / self.level_samples[sampled]
        max_level = max(1, int(terrain.max_terrain_level) - 1)
        for name in self.names:
            self.metric_values.pop(f"mean_level/{name}", None)
        for source_id, mean in enumerate(means.tolist()):
            group_index = self.group_of[source_id]
            if math.isfinite(mean):
                self.metric_values[f"mean_level/{self.names[source_id]}"] = mean
            if math.isfinite(mean) and mean > self.level_threshold * max_level and group_index < len(self.groups) - 1:
                minimum = math.ceil(self.initial_counts[source_id] * self.min_remaining_fraction)
                available = self.current_counts[source_id] - minimum - self.pending[source_id]
                if available > 0:
                    step = max(1, round(self.initial_counts[source_id] * self.release_fraction))
                    self.pending[source_id] += min(step, available)
            self.metric_values[f"pending/{self.names[source_id]}"] = float(self.pending[source_id])
        self.level_sum.zero_()
        self.level_samples.zero_()

    def advance(self, env: Any, step: int) -> None:
        """Accumulate level means and check complete PPO windows."""
        while self.last_step < step:
            boundary = min(step, self.next_check_step)
            self._accumulate(boundary - self.last_step)
            self.last_step = boundary
            if boundary == self.next_check_step:
                self._check_window(env.scene.terrain)
                self.next_check_step += self.check_interval_steps

    def initialize(self, env: Any, step: int) -> None:
        terrain = env.scene.terrain
        first_group = self.groups[0]
        weights = [
            float(terrain.cfg.terrain_generator.sub_terrains[self.names[index]].proportion)
            for index in first_group
        ]
        counts = _split_counts(env.num_envs, weights)
        name_ids = torch.repeat_interleave(
            torch.as_tensor(first_group, device=self.device, dtype=torch.long),
            torch.as_tensor(counts, device=self.device),
        )
        env_ids = torch.arange(env.num_envs, device=self.device)
        self._set_terrain_types(terrain, env_ids, name_ids)
        self.current_counts = counts + [0] * (len(self.names) - len(counts))
        self.initial_counts = self.current_counts.copy()
        self.metric_values.update(
            {f"env_count/{name}": float(self.current_counts[i]) for i, name in enumerate(self.names)}
        )
        self.last_step = step
        self.next_check_step = step + self.check_interval_steps
        self._snapshot(terrain)
        self.initialized = True

    def _target_names(self, group_index: int, count: int) -> torch.Tensor:
        targets = self.groups[group_index]
        start = self.target_cursor[group_index]
        values = [targets[(start + offset) % len(targets)] for offset in range(count)]
        self.target_cursor[group_index] = (start + count) % len(targets)
        return torch.as_tensor(values, device=self.device, dtype=torch.long)

    def process_pending(self, env: Any, reset_ids: torch.Tensor) -> None:
        """Move pending slots only when their source environments are reset."""
        if reset_ids.numel() == 0:
            return
        terrain = env.scene.terrain
        available = torch.ones(reset_ids.numel(), dtype=torch.bool, device=self.device)
        for source_id, amount in enumerate(self.pending):
            if amount <= 0 or self.group_of[source_id] == len(self.groups) - 1:
                continue
            source_mask = (self.assignment[reset_ids] == source_id) & available
            source_ids = reset_ids[source_mask]
            move_count = min(amount, source_ids.numel())
            if move_count == 0:
                continue
            chosen = source_ids[:move_count]
            chosen_mask = torch.nonzero(source_mask, as_tuple=True)[0][:move_count]
            available[chosen_mask] = False
            target_ids = self._target_names(self.group_of[source_id] + 1, move_count)
            # Start transferred environments at the easiest row of their new terrain.
            terrain.terrain_levels[chosen] = 0
            self._set_terrain_types(terrain, chosen, target_ids)
            self.current_counts[source_id] -= move_count
            self.pending[source_id] -= move_count
            self.metric_values[f"env_count/{self.names[source_id]}"] = float(self.current_counts[source_id])
            self.metric_values[f"pending/{self.names[source_id]}"] = float(self.pending[source_id])
            for target_id in torch.unique(target_ids).tolist():
                added = int((target_ids == target_id).sum().item())
                if self.initial_counts[target_id] == 0:
                    self.initial_counts[target_id] = added
                self.current_counts[target_id] += added
                self.metric_values[f"env_count/{self.names[target_id]}"] = float(self.current_counts[target_id])

    def metrics(self) -> dict[str, float]:
        """Expose allocation and level statistics to TensorBoard."""
        return self.metric_values.copy()

def progressive_terrain_env_allocation(
    env: Any,
    env_ids: Sequence[int] | slice,
    terrain_groups: Sequence[Sequence[str]],
    level_threshold: float = 0.9,
    check_interval_iterations: int = 2000,
    steps_per_iteration: int = 24,
    release_fraction: float = 0.2,
    min_remaining_fraction: float = 0.5,
) -> dict[str, float]:
    """Allocate reset environments across progressive terrain groups."""
    terrain = env.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        return {}
    state = getattr(env, "_progressive_terrain_state", None)
    if state is None:
        state = _ProgressiveState(
            env,
            terrain_groups,
            level_threshold,
            check_interval_iterations,
            steps_per_iteration,
            release_fraction,
            min_remaining_fraction,
        )
        env._progressive_terrain_state = state
    reset_ids = _env_ids(env, env_ids)
    current_step = int(getattr(env, "common_step_counter", 0))
    if not state.initialized:
        state.initialize(env, current_step)
        return state.metrics()
    state.advance(env, current_step)
    state.process_pending(env, reset_ids)
    state._snapshot(terrain, reset_ids)
    return state.metrics()
