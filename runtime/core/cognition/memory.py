from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import CognitiveBeliefState, CognitiveObservation, CognitiveOutcome


@dataclass(frozen=True)
class EpisodicMemoryEntry:
    episode_id: str
    context: np.ndarray
    outcome: CognitiveOutcome
    summary: str

    def validate(self) -> None:
        context = np.asarray(self.context, dtype=np.float32)
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if context.ndim != 1 or context.size == 0 or not np.isfinite(context).all():
            raise ValueError("episodic context must be a non-empty finite vector")


@dataclass
class CognitiveMemory:
    """Bounded working and episodic memory for the cognitive loop."""

    working_size: int = 16
    max_episodes: int = 512
    working_decay: float = 0.85
    _working: np.ndarray = field(init=False)
    _episodes: list[EpisodicMemoryEntry] = field(default_factory=list)
    _observation_count: int = 0

    def __post_init__(self) -> None:
        if self.working_size <= 0 or self.max_episodes <= 0:
            raise ValueError("memory sizes must be positive")
        if not 0.0 <= self.working_decay < 1.0:
            raise ValueError("working_decay must be within [0,1)")
        self._working = np.zeros((self.working_size,), dtype=np.float32)

    @property
    def episodes(self) -> tuple[EpisodicMemoryEntry, ...]:
        return tuple(self._episodes)

    def observe(self, observation: CognitiveObservation, recall_count: int = 4) -> CognitiveBeliefState:
        observation.validate()
        context = observation_context_vector(observation, self.working_size)
        self._working = (
            self.working_decay * self._working + (1.0 - self.working_decay) * context
        ).astype(np.float32)
        self._observation_count += 1
        recalled = self.recall(context, limit=recall_count)
        belief = CognitiveBeliefState(
            stamp=observation.stamp,
            goal_robot_xy=observation.goal_robot_xy(),
            working_memory=self._working.copy(),
            recalled_episode_ids=tuple(item.episode_id for item in recalled),
            localization_uncertainty=observation.localization_uncertainty,
            observation_count=self._observation_count,
        )
        belief.validate()
        return belief

    def remember(self, entry: EpisodicMemoryEntry) -> None:
        entry.validate()
        self._episodes.append(entry)
        if len(self._episodes) > self.max_episodes:
            del self._episodes[: len(self._episodes) - self.max_episodes]

    def recall(self, context: np.ndarray, limit: int = 4) -> tuple[EpisodicMemoryEntry, ...]:
        if limit <= 0:
            return ()
        query = np.asarray(context, dtype=np.float32).reshape(-1)
        ranked: list[tuple[float, EpisodicMemoryEntry]] = []
        for entry in self._episodes:
            candidate = np.asarray(entry.context, dtype=np.float32).reshape(-1)
            if candidate.shape != query.shape:
                continue
            denominator = float(np.linalg.norm(query) * np.linalg.norm(candidate))
            similarity = float(np.dot(query, candidate) / denominator) if denominator > 1e-8 else 0.0
            ranked.append((similarity, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return tuple(entry for _, entry in ranked[:limit])


def observation_context_vector(observation: CognitiveObservation, size: int = 16) -> np.ndarray:
    goal = observation.goal_robot_xy()
    local = np.asarray(observation.local_occupancy, dtype=np.float32)
    global_map = np.asarray(observation.global_occupancy, dtype=np.float32)
    route = np.asarray(observation.route_world_xy, dtype=np.float32)
    route_length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()) if len(route) > 1 else 0.0
    command = observation.previous_command
    base = np.asarray(
        [
            goal[0], goal[1], np.linalg.norm(goal), observation.localization_uncertainty,
            np.mean(local > 0.5), np.mean(global_map > 0.5), route_length, len(route),
            len(observation.semantic_objects), command.linear_x if command else 0.0,
            command.angular_z if command else 0.0,
            max(observation.sensor_ages_sec.values(), default=0.0),
            np.sin(observation.pose.yaw_rad), np.cos(observation.pose.yaw_rad),
            observation.pose.x_m, observation.pose.y_m,
        ],
        dtype=np.float32,
    )
    return base[:size].copy() if size <= len(base) else np.pad(base, (0, size - len(base))).astype(np.float32)
