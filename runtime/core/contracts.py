from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RuntimeStamp:
    """Common stamp for data passed between runtime layers."""

    time_sec: float
    frame_id: str
    source: str


@dataclass(frozen=True)
class ObstacleObservation:
    """Obstacle estimate in a declared frame, not a raw segmentation mask."""

    stamp: RuntimeStamp
    label: str
    confidence: float
    centroid_m: np.ndarray
    extent_m: np.ndarray
    covariance: np.ndarray | None = None
    observation_count: int = 1

    def is_valid(self) -> bool:
        centroid = np.asarray(self.centroid_m, dtype=np.float32).reshape(-1)
        extent = np.asarray(self.extent_m, dtype=np.float32).reshape(-1)
        if centroid.size < 2 or extent.size < 2:
            return False
        if not np.isfinite(centroid).all() or not np.isfinite(extent).all():
            return False
        return self.confidence > 0.0 and self.observation_count > 0


@dataclass
class LocalObstacleMap:
    """Planner-facing obstacle map with explicit freshness and frame contract."""

    stamp: RuntimeStamp
    obstacles: list[ObstacleObservation] = field(default_factory=list)
    max_age_sec: float = 1.0

    def fresh_obstacles(self, now_sec: float) -> list[ObstacleObservation]:
        return [
            obstacle
            for obstacle in self.obstacles
            if obstacle.is_valid() and now_sec - obstacle.stamp.time_sec <= self.max_age_sec
        ]


@dataclass(frozen=True)
class PlannerCommand:
    """Controller command chosen by a planner with enough context to audit it."""

    stamp: RuntimeStamp
    linear_x: float
    angular_z: float
    mode: str
    reason: str
    source: Literal["planner", "recovery", "manual", "safety_stop"] = "planner"

    def clipped(self, max_linear: float, max_angular: float) -> "PlannerCommand":
        return PlannerCommand(
            stamp=self.stamp,
            linear_x=float(np.clip(self.linear_x, -max_linear, max_linear)),
            angular_z=float(np.clip(self.angular_z, -max_angular, max_angular)),
            mode=self.mode,
            reason=self.reason,
            source=self.source,
        )
