from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..contracts import PlannerCommand, RuntimeStamp
from ..navigation_intelligence import Pose2DState


COGNITIVE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class OccupancyGridSpec:
    resolution_m: float
    x_min_m: float
    y_min_m: float
    frame_id: str

    def validate(self) -> None:
        if self.resolution_m <= 0.0 or not math.isfinite(self.resolution_m):
            raise ValueError("grid resolution must be finite and positive")
        if not self.frame_id:
            raise ValueError("grid frame_id is required")

    def xy_to_grid(self, x_m: float, y_m: float, shape: tuple[int, int]) -> tuple[int, int] | None:
        col = int(math.floor((x_m - self.x_min_m) / self.resolution_m))
        row = int(math.floor((y_m - self.y_min_m) / self.resolution_m))
        if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
            return None
        return row, col


@dataclass(frozen=True)
class SemanticMemoryObject:
    object_id: int
    label: str
    confidence: float
    centroid_world_xyz: np.ndarray
    extent_xyz: np.ndarray
    observations: int = 1

    def validate(self) -> None:
        centroid = np.asarray(self.centroid_world_xyz, dtype=np.float32).reshape(-1)
        extent = np.asarray(self.extent_xyz, dtype=np.float32).reshape(-1)
        if centroid.size != 3 or extent.size != 3:
            raise ValueError("semantic centroid and extent must have three elements")
        if not np.isfinite(centroid).all() or not np.isfinite(extent).all():
            raise ValueError("semantic object geometry must be finite")
        if not 0.0 <= self.confidence <= 1.0 or self.observations <= 0:
            raise ValueError("semantic confidence and observations are invalid")


@dataclass(frozen=True)
class CognitiveObservation:
    """Versioned snapshot presented to a cognitive policy.

    Arrays are deliberately explicit so live inference and offline training use
    the same frame and shape contracts.
    """

    stamp: RuntimeStamp
    pose: Pose2DState
    goal_world_xy: np.ndarray
    local_occupancy: np.ndarray
    global_occupancy: np.ndarray
    local_grid_spec: OccupancyGridSpec
    global_grid_spec: OccupancyGridSpec
    route_world_xy: np.ndarray
    semantic_objects: tuple[SemanticMemoryObject, ...] = ()
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    previous_command: PlannerCommand | None = None
    localization_uncertainty: float = 0.0
    sensor_ages_sec: dict[str, float] = field(default_factory=dict)
    schema_version: str = COGNITIVE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != COGNITIVE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cognitive schema {self.schema_version}")
        goal = np.asarray(self.goal_world_xy, dtype=np.float32).reshape(-1)
        route = np.asarray(self.route_world_xy, dtype=np.float32)
        local = np.asarray(self.local_occupancy)
        global_map = np.asarray(self.global_occupancy)
        if goal.size != 2 or not np.isfinite(goal).all():
            raise ValueError("goal_world_xy must contain two finite values")
        if route.ndim != 2 or route.shape[1] != 2 or not np.isfinite(route).all():
            raise ValueError("route_world_xy must have shape [N,2] with finite values")
        if local.ndim != 2 or global_map.ndim != 2:
            raise ValueError("occupancy maps must be two-dimensional")
        if local.size == 0 or global_map.size == 0:
            raise ValueError("occupancy maps must not be empty")
        self.local_grid_spec.validate()
        self.global_grid_spec.validate()
        if self.rgb is not None:
            rgb = np.asarray(self.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("rgb must have shape [H,W,3]")
        if self.depth_m is not None and np.asarray(self.depth_m).ndim != 2:
            raise ValueError("depth_m must have shape [H,W]")
        if not math.isfinite(self.localization_uncertainty) or self.localization_uncertainty < 0.0:
            raise ValueError("localization uncertainty must be finite and non-negative")
        if any(not math.isfinite(age) or age < 0.0 for age in self.sensor_ages_sec.values()):
            raise ValueError("sensor ages must be finite and non-negative")
        for semantic_object in self.semantic_objects:
            semantic_object.validate()

    def goal_robot_xy(self) -> np.ndarray:
        dx = float(self.goal_world_xy[0]) - self.pose.x_m
        dy = float(self.goal_world_xy[1]) - self.pose.y_m
        c = math.cos(-self.pose.yaw_rad)
        s = math.sin(-self.pose.yaw_rad)
        return np.asarray([c * dx - s * dy, s * dx + c * dy], dtype=np.float32)


@dataclass(frozen=True)
class CognitiveBeliefState:
    stamp: RuntimeStamp
    goal_robot_xy: np.ndarray
    working_memory: np.ndarray
    recalled_episode_ids: tuple[str, ...]
    localization_uncertainty: float
    observation_count: int

    def validate(self) -> None:
        if np.asarray(self.goal_robot_xy).shape != (2,):
            raise ValueError("belief goal_robot_xy must have shape [2]")
        memory = np.asarray(self.working_memory)
        if memory.ndim != 1 or not np.isfinite(memory).all():
            raise ValueError("working memory must be a finite vector")
        if self.observation_count <= 0:
            raise ValueError("observation_count must be positive")


@dataclass(frozen=True)
class CandidateTrajectory:
    trajectory_id: str
    points_robot_xy: np.ndarray
    predicted_progress_m: float
    collision_probability: float
    exploration_value: float
    uncertainty: float
    score: float
    source: str

    def validate(self) -> None:
        points = np.asarray(self.points_robot_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("candidate trajectory must have shape [H,2]")
        if not np.isfinite(points).all():
            raise ValueError("candidate trajectory points must be finite")
        scalars = (
            self.predicted_progress_m,
            self.collision_probability,
            self.exploration_value,
            self.uncertainty,
            self.score,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("candidate metrics must be finite")
        if not 0.0 <= self.collision_probability <= 1.0 or self.uncertainty < 0.0:
            raise ValueError("candidate risk or uncertainty is invalid")


@dataclass(frozen=True)
class CognitiveDecision:
    stamp: RuntimeStamp
    selected_trajectory_id: str | None
    candidates: tuple[CandidateTrajectory, ...]
    confidence: float
    reason: str
    fallback_requested: bool
    policy_version: str

    @property
    def selected(self) -> CandidateTrajectory | None:
        return next(
            (item for item in self.candidates if item.trajectory_id == self.selected_trajectory_id),
            None,
        )

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("decision confidence must be within [0,1]")
        for candidate in self.candidates:
            candidate.validate()
        if self.selected_trajectory_id is not None and self.selected is None:
            raise ValueError("selected trajectory is not present in candidates")
        if not self.fallback_requested and self.selected is None:
            raise ValueError("a non-fallback decision must select a trajectory")


@dataclass(frozen=True)
class CognitiveOutcome:
    trajectory_id: str
    start_time_sec: float
    end_time_sec: float
    actual_progress_m: float
    collision: bool
    safety_override: bool
    final_goal_distance_m: float

    def prediction_error(self, candidate: CandidateTrajectory) -> float:
        progress_error = abs(candidate.predicted_progress_m - self.actual_progress_m)
        collision_error = abs(candidate.collision_probability - float(self.collision))
        return float(progress_error + collision_error)
