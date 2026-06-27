from __future__ import annotations

import math

import numpy as np

from .contracts import CandidateTrajectory, CognitiveDecision, CognitiveObservation


class DeterministicCognitiveTeacher:
    """Converts the verified global route into an auditable imitation target."""

    policy_version = "deterministic_teacher_v1"

    def decide(self, observation: CognitiveObservation, horizon: int = 5) -> CognitiveDecision:
        observation.validate()
        route = np.asarray(observation.route_world_xy, dtype=np.float32)
        if len(route) < 2:
            return CognitiveDecision(
                observation.stamp, None, (), 0.0, "global route unavailable", True, self.policy_version
            )

        sampled = _sample_polyline(_world_route_to_robot(route, observation), horizon)
        collision = _trajectory_collision(sampled, observation)
        goal = observation.goal_robot_xy()
        progress = float(np.linalg.norm(goal) - np.linalg.norm(goal - sampled[-1]))
        candidate = CandidateTrajectory(
            trajectory_id="teacher_global_route",
            points_robot_xy=sampled,
            predicted_progress_m=progress,
            collision_probability=1.0 if collision else 0.0,
            exploration_value=0.0,
            uncertainty=min(1.0, observation.localization_uncertainty),
            score=-progress + (10.0 if collision else 0.0),
            source=self.policy_version,
        )
        decision = CognitiveDecision(
            stamp=observation.stamp,
            selected_trajectory_id=None if collision else candidate.trajectory_id,
            candidates=(candidate,),
            confidence=max(0.0, 1.0 - candidate.uncertainty),
            reason="teacher route intersects local occupancy" if collision else "teacher follows global route",
            fallback_requested=collision,
            policy_version=self.policy_version,
        )
        decision.validate()
        return decision


def _world_route_to_robot(route: np.ndarray, observation: CognitiveObservation) -> np.ndarray:
    delta = route - np.asarray([observation.pose.x_m, observation.pose.y_m], dtype=np.float32)
    c = math.cos(-observation.pose.yaw_rad)
    s = math.sin(-observation.pose.yaw_rad)
    return np.stack(
        [c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1]], axis=1
    )


def _sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("trajectory horizon must be positive")
    distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if distances[-1] <= 1e-6:
        return np.repeat(points[-1:], count, axis=0).astype(np.float32)
    samples = np.linspace(0.0, min(float(distances[-1]), 2.5), count + 1)[1:]
    return np.stack(
        [np.interp(samples, distances, points[:, 0]), np.interp(samples, distances, points[:, 1])],
        axis=1,
    ).astype(np.float32)


def _trajectory_collision(points: np.ndarray, observation: CognitiveObservation) -> bool:
    grid = np.asarray(observation.local_occupancy) > 0.5
    for point in points:
        cell = observation.local_grid_spec.xy_to_grid(float(point[0]), float(point[1]), grid.shape)
        if cell is not None and bool(grid[cell]):
            return True
    return False
