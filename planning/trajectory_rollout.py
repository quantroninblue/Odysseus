from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from runtime.core.contracts import PlannerCommand, RuntimeStamp
from .local_costmap import LocalCostmap


@dataclass(frozen=True)
class CandidateEvaluation:
    linear_x: float
    angular_z: float
    accepted: bool
    score: float
    reason: str
    min_clearance_m: float


@dataclass(frozen=True)
class RolloutResult:
    command: PlannerCommand
    candidates: list[CandidateEvaluation] = field(default_factory=list)


class TrajectoryRolloutPlanner:
    def __init__(
        self,
        *,
        horizon_sec: float = 1.8,
        dt_sec: float = 0.15,
        max_linear_mps: float = 0.34,
        max_angular_radps: float = 0.85,
    ):
        self.horizon_sec = float(horizon_sec)
        self.dt_sec = float(dt_sec)
        self.max_linear_mps = float(max_linear_mps)
        self.max_angular_radps = float(max_angular_radps)

    def plan(
        self,
        costmap: LocalCostmap,
        *,
        goal_heading_error: float,
        now_sec: float,
        detour_hint: float = 1.0,
        frame_id: str = "base_link",
    ) -> RolloutResult:
        linear_options = [0.0, 0.08, 0.16, 0.24, self.max_linear_mps]
        angular_options = [-0.85, -0.55, -0.30, -0.12, 0.0, 0.12, 0.30, 0.55, 0.85]
        evaluations: list[CandidateEvaluation] = []
        goal = float(np.clip(goal_heading_error, -1.2, 1.2))
        detour = 1.0 if detour_hint >= 0.0 else -1.0
        corridor_clearance = self._forward_corridor_clearance(costmap)

        for linear in linear_options:
            for angular in angular_options:
                angular = float(np.clip(angular, -self.max_angular_radps, self.max_angular_radps))
                evaluation = self._evaluate(costmap, float(linear), angular, goal, detour, corridor_clearance)
                evaluations.append(evaluation)

        accepted = [item for item in evaluations if item.accepted]
        if accepted:
            best = min(accepted, key=lambda item: item.score)
            reason = f"selected score={best.score:.3f} clearance={best.min_clearance_m:.2f}"
            mode = "rollout_drive" if best.linear_x > 0.02 else "rollout_turn"
            command = PlannerCommand(
                stamp=RuntimeStamp(now_sec, frame_id, "trajectory_rollout"),
                linear_x=best.linear_x,
                angular_z=best.angular_z,
                mode=mode,
                reason=reason,
            ).clipped(self.max_linear_mps, self.max_angular_radps)
            return RolloutResult(command=command, candidates=evaluations)

        angular = 0.55 * detour
        command = PlannerCommand(
            stamp=RuntimeStamp(now_sec, frame_id, "trajectory_rollout"),
            linear_x=0.0,
            angular_z=angular,
            mode="rollout_safety_turn",
            reason="all forward rollouts collide",
            source="safety_stop",
        ).clipped(self.max_linear_mps, self.max_angular_radps)
        return RolloutResult(command=command, candidates=evaluations)

    def _evaluate(
        self,
        costmap: LocalCostmap,
        linear: float,
        angular: float,
        goal_heading_error: float,
        detour_hint: float,
        corridor_clearance: float,
    ) -> CandidateEvaluation:
        x = 0.0
        y = 0.0
        yaw = 0.0
        min_clearance = costmap.clearance_m(x, y)
        steps = max(1, int(self.horizon_sec / self.dt_sec))
        collision = False
        for _ in range(steps):
            x += linear * math.cos(yaw) * self.dt_sec
            y += linear * math.sin(yaw) * self.dt_sec
            yaw += angular * self.dt_sec
            min_clearance = min(min_clearance, costmap.clearance_m(x, y))
            if x > 0.05 and costmap.is_occupied(x, y):
                collision = True
                break

        if collision:
            return CandidateEvaluation(linear, angular, False, float("inf"), "collision", min_clearance)

        target_progress = x * math.cos(goal_heading_error) + y * math.sin(goal_heading_error)
        final_heading_error = abs(math.atan2(math.sin(goal_heading_error - yaw), math.cos(goal_heading_error - yaw)))
        clearance_penalty = 0.55 / max(min_clearance, 0.20)
        turn_cost = 0.10 * abs(angular)
        detour_cost = 0.04 * abs(angular - 0.35 * detour_hint)
        blocked_ahead = corridor_clearance < 2.2
        stop_cost = 0.45 if linear <= 0.01 and min_clearance > 0.9 else 0.0
        if linear <= 0.01 and blocked_ahead:
            stop_cost += 1.4
            if abs(angular) < 0.10:
                stop_cost += 2.0
        straight_hazard_cost = 0.0
        if blocked_ahead and linear > 0.02 and abs(angular) < 0.20:
            straight_hazard_cost = 1.2
        active_bypass_reward = 0.38 * abs(angular) if blocked_ahead and abs(angular) > 0.20 else 0.0
        speed_reward = 0.12 * linear
        score = (
            -2.0 * target_progress
            + 0.65 * final_heading_error
            + clearance_penalty
            + turn_cost
            + detour_cost
            + stop_cost
            + straight_hazard_cost
            - active_bypass_reward
            - speed_reward
        )
        return CandidateEvaluation(linear, angular, True, float(score), "ok", min_clearance)

    def _forward_corridor_clearance(self, costmap: LocalCostmap) -> float:
        if costmap.raw_points_xy.size == 0:
            return 8.0
        points = costmap.raw_points_xy
        corridor = points[(points[:, 0] > 0.15) & (points[:, 0] < 3.0) & (np.abs(points[:, 1]) < 0.62)]
        if corridor.size == 0:
            return 8.0
        return float(np.min(corridor[:, 0]))
