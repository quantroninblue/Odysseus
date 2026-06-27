from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from planning.trajectory_rollout import CandidateEvaluation
from runtime.core.cognition import CognitiveObservation
from runtime.core.contracts import PlannerCommand, RuntimeStamp

from .contracts import OdysseusOutcome, RolloutCandidateRecord, build_causal_feature_vector
from .shadow import OdysseusShadowRunner
from .world_memory import OdysseusWorldMemory


@dataclass(frozen=True)
class OdysseusNavigationDecision:
    command: PlannerCommand
    mode: str
    reason: str
    trace_id: str
    selected_candidate_id: str | None
    closed_sample_path: str = ""


@dataclass
class _ActiveDecision:
    trace_id: str
    time_sec: float
    goal_distance_m: float
    command: PlannerCommand
    selected_candidate_id: str | None
    depth_risk: float
    pose_xy: tuple[float, float]


class OdysseusNavigator:
    """Persistent Odysseus navigation loop.

    Odysseus owns maneuver selection and adaptation over time. The deterministic
    rollout planner supplies the action set and collision filtering; navigation
    intelligence remains the final command-level safety authority downstream.
    """

    policy_version = "odysseus_navigation_v1"

    def __init__(
        self,
        *,
        shadow_runner: OdysseusShadowRunner | None = None,
        world_memory: OdysseusWorldMemory | None = None,
        memory_path: str | None = None,
        autosave_memory: bool = True,
        progress_epsilon_m: float = 0.04,
        blocked_window: int = 5,
    ):
        if blocked_window <= 0:
            raise ValueError("blocked_window must be positive")
        self.shadow_runner = shadow_runner or OdysseusShadowRunner()
        self.world_memory = world_memory or OdysseusWorldMemory.load_or_empty(memory_path)
        self.memory_path = memory_path
        self.autosave_memory = bool(autosave_memory)
        self.progress_epsilon_m = float(progress_epsilon_m)
        self.blocked_window = int(blocked_window)
        self._active: _ActiveDecision | None = None
        self._recent_progress: deque[float] = deque(maxlen=blocked_window)
        self._recent_modes: deque[str] = deque(maxlen=blocked_window)
        self._last_pose_xy: tuple[float, float] | None = None
        self._mode = "advance"
        self._mode_until = 0.0
        self._retrace_sign = 1.0
        self._trace_index = 0
        self.closed_samples = 0

    @property
    def mode(self) -> str:
        return self._mode

    def decide(
        self,
        observation: CognitiveObservation,
        candidates: list[CandidateEvaluation],
        *,
        deterministic_command: PlannerCommand,
        goal_distance_m: float,
        semantic_forward_m: float = 8.0,
        semantic_lateral_m: float = 0.0,
        progress_distance_m: float = 0.0,
        nav_safety_action: str = "ALLOW",
        nav_motion_state: str = "OK",
    ) -> OdysseusNavigationDecision:
        observation.validate()
        now = observation.stamp.time_sec
        closed_sample_path = self._close_previous(
            observation,
            goal_distance_m=goal_distance_m,
            semantic_forward_m=semantic_forward_m,
            nav_safety_action=nav_safety_action,
            nav_motion_state=nav_motion_state,
        )
        records = self._candidate_records(candidates, deterministic_command)
        accepted = [record for record in records if record.accepted]
        if not accepted:
            command = deterministic_command
            trace = self.shadow_runner.observe(
                observation,
                selected_candidate=None,
                candidate_records=tuple(records),
                semantic_forward_m=semantic_forward_m,
                semantic_lateral_m=semantic_lateral_m,
                progress_distance_m=progress_distance_m,
                trace_id=self._new_trace_id(now),
            ).trace
            self._active = _ActiveDecision(
                trace.trace_id,
                now,
                goal_distance_m,
                command,
                None,
                semantic_forward_m,
                (observation.pose.x_m, observation.pose.y_m),
            )
            return OdysseusNavigationDecision(
                command=command,
                mode="safety_turn",
                reason="no accepted rollout candidates; using deterministic safety command",
                trace_id=trace.trace_id,
                selected_candidate_id=None,
                closed_sample_path=closed_sample_path,
            )

        self._update_mode(now, semantic_forward_m, nav_safety_action, nav_motion_state)
        selected = self._select_candidate(accepted, records, observation, goal_distance_m, semantic_forward_m, semantic_lateral_m, progress_distance_m)
        for index, record in enumerate(records):
            records[index] = RolloutCandidateRecord(
                candidate_id=record.candidate_id,
                linear_x=record.linear_x,
                angular_z=record.angular_z,
                accepted=record.accepted,
                deterministic_score=record.deterministic_score,
                min_clearance_m=record.min_clearance_m,
                reason=record.reason,
                selected=record.candidate_id == selected.candidate_id,
            )
        trace = self.shadow_runner.observe(
            observation,
            selected_candidate=selected,
            candidate_records=tuple(records),
            semantic_forward_m=semantic_forward_m,
            semantic_lateral_m=semantic_lateral_m,
            progress_distance_m=progress_distance_m,
            trace_id=self._new_trace_id(now),
        ).trace
        command = PlannerCommand(
            stamp=RuntimeStamp(now, deterministic_command.stamp.frame_id, "odysseus"),
            linear_x=selected.linear_x,
            angular_z=selected.angular_z,
            mode=f"odysseus_{self._mode}",
            reason=self._decision_reason(selected),
            source="planner",
        )
        self._active = _ActiveDecision(
            trace.trace_id,
            now,
            goal_distance_m,
            command,
            selected.candidate_id,
            semantic_forward_m,
            (observation.pose.x_m, observation.pose.y_m),
        )
        self._recent_modes.append(self._mode)
        return OdysseusNavigationDecision(
            command=command,
            mode=self._mode,
            reason=command.reason,
            trace_id=trace.trace_id,
            selected_candidate_id=selected.candidate_id,
            closed_sample_path=closed_sample_path,
        )

    def remembered_no_go_points(self) -> np.ndarray:
        return self.world_memory.remembered_no_go_points()

    def remembered_success_points(self) -> np.ndarray:
        return self.world_memory.remembered_success_points()

    def save_memory(self) -> None:
        if self.memory_path:
            self.world_memory.save(self.memory_path)

    def _candidate_records(
        self,
        candidates: list[CandidateEvaluation],
        deterministic_command: PlannerCommand,
    ) -> list[RolloutCandidateRecord]:
        records = []
        for index, candidate in enumerate(candidates):
            selected = (
                abs(candidate.linear_x - deterministic_command.linear_x) < 1e-5
                and abs(candidate.angular_z - deterministic_command.angular_z) < 1e-5
            )
            records.append(
                RolloutCandidateRecord.from_evaluation(
                    candidate,
                    candidate_id=f"rollout_{index:03d}",
                    selected=selected,
                )
            )
        return records

    def _close_previous(
        self,
        observation: CognitiveObservation,
        *,
        goal_distance_m: float,
        semantic_forward_m: float,
        nav_safety_action: str,
        nav_motion_state: str,
    ) -> str:
        if self._active is None:
            self._last_pose_xy = (observation.pose.x_m, observation.pose.y_m)
            return ""
        progress = self._active.goal_distance_m - goal_distance_m
        self._recent_progress.append(progress)
        pose_delta = math.hypot(
            observation.pose.x_m - self._active.pose_xy[0],
            observation.pose.y_m - self._active.pose_xy[1],
        )
        commanded_forward = self._active.command.linear_x > 0.08
        stuck = commanded_forward and pose_delta < 0.02 and progress < self.progress_epsilon_m
        safety_override = nav_safety_action not in {"ALLOW", ""}
        stale_sensor = "STALE" in nav_motion_state.upper()
        localization_diverged = "DIVERG" in nav_motion_state.upper() or "RELOCALIZE" in nav_safety_action.upper()
        sudden_obstacle = semantic_forward_m < 1.0 and semantic_forward_m + 0.35 < self._active.depth_risk
        collision_proxy = safety_override and semantic_forward_m < 0.75
        cause = self._infer_cause(
            progress=progress,
            stuck=stuck,
            safety_override=safety_override,
            stale_sensor=stale_sensor,
            localization_diverged=localization_diverged,
            sudden_obstacle=sudden_obstacle,
            collision_proxy=collision_proxy,
        )
        severity = float(
            np.clip(
                (0.35 if stuck else 0.0)
                + (0.35 if safety_override else 0.0)
                + (0.25 if sudden_obstacle else 0.0)
                + max(0.0, -progress),
                0.0,
                1.0,
            )
        )
        outcome = OdysseusOutcome(
            progress_delta_m=progress,
            collision=collision_proxy,
            stuck=stuck,
            safety_override=safety_override,
            localization_diverged=localization_diverged,
            stale_sensor=stale_sensor,
            final_goal_distance_m=goal_distance_m,
            failure_cause=cause,
            severity=severity,
        )
        active = self._active
        sample_path = ""
        try:
            result = self.shadow_runner.close_episode(
                active.trace_id,
                outcome,
                metadata={
                    "closed_by": self.policy_version,
                    "mode": active.command.mode,
                    "nav_safety_action": nav_safety_action,
                    "nav_motion_state": nav_motion_state,
                },
            )
            self.closed_samples += 1 if result.sample_path is not None else 0
            sample_path = str(result.sample_path) if result.sample_path is not None else ""
        except KeyError:
            # A previous persistence failure or legacy no-candidate trace must not
            # disable online control. Keep learning through world memory and move on.
            sample_path = ""
        self._last_pose_xy = (observation.pose.x_m, observation.pose.y_m)
        action_bucket = self._action_bucket(active.command.linear_x, active.command.angular_z)
        self.world_memory.record_outcome(
            pose_xy=active.pose_xy,
            command_mode=active.command.mode,
            action_bucket=action_bucket,
            outcome=outcome,
            time_sec=observation.stamp.time_sec,
        )
        if self.autosave_memory and self.memory_path:
            self.world_memory.save(self.memory_path)
        return sample_path

    def _infer_cause(
        self,
        *,
        progress: float,
        stuck: bool,
        safety_override: bool,
        stale_sensor: bool,
        localization_diverged: bool,
        sudden_obstacle: bool,
        collision_proxy: bool,
    ) -> str:
        if stale_sensor:
            return "sensor_stale"
        if localization_diverged:
            return "pose_drift"
        if collision_proxy or sudden_obstacle:
            return "thin_obstacle_missed"
        if stuck:
            return "local_minimum"
        if safety_override:
            return "overconfident_clearance"
        if progress < -0.15:
            return "bad_detour_side"
        return "success"

    def _update_mode(
        self,
        now: float,
        semantic_forward_m: float,
        nav_safety_action: str,
        nav_motion_state: str,
    ) -> None:
        if now < self._mode_until and self._mode in {"retrace", "recover", "explore"}:
            return
        recent = tuple(self._recent_progress)
        poor_progress = len(recent) >= self.blocked_window and sum(value > self.progress_epsilon_m for value in recent) <= 1
        losing_ground = len(recent) >= 2 and sum(value < -0.04 for value in recent[-2:]) >= 2
        safety_trigger = nav_safety_action not in {"ALLOW", ""} or nav_motion_state.upper() not in {"OK", "MOVING", "UNKNOWN"}
        if safety_trigger or semantic_forward_m < 0.85:
            self._mode = "recover"
            self._mode_until = now + 1.2
            return
        if losing_ground:
            self._mode = "retrace"
            self._mode_until = now + 2.0
            self._retrace_sign *= -1.0
            return
        if poor_progress:
            self._mode = "explore"
            self._mode_until = now + 2.5
            self._retrace_sign *= -1.0
            return
        self._mode = "advance"
        self._mode_until = now + 0.4

    def _select_candidate(
        self,
        accepted: list[RolloutCandidateRecord],
        all_records: list[RolloutCandidateRecord],
        observation: CognitiveObservation,
        goal_distance_m: float,
        semantic_forward_m: float,
        semantic_lateral_m: float,
        progress_distance_m: float,
    ) -> RolloutCandidateRecord:
        goal = observation.goal_robot_xy()
        goal_heading = math.atan2(float(goal[1]), float(goal[0])) if np.linalg.norm(goal) > 1e-6 else 0.0

        def score(candidate: RolloutCandidateRecord) -> float:
            progress = candidate.linear_x * math.cos(goal_heading) - 0.08 * abs(candidate.angular_z)
            clearance_bonus = 0.16 * min(candidate.min_clearance_m, 3.0)
            projected_x, projected_y = self._project_world_xy(observation, candidate)
            memory_bias = self.world_memory.spatial_bias(projected_x, projected_y, radius_m=1.25)
            action_bucket = self._action_bucket(candidate.linear_x, candidate.angular_z)
            behavior_bias = self.world_memory.behavior_value(self._mode, action_bucket)
            behavior_bias += self.world_memory.behavior_value(f"odysseus_{self._mode}", action_bucket)
            base = progress + clearance_bonus - 0.12 * candidate.deterministic_score + 0.85 * memory_bias + 1.20 * behavior_bias
            if self._mode == "advance":
                base += 0.20 * candidate.linear_x - 0.05 * abs(candidate.angular_z)
            elif self._mode == "recover":
                base += 0.28 * abs(candidate.angular_z) - 0.45 * max(candidate.linear_x, 0.0)
            elif self._mode == "retrace":
                base += 0.35 * max(-candidate.linear_x, 0.0) + 0.18 * self._retrace_sign * candidate.angular_z
            elif self._mode == "explore":
                base += 0.30 * self._retrace_sign * candidate.angular_z + 0.08 * candidate.min_clearance_m
            if goal_distance_m < 1.0:
                base -= 0.15 * abs(candidate.angular_z)
            attribution = self._candidate_attribution(
                observation,
                candidate,
                all_records,
                semantic_forward_m,
                semantic_lateral_m,
                progress_distance_m,
            )
            if attribution is not None:
                base += 0.45 * attribution.progress_delta_m
                base -= 1.10 * attribution.collision_probability
                base -= 0.85 * attribution.stuck_probability
                base -= 0.60 * attribution.safety_override_probability
                base -= 0.55 * attribution.severity
            return float(base)

        return max(accepted, key=score)

    def _candidate_attribution(
        self,
        observation: CognitiveObservation,
        candidate: RolloutCandidateRecord,
        all_records: list[RolloutCandidateRecord],
        semantic_forward_m: float,
        semantic_lateral_m: float,
        progress_distance_m: float,
    ):
        if self.shadow_runner.model is None:
            return None
        marked = [
            RolloutCandidateRecord(
                candidate_id=record.candidate_id,
                linear_x=record.linear_x,
                angular_z=record.angular_z,
                accepted=record.accepted,
                deterministic_score=record.deterministic_score,
                min_clearance_m=record.min_clearance_m,
                reason=record.reason,
                selected=record.candidate_id == candidate.candidate_id,
            )
            for record in all_records
        ]
        selected_candidate = next(
            record for record in marked if record.candidate_id == candidate.candidate_id
        )
        features = build_causal_feature_vector(
            observation,
            selected_candidate,
            tuple(marked),
            semantic_forward_m=semantic_forward_m,
            semantic_lateral_m=semantic_lateral_m,
            progress_distance_m=progress_distance_m,
        )
        return self.shadow_runner.predict_attribution(features)

    def _project_world_xy(
        self,
        observation: CognitiveObservation,
        candidate: RolloutCandidateRecord,
        horizon_sec: float = 2.2,
    ) -> tuple[float, float]:
        yaw = observation.pose.yaw_rad + candidate.angular_z * horizon_sec * 0.5
        distance = candidate.linear_x * horizon_sec
        return (
            observation.pose.x_m + distance * math.cos(yaw),
            observation.pose.y_m + distance * math.sin(yaw),
        )

    def _action_bucket(self, linear_x: float, angular_z: float) -> str:
        if linear_x < -0.03:
            speed = "reverse"
        elif linear_x < 0.04:
            speed = "turn"
        elif linear_x < 0.18:
            speed = "slow"
        else:
            speed = "drive"
        if angular_z < -0.25:
            turn = "right"
        elif angular_z > 0.25:
            turn = "left"
        else:
            turn = "straight"
        return f"{speed}_{turn}"

    def _decision_reason(self, selected: RolloutCandidateRecord) -> str:
        return (
            f"Odysseus {self._mode}: selected {selected.candidate_id} "
            f"v={selected.linear_x:.2f} w={selected.angular_z:.2f} "
            f"clearance={selected.min_clearance_m:.2f} score={selected.deterministic_score:.2f}"
        )

    def _new_trace_id(self, now: float) -> str:
        self._trace_index += 1
        return f"odysseus_nav_{now:.3f}_{self._trace_index:08d}"
