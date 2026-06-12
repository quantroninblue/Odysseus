from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math

import numpy as np

from .contracts import PlannerCommand, RuntimeStamp


MotionState = Literal[
    "OK",
    "MOVING",
    "BLOCKED",
    "SLIPPING",
    "STALE_SENSORS",
    "POSE_DIVERGENCE",
]
SafetyAction = Literal["ALLOW", "SLOW", "STOP", "REVERSE", "RECOVERY_TURN", "RELOCALIZE"]


@dataclass(frozen=True)
class Pose2DState:
    time_sec: float
    x_m: float
    y_m: float
    yaw_rad: float
    source: str
    covariance_trace: float | None = None
    status: str = "OK"

    def distance_to(self, other: "Pose2DState") -> float:
        return float(math.hypot(self.x_m - other.x_m, self.y_m - other.y_m))


@dataclass(frozen=True)
class DepthSceneSignature:
    time_sec: float
    front_m: float
    lower_front_m: float
    left_m: float
    right_m: float
    lower_left_m: float
    lower_right_m: float

    @property
    def nearest_front_m(self) -> float:
        return float(min(self.front_m, self.lower_front_m))

    @property
    def nearest_side_m(self) -> float:
        return float(min(self.left_m, self.right_m, self.lower_left_m, self.lower_right_m))

    def max_delta(self, other: "DepthSceneSignature") -> float:
        values = [
            abs(self.front_m - other.front_m),
            abs(self.lower_front_m - other.lower_front_m),
            abs(self.left_m - other.left_m),
            abs(self.right_m - other.right_m),
            abs(self.lower_left_m - other.lower_left_m),
            abs(self.lower_right_m - other.lower_right_m),
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(max(finite)) if finite else 0.0


@dataclass(frozen=True)
class NavigationIntelligenceInput:
    now_sec: float
    proposed_command: PlannerCommand
    control_pose: Pose2DState | None
    depth_signature: DepthSceneSignature | None
    visual_pose: Pose2DState | None = None
    depth_age_sec: float = 0.0
    control_pose_age_sec: float = 0.0
    visual_pose_age_sec: float | None = None
    in_recovery: bool = False
    goal_distance_m: float | None = None
    learned_risk_score: float | None = None
    learned_risk_action: str = "UNAVAILABLE"
    learned_risk_reason: str = "no learned risk model"


@dataclass(frozen=True)
class NavigationDecision:
    motion_state: MotionState
    safety_action: SafetyAction
    confidence: float
    reason: str
    override_command: PlannerCommand | None = None
    diagnostics: dict[str, float | str] = field(default_factory=dict)

    @property
    def allows_command(self) -> bool:
        return self.override_command is None and self.safety_action in {"ALLOW", "SLOW"}


@dataclass
class _ForwardWatch:
    start_sec: float
    control_pose: Pose2DState | None
    visual_pose: Pose2DState | None
    depth_signature: DepthSceneSignature | None


class NavigationIntelligence:
    """Cross-check commanded motion against odom, visual pose, and depth evidence.

    This class is deliberately ROS-independent. It is the safety/intelligence
    authority above a planner: planners propose commands, this layer rejects or
    modifies commands when sensor evidence contradicts the estimated state.
    """

    def __init__(
        self,
        *,
        forward_speed_threshold_mps: float = 0.10,
        stale_sensor_sec: float = 1.0,
        blocked_watch_sec: float = 1.6,
        min_odom_progress_m: float = 0.24,
        min_visual_progress_m: float = 0.10,
        max_static_scene_delta_m: float = 0.12,
        obstacle_context_m: float = 4.25,
        immediate_stop_m: float = 0.72,
        pose_divergence_warn_m: float = 0.45,
        pose_divergence_relocalize_m: float = 0.85,
    ):
        self.forward_speed_threshold_mps = float(forward_speed_threshold_mps)
        self.stale_sensor_sec = float(stale_sensor_sec)
        self.blocked_watch_sec = float(blocked_watch_sec)
        self.min_odom_progress_m = float(min_odom_progress_m)
        self.min_visual_progress_m = float(min_visual_progress_m)
        self.max_static_scene_delta_m = float(max_static_scene_delta_m)
        self.obstacle_context_m = float(obstacle_context_m)
        self.immediate_stop_m = float(immediate_stop_m)
        self.pose_divergence_warn_m = float(pose_divergence_warn_m)
        self.pose_divergence_relocalize_m = float(pose_divergence_relocalize_m)
        self._forward_watch: _ForwardWatch | None = None
        self._last_decision = NavigationDecision("OK", "ALLOW", 1.0, "initialized")

    def reset(self) -> None:
        self._forward_watch = None
        self._last_decision = NavigationDecision("OK", "ALLOW", 1.0, "reset")

    @property
    def last_decision(self) -> NavigationDecision:
        return self._last_decision

    def update(self, sample: NavigationIntelligenceInput) -> NavigationDecision:
        stale = self._stale_decision(sample)
        if stale is not None:
            self._reset_forward_watch()
            return self._remember(stale)

        if sample.depth_signature is not None and sample.depth_signature.nearest_front_m < self.immediate_stop_m:
            self._reset_forward_watch()
            return self._remember(
                self._override(
                    sample,
                    motion_state="BLOCKED",
                    safety_action="STOP",
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=f"obstacle inside immediate stop distance {sample.depth_signature.nearest_front_m:.2f}m",
                    confidence=0.98,
                    source="safety_stop",
                )
            )

        divergence = self._pose_divergence_decision(sample)
        if divergence is not None:
            self._reset_forward_watch()
            return self._remember(divergence)

        if sample.in_recovery:
            self._reset_forward_watch()
            return self._remember(NavigationDecision("OK", "ALLOW", 0.85, "recovery command has authority"))

        learned = self._learned_risk_decision(sample)
        if learned is not None:
            return self._remember(learned)

        if sample.proposed_command.linear_x <= self.forward_speed_threshold_mps:
            self._reset_forward_watch()
            return self._remember(NavigationDecision("OK", "ALLOW", 1.0, "no sustained forward command"))

        blocked = self._blocked_motion_decision(sample)
        if blocked is not None:
            return self._remember(blocked)

        return self._remember(NavigationDecision("MOVING", "ALLOW", 0.90, "motion evidence acceptable", self._none(), self._diagnostics(sample)))

    def _learned_risk_decision(self, sample: NavigationIntelligenceInput) -> NavigationDecision | None:
        if sample.learned_risk_score is None or sample.proposed_command.linear_x <= self.forward_speed_threshold_mps:
            return None
        risk = float(sample.learned_risk_score)
        nearest_front = sample.depth_signature.nearest_front_m if sample.depth_signature is not None else float("inf")
        diagnostics = self._diagnostics(sample)
        diagnostics["learned_risk_score"] = risk
        diagnostics["learned_risk_action"] = sample.learned_risk_action
        if risk >= 0.82 and nearest_front < self.obstacle_context_m:
            return self._override(
                sample,
                motion_state="SLIPPING",
                safety_action="STOP",
                linear_x=0.0,
                angular_z=0.0,
                reason=f"learned GRU risk high near obstacle: {sample.learned_risk_reason}",
                confidence=min(0.95, max(0.50, risk)),
                source="safety_stop",
                diagnostics=diagnostics,
            )
        if risk >= 0.62:
            override = PlannerCommand(
                stamp=sample.proposed_command.stamp,
                linear_x=min(sample.proposed_command.linear_x, 0.08),
                angular_z=sample.proposed_command.angular_z,
                mode="nav_intel_slow_learned_risk",
                reason=f"learned GRU risk caution: {sample.learned_risk_reason}",
                source="safety_stop",
            )
            return NavigationDecision(
                motion_state="SLIPPING",
                safety_action="SLOW",
                confidence=min(0.90, max(0.45, risk)),
                reason=override.reason,
                override_command=override,
                diagnostics=diagnostics,
            )
        return None

    def _blocked_motion_decision(self, sample: NavigationIntelligenceInput) -> NavigationDecision | None:
        if self._forward_watch is None:
            self._forward_watch = _ForwardWatch(
                start_sec=sample.now_sec,
                control_pose=sample.control_pose,
                visual_pose=sample.visual_pose,
                depth_signature=sample.depth_signature,
            )
            return None

        watch = self._forward_watch
        elapsed = sample.now_sec - watch.start_sec
        if elapsed < self.blocked_watch_sec:
            return None

        control_progress = self._pose_progress(watch.control_pose, sample.control_pose)
        visual_progress = self._pose_progress(watch.visual_pose, sample.visual_pose)
        depth_delta = self._depth_delta(watch.depth_signature, sample.depth_signature)
        nearest_front = sample.depth_signature.nearest_front_m if sample.depth_signature is not None else float("inf")
        obstacle_context = nearest_front < self.obstacle_context_m
        odom_claims_motion = control_progress >= self.min_odom_progress_m
        visual_confirms_motion = visual_progress >= self.min_visual_progress_m
        scene_static = depth_delta <= self.max_static_scene_delta_m
        visual_untrusted = sample.visual_pose is None or self._visual_pose_stale(sample) or not visual_confirms_motion

        diagnostics = self._diagnostics(sample)
        diagnostics.update(
            {
                "watch_elapsed_sec": float(elapsed),
                "control_progress_m": float(control_progress),
                "visual_progress_m": float(visual_progress),
                "depth_scene_delta_m": float(depth_delta),
                "nearest_front_m": float(nearest_front),
            }
        )

        if obstacle_context and odom_claims_motion and scene_static and visual_untrusted:
            return self._override(
                sample,
                motion_state="BLOCKED",
                safety_action="REVERSE",
                linear_x=-0.14,
                angular_z=0.0,
                reason=(
                    "odom reports forward progress while depth scene is static near an obstacle; "
                    "visual pose did not confirm motion"
                ),
                confidence=0.94,
                source="safety_stop",
                diagnostics=diagnostics,
            )

        if control_progress < self.min_odom_progress_m and scene_static and obstacle_context:
            return self._override(
                sample,
                motion_state="BLOCKED",
                safety_action="REVERSE",
                linear_x=-0.14,
                angular_z=0.0,
                reason="commanded forward motion produced no odom progress and no scene change near obstacle",
                confidence=0.90,
                source="safety_stop",
                diagnostics=diagnostics,
            )

        self._forward_watch = _ForwardWatch(
            start_sec=sample.now_sec,
            control_pose=sample.control_pose,
            visual_pose=sample.visual_pose,
            depth_signature=sample.depth_signature,
        )
        return None

    def _stale_decision(self, sample: NavigationIntelligenceInput) -> NavigationDecision | None:
        if sample.depth_signature is None or sample.depth_age_sec > self.stale_sensor_sec:
            return self._override(
                sample,
                motion_state="STALE_SENSORS",
                safety_action="STOP",
                linear_x=0.0,
                angular_z=0.0,
                reason=f"stale depth age={sample.depth_age_sec:.2f}s",
                confidence=1.0,
                source="safety_stop",
            )
        if sample.control_pose is None or sample.control_pose_age_sec > self.stale_sensor_sec:
            return self._override(
                sample,
                motion_state="STALE_SENSORS",
                safety_action="STOP",
                linear_x=0.0,
                angular_z=0.0,
                reason=f"stale control pose age={sample.control_pose_age_sec:.2f}s",
                confidence=1.0,
                source="safety_stop",
            )
        return None

    def _pose_divergence_decision(self, sample: NavigationIntelligenceInput) -> NavigationDecision | None:
        if sample.control_pose is None or sample.visual_pose is None or self._visual_pose_stale(sample):
            return None
        divergence = sample.control_pose.distance_to(sample.visual_pose)
        if divergence >= self.pose_divergence_relocalize_m:
            return self._override(
                sample,
                motion_state="POSE_DIVERGENCE",
                safety_action="RELOCALIZE",
                linear_x=0.0,
                angular_z=0.0,
                reason=f"control pose and visual pose diverged by {divergence:.2f}m",
                confidence=0.95,
                source="safety_stop",
                diagnostics={"pose_divergence_m": float(divergence)},
            )
        if divergence >= self.pose_divergence_warn_m and sample.proposed_command.linear_x > 0.05:
            slowed = sample.proposed_command.clipped(abs(sample.proposed_command.linear_x), abs(sample.proposed_command.angular_z) + 1e-6)
            override = PlannerCommand(
                stamp=sample.proposed_command.stamp,
                linear_x=min(slowed.linear_x, 0.08),
                angular_z=slowed.angular_z,
                mode="nav_intel_slow_pose_divergence",
                reason=f"control pose and visual pose diverged by {divergence:.2f}m",
                source="safety_stop",
            )
            return NavigationDecision(
                motion_state="POSE_DIVERGENCE",
                safety_action="SLOW",
                confidence=0.82,
                reason=override.reason,
                override_command=override,
                diagnostics={"pose_divergence_m": float(divergence)},
            )
        return None

    def _override(
        self,
        sample: NavigationIntelligenceInput,
        *,
        motion_state: MotionState,
        safety_action: SafetyAction,
        linear_x: float,
        angular_z: float,
        reason: str,
        confidence: float,
        source: str,
        diagnostics: dict[str, float | str] | None = None,
    ) -> NavigationDecision:
        command = PlannerCommand(
            stamp=RuntimeStamp(sample.now_sec, sample.proposed_command.stamp.frame_id, "navigation_intelligence"),
            linear_x=float(linear_x),
            angular_z=float(angular_z),
            mode=f"nav_intel_{safety_action.lower()}",
            reason=reason,
            source=source,
        )
        return NavigationDecision(
            motion_state=motion_state,
            safety_action=safety_action,
            confidence=float(confidence),
            reason=reason,
            override_command=command,
            diagnostics=diagnostics or self._diagnostics(sample),
        )

    def _remember(self, decision: NavigationDecision) -> NavigationDecision:
        self._last_decision = decision
        return decision

    def _reset_forward_watch(self) -> None:
        self._forward_watch = None

    @staticmethod
    def _none() -> None:
        return None

    @staticmethod
    def _pose_progress(start: Pose2DState | None, current: Pose2DState | None) -> float:
        if start is None or current is None:
            return 0.0
        return current.distance_to(start)

    @staticmethod
    def _depth_delta(start: DepthSceneSignature | None, current: DepthSceneSignature | None) -> float:
        if start is None or current is None:
            return float("inf")
        return current.max_delta(start)

    @staticmethod
    def _visual_pose_stale(sample: NavigationIntelligenceInput) -> bool:
        if sample.visual_pose is None:
            return True
        if sample.visual_pose_age_sec is None:
            return False
        return sample.visual_pose_age_sec > 1.0

    @staticmethod
    def _diagnostics(sample: NavigationIntelligenceInput) -> dict[str, float | str]:
        diagnostics: dict[str, float | str] = {
            "depth_age_sec": float(sample.depth_age_sec),
            "control_pose_age_sec": float(sample.control_pose_age_sec),
            "proposed_linear_x": float(sample.proposed_command.linear_x),
            "proposed_angular_z": float(sample.proposed_command.angular_z),
            "learned_risk_score": float(sample.learned_risk_score or 0.0),
            "learned_risk_action": sample.learned_risk_action,
        }
        if sample.depth_signature is not None:
            diagnostics["front_m"] = float(sample.depth_signature.front_m)
            diagnostics["lower_front_m"] = float(sample.depth_signature.lower_front_m)
        if sample.control_pose is not None:
            diagnostics["control_pose_source"] = sample.control_pose.source
        if sample.visual_pose is not None:
            diagnostics["visual_pose_source"] = sample.visual_pose.source
        return diagnostics
