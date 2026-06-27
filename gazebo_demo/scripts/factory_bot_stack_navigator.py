#!/usr/bin/env python3
"""Global/local perception and VSLAM demo navigator for factory_bot.

Inputs:
- /camera/depth/image_raw for obstacle sectors
- /odometry/filtered from odom+IMU EKF for control pose, with /odom fallback
- /semantic_spatial/diagnostics and /semantic_spatial/objects for stack logs

Output:
- ROS velocity commands on /factory_bot/cmd_vel, bridged to Gazebo Sim

This is a live demo navigator: it knows the target coordinate, but it does not
know obstacle locations ahead of time. It accumulates depth obstacles in a
world-frame costmap, follows an A* route, and uses local trajectory rollout and
navigation intelligence to execute that route safely.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Optional

from rclpy.executors import ExternalShutdownException

import numpy as np

from planning.global_costmap import GlobalCostmap
from planning.global_route import (
    AStarRoutePlanner,
    GlobalRoute,
    remaining_route_length,
    select_lookahead_waypoint,
)
from planning.local_costmap import build_local_costmap_from_depth
from planning.trajectory_rollout import CandidateEvaluation, TrajectoryRolloutPlanner
from runtime.core.cognition import CognitiveObservation, CognitiveShadowRunner, OccupancyGridSpec, SemanticMemoryObject
from runtime.core.odysseus import OdysseusNavigator, OdysseusShadowRunner, load_odysseus_checkpoint
from runtime.core.contracts import PlannerCommand, RuntimeStamp
from runtime.core.navigation_intelligence import (
    DepthSceneSignature,
    NavigationIntelligence,
    NavigationIntelligenceInput,
    Pose2DState,
)
from runtime.core.navigation_learning import NavigationLearningMemory, NavigationRiskPredictor

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

ROS_CMD_TOPIC = "/factory_bot/cmd_vel"
TARGET_X = 14.0
TARGET_Y = 10.2
MAX_FORWARD_SPEED = 0.34
LOW_OBSTACLE_SLOW_DIST = 3.60
LOW_OBSTACLE_REROUTE_DIST = 2.75
NEAR_OBSTACLE_DIST = 1.25
ESCAPE_OBSTACLE_DIST = 0.90
CRITICAL_SIDE_CLEARANCE_M = 0.78
GOAL_REACQUIRE_HEADING_ERR_RAD = 1.35
GOAL_REACQUIRE_WRAP_ERR_RAD = 2.65
DETOUR_LOCK_SEC = 2.0
NORMAL_MAX_ANGULAR_SPEED = 0.68
EMERGENCY_MAX_ANGULAR_SPEED = 0.95
ANGULAR_SLEW_PER_TICK = 0.32
LOW_FRONT_HARD_STOP_DIST = 3.05
LOW_OBSTACLE_BYPASS_SEC = 5.0
LOW_OBSTACLE_BYPASS_HEADING_BIAS = 0.65
EMERGENCY_MODES = {
    "blocked_turn",
    "escape_reverse",
    "escape_turn",
    "goal_reacquire",
    "side_clearance_turn",
    "low_front_bypass_start",
    "nav_intel_stop",
    "nav_intel_reverse",
    "nav_intel_recovery_turn",
    "nav_intel_relocalize",
    "odysseus_safety_turn",
    "odysseus_explore",
    "odysseus_retrace",
    "odysseus_recover",
}


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    source: str
    stamp: float


@dataclass
class LocalPlan:
    heading_error: float
    clearance: float
    mode: str


class FactoryBotStackNavigator(Node):
    def __init__(self):
        super().__init__("factory_bot_stack_navigator")
        self.depth = None
        self.depth_encoding = ""
        self.rgb = None
        self.rgb_encoding = ""
        self.target_marker_seen = False
        self.pose: Optional[Pose2D] = None
        self.odom_pose: Optional[Pose2D] = None
        self.filtered_odom_pose: Optional[Pose2D] = None
        self.vslam_pose: Optional[Pose2D] = None
        self.last_diag = ""
        self.last_objects = ""
        self.semantic_objects = []
        self.last_log = 0.0
        self.last_depth_stamp = 0.0
        self.last_pose_stamp = 0.0
        self.last_filtered_odom_stamp = 0.0
        self.last_vslam_stamp = 0.0
        self.integral_heading = 0.0
        self.prev_heading_error = 0.0
        self.detour_sign = 1.0
        self.detour_locked_until = 0.0
        self.goal_turn_sign = 1.0
        self.arrived = False
        self.best_distance = float("inf")
        self.last_best_time = 0.0
        self.last_commanded_linear = 0.0
        self.last_commanded_angular = 0.0
        self.recovery_until = 0.0
        self.recovery_reverse_until = 0.0
        self.recovery_phase = ""
        self.recovery_turn_sign = 1.0
        self.detour_until = 0.0
        self.low_obstacle_avoid_until = 0.0
        self.low_obstacle_turn_sign = 1.0
        self.last_plan = LocalPlan(0.0, 8.0, "direct")
        self.last_rollout_mode = "none"
        self.last_rollout_reason = ""
        self.last_nav_motion_state = "OK"
        self.last_nav_safety_action = "ALLOW"
        self.last_nav_confidence = 1.0
        self.last_nav_reason = "initialized"
        self.last_learned_risk_score = 0.0
        self.last_learned_risk_action = "UNAVAILABLE"
        self.last_learned_risk_reason = "no learned risk model"
        self.nav_intelligence = NavigationIntelligence()
        self.nav_learning_memory = NavigationLearningMemory()
        self.nav_risk_predictor = NavigationRiskPredictor()
        self.configure_navigation_learning()
        self.rollout_planner = TrajectoryRolloutPlanner(
            max_linear_mps=MAX_FORWARD_SPEED,
            max_angular_radps=NORMAL_MAX_ANGULAR_SPEED,
        )
        self.global_costmap = GlobalCostmap(
            x_min_m=-4.0,
            x_max_m=max(22.0, TARGET_X + 5.0),
            y_min_m=-5.0,
            y_max_m=max(18.0, TARGET_Y + 5.0),
            resolution_m=0.18,
        )
        self.global_route_planner = AStarRoutePlanner(inflation_radius_m=0.56)
        self.global_route = GlobalRoute(status="waiting", reason="waiting for pose and depth")
        self.global_waypoint: tuple[float, float] | None = None
        self.last_global_plan_time = 0.0
        cognitive_dataset_dir = os.environ.get("COGNITIVE_DATASET_DIR", "").strip()
        self.cognitive_shadow = CognitiveShadowRunner(dataset_directory=cognitive_dataset_dir or None)
        self.last_cognitive_decision = "waiting"
        self.last_cognitive_confidence = 0.0
        self.last_cognitive_memory_count = 0
        self.last_cognitive_sample = ""
        odysseus_dataset_dir = os.environ.get("ODYSSEUS_DATASET_DIR", "").strip()
        odysseus_model = None
        odysseus_checkpoint = os.environ.get("ODYSSEUS_ATTRIBUTOR_CHECKPOINT", "").strip()
        if odysseus_checkpoint:
            try:
                odysseus_model, _checkpoint = load_odysseus_checkpoint(odysseus_checkpoint)
                self.get_logger().info(f"loaded Odysseus causal attributor checkpoint: {odysseus_checkpoint}")
            except Exception as exc:
                self.get_logger().warn(f"could not load Odysseus checkpoint {odysseus_checkpoint}: {exc}")
        odysseus_memory_path = os.environ.get(
            "ODYSSEUS_MEMORY_PATH", "artifacts/odysseus_world_memory.json"
        ).strip()
        self.odysseus_navigator = OdysseusNavigator(
            shadow_runner=OdysseusShadowRunner(model=odysseus_model, dataset_directory=odysseus_dataset_dir or None),
            memory_path=odysseus_memory_path or None,
        )
        remembered_no_go = self.odysseus_navigator.remembered_no_go_points()
        if remembered_no_go.size:
            self.global_costmap.add_obstacles_world(remembered_no_go, evidence=2.4)
        self.last_odysseus_mode = "waiting"
        self.last_odysseus_reason = "waiting for rollout candidates"
        self.last_odysseus_trace = ""
        self.last_odysseus_candidate = ""
        self.last_odysseus_sample = ""
        self.last_motion_check_time = 0.0
        self.last_motion_check_x = 0.0
        self.last_motion_check_y = 0.0
        self.stuck_since = 0.0
        self.command_log_file = None
        self.command_log_writer = None
        self.command_log_path = self.open_command_log()

        self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_rgb, 10)
        self.create_subscription(Odometry, "/semantic_spatial/visual_odometry", self.on_vo, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.on_filtered_odom, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/semantic_spatial/diagnostics", self.on_diag, 10)
        self.create_subscription(String, "/semantic_spatial/objects", self.on_objects, 10)
        self.cmd_pub = self.create_publisher(Twist, ROS_CMD_TOPIC, 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info("Factory bot stack navigator waiting for depth + filtered odom/VSLAM")
        self.get_logger().info(f"navigator command log: {self.command_log_path}")

    def configure_navigation_learning(self) -> None:
        checkpoint = os.environ.get("NAV_GRU_RISK_CHECKPOINT", "").strip()
        if not checkpoint:
            return
        try:
            self.nav_risk_predictor.load(checkpoint)
            self.get_logger().info(f"loaded GRU navigation risk checkpoint: {checkpoint}")
        except Exception as exc:
            self.get_logger().warn(f"could not load GRU navigation risk checkpoint {checkpoint}: {exc}")

    def open_command_log(self) -> Path:
        log_dir = Path("runtime_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"navigator_commands_{stamp}_{os.getpid()}.csv"
        self.command_log_file = path.open("w", newline="", encoding="utf-8")
        fields = [
            "time_sec",
            "mode",
            "cmd_v",
            "cmd_w",
            "pose_x",
            "pose_y",
            "pose_yaw",
            "pose_source",
            "goal_dist",
            "heading_err",
            "front",
            "lower_front",
            "left",
            "right",
            "lower_left",
            "lower_right",
            "plan_mode",
            "plan_clearance",
            "plan_heading",
            "rollout_mode",
            "rollout_reason",
            "global_route_status",
            "global_route_reason",
            "global_route_length",
            "global_waypoint_x",
            "global_waypoint_y",
            "cognitive_decision",
            "cognitive_confidence",
            "cognitive_memory_count",
            "cognitive_sample",
            "odysseus_mode",
            "odysseus_reason",
            "odysseus_trace",
            "odysseus_candidate",
            "odysseus_sample",
            "nav_motion_state",
            "nav_safety_action",
            "nav_confidence",
            "nav_reason",
            "learned_risk_score",
            "learned_risk_action",
            "learned_risk_reason",
            "semantic_forward",
            "semantic_lateral",
        ]
        self.command_log_writer = csv.DictWriter(self.command_log_file, fieldnames=fields)
        self.command_log_writer.writeheader()
        self.command_log_file.flush()
        return path

    def on_depth(self, msg: Image) -> None:
        self.depth_encoding = msg.encoding
        self.depth = image_to_depth_m(msg)
        self.last_depth_stamp = self.get_clock().now().nanoseconds * 1e-9

    def on_rgb(self, msg: Image) -> None:
        self.rgb_encoding = msg.encoding
        self.rgb = image_to_rgb(msg)
        self.target_marker_seen = green_target_visible(self.rgb)

    def on_vo(self, msg: Odometry) -> None:
        self.vslam_pose = odom_to_pose2d(msg, "vslam")
        self.last_vslam_stamp = self.get_clock().now().nanoseconds * 1e-9

    def on_filtered_odom(self, msg: Odometry) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.filtered_odom_pose = odom_to_pose2d(msg, "ekf_odom_imu")
        self.pose = self.filtered_odom_pose
        self.last_filtered_odom_stamp = now
        self.last_pose_stamp = now

    def on_odom(self, msg: Odometry) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.odom_pose = odom_to_pose2d(msg, "odom_fallback")
        if self.filtered_odom_pose is None or now - self.last_filtered_odom_stamp > 0.5:
            self.pose = self.odom_pose
            self.last_pose_stamp = now

    def on_diag(self, msg: String) -> None:
        self.last_diag = msg.data

    def on_objects(self, msg: String) -> None:
        self.last_objects = msg.data
        try:
            parsed = json.loads(msg.data) if msg.data else []
        except json.JSONDecodeError:
            parsed = []
        self.semantic_objects = parsed if isinstance(parsed, list) else []

    def tick(self) -> None:
        if self.arrived:
            self.publish_cmd(0.0, 0.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.depth is None:
            self.log_waiting("depth")
            self.publish_cmd(0.0, 0.0)
            return
        if now - self.last_depth_stamp > 1.0:
            self.log_waiting("fresh depth")
            self.publish_cmd(0.0, 0.0)
            return
        if self.pose is None:
            self.log_waiting("filtered odom/VSLAM")
            self.publish_cmd(0.0, 0.0)
            return
        if now - self.last_pose_stamp > 1.0:
            self.log_waiting("fresh filtered odom/VSLAM")
            self.publish_cmd(0.0, 0.0)
            return

        sectors = depth_sectors(self.depth)
        dx = TARGET_X - self.pose.x
        dy = TARGET_Y - self.pose.y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        heading_error = wrap_angle(target_yaw - self.pose.yaw)
        local_costmap = None
        try:
            local_costmap = build_local_costmap_from_depth(
                self.depth,
                x_max_m=5.0,
                y_max_m=2.4,
                resolution_m=0.08,
                inflation_radius_m=0.48,
            )
            self.global_costmap.update_from_local_points(
                local_costmap.raw_points_xy,
                pose_x_m=self.pose.x,
                pose_y_m=self.pose.y,
                pose_yaw_rad=self.pose.yaw,
                now_sec=now,
            )
            remembered_no_go = self.odysseus_navigator.remembered_no_go_points()
            if remembered_no_go.size:
                self.global_costmap.add_obstacles_world(remembered_no_go, evidence=0.35)
            if self.last_global_plan_time <= 0.0 or now - self.last_global_plan_time >= 0.6:
                self.global_route = self.global_route_planner.plan(
                    self.global_costmap,
                    start_xy=(self.pose.x, self.pose.y),
                    goal_xy=(TARGET_X, TARGET_Y),
                )
                self.last_global_plan_time = now
            self.global_waypoint = select_lookahead_waypoint(
                self.global_route, (self.pose.x, self.pose.y), lookahead_m=1.2
            )
            if self.global_waypoint is not None:
                waypoint_yaw = math.atan2(
                    self.global_waypoint[1] - self.pose.y,
                    self.global_waypoint[0] - self.pose.x,
                )
                heading_error = wrap_angle(waypoint_yaw - self.pose.yaw)
        except Exception as exc:
            self.global_route = GlobalRoute(status="mapping_error", reason=str(exc)[:120])
            self.global_waypoint = None

        self.run_cognitive_shadow(now, local_costmap)

        front = sectors["front"]
        left = sectors["left"]
        right = sectors["right"]
        lower_front = sectors["lower_front"]
        lower_left = sectors["lower_left"]
        lower_right = sectors["lower_right"]
        front_close = min(front, lower_front)

        if (distance < 0.35 and front_close > 1.10) or (distance < 0.85 and self.target_marker_seen):
            self.arrived = True
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info(
                f"ARRIVED target=({TARGET_X:.1f},{TARGET_Y:.1f}) pose=({self.pose.x:.2f},{self.pose.y:.2f}) "
                f"target_marker_seen={self.target_marker_seen} front_close={front_close:.2f}m"
            )
            return
        left_clear = min(left, lower_left)
        right_clear = min(right, lower_right)
        plan = depth_occupancy_plan(self.depth, heading_error, self.detour_sign)
        self.last_plan = plan
        semantic_forward, semantic_lateral = self.semantic_object_hazard()
        front_clear = min(front, plan.clearance, lower_front)
        near = min(front_close, left_clear, right_clear)
        stuck = self.update_stuck_state(now, distance)

        route_active = self.global_route.status == "route_ready"
        progress_distance = remaining_route_length(
            self.global_route, (self.pose.x, self.pose.y)
        ) if route_active else distance
        if not math.isfinite(progress_distance):
            progress_distance = distance
        route_became_longer = route_active and progress_distance > self.best_distance + 0.50
        if self.best_distance == float("inf") or route_became_longer:
            self.best_distance = progress_distance
            self.last_best_time = now
        elif progress_distance < self.best_distance - 0.10:
            self.best_distance = progress_distance
            self.last_best_time = now

        if now < self.recovery_until:
            if now < self.recovery_reverse_until:
                mode = "escape_reverse"
                linear = -0.16
                angular = -0.35 * self.recovery_turn_sign
            else:
                mode = "escape_turn"
                linear = 0.02
                angular = 0.95 * self.recovery_turn_sign
            self.publish_and_log(now, mode, sectors, distance, heading_error, linear, angular)
            return

        boxed_in = front_close < ESCAPE_OBSTACLE_DIST
        stalled = progress_distance > 0.9 and self.last_commanded_linear > 0.12 and now - self.last_best_time > 3.0
        off_course = (
            not route_active
            and progress_distance > self.best_distance + 0.75
            and now - self.last_best_time > 4.5
        )
        if boxed_in or stuck or stalled or off_course:
            turn_sign = self.choose_detour_sign(now, left_clear, right_clear, plan.heading_error, heading_error, lock_sec=3.0)
            self.start_recovery(now, turn_sign)
            self.publish_and_log(now, "escape_reverse", sectors, distance, heading_error, -0.16, -0.35 * turn_sign)
            return

        critical_side = min(left_clear, right_clear) < CRITICAL_SIDE_CLEARANCE_M
        if critical_side and front_close > NEAR_OBSTACLE_DIST:
            turn_sign = 1.0 if right_clear < left_clear else -1.0
            self.lock_detour_sign(now, turn_sign, 1.4)
            self.integral_heading = 0.0
            self.prev_heading_error = 0.75 * turn_sign
            self.publish_and_log(now, "side_clearance_turn", sectors, distance, heading_error, 0.0, 0.70 * turn_sign)
            return

        if abs(heading_error) > GOAL_REACQUIRE_HEADING_ERR_RAD and front_close > NEAR_OBSTACLE_DIST:
            if abs(heading_error) < GOAL_REACQUIRE_WRAP_ERR_RAD:
                self.goal_turn_sign = 1.0 if heading_error > 0.0 else -1.0
            turn_sign = self.goal_turn_sign
            if min(left_clear, right_clear) < 1.05 and abs(left_clear - right_clear) > 0.20:
                turn_sign = 1.0 if right_clear < left_clear else -1.0
            self.lock_detour_sign(now, turn_sign, 1.2)
            self.integral_heading = 0.0
            self.prev_heading_error = 0.85 * turn_sign
            self.publish_and_log(now, "goal_reacquire", sectors, distance, heading_error, 0.0, 0.78 * turn_sign)
            return

        if abs(heading_error) <= GOAL_REACQUIRE_HEADING_ERR_RAD:
            try:
                if local_costmap is None:
                    raise RuntimeError("local costmap unavailable")
                rollout = self.rollout_planner.plan(
                    local_costmap,
                    goal_heading_error=heading_error,
                    now_sec=now,
                    detour_hint=self.detour_sign,
                )
                command = rollout.command
                observation = self.build_cognitive_observation(now, local_costmap)
                if observation is not None:
                    odysseus = self.odysseus_navigator.decide(
                        observation,
                        rollout.candidates,
                        deterministic_command=command,
                        goal_distance_m=distance,
                        semantic_forward_m=semantic_forward,
                        semantic_lateral_m=semantic_lateral,
                        progress_distance_m=progress_distance,
                        nav_safety_action=self.last_nav_safety_action,
                        nav_motion_state=self.last_nav_motion_state,
                    )
                    command = odysseus.command
                    self.last_odysseus_mode = odysseus.mode
                    self.last_odysseus_reason = odysseus.reason[:120]
                    self.last_odysseus_trace = odysseus.trace_id
                    self.last_odysseus_candidate = odysseus.selected_candidate_id or ""
                    self.last_odysseus_sample = odysseus.closed_sample_path
                self.last_rollout_mode = command.mode
                self.last_rollout_reason = command.reason[:120]
                if abs(command.angular_z) > 0.10:
                    self.lock_detour_sign(now, 1.0 if command.angular_z > 0.0 else -1.0, 0.9)
                if command.mode in {"rollout_safety_turn", "odysseus_recover", "odysseus_retrace", "odysseus_explore"}:
                    self.integral_heading = 0.0
                    self.prev_heading_error = command.angular_z
                self.publish_and_log(now, command.mode, sectors, distance, heading_error, command.linear_x, command.angular_z)
                return
            except Exception as exc:
                self.last_rollout_mode = "fallback"
                self.last_rollout_reason = str(exc)[:120]

        low_depth_hazard = lower_front < LOW_FRONT_HARD_STOP_DIST and front > 4.0

        mode = "clear_path"
        control_heading_error = heading_error
        if now < self.low_obstacle_avoid_until and front_close > ESCAPE_OBSTACLE_DIST:
            turn_sign = self.low_obstacle_turn_sign
            linear = 0.12 if lower_front > LOW_OBSTACLE_REROUTE_DIST else 0.07
            control_heading_error = heading_error + LOW_OBSTACLE_BYPASS_HEADING_BIAS * turn_sign
            obstacle_bias = 0.24 * turn_sign
            self.detour_until = max(self.detour_until, now + 0.8)
            mode = "low_obstacle_bypass"
        elif low_depth_hazard:
            turn_sign = self.choose_detour_sign(
                now,
                left_clear,
                right_clear,
                plan.heading_error,
                heading_error,
                lock_sec=LOW_OBSTACLE_BYPASS_SEC,
            )
            self.low_obstacle_turn_sign = turn_sign
            self.low_obstacle_avoid_until = now + LOW_OBSTACLE_BYPASS_SEC
            self.lock_detour_sign(now, turn_sign, LOW_OBSTACLE_BYPASS_SEC)
            self.integral_heading = 0.0
            self.prev_heading_error = 0.75 * turn_sign
            self.detour_until = now + LOW_OBSTACLE_BYPASS_SEC
            self.publish_and_log(now, "low_front_bypass_start", sectors, distance, heading_error, 0.0, 0.88 * turn_sign)
            return
        elif front_close < NEAR_OBSTACLE_DIST:
            turn_sign = self.choose_detour_sign(now, left_clear, right_clear, plan.heading_error, heading_error, lock_sec=2.4)
            linear = 0.00
            control_heading_error = 0.85 * turn_sign
            obstacle_bias = 0.85 * turn_sign
            self.detour_until = now + 1.8
            mode = "blocked_turn"
        elif lower_front < LOW_OBSTACLE_REROUTE_DIST or front_clear < 2.25:
            turn_sign = self.choose_detour_sign(now, left_clear, right_clear, plan.heading_error, heading_error, lock_sec=DETOUR_LOCK_SEC)
            linear = 0.06
            control_heading_error = max(abs(plan.heading_error), 0.62) * turn_sign
            obstacle_bias = 0.48 * turn_sign
            self.detour_until = now + 1.8
            mode = "reroute_low_obstacle" if lower_front < LOW_OBSTACLE_REROUTE_DIST else "reroute_near_obstacle"
        elif plan.mode == "occupancy_detour" and plan.clearance < 2.9:
            turn_sign = self.choose_detour_sign(now, left_clear, right_clear, plan.heading_error, heading_error, lock_sec=2.0)
            linear = 0.10
            control_heading_error = plan.heading_error if abs(plan.heading_error) > 0.12 else 0.45 * turn_sign
            obstacle_bias = 0.38 * turn_sign
            self.detour_until = now + 1.6
            mode = "depth_occupancy_reroute"
        elif now < self.detour_until and abs(heading_error) < 1.2:
            linear = 0.12
            control_heading_error = heading_error + 0.34 * self.detour_sign
            obstacle_bias = 0.18 * self.detour_sign
            mode = "detour_commit"
        elif lower_front < LOW_OBSTACLE_SLOW_DIST:
            linear = 0.18 if abs(heading_error) < 0.65 else 0.12
            control_heading_error = heading_error
            side_diff = abs(left_clear - right_clear)
            if side_diff > 0.75:
                turn_sign = 1.0 if left_clear > right_clear else -1.0
                obstacle_bias = 0.14 * turn_sign
            elif plan.mode == "occupancy_detour" and abs(plan.heading_error) > 0.18:
                obstacle_bias = 0.10 * (1.0 if plan.heading_error > 0.0 else -1.0)
            else:
                obstacle_bias = 0.0
            mode = "low_obstacle_caution"
        elif near < 1.4:
            linear = 0.08
            obstacle_bias = -0.55 if left_clear < right_clear else 0.55
            mode = "side_obstacle"
        elif front > 3.0 and abs(heading_error) < 0.55:
            linear = MAX_FORWARD_SPEED if not self.target_marker_seen else 0.24
            obstacle_bias = 0.0
            mode = "open_path_speedup"
        else:
            linear = 0.22 if abs(heading_error) < 0.75 else 0.14
            obstacle_bias = 0.0

        if abs(control_heading_error) > 1.2:
            linear = min(linear, 0.06)
        if abs(heading_error) > GOAL_REACQUIRE_HEADING_ERR_RAD:
            linear = min(linear, 0.04)

        self.integral_heading = float(np.clip(self.integral_heading + control_heading_error * 0.2, -1.0, 1.0))
        derivative = (control_heading_error - self.prev_heading_error) / 0.2
        self.prev_heading_error = control_heading_error
        if mode in {"depth_occupancy_reroute", "detour_commit", "low_obstacle_bypass", "reroute_near_obstacle", "blocked_turn", "side_obstacle"}:
            heading_gain = 0.28
            derivative_gain = 0.010
        else:
            heading_gain = 0.85
            derivative_gain = 0.035
        angular = heading_gain * control_heading_error + 0.02 * self.integral_heading + derivative_gain * derivative + obstacle_bias
        angular = float(np.clip(angular, -0.95, 0.95))
        if abs(angular) > 0.80:
            linear = min(linear, 0.10)
        elif abs(angular) > 0.55:
            linear = min(linear, 0.18)
        linear = float(np.clip(linear, -0.18, MAX_FORWARD_SPEED))

        self.publish_and_log(now, mode, sectors, distance, heading_error, linear, angular)

    def start_recovery(self, now: float, turn_sign: float) -> None:
        self.recovery_turn_sign = float(1.0 if turn_sign >= 0.0 else -1.0)
        self.detour_sign = self.recovery_turn_sign
        self.detour_locked_until = now + 4.5
        self.recovery_phase = "escape"
        self.recovery_reverse_until = now + 1.8
        self.recovery_until = now + 4.2
        self.detour_until = now + 4.5
        self.low_obstacle_avoid_until = 0.0
        self.last_best_time = now
        self.stuck_since = 0.0

    def choose_detour_sign(
        self,
        now: float,
        left_clear: float,
        right_clear: float,
        plan_heading_error: float,
        goal_heading_error: float,
        lock_sec: float,
    ) -> float:
        clearance_diff = left_clear - right_clear
        if abs(clearance_diff) > 0.18:
            clearance_sign = 1.0 if clearance_diff > 0.0 else -1.0
        else:
            clearance_sign = 0.0

        if now < self.detour_locked_until:
            if clearance_sign != 0.0 and clearance_sign != self.detour_sign and abs(clearance_diff) > 0.90:
                return self.lock_detour_sign(now, clearance_sign, max(lock_sec, 1.0))
            return self.detour_sign

        if clearance_sign != 0.0:
            sign = clearance_sign
        elif abs(plan_heading_error) > 0.10:
            sign = 1.0 if plan_heading_error > 0.0 else -1.0
        elif abs(goal_heading_error) > 0.10:
            sign = 1.0 if goal_heading_error > 0.0 else -1.0
        else:
            sign = self.detour_sign
        return self.lock_detour_sign(now, sign, lock_sec)

    def lock_detour_sign(self, now: float, sign: float, lock_sec: float) -> float:
        self.detour_sign = float(1.0 if sign >= 0.0 else -1.0)
        self.detour_locked_until = max(self.detour_locked_until, now + lock_sec)
        return self.detour_sign

    def update_stuck_state(self, now: float, distance: float) -> bool:
        if self.pose is None or distance < 0.9:
            self.stuck_since = 0.0
            return False
        if self.last_motion_check_time <= 0.0:
            self.last_motion_check_time = now
            self.last_motion_check_x = self.pose.x
            self.last_motion_check_y = self.pose.y
            return False
        if now - self.last_motion_check_time < 1.1:
            return False

        moved = math.hypot(self.pose.x - self.last_motion_check_x, self.pose.y - self.last_motion_check_y)
        commanded_forward = self.last_commanded_linear > 0.10
        self.last_motion_check_time = now
        self.last_motion_check_x = self.pose.x
        self.last_motion_check_y = self.pose.y
        if commanded_forward and moved < 0.035:
            if self.stuck_since <= 0.0:
                self.stuck_since = now
            return now - self.stuck_since > 1.2
        self.stuck_since = 0.0
        return False

    def build_cognitive_observation(self, now: float, local_costmap) -> CognitiveObservation | None:
        if self.pose is None or local_costmap is None:
            return None
        semantic_memory = []
        for item in self.semantic_objects:
            if not isinstance(item, dict):
                continue
            centroid = item.get("centroid")
            extent = item.get("extent")
            if not isinstance(centroid, list) or len(centroid) < 3 or not isinstance(extent, list) or len(extent) < 3:
                continue
            semantic_memory.append(SemanticMemoryObject(
                object_id=int(item.get("object_id", -1)),
                label=str(item.get("label", "unknown")),
                confidence=float(np.clip(item.get("confidence", 0.0), 0.0, 1.0)),
                centroid_world_xyz=np.asarray(centroid[:3], dtype=np.float32),
                extent_xyz=np.asarray(extent[:3], dtype=np.float32),
                observations=max(1, int(item.get("observations", 1))),
            ))
        route = np.asarray(self.global_route.waypoints, dtype=np.float32).reshape(-1, 2)
        previous = PlannerCommand(
            RuntimeStamp(now, "base_link", "navigator_history"),
            self.last_commanded_linear, self.last_commanded_angular, "previous", "previous command",
        )
        return CognitiveObservation(
            stamp=RuntimeStamp(now, "map", "factory_bot_stack_navigator"),
            pose=pose2d_to_state(self.pose),
            goal_world_xy=np.asarray([TARGET_X, TARGET_Y], dtype=np.float32),
            local_occupancy=local_costmap.grid.astype(np.float32),
            global_occupancy=self.global_costmap.inflated_occupancy(0.56).astype(np.float32),
            local_grid_spec=OccupancyGridSpec(local_costmap.resolution_m, 0.0, -local_costmap.y_max_m, "base_link"),
            global_grid_spec=OccupancyGridSpec(
                self.global_costmap.resolution_m, self.global_costmap.x_min_m, self.global_costmap.y_min_m, "map"
            ),
            route_world_xy=route,
            semantic_objects=tuple(semantic_memory),
            rgb=self.rgb,
            depth_m=self.depth,
            previous_command=previous,
            sensor_ages_sec={"depth": max(0.0, now - self.last_depth_stamp), "pose": max(0.0, now - self.last_pose_stamp)},
        )

    def run_cognitive_shadow(self, now: float, local_costmap) -> None:
        try:
            observation = self.build_cognitive_observation(now, local_costmap)
            if observation is None:
                return
            result = self.cognitive_shadow.observe(observation)
            self.last_cognitive_decision = result.decision.reason[:120]
            self.last_cognitive_confidence = result.decision.confidence
            self.last_cognitive_memory_count = result.belief.observation_count
            self.last_cognitive_sample = str(result.sample_path) if result.sample_path else ""
        except Exception as exc:
            self.last_cognitive_decision = f"shadow_error:{str(exc)[:100]}"
            self.last_cognitive_confidence = 0.0

    def semantic_object_hazard(self) -> tuple[float, float]:
        if self.pose is None or not self.semantic_objects:
            return 8.0, 0.0
        c = math.cos(-self.pose.yaw)
        s = math.sin(-self.pose.yaw)
        best_forward = 8.0
        best_lateral = 0.0
        for obj in self.semantic_objects:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label", "")).lower()
            if "obstacle" not in label:
                continue
            centroid = obj.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 2:
                continue
            dx = float(centroid[0]) - self.pose.x
            dy = float(centroid[1]) - self.pose.y
            forward = c * dx - s * dy
            lateral = s * dx + c * dy
            if 0.2 < forward < 4.5 and abs(lateral) < 1.25 and forward < best_forward:
                best_forward = forward
                best_lateral = lateral
        return best_forward, best_lateral

    def publish_and_log(
        self,
        now: float,
        mode: str,
        sectors: dict[str, float],
        distance: float,
        heading_error: float,
        linear: float,
        angular: float,
    ) -> None:
        mode, linear, angular = self.apply_navigation_intelligence(
            now, mode, sectors, distance, heading_error, linear, angular
        )
        linear, angular = self.shape_command(mode, linear, angular)
        self.publish_cmd(linear, angular)
        self.last_commanded_linear = linear
        self.last_commanded_angular = angular
        semantic_forward, semantic_lateral = self.semantic_object_hazard()
        self.log_command(
            now=now,
            mode=mode,
            linear=linear,
            angular=angular,
            sectors=sectors,
            distance=distance,
            heading_error=heading_error,
            semantic_forward=semantic_forward,
            semantic_lateral=semantic_lateral,
        )
        if now - self.last_log > 0.8:
            self.last_log = now
            diag_summary = summarize_json(self.last_diag)
            object_summary = summarize_objects(self.last_objects)
            self.get_logger().info(
                "perception "
                f"mode={mode} depth_encoding={self.depth_encoding} "
                f"front={sectors['front']:.2f}m lower_front={sectors['lower_front']:.2f}m "
                f"left={sectors['left']:.2f}m right={sectors['right']:.2f}m "
                f"lower_left={sectors['lower_left']:.2f}m lower_right={sectors['lower_right']:.2f}m "
                f"goal_dist={distance:.2f}m heading_err={heading_error:.2f}rad "
                f"plan={self.last_plan.mode}:{self.last_plan.clearance:.2f}m/{self.last_plan.heading_error:.2f}rad "
                f"rollout={self.last_rollout_mode}:{self.last_rollout_reason} "
                f"global_route={self.global_route.status}:{self.global_route.reason} "
                f"global_waypoint={self.global_waypoint if self.global_waypoint is not None else 'none'} "
                f"cognitive_shadow={self.last_cognitive_confidence:.2f}:{self.last_cognitive_decision} "
                f"odysseus={self.last_odysseus_mode}:{self.last_odysseus_reason} "
                f"nav_intel={self.last_nav_motion_state}/{self.last_nav_safety_action}:{self.last_nav_reason} "
                f"learned_risk={self.last_learned_risk_score:.2f}/{self.last_learned_risk_action}:{self.last_learned_risk_reason} "
                f"semantic_hazard={semantic_forward:.2f}m/{semantic_lateral:.2f}m "
                f"cmd_v={linear:.2f} cmd_w={angular:.2f} pose_source={self.pose.source} "
                f"filtered_odom={'yes' if self.filtered_odom_pose is not None else 'no'} "
                f"vslam_pose={'yes' if self.vslam_pose is not None else 'no'} "
                f"target_marker_seen={self.target_marker_seen} rgb_encoding={self.rgb_encoding or 'none'} "
                f"diag={diag_summary} objects={object_summary}"
            )

    def log_command(
        self,
        now: float,
        mode: str,
        linear: float,
        angular: float,
        sectors: dict[str, float],
        distance: float,
        heading_error: float,
        semantic_forward: float,
        semantic_lateral: float,
    ) -> None:
        if self.command_log_writer is None:
            return
        pose = self.pose
        self.command_log_writer.writerow(
            {
                "time_sec": f"{now:.3f}",
                "mode": mode,
                "cmd_v": f"{linear:.4f}",
                "cmd_w": f"{angular:.4f}",
                "pose_x": f"{pose.x:.4f}" if pose is not None else "",
                "pose_y": f"{pose.y:.4f}" if pose is not None else "",
                "pose_yaw": f"{pose.yaw:.4f}" if pose is not None else "",
                "pose_source": pose.source if pose is not None else "",
                "goal_dist": f"{distance:.4f}",
                "heading_err": f"{heading_error:.4f}",
                "front": f"{sectors['front']:.4f}",
                "lower_front": f"{sectors['lower_front']:.4f}",
                "left": f"{sectors['left']:.4f}",
                "right": f"{sectors['right']:.4f}",
                "lower_left": f"{sectors['lower_left']:.4f}",
                "lower_right": f"{sectors['lower_right']:.4f}",
                "plan_mode": self.last_plan.mode,
                "plan_clearance": f"{self.last_plan.clearance:.4f}",
                "plan_heading": f"{self.last_plan.heading_error:.4f}",
                "rollout_mode": self.last_rollout_mode,
                "rollout_reason": self.last_rollout_reason,
                "global_route_status": self.global_route.status,
                "global_route_reason": self.global_route.reason,
                "global_route_length": f"{self.global_route.length_m:.4f}",
                "global_waypoint_x": f"{self.global_waypoint[0]:.4f}" if self.global_waypoint is not None else "",
                "global_waypoint_y": f"{self.global_waypoint[1]:.4f}" if self.global_waypoint is not None else "",
                "cognitive_decision": self.last_cognitive_decision,
                "cognitive_confidence": f"{self.last_cognitive_confidence:.4f}",
                "cognitive_memory_count": self.last_cognitive_memory_count,
                "cognitive_sample": self.last_cognitive_sample,
                "odysseus_mode": self.last_odysseus_mode,
                "odysseus_reason": self.last_odysseus_reason,
                "odysseus_trace": self.last_odysseus_trace,
                "odysseus_candidate": self.last_odysseus_candidate,
                "odysseus_sample": self.last_odysseus_sample,
                "nav_motion_state": self.last_nav_motion_state,
                "nav_safety_action": self.last_nav_safety_action,
                "nav_confidence": f"{self.last_nav_confidence:.4f}",
                "nav_reason": self.last_nav_reason,
                "learned_risk_score": f"{self.last_learned_risk_score:.4f}",
                "learned_risk_action": self.last_learned_risk_action,
                "learned_risk_reason": self.last_learned_risk_reason,
                "semantic_forward": f"{semantic_forward:.4f}",
                "semantic_lateral": f"{semantic_lateral:.4f}",
            }
        )
        if self.command_log_file is not None:
            self.command_log_file.flush()

    def independent_vslam_available(self) -> bool:
        if self.vslam_pose is None or self.last_vslam_stamp <= 0.0:
            return False
        try:
            diag = json.loads(self.last_diag) if self.last_diag else {}
        except json.JSONDecodeError:
            return True
        messages = diag.get("messages") if isinstance(diag, dict) else {}
        if not isinstance(messages, dict):
            return True
        status = str(messages.get("pose_status", "")).upper()
        source = str(messages.get("pose_source", "")).upper()
        if "FALLBACK" in status or "FALLBACK" in source or "LOST" in status:
            return False
        return True

    def apply_navigation_intelligence(
        self,
        now: float,
        mode: str,
        sectors: dict[str, float],
        distance: float,
        heading_error: float,
        linear: float,
        angular: float,
    ) -> tuple[str, float, float]:
        proposed = PlannerCommand(
            stamp=RuntimeStamp(now, "base_link", "factory_bot_stack_navigator"),
            linear_x=float(linear),
            angular_z=float(angular),
            mode=mode,
            reason=f"goal_dist={distance:.2f} heading_err={heading_error:.2f}",
        )
        independent_vslam = self.independent_vslam_available()
        visual_pose_age = now - self.last_vslam_stamp if independent_vslam and self.last_vslam_stamp > 0.0 else None
        visual_pose = pose2d_to_state(self.vslam_pose) if independent_vslam else None
        sample = NavigationIntelligenceInput(
            now_sec=now,
            proposed_command=proposed,
            control_pose=pose2d_to_state(self.pose),
            visual_pose=visual_pose,
            depth_signature=depth_signature_from_sectors(sectors, now),
            depth_age_sec=max(0.0, now - self.last_depth_stamp),
            control_pose_age_sec=max(0.0, now - self.last_pose_stamp),
            visual_pose_age_sec=visual_pose_age,
            in_recovery=mode in EMERGENCY_MODES or now < self.recovery_until,
            goal_distance_m=distance,
        )
        self.nav_learning_memory.observe(sample)
        learned = self.nav_risk_predictor.predict(
            self.nav_learning_memory.recent_sequence(self.nav_risk_predictor.window_size)
        )
        self.last_learned_risk_score = learned.risk_score
        self.last_learned_risk_action = learned.action
        self.last_learned_risk_reason = learned.reason[:120]
        if learned.available:
            sample = replace(
                sample,
                learned_risk_score=learned.risk_score,
                learned_risk_action=learned.action,
                learned_risk_reason=learned.reason,
            )
        decision = self.nav_intelligence.update(sample)
        self.last_nav_motion_state = decision.motion_state
        self.last_nav_safety_action = decision.safety_action
        self.last_nav_confidence = decision.confidence
        self.last_nav_reason = decision.reason[:120]

        if decision.override_command is None:
            return mode, linear, angular

        command = decision.override_command
        if decision.safety_action == "REVERSE":
            left_clear = min(sectors["left"], sectors["lower_left"])
            right_clear = min(sectors["right"], sectors["lower_right"])
            turn_sign = 1.0 if left_clear > right_clear else -1.0
            self.start_recovery(now, turn_sign)
            return "nav_intel_reverse", command.linear_x, -0.35 * turn_sign
        if decision.safety_action == "RECOVERY_TURN":
            turn_sign = self.choose_detour_sign(
                now,
                min(sectors["left"], sectors["lower_left"]),
                min(sectors["right"], sectors["lower_right"]),
                self.last_plan.heading_error,
                heading_error,
                lock_sec=2.0,
            )
            return "nav_intel_recovery_turn", 0.0, 0.80 * turn_sign
        if decision.safety_action == "RELOCALIZE":
            self.integral_heading = 0.0
            self.prev_heading_error = 0.0
            return "nav_intel_relocalize", 0.0, 0.0
        if decision.safety_action == "STOP":
            return "nav_intel_stop", 0.0, 0.0
        return command.mode, command.linear_x, command.angular_z

    def shape_command(self, mode: str, linear: float, angular: float) -> tuple[float, float]:
        max_angular = EMERGENCY_MAX_ANGULAR_SPEED if mode in EMERGENCY_MODES else NORMAL_MAX_ANGULAR_SPEED
        angular = float(np.clip(angular, -max_angular, max_angular))
        if mode not in EMERGENCY_MODES:
            delta = float(np.clip(angular - self.last_commanded_angular, -ANGULAR_SLEW_PER_TICK, ANGULAR_SLEW_PER_TICK))
            angular = self.last_commanded_angular + delta
        return linear, angular

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def close(self) -> None:
        if self.command_log_file is not None:
            self.command_log_file.flush()
            self.command_log_file.close()
            self.command_log_file = None

    def log_waiting(self, missing: str) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log > 1.0:
            self.last_log = now
            self.get_logger().info(f"waiting_for={missing}; navigator will not move without live perception + fused/control pose")


def pose2d_to_state(pose: Optional[Pose2D]) -> Optional[Pose2DState]:
    if pose is None:
        return None
    return Pose2DState(
        time_sec=pose.stamp,
        x_m=pose.x,
        y_m=pose.y,
        yaw_rad=pose.yaw,
        source=pose.source,
    )


def depth_signature_from_sectors(sectors: dict[str, float], now: float) -> DepthSceneSignature:
    return DepthSceneSignature(
        time_sec=now,
        front_m=float(sectors["front"]),
        lower_front_m=float(sectors["lower_front"]),
        left_m=float(sectors["left"]),
        right_m=float(sectors["right"]),
        lower_left_m=float(sectors["lower_left"]),
        lower_right_m=float(sectors["lower_right"]),
    )


def image_to_depth_m(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc == "32fc1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    elif enc in {"16uc1", "mono16"}:
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 0.001
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    arr[~np.isfinite(arr)] = np.nan
    return arr


def image_to_rgb(msg: Image) -> Optional[np.ndarray]:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc in {"rgb8", "bgr8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return arr
    if enc in {"rgba8", "bgra8"}:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4).copy()[:, :, :3]
        if enc == "bgra8":
            arr = arr[:, :, ::-1]
        return arr
    return None


def green_target_visible(rgb: Optional[np.ndarray]) -> bool:
    if rgb is None:
        return False
    h, w = rgb.shape[:2]
    roi = rgb[int(h * 0.18):int(h * 0.75), int(w * 0.25):int(w * 0.75), :].astype(np.int16)
    if roi.size == 0:
        return False
    r = roi[:, :, 0]
    g = roi[:, :, 1]
    b = roi[:, :, 2]
    green = (g > 95) & (g > r * 1.35) & (g > b * 1.25)
    return float(np.count_nonzero(green)) / float(green.size) > 0.012


def depth_occupancy_plan(depth: np.ndarray, goal_heading_error: float, detour_sign: float) -> LocalPlan:
    h, w = depth.shape[:2]
    # Stay above the floor-heavy lower image rows. Low obstacles are still handled by
    # the close-range lower sector checks in depth_sectors().
    band = depth[int(h * 0.20):int(h * 0.56):4, int(w * 0.08):int(w * 0.92):4]
    if band.size == 0:
        return LocalPlan(goal_heading_error, 8.0, "direct")

    ys, xs = np.indices(band.shape)
    x_offset = int(w * 0.08)
    cols = xs.reshape(-1) * 4 + x_offset
    ranges = band.reshape(-1).astype(np.float32)
    valid = np.isfinite(ranges) & (ranges > 0.25) & (ranges < 4.8)
    if np.count_nonzero(valid) < 20:
        return LocalPlan(goal_heading_error, 8.0, "direct")

    ranges = ranges[valid]
    cols = cols[valid]
    hfov = 1.047
    angles = (0.5 - (cols.astype(np.float32) / max(float(w - 1), 1.0))) * hfov
    forward = ranges * np.cos(angles)
    lateral = ranges * np.sin(angles)
    useful = (forward > 0.25) & (forward < 4.8) & (np.abs(lateral) < 2.2)
    if np.count_nonzero(useful) < 20:
        return LocalPlan(goal_heading_error, 8.0, "direct")
    forward = forward[useful]
    lateral = lateral[useful]

    candidates = np.linspace(-0.85, 0.85, 19)
    goal = float(np.clip(goal_heading_error, -0.85, 0.85))
    best_heading = goal
    best_clearance = 0.0
    best_cost = float("inf")
    for heading in candidates:
        corridor_center = forward * math.tan(float(heading))
        corridor_width = 0.42 + 0.10 * np.clip(forward, 0.0, 4.0)
        in_corridor = np.abs(lateral - corridor_center) < corridor_width
        clearance = float(np.percentile(forward[in_corridor], 15.0)) if np.any(in_corridor) else 8.0
        clearance_penalty = 2.1 / max(clearance, 0.25)
        blocked_penalty = 7.0 if clearance < 1.15 else 0.0
        turn_bias = 0.06 * abs(float(heading) - float(detour_sign) * 0.40)
        cost = 1.35 * abs(float(heading) - goal) + clearance_penalty + blocked_penalty + turn_bias
        if cost < best_cost:
            best_cost = cost
            best_heading = float(heading)
            best_clearance = clearance

    mode = "direct" if best_clearance > 3.0 and abs(best_heading - goal) < 0.16 else "occupancy_detour"
    return LocalPlan(float(best_heading), float(best_clearance), mode)


def depth_sectors(depth: np.ndarray) -> dict[str, float]:
    h, w = depth.shape[:2]
    # The upper band sees tall obstacles; the lower-middle band catches low boxes and pallets
    # without looking so far down that floor returns dominate the decision.
    upper_y0, upper_y1 = int(h * 0.18), int(h * 0.52)
    lower_y0, lower_y1 = int(h * 0.42), int(h * 0.68)
    bands = {
        "left": depth[upper_y0:upper_y1, int(w * 0.05):int(w * 0.35)],
        "front": depth[upper_y0:upper_y1, int(w * 0.35):int(w * 0.65)],
        "right": depth[upper_y0:upper_y1, int(w * 0.65):int(w * 0.95)],
        "lower_left": depth[lower_y0:lower_y1, int(w * 0.08):int(w * 0.35)],
        "lower_front": depth[lower_y0:lower_y1, int(w * 0.35):int(w * 0.65)],
        "lower_right": depth[lower_y0:lower_y1, int(w * 0.65):int(w * 0.92)],
    }
    out = {}
    for name, band in bands.items():
        valid = band[np.isfinite(band) & (band > 0.12)]
        if not valid.size:
            out[name] = 8.0
            continue
        broad_percentile = 25.0 if name.startswith("lower_") else 35.0
        broad = float(np.percentile(valid, broad_percentile))
        thin = float(np.percentile(valid, 1.0))
        thin_fraction = float(np.count_nonzero(valid < 1.55)) / float(valid.size)
        # Thin poles can occupy too few pixels to affect broad percentiles. Trust
        # close returns when they occupy even a small but nontrivial part of a sector.
        if thin < 1.55 and thin_fraction > 0.002:
            out[name] = thin
        else:
            out[name] = broad
    return out


def odom_to_pose2d(msg: Odometry, source: str) -> Pose2D:
    q = msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return Pose2D(
        x=float(msg.pose.pose.position.x),
        y=float(msg.pose.pose.position.y),
        yaw=yaw,
        source=source,
        stamp=float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
    )


def summarize_json(raw: str) -> str:
    if not raw:
        return "none"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "unparsed"
    parts = []
    for key in ["health", "tracking_ok", "camera_info_ready", "map_points", "semantic_objects", "last_error"]:
        if key in data:
            parts.append(f"{key}={data[key]}")
    messages = data.get("messages")
    if isinstance(messages, dict):
        for key in ["pose_source", "pose_status"]:
            if key in messages:
                parts.append(f"{key}={messages[key]}")
    return ",".join(parts) if parts else "ok"


def summarize_objects(raw: str) -> str:
    if not raw:
        return "none"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "unparsed"
    if not isinstance(data, list):
        return "none"
    return f"count={len(data)}"


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main() -> None:
    rclpy.init()
    node = FactoryBotStackNavigator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        try:
            node.odysseus_navigator.save_memory()
        except Exception:
            pass
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
