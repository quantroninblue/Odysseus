#!/usr/bin/env python3
"""Reactive perception/VSLAM demo navigator for factory_bot.

Inputs:
- /camera/depth/image_raw for obstacle sectors
- /semantic_spatial/visual_odometry if available, otherwise /odom for pose
- /semantic_spatial/diagnostics and /semantic_spatial/objects for stack logs

Output:
- ROS velocity commands on /factory_bot/cmd_vel, bridged to Gazebo Sim

This is a live demo navigator: it knows the target coordinate, but it does not
know obstacle locations ahead of time. It uses depth perception to slow down,
turn away from obstacles, and speed up when the path is clear.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from rclpy.executors import ExternalShutdownException

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

ROS_CMD_TOPIC = "/factory_bot/cmd_vel"
TARGET_X = 14.0
TARGET_Y = 10.2
MAX_FORWARD_SPEED = 0.30
LOW_OBSTACLE_SLOW_DIST = 4.25
LOW_OBSTACLE_REROUTE_DIST = 3.25
NEAR_OBSTACLE_DIST = 1.35
ESCAPE_OBSTACLE_DIST = 0.95


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
        self.vslam_pose: Optional[Pose2D] = None
        self.last_diag = ""
        self.last_objects = ""
        self.semantic_objects = []
        self.last_log = 0.0
        self.last_depth_stamp = 0.0
        self.last_pose_stamp = 0.0
        self.integral_heading = 0.0
        self.prev_heading_error = 0.0
        self.detour_sign = 1.0
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
        self.last_plan = LocalPlan(0.0, 8.0, "direct")
        self.last_motion_check_time = 0.0
        self.last_motion_check_x = 0.0
        self.last_motion_check_y = 0.0
        self.stuck_since = 0.0

        self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_rgb, 10)
        self.create_subscription(Odometry, "/semantic_spatial/visual_odometry", self.on_vo, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/semantic_spatial/diagnostics", self.on_diag, 10)
        self.create_subscription(String, "/semantic_spatial/objects", self.on_objects, 10)
        self.cmd_pub = self.create_publisher(Twist, ROS_CMD_TOPIC, 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info("Factory bot stack navigator waiting for depth + pose/VSLAM")

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

    def on_odom(self, msg: Odometry) -> None:
        self.odom_pose = odom_to_pose2d(msg, "odom_control")
        self.pose = self.odom_pose
        self.last_pose_stamp = self.get_clock().now().nanoseconds * 1e-9

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
            self.log_waiting("pose/VSLAM")
            self.publish_cmd(0.0, 0.0)
            return
        if now - self.last_pose_stamp > 1.0:
            self.log_waiting("fresh pose/VSLAM")
            self.publish_cmd(0.0, 0.0)
            return

        sectors = depth_sectors(self.depth)
        dx = TARGET_X - self.pose.x
        dy = TARGET_Y - self.pose.y
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        heading_error = wrap_angle(target_yaw - self.pose.yaw)

        if distance < 0.35 or (distance < 0.85 and self.target_marker_seen):
            self.arrived = True
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info(
                f"ARRIVED target=({TARGET_X:.1f},{TARGET_Y:.1f}) pose=({self.pose.x:.2f},{self.pose.y:.2f}) "
                f"target_marker_seen={self.target_marker_seen}"
            )
            return

        front = sectors["front"]
        left = sectors["left"]
        right = sectors["right"]
        lower_front = sectors["lower_front"]
        lower_left = sectors["lower_left"]
        lower_right = sectors["lower_right"]
        front_close = min(front, lower_front)
        left_clear = min(left, lower_left)
        right_clear = min(right, lower_right)
        plan = depth_occupancy_plan(self.depth, heading_error, self.detour_sign)
        self.last_plan = plan
        front_clear = min(front, plan.clearance, lower_front)
        near = min(front_close, left_clear, right_clear)
        stuck = self.update_stuck_state(now, distance)

        if self.best_distance == float("inf"):
            self.best_distance = distance
            self.last_best_time = now
        elif distance < self.best_distance - 0.10:
            self.best_distance = distance
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
        stalled = distance > 0.9 and self.last_commanded_linear > 0.12 and now - self.last_best_time > 3.0
        if boxed_in or stuck or stalled:
            if abs(left_clear - right_clear) > 0.15:
                self.detour_sign = 1.0 if left_clear > right_clear else -1.0
            elif abs(plan.heading_error) > 0.10:
                self.detour_sign = 1.0 if plan.heading_error > 0.0 else -1.0
            elif abs(heading_error) > 0.10:
                self.detour_sign = 1.0 if heading_error > 0.0 else -1.0
            self.start_recovery(now, self.detour_sign)
            self.publish_and_log(now, "escape_reverse", sectors, distance, heading_error, -0.16, -0.35 * self.detour_sign)
            return

        mode = "clear_path"
        control_heading_error = heading_error
        if front_close < NEAR_OBSTACLE_DIST:
            if abs(left_clear - right_clear) > 0.15:
                self.detour_sign = 1.0 if left_clear > right_clear else -1.0
            linear = 0.00
            control_heading_error = 0.85 * self.detour_sign
            obstacle_bias = 0.85 * self.detour_sign
            self.detour_until = now + 1.5
            mode = "blocked_turn"
        elif lower_front < LOW_OBSTACLE_REROUTE_DIST or front_clear < 2.45:
            if abs(left_clear - right_clear) > 0.12:
                self.detour_sign = 1.0 if left_clear > right_clear else -1.0
            elif abs(plan.heading_error) > 0.10:
                self.detour_sign = 1.0 if plan.heading_error > 0.0 else -1.0
            linear = 0.06
            control_heading_error = max(abs(plan.heading_error), 0.75) * self.detour_sign
            obstacle_bias = 0.65 * self.detour_sign
            self.detour_until = now + 2.2
            mode = "reroute_low_obstacle" if lower_front < LOW_OBSTACLE_REROUTE_DIST else "reroute_near_obstacle"
        elif plan.mode == "occupancy_detour" and plan.clearance < 2.7:
            if abs(left_clear - right_clear) > 0.20:
                self.detour_sign = 1.0 if left_clear > right_clear else -1.0
            elif abs(plan.heading_error) > 0.08:
                self.detour_sign = 1.0 if plan.heading_error > 0.0 else -1.0
            linear = 0.14
            control_heading_error = plan.heading_error
            obstacle_bias = 0.35 * self.detour_sign
            self.detour_until = now + 1.0
            mode = "depth_occupancy_reroute"
        elif now < self.detour_until and abs(heading_error) < 1.0:
            linear = 0.12
            control_heading_error = max(abs(heading_error), 0.35) * self.detour_sign
            obstacle_bias = 0.22 * self.detour_sign
            mode = "detour_commit"
        elif lower_front < LOW_OBSTACLE_SLOW_DIST:
            if abs(left_clear - right_clear) > 0.15:
                self.detour_sign = 1.0 if left_clear > right_clear else -1.0
            linear = 0.14
            control_heading_error = max(abs(heading_error), 0.22) * self.detour_sign
            obstacle_bias = 0.22 * self.detour_sign
            mode = "low_obstacle_caution"
        elif near < 1.4:
            linear = 0.12
            obstacle_bias = -0.45 if left_clear < right_clear else 0.45
            mode = "side_obstacle"
        elif front > 3.0 and abs(heading_error) < 0.45:
            linear = MAX_FORWARD_SPEED if not self.target_marker_seen else 0.24
            obstacle_bias = 0.0
            mode = "open_path_speedup"
        else:
            linear = 0.18
            obstacle_bias = 0.0

        if abs(control_heading_error) > 1.2:
            linear = min(linear, 0.08)

        self.integral_heading = float(np.clip(self.integral_heading + control_heading_error * 0.2, -1.0, 1.0))
        derivative = (control_heading_error - self.prev_heading_error) / 0.2
        self.prev_heading_error = control_heading_error
        if mode in {"depth_occupancy_reroute", "detour_commit", "reroute_near_obstacle", "blocked_turn", "side_obstacle"}:
            heading_gain = 0.28
            derivative_gain = 0.010
        else:
            heading_gain = 0.85
            derivative_gain = 0.035
        angular = heading_gain * control_heading_error + 0.02 * self.integral_heading + derivative_gain * derivative + obstacle_bias
        angular = float(np.clip(angular, -0.95, 0.95))
        if abs(angular) > 0.80:
            linear = min(linear, 0.12)
        elif abs(angular) > 0.55:
            linear = min(linear, 0.22)
        linear = float(np.clip(linear, -0.18, MAX_FORWARD_SPEED))

        self.publish_and_log(now, mode, sectors, distance, heading_error, linear, angular)

    def start_recovery(self, now: float, turn_sign: float) -> None:
        self.recovery_turn_sign = float(1.0 if turn_sign >= 0.0 else -1.0)
        self.recovery_phase = "escape"
        self.recovery_reverse_until = now + 1.8
        self.recovery_until = now + 4.2
        self.detour_until = now + 4.5
        self.last_best_time = now
        self.stuck_since = 0.0

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

    def semantic_object_hazard(self) -> tuple[float, float]:
        if self.pose is None or not self.semantic_objects:
            return 8.0, 0.0
        c = math.cos(-self.pose.yaw)
        s = math.sin(-self.pose.yaw)
        best_forward = 8.0
        best_lateral = 0.0
        for obj in self.semantic_objects:
            centroid = obj.get("centroid") if isinstance(obj, dict) else None
            if not isinstance(centroid, list) or len(centroid) < 2:
                continue
            dx = float(centroid[0]) - self.pose.x
            dy = float(centroid[1]) - self.pose.y
            forward = c * dx - s * dy
            lateral = s * dx + c * dy
            if 0.2 < forward < 4.0 and abs(lateral) < 1.15 and forward < best_forward:
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
        self.publish_cmd(linear, angular)
        self.last_commanded_linear = linear
        self.last_commanded_angular = angular
        if now - self.last_log > 0.8:
            self.last_log = now
            diag_summary = summarize_json(self.last_diag)
            object_summary = summarize_objects(self.last_objects)
            self.get_logger().info(
                "perception "
                f"mode={mode} depth_encoding={self.depth_encoding} "
                f"front={sectors['front']:.2f}m lower_front={sectors['lower_front']:.2f}m "
                f"left={sectors['left']:.2f}m right={sectors['right']:.2f}m "
                f"goal_dist={distance:.2f}m heading_err={heading_error:.2f}rad "
                f"plan={self.last_plan.mode}:{self.last_plan.clearance:.2f}m/{self.last_plan.heading_error:.2f}rad "
                f"cmd_v={linear:.2f} cmd_w={angular:.2f} pose_source={self.pose.source} "
                f"vslam_pose={'yes' if self.vslam_pose is not None else 'no'} "
                f"target_marker_seen={self.target_marker_seen} rgb_encoding={self.rgb_encoding or 'none'} "
                f"diag={diag_summary} objects={object_summary}"
            )

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def log_waiting(self, missing: str) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log > 1.0:
            self.last_log = now
            self.get_logger().info(f"waiting_for={missing}; navigator will not move without live perception + pose")


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
    angles = ((cols.astype(np.float32) / max(float(w - 1), 1.0)) - 0.5) * hfov
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
    lower_y0, lower_y1 = int(h * 0.46), int(h * 0.72)
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
        percentile = 25.0 if name.startswith("lower_") else 35.0
        out[name] = float(np.percentile(valid, percentile)) if valid.size else 8.0
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
