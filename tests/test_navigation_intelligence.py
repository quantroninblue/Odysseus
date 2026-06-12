from __future__ import annotations

import unittest

from runtime.core.contracts import PlannerCommand, RuntimeStamp
from runtime.core.navigation_intelligence import (
    DepthSceneSignature,
    NavigationIntelligence,
    NavigationIntelligenceInput,
    Pose2DState,
)


def command(t: float, linear: float = 0.34, angular: float = 0.0) -> PlannerCommand:
    return PlannerCommand(
        stamp=RuntimeStamp(t, "base_link", "test_planner"),
        linear_x=linear,
        angular_z=angular,
        mode="rollout_drive",
        reason="test",
    )


def pose(t: float, x: float, source: str = "ekf_odom_imu") -> Pose2DState:
    return Pose2DState(time_sec=t, x_m=x, y_m=0.0, yaw_rad=0.0, source=source)


def scene(t: float, front: float = 3.75, lower_front: float = 3.78) -> DepthSceneSignature:
    return DepthSceneSignature(
        time_sec=t,
        front_m=front,
        lower_front_m=lower_front,
        left_m=6.17,
        right_m=3.74,
        lower_left_m=6.17,
        lower_right_m=3.71,
    )


class NavigationIntelligenceTests(unittest.TestCase):
    def test_allows_forward_motion_when_scene_is_open(self):
        intelligence = NavigationIntelligence()
        last = None
        for t, x in ((0.0, 0.0), (0.8, 0.24), (1.7, 0.55), (2.5, 0.84)):
            last = intelligence.update(
                NavigationIntelligenceInput(
                    now_sec=t,
                    proposed_command=command(t),
                    control_pose=pose(t, x),
                    depth_signature=scene(t, front=8.0, lower_front=8.0),
                    depth_age_sec=0.02,
                    control_pose_age_sec=0.02,
                )
            )

        self.assertIsNotNone(last)
        self.assertEqual(last.safety_action, "ALLOW")
        self.assertIn(last.motion_state, {"MOVING", "OK"})

    def test_blocks_when_odom_moves_but_depth_scene_is_static_near_obstacle(self):
        intelligence = NavigationIntelligence()
        decision = None
        for t, x in ((0.0, 0.0), (0.6, 0.20), (1.2, 0.41), (1.8, 0.62)):
            decision = intelligence.update(
                NavigationIntelligenceInput(
                    now_sec=t,
                    proposed_command=command(t),
                    control_pose=pose(t, x),
                    depth_signature=scene(t),
                    visual_pose=None,
                    depth_age_sec=0.02,
                    control_pose_age_sec=0.02,
                )
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.motion_state, "BLOCKED")
        self.assertEqual(decision.safety_action, "REVERSE")
        self.assertIsNotNone(decision.override_command)
        self.assertLess(decision.override_command.linear_x, 0.0)
        self.assertIn("depth scene is static", decision.reason)

    def test_visual_motion_confirmation_prevents_false_blocked_detection(self):
        intelligence = NavigationIntelligence()
        decision = None
        for t, x in ((0.0, 0.0), (0.6, 0.20), (1.2, 0.41), (1.8, 0.62)):
            decision = intelligence.update(
                NavigationIntelligenceInput(
                    now_sec=t,
                    proposed_command=command(t),
                    control_pose=pose(t, x),
                    visual_pose=pose(t, x, source="vslam"),
                    visual_pose_age_sec=0.02,
                    depth_signature=scene(t),
                    depth_age_sec=0.02,
                    control_pose_age_sec=0.02,
                )
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.safety_action, "ALLOW")

    def test_stale_depth_stops_command(self):
        intelligence = NavigationIntelligence()

        decision = intelligence.update(
            NavigationIntelligenceInput(
                now_sec=5.0,
                proposed_command=command(5.0),
                control_pose=pose(5.0, 1.0),
                depth_signature=scene(4.0),
                depth_age_sec=1.3,
                control_pose_age_sec=0.02,
            )
        )

        self.assertEqual(decision.motion_state, "STALE_SENSORS")
        self.assertEqual(decision.safety_action, "STOP")
        self.assertEqual(decision.override_command.linear_x, 0.0)

    def test_pose_divergence_requests_relocalization(self):
        intelligence = NavigationIntelligence()

        decision = intelligence.update(
            NavigationIntelligenceInput(
                now_sec=3.0,
                proposed_command=command(3.0),
                control_pose=pose(3.0, 1.2),
                visual_pose=pose(3.0, 0.1, source="vslam"),
                visual_pose_age_sec=0.02,
                depth_signature=scene(3.0, front=5.0, lower_front=5.0),
                depth_age_sec=0.02,
                control_pose_age_sec=0.02,
            )
        )

        self.assertEqual(decision.motion_state, "POSE_DIVERGENCE")
        self.assertEqual(decision.safety_action, "RELOCALIZE")
        self.assertIsNotNone(decision.override_command)
        self.assertEqual(decision.override_command.linear_x, 0.0)


if __name__ == "__main__":
    unittest.main()
