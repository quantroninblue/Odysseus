from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from runtime.core.contracts import PlannerCommand, RuntimeStamp
from runtime.core.navigation_intelligence import (
    DepthSceneSignature,
    NavigationIntelligence,
    NavigationIntelligenceInput,
    Pose2DState,
)
from runtime.core.navigation_learning import (
    FEATURE_NAMES,
    GRUNavigationRiskModel,
    NavigationLearningMemory,
    NavigationRiskPredictor,
    build_sequence_dataset_from_csv,
    torch,
)


def command(t: float, linear: float = 0.34) -> PlannerCommand:
    return PlannerCommand(
        stamp=RuntimeStamp(t, "base_link", "test"),
        linear_x=linear,
        angular_z=0.0,
        mode="rollout_drive",
        reason="test",
    )


def sample(t: float, x: float, front: float = 3.75, goal: float = 10.0) -> NavigationIntelligenceInput:
    return NavigationIntelligenceInput(
        now_sec=t,
        proposed_command=command(t),
        control_pose=Pose2DState(t, x, 0.0, 0.0, "ekf_odom_imu"),
        visual_pose=None,
        depth_signature=DepthSceneSignature(t, front, 3.78, 6.17, 3.74, 6.17, 3.71),
        depth_age_sec=0.02,
        control_pose_age_sec=0.02,
        goal_distance_m=goal,
    )


class NavigationLearningTests(unittest.TestCase):
    def test_engineered_memory_tracks_static_scene_and_command_error(self):
        memory = NavigationLearningMemory()
        frame0 = memory.observe(sample(0.0, 0.0, goal=10.0))
        frame1 = memory.observe(sample(1.0, 0.02, goal=9.7))

        self.assertEqual(frame0.values.shape, (len(FEATURE_NAMES),))
        values = dict(zip(FEATURE_NAMES, frame1.values))
        self.assertGreater(values["static_scene_duration_sec"], 0.0)
        self.assertGreater(values["cumulative_cmd_odom_error_m"], 0.20)
        self.assertLessEqual(values["best_goal_delta_m"], 0.0)

    def test_predictor_reports_unavailable_without_loaded_model(self):
        predictor = NavigationRiskPredictor()
        sequence = np.zeros((predictor.window_size, len(FEATURE_NAMES)), dtype=np.float32)

        assessment = predictor.predict(sequence)

        self.assertFalse(assessment.available)
        self.assertEqual(assessment.action, "UNAVAILABLE")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_gru_predictor_scores_sequence_when_model_is_loaded(self):
        predictor = NavigationRiskPredictor(window_size=4)
        predictor.set_model(GRUNavigationRiskModel(input_size=len(FEATURE_NAMES), hidden_size=8))
        sequence = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)

        assessment = predictor.predict(sequence)

        self.assertTrue(assessment.available)
        self.assertGreaterEqual(assessment.risk_score, 0.0)
        self.assertLessEqual(assessment.risk_score, 1.0)
        self.assertIn(assessment.action, {"LOW_RISK", "CAUTION", "HIGH_RISK"})

    def test_csv_dataset_builder_labels_navigation_fault_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "navigator.csv"
            fields = [
                "time_sec",
                "mode",
                "cmd_v",
                "cmd_w",
                "pose_x",
                "pose_y",
                "goal_dist",
                "front",
                "lower_front",
                "left",
                "right",
                "lower_left",
                "lower_right",
                "nav_motion_state",
                "nav_safety_action",
            ]
            rows = []
            for idx in range(6):
                rows.append(
                    {
                        "time_sec": str(float(idx)),
                        "mode": "rollout_drive" if idx < 5 else "nav_intel_reverse",
                        "cmd_v": "0.34",
                        "cmd_w": "0.0",
                        "pose_x": str(0.2 * idx),
                        "pose_y": "0.0",
                        "goal_dist": str(10.0 - 0.2 * idx),
                        "front": "3.75",
                        "lower_front": "3.78",
                        "left": "6.17",
                        "right": "3.74",
                        "lower_left": "6.17",
                        "lower_right": "3.71",
                        "nav_motion_state": "BLOCKED" if idx == 5 else "MOVING",
                        "nav_safety_action": "REVERSE" if idx == 5 else "ALLOW",
                    }
                )
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            x, y = build_sequence_dataset_from_csv([path], window_size=3)

        self.assertEqual(x.shape, (4, 3, len(FEATURE_NAMES)))
        self.assertEqual(y.shape, (4,))
        self.assertEqual(int(y[-1]), 1)

    def test_learned_high_risk_can_stop_before_deterministic_blocked_timeout(self):
        intelligence = NavigationIntelligence()
        decision = intelligence.update(
            NavigationIntelligenceInput(
                now_sec=0.5,
                proposed_command=command(0.5),
                control_pose=Pose2DState(0.5, 0.2, 0.0, 0.0, "ekf_odom_imu"),
                depth_signature=DepthSceneSignature(0.5, 3.0, 3.1, 5.0, 5.0, 5.0, 5.0),
                depth_age_sec=0.02,
                control_pose_age_sec=0.02,
                learned_risk_score=0.90,
                learned_risk_action="HIGH_RISK",
                learned_risk_reason="test GRU risk",
            )
        )

        self.assertEqual(decision.motion_state, "SLIPPING")
        self.assertEqual(decision.safety_action, "STOP")
        self.assertIn("learned GRU risk", decision.reason)


if __name__ == "__main__":
    unittest.main()
