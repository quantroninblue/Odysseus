from __future__ import annotations

import unittest

import numpy as np

from planning.local_costmap import build_local_costmap_from_depth
from planning.trajectory_rollout import TrajectoryRolloutPlanner


class LocalPlannerTests(unittest.TestCase):
    def test_costmap_inflates_center_obstacle(self):
        depth = np.full((120, 160), 8.0, dtype=np.float32)
        depth[35:70, 78:82] = 1.2

        costmap = build_local_costmap_from_depth(depth, resolution_m=0.08, inflation_radius_m=0.45)

        self.assertTrue(costmap.is_occupied(1.2, 0.0))
        self.assertTrue(costmap.is_occupied(1.2, 0.30))
        self.assertFalse(costmap.is_occupied(1.2, 1.20))

    def test_rollout_drives_forward_in_open_space(self):
        depth = np.full((120, 160), 8.0, dtype=np.float32)
        costmap = build_local_costmap_from_depth(depth)
        planner = TrajectoryRolloutPlanner()

        result = planner.plan(costmap, goal_heading_error=0.0, now_sec=1.0)

        self.assertGreater(result.command.linear_x, 0.20)
        self.assertLess(abs(result.command.angular_z), 0.20)
        self.assertEqual(result.command.mode, "rollout_drive")

    def test_rollout_rejects_straight_path_into_center_obstacle(self):
        depth = np.full((120, 160), 8.0, dtype=np.float32)
        depth[35:70, 77:83] = 1.0
        costmap = build_local_costmap_from_depth(depth, resolution_m=0.08, inflation_radius_m=0.45)
        planner = TrajectoryRolloutPlanner()

        result = planner.plan(costmap, goal_heading_error=0.0, now_sec=1.0, detour_hint=1.0)
        straight_collisions = [
            candidate
            for candidate in result.candidates
            if candidate.linear_x > 0.20 and abs(candidate.angular_z) < 0.05 and not candidate.accepted
        ]

        self.assertTrue(straight_collisions)
        self.assertNotAlmostEqual(result.command.angular_z, 0.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
