from __future__ import annotations

import math
import unittest

import numpy as np

from planning.global_costmap import GlobalCostmap
from planning.global_route import AStarRoutePlanner, select_lookahead_waypoint


class GlobalPlannerTests(unittest.TestCase):
    def make_costmap(self) -> GlobalCostmap:
        return GlobalCostmap(
            x_min_m=-1.0,
            x_max_m=6.0,
            y_min_m=-3.0,
            y_max_m=3.0,
            resolution_m=0.10,
        )

    def test_local_observations_are_transformed_into_world_frame(self):
        costmap = self.make_costmap()
        points = np.array([[1.0, 0.0]], dtype=np.float32)

        costmap.update_from_local_points(
            points,
            pose_x_m=1.0,
            pose_y_m=1.0,
            pose_yaw_rad=math.pi / 2.0,
            now_sec=1.0,
        )
        costmap.update_from_local_points(
            points,
            pose_x_m=1.0,
            pose_y_m=1.0,
            pose_yaw_rad=math.pi / 2.0,
            now_sec=1.1,
        )

        obstacle_cell = costmap.world_to_grid(1.0, 2.0)
        self.assertIsNotNone(obstacle_cell)
        self.assertGreaterEqual(costmap.evidence[obstacle_cell], costmap.occupied_threshold)

    def test_astar_routes_through_wall_gap(self):
        costmap = self.make_costmap()
        wall = np.array(
            [[2.5, y] for y in np.arange(-2.5, 2.6, 0.10) if not 0.8 < y < 1.4],
            dtype=np.float32,
        )
        costmap.add_obstacles_world(wall)
        planner = AStarRoutePlanner(inflation_radius_m=0.18)

        route = planner.plan(costmap, start_xy=(0.0, 0.0), goal_xy=(5.0, 0.0))

        self.assertEqual(route.status, "route_ready")
        self.assertGreater(max(y for _, y in route.waypoints), 0.65)
        self.assertGreater(route.length_m, 5.0)
        waypoint = select_lookahead_waypoint(route, (0.0, 0.0))
        self.assertIsNotNone(waypoint)

    def test_astar_returns_direct_route_in_empty_map(self):
        costmap = self.make_costmap()
        route = AStarRoutePlanner(inflation_radius_m=0.20).plan(
            costmap,
            start_xy=(0.0, 0.0),
            goal_xy=(4.0, 0.0),
        )

        self.assertEqual(route.status, "route_ready")
        self.assertEqual(len(route.waypoints), 2)
        self.assertAlmostEqual(route.length_m, 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
