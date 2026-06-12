from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "gazebo_demo"
    / "scripts"
    / "factory_bot_stack_navigator.py"
)

try:
    spec = importlib.util.spec_from_file_location("factory_bot_stack_navigator", SCRIPT)
    navigator = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = navigator
    spec.loader.exec_module(navigator)
    NAVIGATOR_AVAILABLE = True
except ImportError:
    NAVIGATOR_AVAILABLE = False


@unittest.skipUnless(NAVIGATOR_AVAILABLE, "ROS2 Python packages are not available")
class GazeboNavigatorPolicyTests(unittest.TestCase):
    def test_occupancy_plan_left_gap_has_positive_ros_heading(self):
        depth = np.full((100, 100), 8.0, dtype=np.float32)
        depth[20:56, 45:90] = 1.0

        plan = navigator.depth_occupancy_plan(
            depth,
            goal_heading_error=0.0,
            detour_sign=1.0,
        )

        self.assertEqual(plan.mode, "occupancy_detour")
        self.assertGreater(plan.heading_error, 0.0)

    def test_depth_sectors_detect_thin_close_front_pole(self):
        depth = np.full((100, 100), 6.0, dtype=np.float32)
        depth[30:66, 49:51] = 0.85

        sectors = navigator.depth_sectors(depth)

        self.assertLess(sectors["front"], 1.0)



if __name__ == "__main__":
    unittest.main()
