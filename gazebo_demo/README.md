# Gazebo Demo

This folder contains a self-contained Gazebo Sim factory obstacle world and a live factory_bot navigation demo wired into the perception/VSLAM runtime topics.

For committed architecture state, see `../README.md` and `../notes/architecture_contracts.md`.

World file:

```text
gazebo_demo/factory_obstacle_demo.world
```

The world contains a warehouse/factory floor, boundary walls, racks, pallets, crates, barrels, safety posts, a marked start zone, a visible green arrival target, and `factory_bot`, a boxy tracked robot used for internal demo runs.

The demo intentionally does not preload obstacle locations into the navigator. The robot receives a target direction/displacement, consumes live RGB, depth, odom, and semantic stack diagnostics, then adjusts speed and steering from what the sensors report.

## Runtime Commands

Use four terminals for the full demo. Keep Terminal 4 visible during screen recording if you want perception/navigation logs on screen.

Terminal 1: open Gazebo Sim with the factory world.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
gz sim -r gazebo_demo/factory_obstacle_demo.world
```

The `-r` flag starts the simulation running immediately.

Terminal 2: bridge Gazebo camera, depth, IMU, odom, and command topics into ROS.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
ros2 launch gazebo_demo/launch/factory_bot_bridge.launch.py
```

Terminal 3: run odometry+IMU EKF fusion plus the perception/VSLAM stack against the bridged Gazebo topics.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
source install/setup.bash
ros2 launch semantic_spatial_mapping_ros gazebo_fused_runtime.launch.py
```

For the older no-EKF path, use `gazebo_runtime.launch.py` instead.

If the ROS workspace has not been built yet:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
colcon build
source install/setup.bash
```

Terminal 4: run the rollout navigator with navigation intelligence.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
source install/setup.bash
python3 gazebo_demo/scripts/factory_bot_stack_navigator.py
```

The navigator logs live perception state, including depth sectors, rollout decision, navigation intelligence decision, mode, distance to goal, heading error, command velocity, pose source, target marker visibility, stack diagnostics, and semantic object counts. It also writes every published command to `runtime_logs/navigator_commands_<timestamp>_<pid>.csv` with mode, pose, depth sectors, local plan, rollout reason, navigation intelligence state/action/reason, semantic hazard snapshot, and `cmd_v/cmd_w`.

The navigator builds an inflated local obstacle map from live depth and uses a short-horizon trajectory rollout planner for normal motion. A navigation intelligence layer then cross-checks proposed commands against odometry, independent VSLAM, and depth-scene change before publishing `cmd_vel`. It keeps emergency recovery checks for boxed-in, stale, false-odom blocked, and badly off-course cases.

The navigator subscribes to `/semantic_spatial/visual_odometry` for stack
visibility. It uses `/odometry/filtered` from the odom+IMU EKF as the control pose
when available, and falls back to raw `/odom` if the EKF output is stale or absent.
The semantic runtime uses the same preferred-filtered-odom/raw-odom-fallback
contract in the fused Gazebo profile.

Optional learned risk model workflow:

```bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
python3 tools/build_navigation_learning_dataset.py runtime_logs/navigator_commands_*.csv -o artifacts/navigation_risk_dataset.npz --window-size 16
PYTHONPATH=. .venv/bin/python tools/train_navigation_risk_gru.py artifacts/navigation_risk_dataset.npz -o artifacts/nav_gru_risk.pt --epochs 30
NAV_GRU_RISK_CHECKPOINT=artifacts/nav_gru_risk.pt python3 gazebo_demo/scripts/factory_bot_stack_navigator.py
```

The GRU model is a risk predictor, not the final safety authority. The navigator still keeps deterministic safety checks for stale sensors, false odometry progress, immediate obstacles, and pose divergence. Train only from labeled good/faulty runs; untrained smoke-test checkpoints are not useful for live behavior.

## Topic Contract

Gazebo publishes scoped transport topics; `factory_bot_bridge.launch.py` maps them to the stable ROS topics used by the stack.

ROS topics used by the perception/VSLAM stack:

- `/camera/color/image_raw`
- `/camera/color/camera_info`
- `/camera/depth/image_raw`
- `/camera/depth/camera_info`
- `/odom`
- `/imu/data`
- `/odometry/filtered`
- `/semantic_spatial/diagnostics`
- `/semantic_spatial/objects`
- `/semantic_spatial/visual_odometry`

Command path:

- navigator publishes ROS `geometry_msgs/msg/Twist` on `/factory_bot/cmd_vel`
- bridge forwards it to Gazebo `/model/factory_bot/cmd_vel`
- Gazebo DiffDrive publishes `/model/factory_bot/odometry`, bridged back to ROS `/odom`
- Gazebo `base_imu` publishes `/model/factory_bot/imu`, bridged back to ROS `/imu/data`
- `robot_localization` fuses `/odom` and `/imu/data` into ROS `/odometry/filtered`

## What To Watch In Terminal 4

The earlier failure mode was: `cmd_v` stayed positive, `/odometry/filtered` claimed progress, but the front/lower-front depth scene stayed nearly unchanged while the robot pushed into poles. In the `Vector_NN` path, the expected safety signature near that failure is:

```text
nav_intel=BLOCKED/REVERSE:odom reports forward progress while depth scene is static near an obstacle
mode=nav_intel_reverse
```

If `learned_risk` is `UNAVAILABLE`, the GRU checkpoint is not loaded and deterministic safety is still active. If a checkpoint is loaded, `learned_risk=<score>/CAUTION` or `HIGH_RISK` may slow or stop earlier, but deterministic safety remains the hard guardrail.

## Manual Checks

List Gazebo topics:

```bash
gz topic -l | grep factory_bot
```

List ROS topics after the bridge is running:

```bash
ros2 topic list
```

Stop the robot manually:

```bash
ros2 topic pub --once /factory_bot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

Send a manual forward command for actuator testing only:

```bash
ros2 topic pub /factory_bot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.25}, angular: {z: 0.0}}" -r 10
```

Stop it with Ctrl-C, then send the zero command above.

## Notes

- `factory_bot_demo_driver.py` is kept as a simple actuator smoke test, but the intended demo path is `factory_bot_stack_navigator.py` with the ROS bridge and perception/VSLAM stack running.
- The target used by the navigator is an odom-frame displacement from the spawn area to the green arrival target across the warehouse.
- The stack logs still provide the deeper runtime report files under `runtime_logs/` when the ROS node is running.
