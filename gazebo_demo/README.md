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

Terminal 4: run Odysseus navigation with deterministic safety authority and automatic learning.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
source install/setup.bash
python3 gazebo_demo/scripts/run_odysseus_auto.py
```

`run_odysseus_auto.py` creates the Odysseus sample directory, loads/saves
persistent world memory, loads an existing causal-attributor checkpoint when
available, runs the navigator, and trains a refreshed checkpoint after you stop
Terminal 4 if enough samples exist. It automatically uses `.venv/bin/python`
when that interpreter has both `rclpy` and `torch`, so live Odysseus can load the
MLP attributor instead of only training after shutdown. Use `--no-train` when
you want navigation only.

The navigator logs live perception state, including depth sectors, Odysseus mode/reason/trace, rollout evidence, navigation intelligence decision, distance to goal, heading error, command velocity, pose source, target marker visibility, stack diagnostics, and semantic object counts. It also writes every published command to `runtime_logs/navigator_commands_<timestamp>_<pid>.csv` with mode, pose, depth sectors, local plan, rollout reason, Odysseus trace/candidate/sample, navigation intelligence state/action/reason, semantic hazard snapshot, and `cmd_v/cmd_w`.

The navigator transforms live depth obstacles into a persistent world-frame costmap, inflates them by the robot safety radius, and runs A* from the current control pose to the target. A lookahead waypoint from that global route guides short-horizon trajectory rollout generation on the live local depth costmap. Odysseus owns the continuing navigation decision loop over those collision-checked candidates: it observes the previous choice's outcome, detects surprise or poor progress, and can switch between advance, recover, retrace, and explore behavior. Navigation intelligence then cross-checks Odysseus commands against odometry, independent VSLAM, and depth-scene change before publishing `cmd_vel`.

The global map starts empty; no factory obstacle coordinates are preloaded. Repeated depth observations add occupancy evidence, free-space rays clear stale evidence, and old obstacle evidence decays. Terminal 4 reports `global_route=route_ready` and the active `global_waypoint`, and the command CSV includes the global route status, reason, length, and waypoint.

The auto-runner writes owned causal episode samples to `artifacts/odysseus_samples`, keeps persistent world memory in `artifacts/odysseus_world_memory.json`, and updates `artifacts/odysseus_attributor.pt` after a run. Keep the memory file between runs to let Odysseus avoid previously failed routes in the same world. Each sample ties the state and candidate set to what actually happened on the next tick: progress, stuck/safety events, localization/stale-sensor signals, inferred failure cause, and severity.

The navigator also runs the Neural Cognitive Core foundation as a supporting world-model data path. It builds a versioned observation from RGB-D, pose, maps, route, semantics, and command history; updates memory; and logs a deterministic teacher decision. Set `COGNITIVE_DATASET_DIR=artifacts/cognitive_samples` to record owned temporal NPZ samples for cognitive-world-model training.

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

At startup the auto-runner should print the navigator interpreter. On this workstation it should prefer the venv:

```text
[odysseus-auto] navigator python: /home/neel-mukherjee/Desktop/semantic_spatial_mapping/.venv/bin/python
```

During navigation, useful fields are:

```text
odysseus=<mode>:<reason>
rollout=<mode>:<reason>
global_route=<status>:<reason>
nav_intel=<motion_state>/<safety_action>:<reason>
cmd_v=<linear> cmd_w=<angular>
```

Expected Odysseus behavior in clutter is not a single perfect line. You should see it advance, recover, retrace, and explore as it learns local structure. If it hits a close obstacle, reverse recovery candidates are now available and navigation intelligence allows recovery reverse/turn commands even inside the immediate stop zone.

Bad signs to investigate:

```text
rollout=fallback:'unknown Odysseus trace: ...'
```

This was fixed in the current version; if it reappears, Odysseus trace closure regressed. Repeated `nav_intel=BLOCKED/STOP` with no `odysseus_recover` or reverse motion means the robot is physically boxed in or the local costmap is rejecting all escape candidates.

If `learned_risk` is `UNAVAILABLE`, that only refers to the legacy GRU risk model. Odysseus uses its own causal attributor via `artifacts/odysseus_attributor.pt` when available.

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
