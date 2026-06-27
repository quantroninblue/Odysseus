# Odysseus Navigation AI

Odysseus is an embodied navigation AI for obstacle-laden factory environments. It uses RGB-D perception, semantic mapping, VSLAM/odometry, persistent occupancy mapping, A* route search, local trajectory rollout, causal attribution, and cross-run world memory to reach a designated destination while learning from failed paths.

Perception, mapping, VSLAM, and planning are the sensorimotor substrate. Odysseus is the product layer that owns the continuing navigation loop: it observes the world, chooses from collision-checked candidate motions, sees what actually happened, updates memory, marks bad regions, retraces or explores when needed, and records causal training samples.

## What Odysseus Does

```text
RGB-D + CameraInfo + odometry/IMU/VSLAM
    -> semantic RGB-D mapping and object memory
    -> persistent world-frame occupancy costmap
    -> A* global route and lookahead waypoint
    -> inflated local depth costmap
    -> collision-checked rollout candidates
    -> Odysseus embodied navigation loop
       (advance / recover / retrace / explore)
    -> causal outcome closure + persistent world memory
    -> optional MLP causal attributor checkpoint
    -> navigation intelligence hard safety authority
    -> cmd_vel + logs + owned training samples
```

Odysseus currently learns in three ways:

- Online world memory: failed cells become no-go regions, successful cells become corridor hints, and behavior values are updated every tick.
- Causal samples: each decision is closed with observed progress, stuck/safety events, localization/stale-sensor signals, inferred failure cause, and severity.
- Causal MLP: `runtime/core/odysseus/attribution.py` trains a PyTorch MLP that predicts failure cause, progress, stuck/collision/safety risk, localization risk, stale-sensor risk, and severity. The auto-runner loads the latest checkpoint when present.

Hard safety is still deterministic. Odysseus may make policy mistakes so it can learn, but navigation intelligence can still stop, reverse, recover, or request relocalization before `cmd_vel` is published.

## Repository Layout

```text
runtime/core/odysseus/        Odysseus contracts, world memory, causal MLP, navigator
planning/                     global costmap, A*, local costmap, trajectory rollout
runtime/core/cognition/       neural world-model foundation and teacher data path
runtime/core/                 runtime contracts, diagnostics, safety authority, learning
ros/semantic_spatial_mapping_ros/
                              ROS2 package, launch files, node, publishers, converters
motion/vo/                    visual odometry and VSLAM scaffolding
mapping/global_map/           bounded world map and semantic object map
geometry/                     RGB-D geometry, camera models, transforms, point clouds
gazebo_demo/                  factory world, bridge launch, Odysseus auto-runner
tests/                        headless planner, Odysseus, runtime, safety, ROS converter tests
tools/                        Odysseus, cognitive model, and legacy GRU training CLIs
```

## Install

Recommended local setup on Ubuntu with ROS2 Jazzy:

```bash
cd /home/neel-mukherjee/Desktop/odysseus
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy pytest torch
```

Install/source ROS2 Jazzy with at least:

```text
rclpy sensor_msgs nav_msgs geometry_msgs std_msgs tf2_ros robot_localization launch launch_ros rosbag2
```

Build the ROS package from the repo root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select semantic_spatial_mapping_ros
source install/setup.bash
```

If using `requirements/python_requirements.txt`, convert it from UTF-16 LE before installing:

```bash
iconv -f UTF-16LE -t UTF-8 requirements/python_requirements.txt > /tmp/odysseus_python_requirements.txt
pip install -r /tmp/odysseus_python_requirements.txt
```

## Run Odysseus In Gazebo

Use four terminals. Terminal 4 is automated: it creates sample/memory directories, loads a causal checkpoint if present, runs Odysseus, and trains a refreshed checkpoint after shutdown when enough samples exist.

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/odysseus
gz sim -r gazebo_demo/factory_obstacle_demo.world
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/odysseus
ros2 launch gazebo_demo/launch/factory_bot_bridge.launch.py
```

Terminal 3:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/odysseus
colcon build --packages-select semantic_spatial_mapping_ros
source install/setup.bash
ros2 launch semantic_spatial_mapping_ros gazebo_fused_runtime.launch.py
```

Terminal 4:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/odysseus
source install/setup.bash
python3 gazebo_demo/scripts/run_odysseus_auto.py
```

Useful Terminal 4 options:

```bash
python3 gazebo_demo/scripts/run_odysseus_auto.py --no-train
python3 gazebo_demo/scripts/run_odysseus_auto.py --epochs 10 --min-samples 8
```

Default Odysseus artifacts:

```text
artifacts/odysseus_samples/              owned causal samples
artifacts/odysseus_world_memory.json     persistent no-go/success/action memory
artifacts/odysseus_attributor.pt         causal MLP checkpoint
runtime_logs/navigator_commands_*.csv    command and decision audit log
```

Run the same world again without deleting `artifacts/odysseus_world_memory.json` to let Odysseus start with prior no-go regions and behavior memory.

## Test

Headless validation:

```bash
python3 -m compileall planning runtime/core mapping/global_map motion/vo segmentation ros/semantic_spatial_mapping_ros/semantic_spatial_mapping_ros tools tests
python3 -m pytest tests -q
python3 -m runtime.core.runtime_validation
```

ML-specific Odysseus validation with the local venv:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_odysseus.py -q
```

Expected current result:

```text
48 passed, 3 skipped
7 passed in .venv Odysseus tests
runtime validation passed
```

ROS build validation:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select semantic_spatial_mapping_ros
```

## Container Deployment

Build the container:

```bash
docker build -t odysseus-navigation:latest .
```

Start an interactive container with host ROS/Gazebo networking:

```bash
docker compose run --rm odysseus
```

Inside the container:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/odysseus_ws/install/setup.bash
cd /opt/odysseus
python3 -m pytest tests -q
```

For GUI Gazebo from Docker, allow X11 access on the host first if needed:

```bash
xhost +local:docker
```

The container is intended as a reproducible ROS2/Odysseus deployment environment. Generated logs, datasets, checkpoints, and local memory stay out of git through `.gitignore` and `.dockerignore`.

## Embedded Deployment Direction

For embedded robots, Odysseus should run with the same contracts and different sensor topics/config:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch semantic_spatial_mapping_ros embedded_fused_runtime.launch.py
python3 gazebo_demo/scripts/run_odysseus_auto.py --no-train
```

Before real factory deployment, update camera extrinsics, validate odometry/IMU/VSLAM frame contracts, tune QoS/sync, and run with conservative safety thresholds. Real-world training artifacts should remain in `artifacts/` or an external artifact store, not in git.

## Architecture Notes

Core Odysseus modules:

- `runtime/core/odysseus/contracts.py`: schema, rollout records, causal labels, trace and sample persistence.
- `runtime/core/odysseus/world_memory.py`: persistent no-go cells, successful corridors, online behavior values.
- `runtime/core/odysseus/navigation.py`: continuing navigation loop and mode switching.
- `runtime/core/odysseus/attribution.py`: PyTorch causal MLP and checkpoint handling.
- `runtime/core/odysseus/shadow.py`: trace closure and owned sample writing.
- `gazebo_demo/scripts/run_odysseus_auto.py`: automatic run/train wrapper.

Supporting learning paths:

- `runtime/core/cognition/`: neural world-model foundation and deterministic teacher data path.
- `runtime/core/navigation_learning.py`: legacy optional GRU risk predictor from engineered CSV sequences.

Architecture contracts and acceptance criteria live in:

```text
notes/architecture_contracts.md
notes/runtime_validation_checklist.md
```

## Runtime Logging

Every ROS2 stack run creates a text report:

```text
runtime_logs/<timestamp>_<profile>_<pid>/runtime_report.txt
```

Embedded profiles also try to record a rosbag:

```text
runtime_logs/<timestamp>_<profile>_<pid>/rosbag/
```

`runtime_logs/` is ignored by git to keep the repository clean. These reports are still the primary field-run evidence when debugging sensor topics, encodings, CameraInfo, TF, calibration, perception, mapping, VSLAM behavior, navigation intelligence, or learned-risk training data.

Runtime logging config:

```yaml
logging:
  enabled: true
  directory: runtime_logs
  frame_log_period: 1
  record_embedded_rosbag: true
  rosbag_topics: []
```

The report includes the resolved config, topic list, image/depth message metadata, depth validity ratios, CameraInfo intrinsics, pose status, diagnostics, semantic object counts, exceptions, and shutdown summary.

## ROS2 Topics

Default input topics are configurable in YAML:

```text
RGB image:        /camera/color/image_raw or /vctr/rgb_raw
Depth image:      /camera/depth/image_raw or /vctr/depth_raw
IMU:              /imu/data
RGB CameraInfo:   /camera/color/camera_info or /vctr/rgb/camera_info
Depth CameraInfo: /camera/depth/camera_info or /vctr/depth/camera_info
Pose source:      /odom, /odometry/filtered, /pose, TF, internal VSLAM, or identity fallback
Fallback odom:    /odom when /odometry/filtered is unavailable or stale
```

Default output topics:

```text
/semantic_spatial/points
/semantic_spatial/objects
/semantic_spatial/map
/semantic_spatial/diagnostics
/semantic_spatial/visual_odometry
/semantic_spatial/debug_overlay
```

## Pose Contract

The runtime consumes camera pose as:

```text
T_world_cam
```

ROS pose inputs declare their contract:

```yaml
pose:
  input_pose: world_camera
```

or:

```yaml
pose:
  input_pose: world_base
```

For `world_base`, the runtime computes:

```text
T_world_cam = T_world_base @ T_base_camera
```

using:

```yaml
extrinsics:
  base_to_camera:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
```

Replace identity extrinsics with measured robot calibration before trusting map geometry.

## Current Stack

| Area | Current state |
| --- | --- |
| Odysseus navigation | Owns rollout selection, advance/recover/retrace/explore mode switching, causal sample closure, and persistent world memory |
| World memory | Stores no-go regions, successful corridors, action values, and cross-run behavior evidence in `artifacts/odysseus_world_memory.json` |
| Causal attribution | PyTorch MLP predicts failure cause, progress, stuck/collision/safety risk, localization risk, stale-sensor risk, and severity |
| Global planning | Builds a persistent world-frame occupancy costmap and runs A* to produce route waypoints |
| Local planning | Generates collision-checked forward, turning, and reverse recovery rollouts from live depth geometry |
| Safety authority | Deterministic navigation intelligence checks stale sensors, immediate obstacles, false progress, pose divergence, and recovery authority before `cmd_vel` |
| Perception/VSLAM | Provides RGB-D geometry, semantic objects, bounded maps, odometry/VSLAM pose evidence, diagnostics, and ROS2 topic outputs |
| Automation | `run_odysseus_auto.py` creates artifacts, loads checkpoints, runs navigation, and trains Odysseus after shutdown when samples exist |
| Deployment | Local ROS2 Jazzy workflow plus Docker/Docker Compose baseline for reproducible installation |
| Tests | Headless planner, Odysseus, runtime, safety, ROS converter, and validation checks documented above |

This README describes the current Odysseus stack. Older integration notes and previous roadmap material have been removed from the public README so the repository presents the product as it exists now.
