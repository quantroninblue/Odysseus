# Odysseus Navigation AI

Odysseus is an embodied navigation AI for obstacle-laden factory environments. It uses RGB-D perception, semantic mapping, VSLAM/odometry, persistent occupancy mapping, A* route search, local trajectory rollout, causal attribution, and cross-run world memory to reach a designated destination while learning from failed paths.

The repository name is still `semantic_spatial_mapping`, but the current product layer is **Odysseus**. Perception, mapping, VSLAM, and planning are the sensorimotor substrate. Odysseus owns the continuing navigation loop: it observes the world, chooses from collision-checked candidate motions, sees what actually happened, updates memory, marks bad regions, retraces or explores when needed, and records causal training samples.

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
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
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

The historical `requirements/python_requirements.txt` is UTF-16 LE. Convert before using it:

```bash
iconv -f UTF-16LE -t UTF-8 requirements/python_requirements.txt > /tmp/ssm_python_requirements.txt
pip install -r /tmp/ssm_python_requirements.txt
```

## Run Odysseus In Gazebo

Use four terminals. Terminal 4 is automated: it creates sample/memory directories, loads a causal checkpoint if present, runs Odysseus, and trains a refreshed checkpoint after shutdown when enough samples exist.

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
gz sim -r gazebo_demo/factory_obstacle_demo.world
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
ros2 launch gazebo_demo/launch/factory_bot_bridge.launch.py
```

Terminal 3:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
colcon build --packages-select semantic_spatial_mapping_ros
source install/setup.bash
ros2 launch semantic_spatial_mapping_ros gazebo_fused_runtime.launch.py
```

Terminal 4:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
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

## Current Capabilities

| Area | Current state |
| --- | --- |
| RGB-D geometry | Vectorized point extraction, depth units, projection helpers |
| Runtime core | Config-driven deployment runtime with diagnostics and degraded modes |
| ROS2 package | Launch files, node, converters, sync, pose adapters, publishers |
| Pose sources | Odometry, PoseStamped, TF, internal VSLAM, identity fallback |
| Perception | Backend interface with disabled, mock, and YOLO providers |
| Mask handling | Area, border-touch, depth-support, and max-mask filters |
| Semantic map | Bounded point map plus object/entity map with IDs and observations |
| VSLAM | RGB-D PnP path with depth landmarks and monocular fallback |
| Logging | Per-run text reports; embedded rosbag recording attempt |
| Tests | Core runtime, ROS converters, failure modes, RGB-D VO PnP |

## Known Gaps Before Calling It Finished

- Needs `colcon build` and ROS launch validation in a sourced ROS2 workspace.
- Needs Gazebo, rosbag, and embedded hardware runtime feedback.
- Camera optical frame, base-to-camera, and depth-to-RGB extrinsics need real calibration checks.
- QoS and sync behavior may need sensor-specific tuning.
- Internal VSLAM still needs robust local BA, relocalization, loop validation, and pose graph correction.
- Semantic object fusion is currently nearest-class/centroid based; it needs runtime evaluation.
- Runtime performance needs profiling on embedded hardware.

## 05.06.26 Integration Report

### Starting Point

Before the v2.0 integration pass, the repository already had modular research/reference components:

- `geometry/` for RGB-D backprojection, transforms, point clouds, and OBB support.
- `segmentation/` with a YOLOv8 segmentation reference.
- `motion/vo/` with a monocular visual odometry frontend.
- `mapping/global_map/` with world-frame point accumulation.
- `runtime/python_reference/` with a Python reference semantic runtime.
- `tracking/` with temporal filtering and state-estimation references.
- `ingestion/` and replay utilities for dataset/rosbag-style data paths.

At that stage, the repo was organized around modular perception and VO prototypes. The ROS deployment layer was not yet a working runtime package, pose-source selection was not generalized, world-map storage was append-oriented, RGB-D extraction used more prototype-style loops, and the VO backend pieces from the external VSLAM reference had not been integrated into this repo's `motion/vo` package.

### Changes Added In v2.0

Runtime core:

- Added `runtime/core/` as the deployment-facing runtime layer.
- Added structured runtime config loading and validation.
- Added diagnostics, health state, degraded modes, pose providers, runtime output types, and lifecycle hooks.
- Added runtime logger that writes `runtime_logs/<session>/runtime_report.txt` on every ROS2 run.
- Added embedded-only rosbag recording attempt through `ros2 bag record`.

ROS2 deployment:

- Added `ros/semantic_spatial_mapping_ros/` package.
- Added Gazebo and embedded OAK-D style YAML profiles.
- Added launch files for both profiles.
- Added `semantic_spatial_node`.
- Added ROS image/CameraInfo conversion without requiring `cv_bridge`.
- Added approximate RGB-D sync.
- Added ROS pose adapters for Odometry, PoseStamped, TF, internal VSLAM, and identity fallback.
- Added publishers for semantic points, semantic objects, map points, diagnostics, VO odometry, and optional overlay.
- Added sensor-data QoS for camera streams.

Geometry and mapping:

- Vectorized point cloud generation and object point extraction.
- Added configurable depth units.
- Added depth-to-RGB point projection helpers.
- Replaced append-only world map behavior with bounded voxel-downsampled storage.
- Added semantic object map with IDs, labels, confidence, centroid, extent, covariance, observation count, and bounded fused points.

Perception:

- Added perception contracts: `InstanceMask`, `SemanticFrame`, and `ObjectGeometry`.
- Added segmentation provider abstraction: disabled, mock, YOLO.
- Preserved YOLO class IDs, labels, confidences, and boxes.
- Added mask filtering by area, depth support, border contact, and max masks per frame.
- Added point outlier filtering before map/object fusion.

VSLAM and VO:

- Integrated external VSLAM backend modules into `motion/vo/`.
- Preserved the existing `VisualOdometry.update(...) -> PoseUpdate` API used by runtime code.
- Added covisibility graph plumbing.
- Added RGB-D scale handling.
- Added RGB-D PnP tracking path using keyframe depth and current 2D features.
- Added depth-initialized RGB-D landmarks when PnP succeeds.
- Kept monocular essential-matrix tracking as fallback.
- Added VO diagnostics for tracking method, depth support ratio, and reprojection error.
- Made optional visualization imports lazy.

Reliability and debugging:

- Added config validation for topics, frames, depth ranges, pose source, pose contract, extrinsics, resource limits, and logging settings.
- Added external pose staleness checks.
- Added runtime pose guards for NaN/non-finite poses, huge translations, and sudden jumps.
- Added CameraInfo staleness filtering.
- Added resource caps for masks, per-frame points, and per-object points.
- Added runtime validation checklist in `notes/runtime_validation_checklist.md`.

Tests:

- Added synthetic runtime validation.
- Added unit tests for config validation, stale pose, missing depth, missing CameraInfo, empty segmentation, bad extrinsics, NaN pose, pose jumps, object fusion, point caps, runtime logging, and default rosbag topics.
- Added ROS converter tests that run when ROS message packages are available.
- Added RGB-D VO PnP test.

### Verification Run

The following local checks passed:

```bash
python3 -m compileall runtime/core planning mapping/global_map motion/vo segmentation ros/semantic_spatial_mapping_ros/semantic_spatial_mapping_ros tools tests
python3 -m pytest tests
python3 -m runtime.core.runtime_validation
colcon build --packages-select semantic_spatial_mapping_ros
git check-ignore -v runtime_logs || true
```

The unit test suite currently reports:

```text
20 tests OK
```

### Current Interpretation

The v2.0 integration moved the repository from a modular perception/VO reference codebase into a first deployable architecture pass for a hardware/software-agnostic perception + VSLAM stack. The next milestone is runtime validation with Gazebo, rosbag replay, and embedded hardware so the stack can be hardened against real TF trees, topic QoS, depth encodings, CameraInfo timing, calibration, object fusion behavior, and VSLAM tracking quality.

## Historical Snapshot: Earlier README Direction

The earlier README described the repository as a modular semantic spatial mapping framework integrating:

- RGB-D geometry
- semantic segmentation
- persistent object tracking
- monocular visual odometry
- world-frame spatial accumulation
- semantic point cloud extraction

The earlier pipeline was:

```text
RGB Frame
Depth Frame
    -> Segmentation
    -> Semantic Masks
    -> RGB-D Pointcloud Extraction
    -> Visual Odometry
    -> World-Frame Projection
    -> Persistent Semantic Spatial Map
```

The earlier roadmap focused on:

- static scene stability
- rotation and translation consistency
- semantic persistence
- Open3D visualization
- RGB-D metric scale grounding
- persistent semantic entities
- relocalization
- local semantic mapping
- pose graph optimization
- loop closure
- navigation/manipulation integration

That direction remains the project direction. The current README records the deployment runtime, ROS2 integration, logging, testing, and object/VSLAM upgrades added on `05.06.26`.
