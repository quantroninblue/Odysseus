# Architecture Contracts

This repository should be treated as a layered robotics stack, not a collection of scripts. Each layer owns a specific representation and downstream layers should consume that representation instead of reaching backward into raw internals.

## Data Flow

```text
sensor drivers / Gazebo bridge / rosbag replay
  -> synchronized FramePacket + CameraCalibration
  -> pose provider / EKF / VSLAM PoseEstimate
  -> segmentation result
  -> SemanticFrame / InstanceMask
  -> object point extraction + world projection
  -> SemanticObjectMap + WorldMap
  -> LocalObstacleMap / planner-facing obstacle contract
  -> planner/controller PlannerCommand proposal
  -> NavigationLearningMemory engineered sequence features + optional GRU risk predictor
  -> NavigationIntelligence motion consistency and safety authority
  -> ROS cmd_vel + command audit log
```

## Layer Rules

- Sensor ingestion owns topic names, encodings, timestamps, camera info, and frame IDs.
- Pose providers own `T_world_cam` or the declared configured pose contract. Consumers must not infer frame semantics from topic names.
- Segmentation owns masks and labels only. It does not decide robot motion.
- Mapping owns semantic object identity, fusion, confidence, and observation history.
- Planning must consume planner-facing obstacle state or live depth/costmap evidence. It must not steer directly from raw semantic centroids.
- Control commands must be bounded, auditable, and logged with the mode/reason and perception snapshot that produced them.
- Navigation intelligence owns cross-checks between commanded motion, odometry, VSLAM, and depth scene change. It may override planner commands when those signals contradict each other.
- Navigation learning may predict risk from recent engineered sequence features, but it does not bypass deterministic safety rules. It can slow/stop earlier inside the safety envelope.
- Fallbacks must be explicit: missing IMU, missing filtered odom, missing VSLAM, stale depth, and stale objects must all degrade by contract, not by accidental None checks.

## Current Implementation Status

Implemented now:

- `runtime/core/types.py` defines frame and runtime packet contracts.
- `runtime/core/contracts.py` defines planner-facing obstacle and command contracts.
- Gazebo and embedded fused configs declare primary and fallback odometry topics.
- The ROS runtime subscribes to preferred odometry and fallback odometry.
- Gazebo depth segmentation emits low obstacle labels for short/thin obstacles.
- `planning/local_costmap.py` builds an inflated local obstacle map from live depth.
- `planning/trajectory_rollout.py` chooses bounded `PlannerCommand` outputs from collision-checked short-horizon rollouts.
- `runtime/core/navigation_intelligence.py` classifies stale sensors, blocked motion, odometry slip suspicion, and pose divergence from command/pose/depth/VSLAM evidence.
- `runtime/core/navigation_learning.py` builds long-horizon engineered memory features and defines a lightweight PyTorch GRU risk classifier for learning from faulty run logs.
- `tools/build_navigation_learning_dataset.py` converts navigator CSV logs into sequence datasets; `tools/train_navigation_risk_gru.py` trains checkpoints for `NAV_GRU_RISK_CHECKPOINT`.
- The demo navigator uses the rollout planner for normal motion, passes proposed commands through navigation intelligence before publishing, and logs every published command to `runtime_logs/navigator_commands_*.csv`.
- The demo navigator no longer lets raw semantic object centroids reset avoidance state; semantics are mapped and logged, not used as direct control authority.

Still required before relying on this stack for field runs:

- Add obstacle decay and confidence gating to planner-facing obstacle state beyond the live depth costmap.
- Add bag replay tests for Gazebo pole, pallet, tall rack, and no-obstacle corridors.
- Add timestamp/latency budgets across every edge of the pipeline.
- Add bag replay regression cases that assert navigation intelligence triggers on static-depth/false-odom pole contact runs.
- Collect enough labeled faulty/successful runs to train and calibrate the GRU risk model beyond smoke tests.
- Add per-layer diagnostics for stale data, dropped frames, rejected masks, rejected obstacles, and planner collision checks.
- Validate all frame transforms and camera extrinsics against real calibration data.

## Acceptance Criteria

Each live run should be reproducible from logs alone:

- sensor topics, encodings, frame IDs, camera info, and rates
- pose source, age, frame contract, and fallback source
- segmentation masks accepted/rejected with reasons
- objects fused, aged, decayed, or removed
- local obstacle map published or logged
- planner candidates accepted/rejected with collision reason
- final `cmd_vel` command, mode, reason, and limits applied
