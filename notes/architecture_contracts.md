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
  -> collision-checked rollout candidate set
  -> OdysseusNavigator persistent embodied navigation decision
  -> Odysseus causal episode closure + optional owned sample
  -> Cognitive world-model teacher/shadow data path
  -> NavigationLearningMemory engineered sequence features + optional GRU risk predictor
  -> NavigationIntelligence motion consistency and hard safety authority
  -> ROS cmd_vel + command audit log
```

## Layer Rules

- Sensor ingestion owns topic names, encodings, timestamps, camera info, and frame IDs.
- Pose providers own `T_world_cam` or the declared configured pose contract. Consumers must not infer frame semantics from topic names.
- Segmentation owns masks and labels only. It does not decide robot motion.
- Mapping owns semantic object identity, fusion, confidence, and observation history.
- Planning must consume planner-facing obstacle state or live depth/costmap evidence. It must not steer directly from raw semantic centroids.
- Control commands must be bounded, auditable, and logged with the mode/reason and perception snapshot that produced them.
- Odysseus owns continuing navigation intent: it chooses among collision-checked rollout candidates, observes the next outcome, detects surprise or poor progress, can switch between advance, recover, retrace, and explore behavior, and persists cross-run world memory of no-go regions, successful corridors, and online behavior values.
- Navigation intelligence owns cross-checks between commanded motion, odometry, VSLAM, and depth scene change. It may override Odysseus commands when those signals contradict each other.
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
- `planning/global_costmap.py` accumulates decaying world-frame occupancy evidence from local depth observations.
- `planning/global_route.py` runs inflated-grid A* and provides a route lookahead waypoint to the local rollout planner.
- `runtime/core/navigation_intelligence.py` classifies stale sensors, blocked motion, odometry slip suspicion, and pose divergence from command/pose/depth/VSLAM evidence.
- `runtime/core/navigation_learning.py` builds long-horizon engineered memory features and defines a lightweight PyTorch GRU risk classifier for learning from faulty run logs.
- `runtime/core/odysseus/` defines the embodied navigation layer, causal outcome schema, MLP attributor, owned NPZ samples, training losses, checkpoints, and persistent online decision loop.
- `runtime/core/cognition/contracts.py` defines the versioned Neural Cognitive Core/world-model data path.
- `runtime/core/cognition/memory.py` provides bounded working and episodic memory.
- `runtime/core/cognition/teacher.py` converts verified global routes into safe imitation targets.
- `runtime/core/cognition/shadow.py` records owned samples without affecting commands.
- `runtime/core/cognition/model.py` proposes and evaluates futures through a learned latent transition.
- `runtime/core/cognition/training.py` supplies imitation/outcome losses and versioned checkpoints.
- `tools/build_navigation_learning_dataset.py` converts navigator CSV logs into sequence datasets; `tools/train_navigation_risk_gru.py` trains checkpoints for `NAV_GRU_RISK_CHECKPOINT`.
- `tools/train_odysseus_attributor.py` trains Odysseus causal-attribution checkpoints from owned episode NPZ samples.
- The demo navigator lets Odysseus choose normal motion from collision-checked rollout candidates, passes Odysseus commands through navigation intelligence before publishing, and logs every published command to `runtime_logs/navigator_commands_*.csv`.
- The demo navigator no longer lets raw semantic object centroids reset avoidance state; semantics are mapped and logged, not used as direct control authority.

Still required before relying on this stack for field runs:

- Add obstacle decay and confidence gating to planner-facing obstacle state beyond the live depth costmap.
- Add bag replay tests for Gazebo pole, pallet, tall rack, and no-obstacle corridors.
- Add timestamp/latency budgets across every edge of the pipeline.
- Add bag replay regression cases that assert navigation intelligence triggers on static-depth/false-odom pole contact runs.
- Collect enough Odysseus owned outcome samples to train and calibrate the causal attributor, then use GRU risk learning only as a legacy/supporting signal.
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
