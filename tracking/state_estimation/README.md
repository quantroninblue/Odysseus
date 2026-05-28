# State Estimation Subsystem

## Purpose

The state_estimation subsystem provides probabilistic motion estimation and temporal stabilization for object tracking.

This subsystem is responsible for:
- motion prediction
- temporal smoothing
- velocity estimation
- future-state estimation
- uncertainty-aware tracking

State estimation exists to stabilize tracking under:
- noisy detections
- intermittent segmentation
- partial occlusions
- object jitter
- frame-to-frame geometric instability

---

# Current Architecture Role

This subsystem operates between:
- raw detections
and
- association/tracking logic

Pipeline role:

Detections
    ->
State Estimation
    ->
Association
    ->
Persistent Tracks

---

# Current Implemented Components

## kalman_filter.py
Constant-velocity 2D Kalman filter.

Current state vector:

[x, y, vx, vy]

where:
- x  -> object center x
- y  -> object center y
- vx -> object velocity x
- vy -> object velocity y

---

# Current Capabilities

- Constant-velocity motion prediction
- Temporal smoothing
- Velocity estimation
- Motion-aware state propagation
- Reduced frame-to-frame jitter

---

# Current Limitations

- No covariance visualization
- No adaptive process noise
- No nonlinear motion modeling
- No orientation-state filtering
- No depth-aware motion estimation
- No multi-hypothesis tracking
- No acceleration modeling

---

# Near-Term Goals

- Integrate Kalman prediction into TrackState
- Use predicted state during association
- Improve persistent ID stability
- Reduce track fragmentation
- Improve missed-frame recovery

---

# Mid-Term Goals

- Orientation-aware filtering
- 3D motion estimation
- Depth-aware state estimation
- Sensor-fusion-ready architecture
- Uncertainty-aware gating

---

# Long-Term Goals

- Extended Kalman Filters (EKF)
- Unscented Kalman Filters (UKF)
- Particle filters
- Multi-object probabilistic tracking
- SLAM-compatible state estimation

---

# Design Principle

State estimation
!=
association

This subsystem predicts motion.

Association decides identity correspondence.

These responsibilities remain intentionally separated.