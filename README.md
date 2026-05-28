# semantic_spatial_mapping

## Overview

`semantic_spatial_mapping` is a modular robotics perception and semantic spatial mapping framework integrating:

* RGBD geometry
* semantic segmentation
* persistent object tracking
* monocular visual odometry
* world-frame spatial accumulation
* semantic pointcloud extraction

The system is designed as a robotics-oriented spatial runtime architecture rather than a standalone computer vision pipeline.

The primary architectural goal is:

```text
RGBD Perception
    +
Semantic Understanding
    +
Visual Motion Estimation
    +
Persistent Spatial World Modeling
```

This repository serves as the integration layer between:

* a modular RGBD semantic perception stack
* a monocular visual odometry frontend

---

# Core Runtime Pipeline

```text
RGB Frame
Depth Frame
    ↓
Segmentation
    ↓
Semantic Masks
    ↓
RGBD Pointcloud Extraction
    ↓
Visual Odometry
    ↓
World-Frame Projection
    ↓
Persistent Semantic Spatial Map
```

---

# Repository Structure

```text
semantic_spatial_mapping/
├── configs/
├── geometry/
├── geometry_validation/
├── ingestion/
├── mapping/
├── motion/
├── replay_validation/
├── requirements/
├── ros/
├── runtime/
├── segmentation/
├── tracking/
├── utils/
├── visualization/
└── world/
```

---

# Subsystem Breakdown

## geometry/

Core spatial reasoning and geometric processing.

### Responsibilities

* camera intrinsics
* RGBD reprojection
* depth backprojection
* pointcloud extraction
* coordinate transforms
* oriented bounding boxes
* geometric estimation utilities

### Architectural Role

This subsystem defines:

```text
pixel space
→ camera space
→ world space
```

transform semantics.

---

## segmentation/

Semantic perception subsystem.

### Current Features

* YOLOv8 segmentation
* semantic occupancy masks
* mask cleanup
* OBB estimation

### Future Direction

* semantic landmarks
* category-aware mapping
* semantic scene understanding
* object identity persistence

---

## tracking/

Persistent temporal object-state estimation.

### Current Features

* temporal persistence
* Kalman filtering
* motion prediction
* dropout handling
* track continuity

### Architectural Role

This subsystem represents the beginning of persistent world-state modeling.

---

## motion/vo/

Monocular visual odometry frontend.

### Responsibilities

* feature extraction
* feature matching
* epipolar geometry
* relative pose estimation
* trajectory accumulation
* sparse triangulation
* keyframe management

### Important Design Decision

VO is treated as a reusable motion-estimation subsystem.

It does NOT own:

* runtime orchestration
* visualization
* semantic mapping
* robotics infrastructure

---

## world/

World-frame coordinate management.

### Current Responsibilities

* camera-frame → world-frame transforms
* persistent coordinate semantics

### Future Direction

* transform trees
* world anchors
* semantic frames
* map-frame consistency

---

## mapping/

Persistent semantic spatial memory.

### Current Responsibilities

* accumulated world-frame pointclouds

### Future Direction

* semantic maps
* relocalization
* semantic landmarks
* loop closure anchors
* occupancy structures

---

## runtime/

Integrated runtime orchestration.

### Current Runtime

```text
runtime/python_reference/semantic_spatial_runtime.py
```

This runtime integrates:

* segmentation
* RGBD geometry
* VO pose estimation
* world-frame projection
* persistent semantic accumulation

---

# Current System Capabilities

| Capability                     | Status                       |
| ------------------------------ | ---------------------------- |
| RGBD geometry                  | Implemented                  |
| Semantic segmentation          | Implemented                  |
| Semantic pointcloud extraction | Implemented                  |
| Persistent tracking            | Implemented                  |
| Monocular VO                   | Integrated                   |
| World-frame accumulation       | Implemented                  |
| Semantic spatial mapping       | Initial integration complete |

---

# Current System Limitations

| Limitation                          | Notes                              |
| ----------------------------------- | ---------------------------------- |
| No loop closure                     | Backend SLAM not implemented yet   |
| No bundle adjustment                | Frontend-only motion estimation    |
| Monocular scale drift               | Temporary heuristic currently used |
| No dense global optimization        | Sparse accumulation only           |
| No persistent semantic entities yet | Tracking still mostly frame-local  |

---

# Immediate Validation Goals

## 1. Static Scene Stability

Expected:

* world geometry remains stable
* no transform explosions
* no severe drift

---

## 2. Rotation Consistency

Expected:

* camera rotates
* world geometry remains stationary

---

## 3. Slow Translation Consistency

Expected:

* object remains fixed in world coordinates
* trajectory evolves coherently

---

## 4. Semantic Persistence

Expected:

* previously observed geometry remains accumulated

---

# Future Roadmap

## Near-Term

* Open3D world-map visualization
* RGBD metric scale grounding
* persistent semantic entities
* semantic world-frame anchoring
* trajectory stabilization

## Mid-Term

* semantic landmarks
* relocalization
* local semantic mapping
* pose graph optimization
* map consistency validation

## Long-Term

* semantic SLAM backend
* loop closure
* semantic world models
* navigation integration
* manipulation-aware spatial reasoning
* robotics deployment runtime

---

# Environment Setup

---

# Installation

```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements/python_requirements.txt
```

---

# Architectural Philosophy

This repository is intended to become:

```text
modular semantic spatial intelligence infrastructure
```

* explicit subsystem boundaries
* reusable geometry layers
* persistent spatial representations
* world-frame reasoning
* robotics-oriented runtime composition

---

# Current Integration Milestone

The system currently integrates:

```text
Semantic RGBD Geometry
    +
Monocular VO Pose Estimation
    +
World-Frame Spatial Accumulation
```

inside a unified runtime architecture.

