# geometry/backprojection

This module contains the depth-to-3D lifting subsystem extracted from `vision_node2.py`.

Its responsibility is converting 2D depth image regions into sparse 3D point clouds.

Current functionality:
- Depth ROI extraction
- Sparse point sampling
- Camera intrinsics projection
- Camera-frame XYZ reconstruction
- Homogeneous coordinate generation
- World-frame transformation using SE(3) matrices
- Optional exclusion-region masking
- Depth-range filtering

Current files:
- `backprojection_reference.py`
    Original modular Python extraction for experimentation and validation

- `backprojection_validation.py`
    Runtime validation and debugging tests

- `backprojection.hpp`
    C++ interface for deployment/runtime migration

- `backprojection.cpp`
    C++ implementation of sparse depth lifting

This module is one of the core geometry layers of the perception stack.

It sits between:
- depth ingestion
and
- higher-level geometry estimation systems

Downstream systems using this module later include:
- point cloud geometry
- surface estimation
- oriented bounding boxes
- pose estimation
- grasp reasoning