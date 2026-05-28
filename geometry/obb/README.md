# geometry/obb

This module contains oriented bounding box (OBB) geometry estimation logic.

Purpose:
- Estimate object orientation
- Estimate object dimensions
- Extract grasp-aligned geometry primitives
- Convert segmentation masks into geometric object representations

Current functionality:
- Binary contour extraction
- Largest-contour selection
- OpenCV minAreaRect fitting
- 2D oriented bounding box estimation
- Object yaw estimation
- Pixel-space dimension estimation
- Metric dimension estimation using camera intrinsics and depth

Current files:
- `obb_reference.py`
    Reference Python implementation extracted from vision_node2.py

- `obb_validation.py`
    Validation and visualization tests for OBB estimation

- `obb.hpp`
    C++ interface for deployment-oriented migration

- `obb.cpp`
    C++ implementation of OBB geometry estimation

This module sits downstream of:
- segmentation
- depth estimation
- point cloud extraction

and upstream of:
- grasp planning
- pose estimation
- manipulation geometry
- object tracking

Current implementation focuses on:
2D contour-driven OBB estimation.

Future expansion may include:
- PCA-based orientation estimation
- 3D oriented bounding boxes
- surface normal estimation
- grasp-frame alignment
- object principal axes