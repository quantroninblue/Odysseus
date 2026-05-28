# segmentation

This module contains semantic segmentation infrastructure for the modular robotics perception stack.

Purpose:
- Generate object masks from RGB frames
- Isolate semantic object regions
- Provide geometry-ready masks for downstream processing
- Enable replay-driven semantic perception

Current functionality:
- YOLOv8-seg inference
- Binary mask extraction
- Segmentation overlay generation
- Inference timing measurement
- Replay-compatible frame processing

Current files:
- `segmentation_reference.py`
    Python reference implementation using YOLOv8-seg

- `segmentation_validation.py`
    Validation script for segmentation inference and mask generation

- `segmentation.hpp`
    C++ segmentation interface

- `segmentation.cpp`
    C++ segmentation implementation

Current outputs:
- segmentation masks
- semantic overlays
- inference timing statistics

Downstream consumers:
- OBB estimation
- backprojection
- point cloud generation
- tracking
- visualization

Current architecture:
ReplayFramePacket
    ↓
Segmentation
    ↓
Semantic masks
    ↓
Geometry pipeline

This module currently focuses only on:
- semantic mask generation
- deterministic replay-driven segmentation

Advanced scheduling, async execution, depth fusion,
and mask refinement are handled separately in:
- workers
- depth_fusion
- mask_processing