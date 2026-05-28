# Geometry Transforms Module

## Purpose

This module contains transform and coordinate-frame utilities extracted from `vision_node2.py` (previous monolithic python code file)

Current responsibilities include:
- homogeneous transform utilities
- quaternion/rotation conversions
- TF matrix conversion helpers
- coordinate-frame transforms
- camera/world frame alignment helpers

---

# Module Files

```text
geometry/transforms/
│
├── transforms_reference.py
├── transforms.hpp
├── transforms.cpp
├── transform_validation.py
└── README.md