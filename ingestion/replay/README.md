# ingestion/replay

This module contains deterministic offline replay ingestion infrastructure.

Purpose:
- Replay recorded robotics sessions
- Simulate live sensor streams offline
- Feed geometry/perception pipelines from recorded data
- Enable deterministic debugging and experimentation

Current functionality:
- RGB video replay
- Depth video replay
- Synchronized frame stepping
- Replay packet generation
- Offline sensor simulation
- Deterministic frame iteration

Current files:
- `replay_loader.py`
    Python reference replay engine using OpenCV VideoCapture

- `replay_packet.py`
    Shared transport structure for replayed sensor data

- `replay_validation.py`
    Validation script for replay stepping and synchronized playback

- `replay_loader.hpp`
    C++ replay loader interface

- `replay_loader.cpp`
    C++ replay loader implementation

Replay packets currently contain:
- RGB frames
- depth frames
- timestamps
- metadata hooks

This module sits upstream of:
- segmentation
- geometry
- tracking
- visualization

The replay system enables:
- offline geometry debugging
- replay-driven experimentation
- perception observability
- reproducible testing
- deployment-independent pipeline development