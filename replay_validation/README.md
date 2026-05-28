# Replay Validation Runtime

## Purpose

Replay Validation is the orchestration-layer integration runtime for the modular perception stack.

This subsystem validates interoperability between:
- replay ingestion
- segmentation
- geometry extraction
- tracking
- visualization
- telemetry
- export pipelines

Replay Validation is NOT a subsystem implementation layer.

It does NOT implement:
- segmentation logic
- geometry logic
- tracking logic
- visualization internals

Instead, it imports and orchestrates modular subsystem APIs.

---

# Architectural Philosophy

Replay Validation exists to prevent regeneration of the original monolithic runtime architecture.

Subsystems remain independently testable and independently maintainable.

Replay Validation ONLY performs:
- subsystem orchestration
- dataflow routing
- runtime integration
- visualization integration
- replay export

---
# Design Principle

Subsystem validation
!=
Integration validation

Replay Validation exists specifically for:
integration validation.