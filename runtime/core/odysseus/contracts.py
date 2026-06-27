from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import os
import tempfile

import numpy as np

from planning.trajectory_rollout import CandidateEvaluation
from runtime.core.cognition import CognitiveObservation, observation_context_vector


ODYSSEUS_SCHEMA_VERSION = "1.0"
ODYSSEUS_FEATURE_SIZE = 40
ODYSSEUS_CAUSE_LABELS: tuple[str, ...] = (
    "success",
    "thin_obstacle_missed",
    "bad_detour_side",
    "stale_global_obstacle",
    "local_minimum",
    "pose_drift",
    "sensor_stale",
    "overconfident_clearance",
    "route_blocked",
    "semantic_hazard_underweighted",
    "unknown_failure",
)
_CAUSE_TO_INDEX = {label: index for index, label in enumerate(ODYSSEUS_CAUSE_LABELS)}


@dataclass(frozen=True)
class RolloutCandidateRecord:
    candidate_id: str
    linear_x: float
    angular_z: float
    accepted: bool
    deterministic_score: float
    min_clearance_m: float
    reason: str
    selected: bool = False

    @classmethod
    def from_evaluation(
        cls,
        evaluation: CandidateEvaluation,
        *,
        candidate_id: str,
        selected: bool = False,
    ) -> "RolloutCandidateRecord":
        score = float(evaluation.score)
        if not math.isfinite(score):
            score = 1.0e6
        return cls(
            candidate_id=candidate_id,
            linear_x=float(evaluation.linear_x),
            angular_z=float(evaluation.angular_z),
            accepted=bool(evaluation.accepted),
            deterministic_score=score,
            min_clearance_m=float(evaluation.min_clearance_m),
            reason=evaluation.reason,
            selected=selected,
        )

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        values = (self.linear_x, self.angular_z, self.deterministic_score, self.min_clearance_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rollout candidate values must be finite")
        if self.min_clearance_m < 0.0:
            raise ValueError("rollout candidate clearance must be non-negative")

    def feature_vector(self) -> np.ndarray:
        self.validate()
        return np.asarray(
            [
                self.linear_x,
                self.angular_z,
                1.0 if self.accepted else 0.0,
                np.clip(self.deterministic_score, -100.0, 100.0) / 100.0,
                np.clip(self.min_clearance_m, 0.0, 8.0) / 8.0,
                1.0 if self.selected else 0.0,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class OdysseusOutcome:
    progress_delta_m: float
    collision: bool
    stuck: bool
    safety_override: bool
    localization_diverged: bool
    stale_sensor: bool
    final_goal_distance_m: float
    failure_cause: str = "success"
    severity: float = 0.0

    def validate(self) -> None:
        if self.failure_cause not in _CAUSE_TO_INDEX:
            raise ValueError(f"unsupported Odysseus failure cause: {self.failure_cause}")
        values = (self.progress_delta_m, self.final_goal_distance_m, self.severity)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("outcome values must be finite")
        if self.final_goal_distance_m < 0.0 or not 0.0 <= self.severity <= 1.0:
            raise ValueError("outcome distance or severity is invalid")
        if not self.failed and self.failure_cause != "success":
            raise ValueError("successful outcomes must use the success cause")

    @property
    def failed(self) -> bool:
        return bool(
            self.collision
            or self.stuck
            or self.safety_override
            or self.localization_diverged
            or self.stale_sensor
            or self.progress_delta_m < -0.15
        )

    @property
    def cause_index(self) -> int:
        return _CAUSE_TO_INDEX[self.failure_cause]

    def target_vector(self) -> np.ndarray:
        self.validate()
        return np.asarray(
            [
                self.progress_delta_m,
                float(self.collision),
                float(self.stuck),
                float(self.safety_override),
                float(self.localization_diverged),
                float(self.stale_sensor),
                np.clip(self.final_goal_distance_m, 0.0, 50.0) / 50.0,
                self.severity,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class FailureAttribution:
    cause: str
    confidence: float
    severity: float
    progress_delta_m: float
    collision_probability: float
    stuck_probability: float
    safety_override_probability: float
    localization_risk: float
    stale_sensor_risk: float
    reason: str

    def validate(self) -> None:
        if self.cause not in _CAUSE_TO_INDEX:
            raise ValueError(f"unsupported attribution cause: {self.cause}")
        probabilities = (
            self.confidence,
            self.severity,
            self.collision_probability,
            self.stuck_probability,
            self.safety_override_probability,
            self.localization_risk,
            self.stale_sensor_risk,
        )
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("attribution probabilities must be within [0,1]")
        if not math.isfinite(self.progress_delta_m):
            raise ValueError("attribution progress must be finite")


@dataclass(frozen=True)
class OdysseusDecisionTrace:
    trace_id: str
    time_sec: float
    feature_vector: np.ndarray
    selected_candidate_id: str | None
    candidate_count: int
    attribution: FailureAttribution | None = None
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def validate(self) -> None:
        features = np.asarray(self.feature_vector, dtype=np.float32).reshape(-1)
        if not self.trace_id:
            raise ValueError("trace_id is required")
        if features.shape != (ODYSSEUS_FEATURE_SIZE,) or not np.isfinite(features).all():
            raise ValueError(f"Odysseus traces require {ODYSSEUS_FEATURE_SIZE} finite features")
        if self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if self.attribution is not None:
            self.attribution.validate()


@dataclass(frozen=True)
class CausalTrainingSample:
    feature_vector: np.ndarray
    cause_index: int
    outcome_targets: np.ndarray
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)
    schema_version: str = ODYSSEUS_SCHEMA_VERSION

    def validate(self) -> None:
        features = np.asarray(self.feature_vector, dtype=np.float32).reshape(-1)
        targets = np.asarray(self.outcome_targets, dtype=np.float32).reshape(-1)
        if self.schema_version != ODYSSEUS_SCHEMA_VERSION:
            raise ValueError(f"unsupported Odysseus schema {self.schema_version}")
        if features.shape != (ODYSSEUS_FEATURE_SIZE,) or not np.isfinite(features).all():
            raise ValueError(f"feature_vector must contain {ODYSSEUS_FEATURE_SIZE} finite values")
        if not 0 <= int(self.cause_index) < len(ODYSSEUS_CAUSE_LABELS):
            raise ValueError("cause index is out of range")
        if targets.shape != (8,) or not np.isfinite(targets).all():
            raise ValueError("outcome_targets must contain eight finite values")


def build_causal_feature_vector(
    observation: CognitiveObservation,
    selected_candidate: RolloutCandidateRecord | None,
    candidate_records: tuple[RolloutCandidateRecord, ...] | list[RolloutCandidateRecord],
    *,
    semantic_forward_m: float = 8.0,
    semantic_lateral_m: float = 0.0,
    progress_distance_m: float = 0.0,
) -> np.ndarray:
    observation.validate()
    candidates = tuple(candidate_records)
    for candidate in candidates:
        candidate.validate()
    selected = selected_candidate
    if selected is None:
        selected = next((candidate for candidate in candidates if candidate.selected), None)
    selected_features = (
        selected.feature_vector()
        if selected is not None
        else np.zeros((6,), dtype=np.float32)
    )
    accepted = [candidate for candidate in candidates if candidate.accepted]
    rejected = len(candidates) - len(accepted)
    scores = np.asarray([candidate.deterministic_score for candidate in accepted], dtype=np.float32)
    clearances = np.asarray([candidate.min_clearance_m for candidate in candidates], dtype=np.float32)
    angular = np.asarray([abs(candidate.angular_z) for candidate in candidates], dtype=np.float32)
    linear = np.asarray([candidate.linear_x for candidate in candidates], dtype=np.float32)
    best_score = float(np.min(scores)) if scores.size else 100.0
    selected_score = selected.deterministic_score if selected is not None else best_score
    best_clearance = float(np.max(clearances)) if clearances.size else 0.0
    min_clearance = float(np.min(clearances)) if clearances.size else 0.0
    base = observation_context_vector(observation, 16)
    summary = np.asarray(
        [
            len(candidates) / 64.0,
            len(accepted) / max(1.0, float(len(candidates))),
            rejected / max(1.0, float(len(candidates))),
            np.clip(best_score, -100.0, 100.0) / 100.0,
            np.clip(selected_score - best_score, -100.0, 100.0) / 100.0,
            np.clip(best_clearance, 0.0, 8.0) / 8.0,
            np.clip(min_clearance, 0.0, 8.0) / 8.0,
            float(np.mean(linear)) if linear.size else 0.0,
            float(np.mean(angular)) if angular.size else 0.0,
            float(np.max(angular)) if angular.size else 0.0,
            np.clip(semantic_forward_m, 0.0, 8.0) / 8.0,
            np.clip(semantic_lateral_m, -4.0, 4.0) / 4.0,
            np.clip(progress_distance_m, 0.0, 50.0) / 50.0,
            1.0 if selected is None else 0.0,
            float(np.mean(clearances < 0.75)) if clearances.size else 0.0,
            float(np.mean(scores < 0.0)) if scores.size else 0.0,
            float(np.std(scores)) / 10.0 if scores.size > 1 else 0.0,
            float(np.std(clearances)) / 8.0 if clearances.size > 1 else 0.0,
        ],
        dtype=np.float32,
    )
    features = np.concatenate([base, selected_features, summary], axis=0).astype(np.float32)
    if features.shape != (ODYSSEUS_FEATURE_SIZE,):
        raise RuntimeError(f"internal feature size mismatch: {features.shape}")
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


def make_causal_sample(
    trace: OdysseusDecisionTrace,
    outcome: OdysseusOutcome,
    *,
    metadata: dict[str, str | float | int | bool] | None = None,
) -> CausalTrainingSample:
    trace.validate()
    outcome.validate()
    merged = dict(trace.metadata)
    merged.update(metadata or {})
    merged.update(
        {
            "trace_id": trace.trace_id,
            "time_sec": trace.time_sec,
            "selected_candidate_id": trace.selected_candidate_id or "",
            "candidate_count": trace.candidate_count,
            "failure_cause": outcome.failure_cause,
        }
    )
    sample = CausalTrainingSample(
        feature_vector=trace.feature_vector,
        cause_index=outcome.cause_index,
        outcome_targets=outcome.target_vector(),
        metadata=merged,
    )
    sample.validate()
    return sample


def save_causal_sample(sample: CausalTrainingSample, output_path: str | Path) -> Path:
    sample.validate()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(sample.metadata, sort_keys=True, separators=(",", ":"))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                feature_vector=np.asarray(sample.feature_vector, dtype=np.float32),
                cause_index=np.asarray(int(sample.cause_index), dtype=np.int64),
                outcome_targets=np.asarray(sample.outcome_targets, dtype=np.float32),
                metadata_json=np.asarray(metadata_json),
                schema_version=np.asarray(sample.schema_version),
            )
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_causal_sample(path: str | Path) -> CausalTrainingSample:
    with np.load(Path(path), allow_pickle=False) as payload:
        sample = CausalTrainingSample(
            feature_vector=payload["feature_vector"],
            cause_index=int(payload["cause_index"].item()),
            outcome_targets=payload["outcome_targets"],
            metadata=json.loads(str(payload["metadata_json"].item())),
            schema_version=str(payload["schema_version"].item()),
        )
    sample.validate()
    return sample
