from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import (
    FailureAttribution,
    OdysseusDecisionTrace,
    OdysseusOutcome,
    RolloutCandidateRecord,
    build_causal_feature_vector,
    make_causal_sample,
    save_causal_sample,
    ODYSSEUS_CAUSE_LABELS,
)
from .attribution import OdysseusCausalAttributor, torch
from runtime.core.cognition import CognitiveObservation


@dataclass(frozen=True)
class OdysseusShadowResult:
    trace: OdysseusDecisionTrace
    sample_path: Path | None = None


class OdysseusShadowRunner:
    """Shadow-only Odysseus layer for causal attribution and episode closure."""

    policy_version = "odysseus_shadow_v1"

    def __init__(
        self,
        *,
        model: OdysseusCausalAttributor | None = None,
        dataset_directory: str | Path | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.dataset_directory = Path(dataset_directory) if dataset_directory else None
        self.device = device
        self._open_traces: dict[str, OdysseusDecisionTrace] = {}
        if self.model is not None and torch is not None:
            self.model.to(device).eval()

    def observe(
        self,
        observation: CognitiveObservation,
        *,
        selected_candidate: RolloutCandidateRecord | None = None,
        candidate_records: tuple[RolloutCandidateRecord, ...] | list[RolloutCandidateRecord] = (),
        semantic_forward_m: float = 8.0,
        semantic_lateral_m: float = 0.0,
        progress_distance_m: float = 0.0,
        trace_id: str | None = None,
    ) -> OdysseusShadowResult:
        features = build_causal_feature_vector(
            observation,
            selected_candidate,
            candidate_records,
            semantic_forward_m=semantic_forward_m,
            semantic_lateral_m=semantic_lateral_m,
            progress_distance_m=progress_distance_m,
        )
        attribution = self._predict(features)
        selected_id = selected_candidate.candidate_id if selected_candidate is not None else None
        trace = OdysseusDecisionTrace(
            trace_id=trace_id or f"odysseus_{observation.stamp.time_sec:.3f}",
            time_sec=observation.stamp.time_sec,
            feature_vector=features,
            selected_candidate_id=selected_id,
            candidate_count=len(candidate_records),
            attribution=attribution,
            metadata={
                "source": "odysseus_shadow",
                "policy_version": self.policy_version,
                "frame_id": observation.stamp.frame_id,
            },
        )
        trace.validate()
        self._open_traces[trace.trace_id] = trace
        return OdysseusShadowResult(trace=trace)

    def close_episode(
        self,
        trace_id: str,
        outcome: OdysseusOutcome,
        *,
        metadata: dict[str, str | float | int | bool] | None = None,
    ) -> OdysseusShadowResult:
        if trace_id not in self._open_traces:
            raise KeyError(f"unknown Odysseus trace: {trace_id}")
        trace = self._open_traces.pop(trace_id)
        sample_path = None
        if self.dataset_directory is not None:
            sample = make_causal_sample(trace, outcome, metadata=metadata)
            name = f"odysseus_{trace.time_sec:.3f}_{trace.trace_id}.npz"
            sample_path = save_causal_sample(sample, self.dataset_directory / name)
        return OdysseusShadowResult(trace=trace, sample_path=sample_path)

    def predict_attribution(self, features: np.ndarray) -> FailureAttribution | None:
        return self._predict(features)

    def _predict(self, features: np.ndarray) -> FailureAttribution | None:
        if self.model is None or torch is None:
            return None
        with torch.no_grad():
            batch = torch.as_tensor(features[None], dtype=torch.float32, device=self.device)
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs["cause_logits"], dim=-1)[0].detach().cpu().numpy()
            cause_index = int(np.argmax(probabilities))
            binary = torch.sigmoid(outputs["binary_logits"])[0].detach().cpu().numpy()
            attribution = FailureAttribution(
                cause=ODYSSEUS_CAUSE_LABELS[cause_index],
                confidence=float(probabilities[cause_index]),
                severity=float(outputs["severity"][0].detach().cpu()),
                progress_delta_m=float(outputs["progress_delta"][0].detach().cpu()),
                collision_probability=float(binary[0]),
                stuck_probability=float(binary[1]),
                safety_override_probability=float(binary[2]),
                localization_risk=float(binary[3]),
                stale_sensor_risk=float(binary[4]),
                reason="mlp causal attribution",
            )
        attribution.validate()
        return attribution
