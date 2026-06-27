from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import tempfile

import numpy as np

from .contracts import COGNITIVE_SCHEMA_VERSION


@dataclass(frozen=True)
class CognitiveTrainingSample:
    """One owned, versioned temporal sample for cognitive-model training."""

    rgbd_context: np.ndarray
    map_context: np.ndarray
    state_context: np.ndarray
    expert_trajectory_xy: np.ndarray
    outcome_targets: np.ndarray
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)
    schema_version: str = COGNITIVE_SCHEMA_VERSION

    def validate(self) -> None:
        rgbd = np.asarray(self.rgbd_context)
        maps = np.asarray(self.map_context)
        state = np.asarray(self.state_context)
        trajectory = np.asarray(self.expert_trajectory_xy)
        outcomes = np.asarray(self.outcome_targets)
        if self.schema_version != COGNITIVE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cognitive dataset schema {self.schema_version}")
        if rgbd.ndim != 4 or rgbd.shape[1] != 4:
            raise ValueError("rgbd_context must have shape [T,4,H,W]")
        if maps.ndim != 4 or maps.shape[1] != 2:
            raise ValueError("map_context must have shape [T,2,H,W]")
        if state.ndim != 2 or state.shape[0] != rgbd.shape[0] or maps.shape[0] != rgbd.shape[0]:
            raise ValueError("state, map, and RGB-D temporal dimensions must match")
        if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) == 0:
            raise ValueError("expert_trajectory_xy must have shape [H,2]")
        if outcomes.shape != (4,):
            raise ValueError("outcome_targets must be [progress, collision, exploration, uncertainty]")
        for array in (rgbd, maps, state, trajectory, outcomes):
            if not np.isfinite(array).all():
                raise ValueError("training arrays must contain only finite values")


def save_training_sample(sample: CognitiveTrainingSample, output_path: str | Path) -> Path:
    sample.validate()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(sample.metadata, sort_keys=True, separators=(",", ":"))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                rgbd_context=np.asarray(sample.rgbd_context, dtype=np.float32),
                map_context=np.asarray(sample.map_context, dtype=np.float32),
                state_context=np.asarray(sample.state_context, dtype=np.float32),
                expert_trajectory_xy=np.asarray(sample.expert_trajectory_xy, dtype=np.float32),
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


def load_training_sample(path: str | Path) -> CognitiveTrainingSample:
    with np.load(Path(path), allow_pickle=False) as payload:
        sample = CognitiveTrainingSample(
            rgbd_context=payload["rgbd_context"],
            map_context=payload["map_context"],
            state_context=payload["state_context"],
            expert_trajectory_xy=payload["expert_trajectory_xy"],
            outcome_targets=payload["outcome_targets"],
            metadata=json.loads(str(payload["metadata_json"].item())),
            schema_version=str(payload["schema_version"].item()),
        )
    sample.validate()
    return sample
