from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import os
import tempfile

import numpy as np

from .contracts import OdysseusOutcome


ODYSSEUS_MEMORY_VERSION = "1.0"


@dataclass
class SpatialMemoryCell:
    x_m: float
    y_m: float
    value: float
    visits: int = 1
    failures: int = 0
    successes: int = 0
    last_time_sec: float = 0.0
    cause: str = ""

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m, self.value, self.last_time_sec)):
            raise ValueError("spatial memory cell contains non-finite values")
        if self.visits <= 0 or self.failures < 0 or self.successes < 0:
            raise ValueError("spatial memory counts are invalid")


@dataclass
class BehaviorValue:
    value: float = 0.0
    visits: int = 0

    def update(self, reward: float, learning_rate: float = 0.18) -> None:
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        alpha = float(np.clip(learning_rate, 0.001, 1.0))
        self.value = float((1.0 - alpha) * self.value + alpha * reward)
        self.visits += 1


@dataclass
class OdysseusWorldMemory:
    """Persistent cross-run memory for Odysseus navigation.

    This is deliberately lightweight and deterministic: it gives Odysseus a
    durable world-state prior and online value estimates without pretending to be
    the final deep-RL system.
    """

    resolution_m: float = 0.50
    max_cells: int = 4096
    schema_version: str = ODYSSEUS_MEMORY_VERSION
    no_go: dict[str, SpatialMemoryCell] = field(default_factory=dict)
    success: dict[str, SpatialMemoryCell] = field(default_factory=dict)
    behavior_values: dict[str, BehaviorValue] = field(default_factory=dict)

    def key(self, x_m: float, y_m: float) -> str:
        col = int(math.floor(float(x_m) / self.resolution_m))
        row = int(math.floor(float(y_m) / self.resolution_m))
        return f"{row}:{col}"

    def cell_center(self, key: str) -> tuple[float, float]:
        row_s, col_s = key.split(":", 1)
        row = int(row_s)
        col = int(col_s)
        return ((col + 0.5) * self.resolution_m, (row + 0.5) * self.resolution_m)

    def record_outcome(
        self,
        *,
        pose_xy: tuple[float, float],
        command_mode: str,
        action_bucket: str,
        outcome: OdysseusOutcome,
        time_sec: float,
    ) -> float:
        outcome.validate()
        reward = self.reward(outcome)
        behavior_key = f"{command_mode}:{action_bucket}"
        self.behavior_values.setdefault(behavior_key, BehaviorValue()).update(reward)
        key = self.key(*pose_xy)
        target = self.no_go if outcome.failed else self.success
        other = self.success if outcome.failed else self.no_go
        x_m, y_m = self.cell_center(key)
        cell = target.get(key)
        if cell is None:
            cell = SpatialMemoryCell(x_m=x_m, y_m=y_m, value=0.0, last_time_sec=float(time_sec))
            target[key] = cell
        cell.visits += 1
        cell.last_time_sec = float(time_sec)
        cell.cause = outcome.failure_cause
        if outcome.failed:
            cell.failures += 1
            cell.value = float(np.clip(cell.value + 0.45 + outcome.severity, 0.0, 8.0))
        else:
            cell.successes += 1
            cell.value = float(np.clip(cell.value + 0.30 + max(0.0, outcome.progress_delta_m), 0.0, 8.0))
            if key in other:
                other[key].value = max(0.0, other[key].value - 0.35)
        self._prune()
        return reward

    def reward(self, outcome: OdysseusOutcome) -> float:
        penalty = (
            2.0 * float(outcome.collision)
            + 1.3 * float(outcome.stuck)
            + 1.0 * float(outcome.safety_override)
            + 0.9 * float(outcome.localization_diverged)
            + 0.7 * float(outcome.stale_sensor)
            + outcome.severity
        )
        return float(outcome.progress_delta_m - penalty)

    def behavior_value(self, mode: str, action_bucket: str) -> float:
        item = self.behavior_values.get(f"{mode}:{action_bucket}")
        return item.value if item is not None else 0.0

    def spatial_bias(self, x_m: float, y_m: float, radius_m: float = 1.0) -> float:
        penalty = self._nearby_value(self.no_go, x_m, y_m, radius_m)
        reward = self._nearby_value(self.success, x_m, y_m, radius_m)
        return float(reward - penalty)

    def remembered_no_go_points(self, *, minimum_value: float = 0.8) -> np.ndarray:
        points = [
            (cell.x_m, cell.y_m)
            for cell in self.no_go.values()
            if cell.value >= minimum_value
        ]
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)

    def remembered_success_points(self, *, minimum_value: float = 0.8) -> np.ndarray:
        points = [
            (cell.x_m, cell.y_m)
            for cell in self.success.values()
            if cell.value >= minimum_value
        ]
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "resolution_m": self.resolution_m,
            "max_cells": self.max_cells,
            "no_go": {key: asdict(cell) for key, cell in self.no_go.items()},
            "success": {key: asdict(cell) for key, cell in self.success.items()},
            "behavior_values": {key: asdict(value) for key, value in self.behavior_values.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OdysseusWorldMemory":
        if payload.get("schema_version") != ODYSSEUS_MEMORY_VERSION:
            raise ValueError(f"unsupported Odysseus memory schema {payload.get('schema_version')}")
        memory = cls(
            resolution_m=float(payload.get("resolution_m", 0.50)),
            max_cells=int(payload.get("max_cells", 4096)),
            schema_version=str(payload.get("schema_version")),
        )
        memory.no_go = {
            key: SpatialMemoryCell(**value)
            for key, value in dict(payload.get("no_go", {})).items()
        }
        memory.success = {
            key: SpatialMemoryCell(**value)
            for key, value in dict(payload.get("success", {})).items()
        }
        memory.behavior_values = {
            key: BehaviorValue(**value)
            for key, value in dict(payload.get("behavior_values", {})).items()
        }
        memory.validate()
        return memory

    def validate(self) -> None:
        if self.resolution_m <= 0.0 or self.max_cells <= 0:
            raise ValueError("Odysseus memory dimensions are invalid")
        for cell in [*self.no_go.values(), *self.success.values()]:
            cell.validate()

    def save(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"))
            os.replace(temporary_name, output)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return output

    @classmethod
    def load(cls, path: str | Path) -> "OdysseusWorldMemory":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def load_or_empty(cls, path: str | Path | None) -> "OdysseusWorldMemory":
        if path is None:
            return cls()
        memory_path = Path(path)
        if not memory_path.exists():
            return cls()
        return cls.load(memory_path)

    def _nearby_value(
        self,
        cells: dict[str, SpatialMemoryCell],
        x_m: float,
        y_m: float,
        radius_m: float,
    ) -> float:
        radius = max(radius_m, self.resolution_m)
        total = 0.0
        for cell in cells.values():
            distance = math.hypot(cell.x_m - x_m, cell.y_m - y_m)
            if distance > radius:
                continue
            total += cell.value * (1.0 - distance / radius)
        return float(total)

    def _prune(self) -> None:
        for cells in (self.no_go, self.success):
            if len(cells) <= self.max_cells:
                continue
            ranked = sorted(cells.items(), key=lambda item: (item[1].value, item[1].last_time_sec))
            for key, _cell in ranked[: len(cells) - self.max_cells]:
                del cells[key]
