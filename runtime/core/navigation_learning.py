from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
from typing import Iterable

import numpy as np

from .navigation_intelligence import NavigationDecision, NavigationIntelligenceInput

try:  # Torch is optional for ROS deployment; required for training/inference.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in system-python deployment.
    torch = None
    nn = None


FEATURE_NAMES = [
    "cmd_linear_x",
    "cmd_angular_z",
    "front_m",
    "lower_front_m",
    "nearest_front_m",
    "left_clear_m",
    "right_clear_m",
    "depth_scene_delta_m",
    "front_depth_rate_mps",
    "control_progress_m",
    "visual_progress_m",
    "pose_divergence_m",
    "depth_age_sec",
    "control_pose_age_sec",
    "visual_pose_age_sec",
    "time_since_confirmed_progress_sec",
    "static_scene_duration_sec",
    "cumulative_cmd_odom_error_m",
    "recovery_count",
    "best_goal_delta_m",
]


@dataclass(frozen=True)
class NavigationFeatureFrame:
    time_sec: float
    values: np.ndarray
    names: tuple[str, ...] = tuple(FEATURE_NAMES)

    def as_float32(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)


@dataclass(frozen=True)
class LearnedRiskAssessment:
    available: bool
    risk_score: float
    action: str
    reason: str


@dataclass
class NavigationLearningMemory:
    """Engineered long-horizon memory wrapped around a lightweight GRU.

    The GRU handles temporal pattern recognition. This memory object provides
    physics-aware state that a small recurrent model would otherwise struggle to
    retain: progress age, static-scene duration, cumulative command-vs-odom
    error, repeated recovery count, and best goal-distance delta.
    """

    progress_epsilon_m: float = 0.05
    static_scene_delta_m: float = 0.12
    max_history: int = 64
    _last_sample: NavigationIntelligenceInput | None = None
    _last_progress_time: float | None = None
    _static_scene_start: float | None = None
    _cumulative_cmd_odom_error_m: float = 0.0
    _recovery_count: int = 0
    _best_goal_distance_m: float | None = None
    _history: list[NavigationFeatureFrame] = field(default_factory=list)

    def reset(self) -> None:
        self._last_sample = None
        self._last_progress_time = None
        self._static_scene_start = None
        self._cumulative_cmd_odom_error_m = 0.0
        self._recovery_count = 0
        self._best_goal_distance_m = None
        self._history.clear()

    @property
    def history(self) -> tuple[NavigationFeatureFrame, ...]:
        return tuple(self._history)

    def observe(
        self,
        sample: NavigationIntelligenceInput,
        decision: NavigationDecision | None = None,
    ) -> NavigationFeatureFrame:
        previous = self._last_sample
        dt = max(sample.now_sec - previous.now_sec, 1e-6) if previous is not None else 1e-6
        control_progress = _pose_progress(previous, sample, visual=False)
        visual_progress = _pose_progress(previous, sample, visual=True)
        depth_delta = _depth_delta(previous, sample)
        front_rate = _front_depth_rate(previous, sample, dt)

        if control_progress > self.progress_epsilon_m or visual_progress > self.progress_epsilon_m:
            self._last_progress_time = sample.now_sec
        elif self._last_progress_time is None:
            self._last_progress_time = sample.now_sec

        if depth_delta <= self.static_scene_delta_m:
            if self._static_scene_start is None:
                self._static_scene_start = sample.now_sec
        else:
            self._static_scene_start = None

        expected_motion = max(0.0, sample.proposed_command.linear_x) * dt
        self._cumulative_cmd_odom_error_m += max(0.0, expected_motion - control_progress)
        if decision is not None and decision.safety_action in {"REVERSE", "RECOVERY_TURN", "RELOCALIZE"}:
            self._recovery_count += 1

        goal_distance = getattr(sample, "goal_distance_m", None)
        if goal_distance is not None and math.isfinite(goal_distance):
            if self._best_goal_distance_m is None:
                self._best_goal_distance_m = float(goal_distance)
            else:
                self._best_goal_distance_m = min(self._best_goal_distance_m, float(goal_distance))
            best_goal_delta = float(goal_distance) - self._best_goal_distance_m
        else:
            best_goal_delta = 0.0

        signature = sample.depth_signature
        front = signature.front_m if signature is not None else 8.0
        lower_front = signature.lower_front_m if signature is not None else 8.0
        nearest_front = signature.nearest_front_m if signature is not None else 8.0
        left_clear = min(signature.left_m, signature.lower_left_m) if signature is not None else 8.0
        right_clear = min(signature.right_m, signature.lower_right_m) if signature is not None else 8.0
        pose_divergence = _pose_divergence(sample)
        visual_age = sample.visual_pose_age_sec if sample.visual_pose_age_sec is not None else 10.0
        time_since_progress = sample.now_sec - self._last_progress_time if self._last_progress_time is not None else 0.0
        static_scene_duration = sample.now_sec - self._static_scene_start if self._static_scene_start is not None else 0.0

        values = np.array(
            [
                sample.proposed_command.linear_x,
                sample.proposed_command.angular_z,
                front,
                lower_front,
                nearest_front,
                left_clear,
                right_clear,
                depth_delta,
                front_rate,
                control_progress,
                visual_progress,
                pose_divergence,
                sample.depth_age_sec,
                sample.control_pose_age_sec,
                visual_age,
                time_since_progress,
                static_scene_duration,
                self._cumulative_cmd_odom_error_m,
                float(self._recovery_count),
                best_goal_delta,
            ],
            dtype=np.float32,
        )
        frame = NavigationFeatureFrame(sample.now_sec, np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0))
        self._history.append(frame)
        if len(self._history) > self.max_history:
            del self._history[: len(self._history) - self.max_history]
        self._last_sample = sample
        return frame

    def recent_sequence(self, window_size: int) -> np.ndarray:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        frames = self._history[-window_size:]
        if not frames:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        sequence = np.stack([frame.as_float32() for frame in frames], axis=0)
        if sequence.shape[0] < window_size:
            pad = np.zeros((window_size - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([pad, sequence], axis=0)
        return sequence.astype(np.float32, copy=False)


if nn is not None:

    class GRUNavigationRiskModel(nn.Module):
        def __init__(self, input_size: int = len(FEATURE_NAMES), hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, sequence):
            output, _hidden = self.gru(sequence)
            logits = self.head(output[:, -1, :])
            return logits.squeeze(-1)

else:

    class GRUNavigationRiskModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GRUNavigationRiskModel")


@dataclass
class NavigationRiskPredictor:
    window_size: int = 16
    risk_threshold: float = 0.72
    caution_threshold: float = 0.55
    checkpoint_path: Path | None = None
    device: str = "cpu"
    _model: object | None = None

    @property
    def available(self) -> bool:
        return torch is not None and self._model is not None

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to load the navigation risk predictor")
        path = Path(checkpoint_path or self.checkpoint_path or "")
        if not path:
            raise ValueError("checkpoint_path is required")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        hidden_size = int(checkpoint.get("hidden_size", 32)) if isinstance(checkpoint, dict) else 32
        window_size = int(checkpoint.get("window_size", self.window_size)) if isinstance(checkpoint, dict) else self.window_size
        model = GRUNavigationRiskModel(input_size=len(FEATURE_NAMES), hidden_size=hidden_size)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._model = model
        self.window_size = window_size
        self.checkpoint_path = path

    def set_model(self, model: object) -> None:
        self._model = model
        if torch is not None and hasattr(model, "eval"):
            model.eval()

    def predict(self, sequence: np.ndarray) -> LearnedRiskAssessment:
        if torch is None:
            return LearnedRiskAssessment(False, 0.0, "UNAVAILABLE", "PyTorch is not installed")
        if self._model is None:
            return LearnedRiskAssessment(False, 0.0, "UNAVAILABLE", "no GRU checkpoint loaded")
        if sequence.ndim != 2 or sequence.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected sequence shape [T,{len(FEATURE_NAMES)}], got {sequence.shape}")
        tensor = torch.as_tensor(sequence[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            risk = float(torch.sigmoid(logits).detach().cpu().reshape(-1)[0])
        if risk >= self.risk_threshold:
            action = "HIGH_RISK"
        elif risk >= self.caution_threshold:
            action = "CAUTION"
        else:
            action = "LOW_RISK"
        return LearnedRiskAssessment(True, risk, action, f"GRU navigation risk={risk:.3f}")


def build_sequence_dataset_from_csv(
    csv_paths: Iterable[str | Path],
    *,
    window_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for csv_path in csv_paths:
        rows = _read_rows(csv_path)
        if len(rows) < window_size:
            continue
        vectors = [_features_from_row(rows, index) for index in range(len(rows))]
        row_labels = [_label_from_row(row) for row in rows]
        for index in range(window_size - 1, len(rows)):
            sequence = np.stack(vectors[index - window_size + 1:index + 1], axis=0).astype(np.float32)
            label = int(max(row_labels[index - window_size + 1:index + 1]))
            sequences.append(sequence)
            labels.append(label)
    if not sequences:
        return (
            np.zeros((0, window_size, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return np.stack(sequences, axis=0), np.asarray(labels, dtype=np.int64)


def save_sequence_dataset_npz(csv_paths: Iterable[str | Path], output_path: str | Path, *, window_size: int = 16) -> None:
    x, y = build_sequence_dataset_from_csv(csv_paths, window_size=window_size)
    np.savez_compressed(output_path, x=x, y=y, feature_names=np.asarray(FEATURE_NAMES), window_size=window_size)


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _features_from_row(rows: list[dict[str, str]], index: int) -> np.ndarray:
    row = rows[index]
    prev = rows[index - 1] if index > 0 else row
    time_sec = _float(row, "time_sec")
    prev_time = _float(prev, "time_sec", time_sec)
    dt = max(time_sec - prev_time, 1e-6)
    front = _float(row, "front", 8.0)
    lower_front = _float(row, "lower_front", 8.0)
    prev_front = _float(prev, "front", front)
    prev_lower = _float(prev, "lower_front", lower_front)
    depth_delta = max(abs(front - prev_front), abs(lower_front - prev_lower))
    x = _float(row, "pose_x")
    y = _float(row, "pose_y")
    prev_x = _float(prev, "pose_x", x)
    prev_y = _float(prev, "pose_y", y)
    control_progress = float(math.hypot(x - prev_x, y - prev_y))
    cmd_linear = _float(row, "cmd_v")
    expected_motion = max(0.0, cmd_linear) * dt
    cumulative_error = _rolling_command_error(rows, index)
    recovery_count = sum(1 for item in rows[: index + 1] if _label_from_row(item))
    best_goal = min(_float(item, "goal_dist", float("inf")) for item in rows[: index + 1])
    goal = _float(row, "goal_dist", best_goal if math.isfinite(best_goal) else 0.0)
    best_goal_delta = goal - best_goal if math.isfinite(best_goal) else 0.0
    values = np.array(
        [
            cmd_linear,
            _float(row, "cmd_w"),
            front,
            lower_front,
            min(front, lower_front),
            min(_float(row, "left", 8.0), _float(row, "lower_left", 8.0)),
            min(_float(row, "right", 8.0), _float(row, "lower_right", 8.0)),
            depth_delta,
            (min(front, lower_front) - min(prev_front, prev_lower)) / dt,
            control_progress,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,
            _time_since_progress(rows, index),
            _static_scene_duration(rows, index),
            cumulative_error,
            float(recovery_count),
            best_goal_delta,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)


def _label_from_row(row: dict[str, str]) -> int:
    explicit = row.get("label") or row.get("fault_label")
    if explicit is not None and explicit != "":
        return 1 if explicit.strip().lower() in {"1", "true", "fault", "blocked", "slip", "failure"} else 0
    motion_state = (row.get("nav_motion_state") or "").upper()
    safety_action = (row.get("nav_safety_action") or "").upper()
    mode = (row.get("mode") or "").lower()
    if motion_state in {"BLOCKED", "SLIPPING", "POSE_DIVERGENCE", "STALE_SENSORS"}:
        return 1
    if safety_action and safety_action not in {"ALLOW", "SLOW"}:
        return 1
    if mode.startswith("nav_intel_") or mode in {"escape_reverse", "escape_turn", "blocked_turn"}:
        return 1
    return 0


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else float(default)
    except (TypeError, ValueError):
        return float(default)


def _rolling_command_error(rows: list[dict[str, str]], index: int, horizon: int = 16) -> float:
    total = 0.0
    start = max(1, index - horizon + 1)
    for idx in range(start, index + 1):
        row = rows[idx]
        prev = rows[idx - 1]
        dt = max(_float(row, "time_sec") - _float(prev, "time_sec"), 1e-6)
        expected = max(0.0, _float(row, "cmd_v")) * dt
        progress = math.hypot(_float(row, "pose_x") - _float(prev, "pose_x"), _float(row, "pose_y") - _float(prev, "pose_y"))
        total += max(0.0, expected - progress)
    return float(total)


def _time_since_progress(rows: list[dict[str, str]], index: int, epsilon: float = 0.05) -> float:
    current_time = _float(rows[index], "time_sec")
    for idx in range(index, 0, -1):
        progress = math.hypot(
            _float(rows[idx], "pose_x") - _float(rows[idx - 1], "pose_x"),
            _float(rows[idx], "pose_y") - _float(rows[idx - 1], "pose_y"),
        )
        if progress > epsilon:
            return max(0.0, current_time - _float(rows[idx], "time_sec"))
    return max(0.0, current_time - _float(rows[0], "time_sec"))


def _static_scene_duration(rows: list[dict[str, str]], index: int, threshold: float = 0.12) -> float:
    current_time = _float(rows[index], "time_sec")
    for idx in range(index, 0, -1):
        front_delta = abs(_float(rows[idx], "front", 8.0) - _float(rows[idx - 1], "front", 8.0))
        lower_delta = abs(_float(rows[idx], "lower_front", 8.0) - _float(rows[idx - 1], "lower_front", 8.0))
        if max(front_delta, lower_delta) > threshold:
            return max(0.0, current_time - _float(rows[idx], "time_sec"))
    return max(0.0, current_time - _float(rows[0], "time_sec"))


def _pose_progress(previous: NavigationIntelligenceInput | None, current: NavigationIntelligenceInput, *, visual: bool) -> float:
    if previous is None:
        return 0.0
    start = previous.visual_pose if visual else previous.control_pose
    end = current.visual_pose if visual else current.control_pose
    if start is None or end is None:
        return 0.0
    return end.distance_to(start)


def _depth_delta(previous: NavigationIntelligenceInput | None, current: NavigationIntelligenceInput) -> float:
    if previous is None or previous.depth_signature is None or current.depth_signature is None:
        return 0.0
    return current.depth_signature.max_delta(previous.depth_signature)


def _front_depth_rate(previous: NavigationIntelligenceInput | None, current: NavigationIntelligenceInput, dt: float) -> float:
    if previous is None or previous.depth_signature is None or current.depth_signature is None:
        return 0.0
    return (current.depth_signature.nearest_front_m - previous.depth_signature.nearest_front_m) / dt


def _pose_divergence(sample: NavigationIntelligenceInput) -> float:
    if sample.control_pose is None or sample.visual_pose is None:
        return 0.0
    return sample.control_pose.distance_to(sample.visual_pose)
