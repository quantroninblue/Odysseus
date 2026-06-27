from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import CognitiveBeliefState, CognitiveDecision, CognitiveObservation
from .dataset import CognitiveTrainingSample, save_training_sample
from .memory import CognitiveMemory, observation_context_vector
from .teacher import DeterministicCognitiveTeacher


@dataclass(frozen=True)
class CognitiveShadowResult:
    belief: CognitiveBeliefState
    decision: CognitiveDecision
    sample_path: Path | None = None


class CognitiveShadowRunner:
    """Runs cognition without command authority and optionally records owned data."""

    def __init__(
        self,
        *,
        context_length: int = 6,
        image_size: tuple[int, int] = (64, 96),
        map_size: tuple[int, int] = (64, 64),
        dataset_directory: str | Path | None = None,
        record_stride: int = 5,
    ):
        if context_length <= 0 or record_stride <= 0:
            raise ValueError("context_length and record_stride must be positive")
        self.context_length = context_length
        self.image_size = image_size
        self.map_size = map_size
        self.dataset_directory = Path(dataset_directory) if dataset_directory else None
        self.record_stride = record_stride
        self.memory = CognitiveMemory()
        self.teacher = DeterministicCognitiveTeacher()
        self._frames: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=context_length)
        self._step = 0

    def observe(self, observation: CognitiveObservation) -> CognitiveShadowResult:
        belief = self.memory.observe(observation)
        decision = self.teacher.decide(observation)
        self._frames.append(self._tensorize(observation))
        self._step += 1
        sample_path = None
        if (
            self.dataset_directory is not None
            and len(self._frames) == self.context_length
            and self._step % self.record_stride == 0
            and decision.selected is not None
        ):
            rgbd, maps, state = (np.stack(items, axis=0) for items in zip(*self._frames))
            selected = decision.selected
            sample = CognitiveTrainingSample(
                rgbd_context=rgbd,
                map_context=maps,
                state_context=state,
                expert_trajectory_xy=selected.points_robot_xy,
                outcome_targets=np.asarray(
                    [
                        selected.predicted_progress_m,
                        selected.collision_probability,
                        selected.exploration_value,
                        selected.uncertainty,
                    ],
                    dtype=np.float32,
                ),
                metadata={
                    "source": "owned_runtime_teacher",
                    "policy_version": decision.policy_version,
                    "time_sec": observation.stamp.time_sec,
                    "frame_id": observation.stamp.frame_id,
                },
            )
            name = f"cognitive_{observation.stamp.time_sec:.3f}_{self._step:08d}.npz"
            sample_path = save_training_sample(sample, self.dataset_directory / name)
        return CognitiveShadowResult(belief, decision, sample_path)

    def _tensorize(self, observation: CognitiveObservation):
        height, width = self.image_size
        if observation.rgb is None:
            rgb = np.zeros((height, width, 3), dtype=np.float32)
        else:
            rgb = _resize_nearest(np.asarray(observation.rgb), self.image_size).astype(np.float32) / 255.0
        if observation.depth_m is None:
            depth = np.zeros((height, width), dtype=np.float32)
        else:
            depth = _resize_nearest(np.asarray(observation.depth_m), self.image_size).astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 10.0) / 10.0
        rgbd = np.concatenate([np.moveaxis(rgb, -1, 0), depth[None]], axis=0)
        local = _resize_nearest(np.asarray(observation.local_occupancy), self.map_size)
        global_map = _resize_nearest(np.asarray(observation.global_occupancy), self.map_size)
        maps = np.stack([local, global_map], axis=0).astype(np.float32)
        state = observation_context_vector(observation, 16)
        return rgbd.astype(np.float32), maps, state


def _resize_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    output_h, output_w = size
    input_h, input_w = array.shape[:2]
    rows = np.minimum((np.arange(output_h) * input_h / output_h).astype(int), input_h - 1)
    cols = np.minimum((np.arange(output_w) * input_w / output_w).astype(int), input_w - 1)
    return array[rows[:, None], cols[None, :]]
