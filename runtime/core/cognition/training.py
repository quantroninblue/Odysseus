from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import CognitiveTrainingSample
from .model import CognitiveModelConfig, NeuralCognitiveWorldModel, torch


@dataclass(frozen=True)
class CognitiveLossWeights:
    trajectory: float = 2.0
    progress: float = 1.0
    collision: float = 2.0
    exploration: float = 0.5
    uncertainty: float = 0.25
    value: float = 0.5


def stack_training_samples(samples: list[CognitiveTrainingSample], device: str = "cpu"):
    _require_torch()
    if not samples:
        raise ValueError("at least one cognitive training sample is required")
    for sample in samples:
        sample.validate()
    return {
        "rgbd": torch.as_tensor([sample.rgbd_context for sample in samples], dtype=torch.float32, device=device),
        "maps": torch.as_tensor([sample.map_context for sample in samples], dtype=torch.float32, device=device),
        "state": torch.as_tensor([sample.state_context for sample in samples], dtype=torch.float32, device=device),
        "trajectory": torch.as_tensor([sample.expert_trajectory_xy for sample in samples], dtype=torch.float32, device=device),
        "outcomes": torch.as_tensor([sample.outcome_targets for sample in samples], dtype=torch.float32, device=device),
    }


def cognitive_imitation_loss(outputs, expert_trajectory, outcome_targets, weights=None):
    _require_torch()
    weights = weights or CognitiveLossWeights()
    candidates = outputs["trajectories"]
    if candidates.shape[2:] != expert_trajectory.shape[1:]:
        raise ValueError("model and expert trajectory horizons must match")
    trajectory_errors = ((candidates - expert_trajectory[:, None]) ** 2).mean(dim=(-1, -2))
    best_error, best_index = trajectory_errors.min(dim=1)
    batch_index = torch.arange(candidates.shape[0], device=candidates.device)

    def selected(name):
        return outputs[name][batch_index, best_index]

    progress_loss = torch.nn.functional.smooth_l1_loss(selected("predicted_progress"), outcome_targets[:, 0])
    collision_loss = torch.nn.functional.binary_cross_entropy(
        selected("collision_probability"), outcome_targets[:, 1].clamp(0.0, 1.0)
    )
    exploration_loss = torch.nn.functional.smooth_l1_loss(selected("exploration_value"), outcome_targets[:, 2])
    uncertainty_loss = torch.nn.functional.smooth_l1_loss(selected("uncertainty"), outcome_targets[:, 3])
    target_value = outcome_targets[:, 0] - 5.0 * outcome_targets[:, 1] + 0.2 * outcome_targets[:, 2]
    value_loss = torch.nn.functional.smooth_l1_loss(selected("value"), target_value)
    trajectory_loss = best_error.mean()
    total = (
        weights.trajectory * trajectory_loss
        + weights.progress * progress_loss
        + weights.collision * collision_loss
        + weights.exploration * exploration_loss
        + weights.uncertainty * uncertainty_loss
        + weights.value * value_loss
    )
    return total, {
        "trajectory": trajectory_loss.detach(),
        "progress": progress_loss.detach(),
        "collision": collision_loss.detach(),
        "exploration": exploration_loss.detach(),
        "uncertainty": uncertainty_loss.detach(),
        "value": value_loss.detach(),
    }


def save_cognitive_checkpoint(model, output_path: str | Path, *, epoch: int, metadata=None) -> Path:
    _require_torch()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return path


def load_cognitive_checkpoint(path: str | Path, device: str = "cpu"):
    _require_torch()
    checkpoint = torch.load(Path(path), map_location=device)
    config = CognitiveModelConfig(**checkpoint["model_config"])
    model = NeuralCognitiveWorldModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for cognitive-model training")
