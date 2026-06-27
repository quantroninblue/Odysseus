from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contracts import ODYSSEUS_CAUSE_LABELS, ODYSSEUS_FEATURE_SIZE, CausalTrainingSample

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - deployment may omit PyTorch.
    torch = None
    nn = None


@dataclass(frozen=True)
class OdysseusAttributionConfig:
    input_size: int = ODYSSEUS_FEATURE_SIZE
    hidden_size: int = 128
    cause_count: int = len(ODYSSEUS_CAUSE_LABELS)
    dropout: float = 0.08

    def validate(self) -> None:
        if min(self.input_size, self.hidden_size, self.cause_count) <= 0:
            raise ValueError("Odysseus attribution dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Odysseus dropout must be within [0,1)")


@dataclass(frozen=True)
class OdysseusLossWeights:
    cause: float = 2.0
    progress: float = 1.0
    binary_outcomes: float = 1.5
    distance: float = 0.5
    severity: float = 1.0


if nn is not None:

    class OdysseusCausalAttributor(nn.Module):
        """MLP failure attributor trained on Odysseus episode closure records."""

        def __init__(self, config: OdysseusAttributionConfig | None = None):
            super().__init__()
            self.config = config or OdysseusAttributionConfig()
            self.config.validate()
            hidden = self.config.hidden_size
            self.encoder = nn.Sequential(
                nn.Linear(self.config.input_size, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.cause_head = nn.Linear(hidden, self.config.cause_count)
            self.progress_head = nn.Linear(hidden, 1)
            self.binary_head = nn.Linear(hidden, 5)
            self.distance_head = nn.Linear(hidden, 1)
            self.severity_head = nn.Linear(hidden, 1)

        def forward(self, features):
            if features.ndim != 2 or features.shape[1] != self.config.input_size:
                raise ValueError(f"features must have shape [B,{self.config.input_size}]")
            latent = self.encoder(features)
            return {
                "cause_logits": self.cause_head(latent),
                "progress_delta": self.progress_head(latent).squeeze(-1),
                "binary_logits": self.binary_head(latent),
                "final_goal_distance": torch.sigmoid(self.distance_head(latent).squeeze(-1)),
                "severity": torch.sigmoid(self.severity_head(latent).squeeze(-1)),
            }

else:

    class OdysseusCausalAttributor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for OdysseusCausalAttributor")


def stack_causal_samples(samples: list[CausalTrainingSample], device: str = "cpu"):
    _require_torch()
    if not samples:
        raise ValueError("at least one Odysseus causal sample is required")
    for sample in samples:
        sample.validate()
    return {
        "features": torch.as_tensor(np.stack([sample.feature_vector for sample in samples], axis=0), dtype=torch.float32, device=device),
        "cause": torch.as_tensor([sample.cause_index for sample in samples], dtype=torch.long, device=device),
        "outcomes": torch.as_tensor(np.stack([sample.outcome_targets for sample in samples], axis=0), dtype=torch.float32, device=device),
    }


def odysseus_attribution_loss(outputs, cause_targets, outcome_targets, weights=None):
    _require_torch()
    weights = weights or OdysseusLossWeights()
    cause_loss = torch.nn.functional.cross_entropy(outputs["cause_logits"], cause_targets)
    progress_loss = torch.nn.functional.smooth_l1_loss(outputs["progress_delta"], outcome_targets[:, 0])
    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["binary_logits"], outcome_targets[:, 1:6].clamp(0.0, 1.0)
    )
    distance_loss = torch.nn.functional.smooth_l1_loss(outputs["final_goal_distance"], outcome_targets[:, 6])
    severity_loss = torch.nn.functional.smooth_l1_loss(outputs["severity"], outcome_targets[:, 7])
    total = (
        weights.cause * cause_loss
        + weights.progress * progress_loss
        + weights.binary_outcomes * binary_loss
        + weights.distance * distance_loss
        + weights.severity * severity_loss
    )
    return total, {
        "cause": cause_loss.detach(),
        "progress": progress_loss.detach(),
        "binary_outcomes": binary_loss.detach(),
        "distance": distance_loss.detach(),
        "severity": severity_loss.detach(),
    }


def save_odysseus_checkpoint(model, output_path: str | Path, *, epoch: int, metadata=None) -> Path:
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
            "cause_labels": ODYSSEUS_CAUSE_LABELS,
        },
        path,
    )
    return path


def load_odysseus_checkpoint(path: str | Path, device: str = "cpu"):
    _require_torch()
    checkpoint = torch.load(Path(path), map_location=device)
    config = OdysseusAttributionConfig(**checkpoint["model_config"])
    model = OdysseusCausalAttributor(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for Odysseus attribution training")
