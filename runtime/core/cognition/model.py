from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - system deployment may omit PyTorch.
    torch = None
    nn = None


@dataclass(frozen=True)
class CognitiveModelConfig:
    state_size: int = 16
    hidden_size: int = 128
    candidate_count: int = 6
    horizon: int = 5
    max_step_m: float = 0.55

    def validate(self) -> None:
        if min(self.state_size, self.hidden_size, self.candidate_count, self.horizon) <= 0:
            raise ValueError("cognitive model dimensions must be positive")
        if self.max_step_m <= 0.0:
            raise ValueError("max_step_m must be positive")


if nn is not None:

    class _SpatialEncoder(nn.Module):
        def __init__(self, channels: int, output_size: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(channels, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 48, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(48, output_size),
                nn.LayerNorm(output_size),
            )

        def forward(self, value):
            return self.network(value)


    class NeuralCognitiveWorldModel(nn.Module):
        """Compact multimodal model that proposes and evaluates imagined futures."""

        def __init__(self, config: CognitiveModelConfig | None = None):
            super().__init__()
            self.config = config or CognitiveModelConfig()
            self.config.validate()
            hidden = self.config.hidden_size
            self.rgbd_encoder = _SpatialEncoder(4, 64)
            self.map_encoder = _SpatialEncoder(2, 48)
            self.state_encoder = nn.Sequential(
                nn.Linear(self.config.state_size, 48), nn.ReLU(), nn.LayerNorm(48)
            )
            self.fusion = nn.Sequential(nn.Linear(160, hidden), nn.ReLU(), nn.LayerNorm(hidden))
            self.working_memory = nn.GRU(hidden, hidden, batch_first=True)
            self.trajectory_head = nn.Linear(
                hidden, self.config.candidate_count * self.config.horizon * 2
            )
            action_size = self.config.horizon * 2
            self.action_encoder = nn.Sequential(nn.Linear(action_size, hidden), nn.Tanh())
            self.world_transition = nn.GRUCell(hidden, hidden)
            self.outcome_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 5)
            )

        def forward(self, rgbd_context, map_context, state_context, memory=None):
            self._validate_inputs(rgbd_context, map_context, state_context)
            batch, steps = state_context.shape[:2]
            rgbd = self.rgbd_encoder(rgbd_context.reshape(-1, *rgbd_context.shape[2:]))
            maps = self.map_encoder(map_context.reshape(-1, *map_context.shape[2:]))
            state = self.state_encoder(state_context.reshape(-1, state_context.shape[-1]))
            fused = self.fusion(torch.cat([rgbd, maps, state], dim=-1)).reshape(batch, steps, -1)
            remembered, next_memory = self.working_memory(fused, memory)
            context = remembered[:, -1]

            increments = torch.tanh(self.trajectory_head(context)).reshape(
                batch, self.config.candidate_count, self.config.horizon, 2
            ) * self.config.max_step_m
            trajectories = torch.cumsum(increments, dim=2)
            actions = self.action_encoder(trajectories.flatten(start_dim=2))
            expanded_context = context[:, None, :].expand(-1, self.config.candidate_count, -1)
            imagined = self.world_transition(
                actions.reshape(-1, self.config.hidden_size),
                expanded_context.reshape(-1, self.config.hidden_size),
            ).reshape(batch, self.config.candidate_count, self.config.hidden_size)
            raw = self.outcome_head(torch.cat([expanded_context, imagined], dim=-1))
            return {
                "trajectories": trajectories,
                "predicted_progress": raw[..., 0],
                "collision_probability": torch.sigmoid(raw[..., 1]),
                "exploration_value": raw[..., 2],
                "uncertainty": torch.nn.functional.softplus(raw[..., 3]),
                "value": raw[..., 4],
                "memory": next_memory,
            }

        def _validate_inputs(self, rgbd, maps, state) -> None:
            if rgbd.ndim != 5 or rgbd.shape[2] != 4:
                raise ValueError("rgbd_context must have shape [B,T,4,H,W]")
            if maps.ndim != 5 or maps.shape[2] != 2:
                raise ValueError("map_context must have shape [B,T,2,H,W]")
            if state.ndim != 3 or state.shape[2] != self.config.state_size:
                raise ValueError(f"state_context must have shape [B,T,{self.config.state_size}]")
            if rgbd.shape[:2] != maps.shape[:2] or rgbd.shape[:2] != state.shape[:2]:
                raise ValueError("all cognitive inputs must share batch and temporal dimensions")

else:

    class NeuralCognitiveWorldModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for NeuralCognitiveWorldModel")
