#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.core.cognition.dataset import load_training_sample
from runtime.core.cognition.model import CognitiveModelConfig, NeuralCognitiveWorldModel, torch
from runtime.core.cognition.training import (
    cognitive_imitation_loss,
    save_cognitive_checkpoint,
    stack_training_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the proprietary cognitive world model from owned NPZ samples.")
    parser.add_argument("dataset", help="directory containing cognitive_*.npz samples")
    parser.add_argument("-o", "--output", default="artifacts/cognitive_world_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if torch is None:
        raise SystemExit("PyTorch is required; use the project ML environment")
    paths = sorted(Path(args.dataset).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZ samples found in {args.dataset}")
    samples = [load_training_sample(path) for path in paths]
    state_size = int(samples[0].state_context.shape[1])
    horizon = int(samples[0].expert_trajectory_xy.shape[0])
    model = NeuralCognitiveWorldModel(CognitiveModelConfig(state_size=state_size, horizon=horizon)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(samples), args.batch_size):
            batch = stack_training_samples(samples[start : start + args.batch_size], args.device)
            outputs = model(batch["rgbd"], batch["maps"], batch["state"])
            loss, _components = cognitive_imitation_loss(outputs, batch["trajectory"], batch["outcomes"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch:03d} loss={epoch_loss / max(1, batches):.6f}")
    save_cognitive_checkpoint(
        model,
        args.output,
        epoch=args.epochs,
        metadata={"dataset": str(Path(args.dataset).resolve()), "samples": len(samples)},
    )
    print(f"saved cognitive checkpoint: {args.output}")


if __name__ == "__main__":
    main()
