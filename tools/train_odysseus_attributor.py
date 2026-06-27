#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.core.odysseus import (
    OdysseusAttributionConfig,
    OdysseusCausalAttributor,
    load_causal_sample,
    odysseus_attribution_loss,
    save_odysseus_checkpoint,
    stack_causal_samples,
)
from runtime.core.odysseus.attribution import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Odysseus causal attribution from owned episode NPZ samples.")
    parser.add_argument("dataset", help="directory containing odysseus_*.npz samples")
    parser.add_argument("-o", "--output", default="artifacts/odysseus_attributor.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if torch is None:
        raise SystemExit("PyTorch is required; use the project ML environment")
    paths = sorted(Path(args.dataset).glob("odysseus_*.npz"))
    if not paths:
        raise SystemExit(f"no Odysseus NPZ samples found in {args.dataset}")
    samples = [load_causal_sample(path) for path in paths]
    model = OdysseusCausalAttributor(OdysseusAttributionConfig()).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(samples), args.batch_size):
            batch = stack_causal_samples(samples[start : start + args.batch_size], args.device)
            outputs = model(batch["features"])
            loss, _components = odysseus_attribution_loss(outputs, batch["cause"], batch["outcomes"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch:03d} loss={epoch_loss / max(1, batches):.6f}")
    save_odysseus_checkpoint(
        model,
        args.output,
        epoch=args.epochs,
        metadata={"dataset": str(Path(args.dataset).resolve()), "samples": len(samples)},
    )
    print(f"saved Odysseus checkpoint: {args.output}")


if __name__ == "__main__":
    main()
