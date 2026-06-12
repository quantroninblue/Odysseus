#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from runtime.core.navigation_learning import FEATURE_NAMES, GRUNavigationRiskModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU navigation-risk classifier from .npz sequence dataset.")
    parser.add_argument("dataset", help="dataset .npz from build_navigation_learning_dataset.py")
    parser.add_argument("--output", "-o", required=True, help="checkpoint .pt path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required. Use .venv/bin/python after installing torch.") from exc

    rng = np.random.default_rng(args.seed)
    data = np.load(args.dataset, allow_pickle=True)
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != len(FEATURE_NAMES):
        raise SystemExit(f"expected x shape [N,T,{len(FEATURE_NAMES)}], got {x.shape}")
    if x.shape[0] == 0:
        raise SystemExit("dataset contains no sequences")

    order = rng.permutation(x.shape[0])
    x = x[order]
    y = y[order]
    split = max(1, int(0.8 * x.shape[0]))
    train_x, val_x = x[:split], x[split:]
    train_y, val_y = y[:split], y[split:]
    if val_x.shape[0] == 0:
        val_x, val_y = train_x, train_y

    model = GRUNavigationRiskModel(input_size=len(FEATURE_NAMES), hidden_size=args.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positives = max(float(train_y.sum()), 1.0)
    negatives = max(float(train_y.shape[0] - train_y.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for start in range(0, train_x.shape[0], args.batch_size):
            xb = torch.as_tensor(train_x[start:start + args.batch_size], dtype=torch.float32)
            yb = torch.as_tensor(train_y[start:start + args.batch_size], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            logits = model(torch.as_tensor(val_x, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
            pred = probs >= 0.5
            acc = float((pred == (val_y >= 0.5)).mean())
            val_loss = float(criterion(logits, torch.as_tensor(val_y, dtype=torch.float32)).detach().cpu())
        print(f"epoch={epoch:03d} train_loss={np.mean(epoch_losses):.4f} val_loss={val_loss:.4f} val_acc={acc:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "hidden_size": args.hidden_size,
            "window_size": int(x.shape[1]),
        },
        output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
