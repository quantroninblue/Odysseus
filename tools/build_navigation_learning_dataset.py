#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from runtime.core.navigation_learning import save_sequence_dataset_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GRU navigation-risk dataset from navigator command CSV logs.")
    parser.add_argument("csv", nargs="+", help="navigator_commands_*.csv files")
    parser.add_argument("--output", "-o", required=True, help="output .npz path")
    parser.add_argument("--window-size", type=int, default=16, help="sequence length")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_sequence_dataset_npz(args.csv, output, window_size=args.window_size)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
