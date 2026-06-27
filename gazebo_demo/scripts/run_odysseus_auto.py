#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Odysseus navigation with automatic samples, memory, checkpoint loading, and post-run training."
    )
    parser.add_argument("--samples-dir", default="artifacts/odysseus_samples")
    parser.add_argument("--memory-path", default="artifacts/odysseus_world_memory.json")
    parser.add_argument("--checkpoint", default="artifacts/odysseus_attributor.pt")
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-train", action="store_true", help="run Odysseus but skip automatic post-run training")
    args = parser.parse_args()

    samples_dir = (REPO_ROOT / args.samples_dir).resolve()
    memory_path = (REPO_ROOT / args.memory_path).resolve()
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    samples_dir.mkdir(parents=True, exist_ok=True)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(env.get("PYTHONPATH", ""), str(REPO_ROOT))
    env["ODYSSEUS_DATASET_DIR"] = str(samples_dir)
    env["ODYSSEUS_MEMORY_PATH"] = str(memory_path)
    if checkpoint.exists():
        env["ODYSSEUS_ATTRIBUTOR_CHECKPOINT"] = str(checkpoint)
        print(f"[odysseus-auto] loading checkpoint: {checkpoint}", flush=True)
    else:
        env.pop("ODYSSEUS_ATTRIBUTOR_CHECKPOINT", None)
        print("[odysseus-auto] no checkpoint yet; running with online memory only", flush=True)
    print(f"[odysseus-auto] samples: {samples_dir}", flush=True)
    print(f"[odysseus-auto] memory:  {memory_path}", flush=True)

    navigator_python = _navigator_python(env)
    print(f"[odysseus-auto] navigator python: {navigator_python}", flush=True)
    navigator = subprocess.Popen(
        [str(navigator_python), str(REPO_ROOT / "gazebo_demo" / "scripts" / "factory_bot_stack_navigator.py")],
        cwd=REPO_ROOT,
        env=env,
    )
    _wait_for_navigator(navigator)
    if args.no_train:
        print("[odysseus-auto] training skipped by --no-train", flush=True)
        return
    _train_if_ready(args, samples_dir, checkpoint)


def _navigator_python(env: dict[str, str]) -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    if not candidate.exists():
        return Path(sys.executable)
    probe = "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('rclpy') and importlib.util.find_spec('torch') else 1)"
    result = subprocess.run([str(candidate), "-c", probe], cwd=REPO_ROOT, env=env)
    return candidate if result.returncode == 0 else Path(sys.executable)


def _prepend_path(existing: str, path: str) -> str:
    parts = [item for item in existing.split(os.pathsep) if item]
    return os.pathsep.join([path, *[item for item in parts if item != path]])


def _wait_for_navigator(process: subprocess.Popen) -> None:
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[odysseus-auto] stopping navigator; training will run after shutdown", flush=True)
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _train_if_ready(args: argparse.Namespace, samples_dir: Path, checkpoint: Path) -> None:
    samples = sorted(samples_dir.glob("odysseus_*.npz"))
    if len(samples) < args.min_samples:
        print(
            f"[odysseus-auto] only {len(samples)} sample(s); need {args.min_samples} before training",
            flush=True,
        )
        return
    trainer_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not trainer_python.exists():
        print("[odysseus-auto] .venv/bin/python not found; skipping automatic training", flush=True)
        return
    command = [
        str(trainer_python),
        str(REPO_ROOT / "tools" / "train_odysseus_attributor.py"),
        str(samples_dir),
        "--output",
        str(checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    print(f"[odysseus-auto] training on {len(samples)} sample(s)", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, env=_training_env())
    if completed.returncode == 0:
        print(f"[odysseus-auto] updated checkpoint: {checkpoint}", flush=True)
    else:
        print(f"[odysseus-auto] training failed with exit code {completed.returncode}", flush=True)


def _training_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(env.get("PYTHONPATH", ""), str(REPO_ROOT))
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


if __name__ == "__main__":
    main()
