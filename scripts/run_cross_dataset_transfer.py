"""Run cross-dataset transfer experiments across Hard-v2, Challenge, and NeutralKG."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "hard_v2": "data/processed/hard_v2",
    "challenge": "data/processed/challenge",
    "neutral_kg": "data/processed/challenge_neutral_kg",
}

MODELS = ["kcgt_contrastive", "kcgt", "textcnn", "transformer", "text_gat", "text_rgcn"]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", default="outputs/cross_dataset_transfer")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    for seed in args.seeds:
        for source_name, source_dir in DATASETS.items():
            for target_name, target_dir in DATASETS.items():
                if source_name == target_name:
                    continue
                for model in args.models:
                    out = ROOT / args.output_root / f"seed_{seed}" / f"{source_name}_to_{target_name}" / model
                    metric = out / "metrics.json"
                    if args.skip_existing and metric.exists():
                        print(f"Skipping existing: {metric}", flush=True)
                        continue
                    cmd = [
                        args.python,
                        str(ROOT / "scripts" / "train_transfer_neural.py"),
                        "--source-dir",
                        source_dir,
                        "--target-dir",
                        target_dir,
                        "--source-name",
                        source_name,
                        "--target-name",
                        target_name,
                        "--model",
                        model,
                        "--output-dir",
                        str(out),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--seed",
                        str(seed),
                    ]
                    if model == "kcgt_contrastive":
                        cmd.extend(["--positive-mode", "group"])
                    run(cmd)


if __name__ == "__main__":
    main()
