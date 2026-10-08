"""Run training-size sensitivity experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run training-fraction sensitivity suite.")
    parser.add_argument("--fraction-root", default="data/processed/hard_v2_fractions")
    parser.add_argument("--output-root", default="outputs/formal_v2_sensitivity")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fractions", nargs="+", default=["frac_010", "frac_025", "frac_050", "frac_075", "frac_100"])
    parser.add_argument("--models", nargs="+", default=["textcnn", "transformer", "kcgt_contrastive"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for frac in args.fractions:
        processed = ROOT / args.fraction_root / frac
        for model in args.models:
            out = ROOT / args.output_root / frac / model
            metric = out / "metrics.json" if model == "kcgt_contrastive" else out / model / "metrics.json"
            if args.skip_existing and metric.exists():
                print(f"Skipping existing: {metric}", flush=True)
                continue
            if model == "kcgt_contrastive":
                cmd = [
                    args.python,
                    str(ROOT / "scripts" / "train_kcgt_contrastive.py"),
                    "--processed-dir",
                    str(processed),
                    "--output-dir",
                    str(out),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--positive-mode",
                    "group",
                ]
            else:
                cmd = [
                    args.python,
                    str(ROOT / "scripts" / "train_baseline_classification.py"),
                    "--model",
                    model,
                    "--processed-dir",
                    str(processed),
                    "--output-dir",
                    str(out),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                ]
            run(cmd)


if __name__ == "__main__":
    main()
