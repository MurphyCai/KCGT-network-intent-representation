"""Run multi-seed training-size sensitivity experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction-root", default="data/processed/hard_v2_fractions")
    parser.add_argument("--output-root", default="outputs/formal_v2_sensitivity_multiseed")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fractions", nargs="+", default=["frac_010", "frac_025", "frac_050", "frac_100"])
    parser.add_argument("--models", nargs="+", default=["textcnn", "transformer", "text_rgcn", "kcgt_contrastive"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        for frac in args.fractions:
            processed = ROOT / args.fraction_root / frac
            for model in args.models:
                out = ROOT / args.output_root / f"seed_{seed}" / frac / model
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
                        str(seed),
                        "--positive-mode",
                        "group",
                    ]
                else:
                    cmd = [
                        args.python,
                        str(ROOT / "src" / "experiments" / "train_baseline_classification.py"),
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
                        str(seed),
                    ]
                run(cmd)


if __name__ == "__main__":
    main()
