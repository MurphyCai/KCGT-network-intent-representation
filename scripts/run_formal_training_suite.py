"""Run the first formal experiment base on KNI-Intent-Hard-v2."""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run formal baseline/KCGT training suite.")
    parser.add_argument("--processed-dir", default="data/processed/hard_v2")
    parser.add_argument("--output-root", default="outputs/formal_v2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", nargs="+", default=["textcnn", "transformer", "rgcn", "text_rgcn", "hypergraph", "text_hypergraph", "kcgt"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def maybe_run(cmd: list[str], metric_path: Path, skip_existing: bool) -> None:
    if skip_existing and metric_path.exists():
        print(f"Skipping existing: {metric_path}", flush=True)
        return
    run(cmd)


def main() -> None:
    args = parse_args()
    processed = str(ROOT / args.processed_dir)
    out_root = ROOT / args.output_root

    for seed in args.seeds:
        for model in args.models:
            family = "kcgt_supervised" if model == "kcgt" else "baselines"
            out_dir = out_root / f"seed_{seed}" / family
            metric_path = out_dir / model / "metrics.json"
            cmd = [
                args.python,
                str(ROOT / "scripts" / "train_baseline_classification.py"),
                "--model",
                model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--processed-dir",
                processed,
                "--output-dir",
                str(out_dir),
                "--seed",
                str(seed),
            ]
            maybe_run(cmd, metric_path, args.skip_existing)

        con_out = out_root / f"seed_{seed}" / "kcgt_contrastive"
        metric_path = con_out / "metrics.json"
        cmd = [
            args.python,
            str(ROOT / "scripts" / "train_kcgt_contrastive.py"),
            "--processed-dir",
            processed,
            "--output-dir",
            str(con_out),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--positive-mode",
            "group",
        ]
        maybe_run(cmd, metric_path, args.skip_existing)


if __name__ == "__main__":
    main()
