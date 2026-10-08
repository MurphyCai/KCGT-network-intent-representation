"""Evaluate classification robustness for trained neural baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ["text_dropout", "qos_noise", "missing_qos", "kg_edge_drop"]


def checkpoint_path(root: Path, seed: int, model: str) -> Path:
    if model == "kcgt_contrastive":
        return root / f"seed_{seed}" / "kcgt_contrastive" / "best.pt"
    return root / f"seed_{seed}" / "baselines" / model / "best.pt"


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint-root", default="outputs/formal_v2")
    parser.add_argument("--query-dir", default="data/processed/hard_v2_robustness")
    parser.add_argument("--output-root", default="outputs/formal_v2_robustness_classification")
    parser.add_argument("--models", nargs="+", default=["textcnn", "transformer", "text_rgcn", "text_hypergraph", "kcgt_contrastive"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    checkpoint_root = ROOT / args.checkpoint_root
    for seed in args.seeds:
        for model in args.models:
            checkpoint = checkpoint_path(checkpoint_root, seed, model)
            if not checkpoint.exists():
                print(f"Missing checkpoint, skip: {checkpoint}", flush=True)
                continue
            for mode in MODES:
                out_dir = ROOT / args.output_root / f"seed_{seed}" / model / mode
                metric = out_dir / "metrics.json"
                if args.skip_existing and metric.exists():
                    print(f"Skipping existing: {metric}", flush=True)
                    continue
                run(
                    [
                        args.python,
                        str(ROOT / "scripts" / "evaluate_checkpoint_classification.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--jsonl",
                        str(ROOT / args.query_dir / f"{mode}.jsonl"),
                        "--output-dir",
                        str(out_dir),
                    ]
                )


if __name__ == "__main__":
    main()
