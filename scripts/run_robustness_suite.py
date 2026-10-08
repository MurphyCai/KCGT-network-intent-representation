"""Run robustness retrieval evaluations for KCGT checkpoints."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MODES = ["text_dropout", "qos_noise", "missing_qos", "kg_edge_drop"]


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    parser.add_argument("--checkpoint-root", default="outputs/formal_v2")
    parser.add_argument("--query-dir", default="data/processed/hard_v2_robustness")
    parser.add_argument("--gallery", default="data/processed/hard_v2/test.jsonl")
    parser.add_argument("--output-root", default="outputs/formal_v2_robustness")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    for seed in args.seeds:
        checkpoint = checkpoint_root / f"seed_{seed}" / "kcgt_contrastive" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for mode in MODES:
            out_dir = output_root / f"seed_{seed}" / mode
            metrics = out_dir / "retrieval_metrics.json"
            if args.skip_existing and metrics.exists():
                print(f"skip existing {metrics}", flush=True)
                continue
            run(
                [
                    args.python,
                    "scripts/evaluate_kcgt_retrieval.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--query",
                    str(Path(args.query_dir) / f"{mode}.jsonl"),
                    "--gallery",
                    args.gallery,
                    "--output-dir",
                    str(out_dir),
                    "--relevance",
                    "group",
                ],
                cwd,
            )


if __name__ == "__main__":
    main()
