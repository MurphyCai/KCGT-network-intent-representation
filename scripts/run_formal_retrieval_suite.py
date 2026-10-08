"""Run formal KCGT retrieval, few-shot, and unseen evaluations."""

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
    parser = argparse.ArgumentParser(description="Run formal retrieval suite.")
    parser.add_argument("--processed-dir", default="data/processed/hard_v2")
    parser.add_argument("--output-root", default="outputs/formal_v2_retrieval")
    parser.add_argument("--checkpoint-root", default="outputs/formal_v2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def eval_one(args, seed: int, name: str, query: Path, gallery: Path, relevance: str, tsne: bool = False) -> None:
    out_dir = ROOT / args.output_root / f"seed_{seed}" / name
    metric_path = out_dir / "retrieval_metrics.json"
    if args.skip_existing and metric_path.exists():
        print(f"Skipping existing: {metric_path}", flush=True)
        return
    checkpoint = ROOT / args.checkpoint_root / f"seed_{seed}" / "kcgt_contrastive" / "best.pt"
    cmd = [
        args.python,
        str(ROOT / "scripts" / "evaluate_kcgt_retrieval.py"),
        "--checkpoint",
        str(checkpoint),
        "--query",
        str(query),
        "--gallery",
        str(gallery),
        "--output-dir",
        str(out_dir),
        "--relevance",
        relevance,
    ]
    if tsne:
        cmd.append("--tsne")
    run(cmd)


def main() -> None:
    args = parse_args()
    processed = ROOT / args.processed_dir
    for seed in args.seeds:
        eval_one(args, seed, "test_group", processed / "test.jsonl", processed / "test.jsonl", "group", tsne=(seed == args.seeds[0]))
        for k in [1, 5, 10, 20]:
            eval_one(args, seed, f"fewshot_k{k}", processed / "test.jsonl", processed / "few_shot" / f"k{k}_train.jsonl", "class")
        eval_one(args, seed, "unseen_template", processed / "unseen" / "template_test.jsonl", processed / "unseen" / "template_train.jsonl", "class")
        eval_one(args, seed, "unseen_application", processed / "unseen" / "application_test.jsonl", processed / "unseen" / "application_train.jsonl", "class")
        eval_one(args, seed, "unseen_scenario", processed / "unseen" / "scenario_test.jsonl", processed / "unseen" / "scenario_train.jsonl", "class")
        eval_one(args, seed, "unseen_slice", processed / "unseen" / "slice_test.jsonl", processed / "unseen" / "slice_train.jsonl", "class")
        eval_one(args, seed, "unseen_qos_range", processed / "unseen" / "qos_range_test.jsonl", processed / "unseen" / "qos_range_train.jsonl", "class")


if __name__ == "__main__":
    main()
