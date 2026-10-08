"""Run KCGT retrieval experiments on KNI-Intent-Challenge."""

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


def make_gallery(processed: Path) -> Path:
    gallery = processed / "gallery_standard.jsonl"
    if gallery.exists():
        return gallery
    lines = []
    for split in ["train", "val", "test"]:
        lines.extend((processed / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
    gallery.write_text("\n".join(line for line in lines if line.strip()) + "\n", encoding="utf-8")
    return gallery


def eval_one(args, seed: int, name: str, query: Path, gallery: Path, relevance: str, tsne: bool = False) -> None:
    out_dir = ROOT / args.output_root / f"seed_{seed}" / name
    metric = out_dir / "retrieval_metrics.json"
    if args.skip_existing and metric.exists():
        print(f"Skipping existing: {metric}", flush=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--processed-dir", default="data/processed/challenge")
    parser.add_argument("--checkpoint-root", default="outputs/challenge_core")
    parser.add_argument("--output-root", default="outputs/challenge_retrieval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    processed = ROOT / args.processed_dir
    standard_gallery = make_gallery(processed)
    for seed in args.seeds:
        eval_one(args, seed, "test_group", processed / "test.jsonl", processed / "test.jsonl", "group", tsne=(seed == args.seeds[0]))
        eval_one(args, seed, "test_class", processed / "test.jsonl", processed / "test.jsonl", "class")
        eval_one(args, seed, "unseen_composition", processed / "unseen_composition_test.jsonl", standard_gallery, "class")
        pairwise_gallery = processed / "pairwise_hard_negative_gallery.jsonl"
        if pairwise_gallery.exists():
            eval_one(args, seed, "pairwise_hard_negative_gallery", processed / "test.jsonl", pairwise_gallery, "group")
        hard_gallery = processed / "hard_negative_gallery.jsonl"
        if hard_gallery.exists():
            eval_one(args, seed, "hard_negative_gallery", processed / "test.jsonl", hard_gallery, "group")


if __name__ == "__main__":
    main()
