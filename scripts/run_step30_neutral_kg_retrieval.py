"""Run Step 30 NeutralKG retrieval ablations for KCGT checkpoints."""

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


def eval_one(
    args,
    seed: int,
    ablation: str,
    scenario: str,
    query: Path,
    gallery: Path,
    relevance: str,
    use_embedding: bool = False,
    tsne: bool = False,
) -> None:
    suffix = "_fused" if use_embedding else "_projection"
    out = ROOT / args.output_root / f"seed_{seed}" / ablation / f"{scenario}{suffix}"
    metric = out / "retrieval_metrics.json"
    if args.skip_existing and metric.exists():
        print(f"Skipping existing: {metric}", flush=True)
        return
    checkpoint = ROOT / args.checkpoint_root / f"seed_{seed}" / ablation / "best.pt"
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
        str(out),
        "--relevance",
        relevance,
    ]
    if use_embedding:
        cmd.append("--use-embedding")
    if tsne:
        cmd.append("--tsne")
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--checkpoint-root", default="outputs/step30_neutral_kg_ablation")
    parser.add_argument("--output-root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    processed = ROOT / args.processed_dir
    ablations = [
        "full_group",
        "no_contrastive",
        "positive_class",
        "positive_both",
        "no_graph",
        "no_qos",
        "concat_fusion",
        "graph_gcn",
        "graph_gat",
    ]
    scenarios = [
        ("test_group", processed / "test.jsonl", processed / "test.jsonl", "group"),
        ("pairwise_hard_negative", processed / "test.jsonl", processed / "pairwise_hard_negative_gallery.jsonl", "group"),
        ("test_class", processed / "test.jsonl", processed / "test.jsonl", "class"),
        ("unseen_composition", processed / "unseen_composition_test.jsonl", processed / "gallery_standard.jsonl", "class"),
    ]
    for seed in args.seeds:
        for ablation in ablations:
            for scenario, query, gallery, relevance in scenarios:
                eval_one(
                    args,
                    seed,
                    ablation,
                    scenario,
                    query,
                    gallery,
                    relevance,
                    use_embedding=False,
                    tsne=(seed == args.seeds[0] and ablation == "full_group" and scenario == "test_group"),
                )
            if ablation == "full_group":
                for scenario, query, gallery, relevance in scenarios:
                    eval_one(args, seed, ablation, scenario, query, gallery, relevance, use_embedding=True)


if __name__ == "__main__":
    main()
