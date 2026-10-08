"""Run KCGT retrieval under cross-protocol / schema-shift transfer.

The checkpoints are trained by `train_transfer_neural.py` on a source protocol
and evaluated on a target protocol. This is intentionally named schema-shift
transfer because Challenge-NeutralKG changes the KG schema and should not be
interpreted as an ordinary in-distribution generalization split.
"""

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


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def eval_one(
    args,
    checkpoint: Path,
    seed: int,
    source: str,
    target: str,
    scenario: str,
    query: Path,
    gallery: Path,
    relevance: str,
    use_embedding: bool = False,
) -> None:
    embedding = "fused" if use_embedding else "projection"
    out = ROOT / args.output_root / f"seed_{seed}" / f"{source}_to_{target}" / "kcgt_contrastive" / f"{scenario}_{embedding}"
    metric = out / "retrieval_metrics.json"
    if args.skip_existing and metric.exists():
        print(f"Skipping existing: {metric}", flush=True)
        return
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
    run(cmd)


def scenarios_for(target_dir: Path) -> list[tuple[str, Path, Path, str]]:
    scenarios = [
        ("target_test_group", target_dir / "test.jsonl", target_dir / "test.jsonl", "group"),
        ("target_test_class", target_dir / "test.jsonl", target_dir / "test.jsonl", "class"),
    ]
    gallery_standard = target_dir / "gallery_standard.jsonl"
    unseen = target_dir / "unseen_composition_test.jsonl"
    if gallery_standard.exists() and unseen.exists():
        scenarios.append(("target_unseen_composition", unseen, gallery_standard, "class"))
    pairwise = target_dir / "pairwise_hard_negative_gallery.jsonl"
    if pairwise.exists():
        scenarios.append(("target_pairwise_hard_negative", target_dir / "test.jsonl", pairwise, "group"))
    hard = target_dir / "hard_negative_gallery.jsonl"
    if hard.exists() and hard != pairwise:
        scenarios.append(("target_hard_negative", target_dir / "test.jsonl", hard, "group"))
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint-root", default="outputs/cross_dataset_transfer")
    parser.add_argument("--output-root", default="outputs/cross_protocol_transfer_retrieval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-fused", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        for source in DATASETS:
            for target, target_rel in DATASETS.items():
                if source == target:
                    continue
                checkpoint = ROOT / args.checkpoint_root / f"seed_{seed}" / f"{source}_to_{target}" / "kcgt_contrastive" / "best.pt"
                if not checkpoint.exists():
                    raise SystemExit(f"Missing checkpoint: {checkpoint}")
                target_dir = ROOT / target_rel
                for scenario, query, gallery, relevance in scenarios_for(target_dir):
                    eval_one(args, checkpoint, seed, source, target, scenario, query, gallery, relevance, use_embedding=False)
                    if args.include_fused and scenario in {"target_test_group", "target_pairwise_hard_negative"}:
                        eval_one(args, checkpoint, seed, source, target, scenario, query, gallery, relevance, use_embedding=True)


if __name__ == "__main__":
    main()
