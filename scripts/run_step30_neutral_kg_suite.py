"""Run Step 30 NeutralKG classification, ablation, and stronger baselines."""

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


def maybe_run(cmd: list[str], metric: Path, skip_existing: bool) -> None:
    if skip_existing and metric.exists():
        print(f"Skipping existing: {metric}", flush=True)
        return
    run(cmd)


def run_kcgt_ablation(args, seed: int, name: str, extra: list[str]) -> None:
    out = ROOT / args.ablation_root / f"seed_{seed}" / name
    metric = out / "metrics.json"
    cmd = [
        args.python,
        str(ROOT / "scripts" / "train_kcgt_contrastive.py"),
        "--processed-dir",
        str(ROOT / args.processed_dir),
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(seed),
    ] + extra
    maybe_run(cmd, metric, args.skip_existing)


def run_baseline(args, seed: int, model: str) -> None:
    out = ROOT / args.baseline_root / f"seed_{seed}" / "baselines"
    metric = out / model / "metrics.json"
    cmd = [
        args.python,
        str(ROOT / "src" / "experiments" / "train_baseline_classification.py"),
        "--model",
        model,
        "--processed-dir",
        str(ROOT / args.processed_dir),
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(seed),
    ]
    maybe_run(cmd, metric, args.skip_existing)


def run_pretrained(args, encoder: str, model_name: str, name: str) -> None:
    out = ROOT / args.baseline_root / "seed_1" / "pretrained"
    metric = out / name / "metrics.json"
    cmd = [
        args.python,
        str(ROOT / "scripts" / "train_pretrained_text_baseline.py"),
        "--processed-dir",
        str(ROOT / args.processed_dir),
        "--output-dir",
        str(out),
        "--encoder",
        encoder,
        "--model-name",
        model_name,
        "--name",
        name,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        "1",
    ]
    maybe_run(cmd, metric, args.skip_existing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--ablation-root", default="outputs/step30_neutral_kg_ablation")
    parser.add_argument("--baseline-root", default="outputs/step30_neutral_kg_baselines")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-pretrained", action="store_true")
    args = parser.parse_args()

    ablations = [
        ("full_group", ["--positive-mode", "group"]),
        ("no_contrastive", ["--positive-mode", "group", "--contrastive-weight", "0.0"]),
        ("positive_class", ["--positive-mode", "class"]),
        ("positive_both", ["--positive-mode", "both"]),
        ("no_graph", ["--positive-mode", "group", "--disable-graph"]),
        ("no_qos", ["--positive-mode", "group", "--disable-qos"]),
        ("concat_fusion", ["--positive-mode", "group", "--fusion-type", "concat"]),
        ("graph_gcn", ["--positive-mode", "group", "--graph-model", "gcn"]),
        ("graph_gat", ["--positive-mode", "group", "--graph-model", "gat"]),
    ]
    for seed in args.seeds:
        for name, extra in ablations:
            run_kcgt_ablation(args, seed, name, extra)

    stronger_baselines = [
        "qos_mlp",
        "lstm",
        "gcn",
        "gat",
        "rgcn",
        "graph_transformer",
        "hypergraph",
        "text_gcn",
        "text_gat",
        "text_graph_transformer",
    ]
    for seed in args.seeds:
        for model in stronger_baselines:
            run_baseline(args, seed, model)

    if args.run_pretrained:
        run_pretrained(args, "hf", "google/bert_uncased_L-2_H-128_A-2", "bert_mini")
        run_pretrained(args, "sbert", "sentence-transformers/all-MiniLM-L6-v2", "sbert_minilm")


if __name__ == "__main__":
    main()
