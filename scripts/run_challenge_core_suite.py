"""Run core classification experiments on KNI-Intent-Challenge."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NEURAL_BASELINES = ["textcnn", "transformer", "text_rgcn", "text_hypergraph"]


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--processed-dir", default="data/processed/challenge")
    parser.add_argument("--output-root", default="outputs/challenge_core")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-pretrained", action="store_true")
    args = parser.parse_args()

    processed = ROOT / args.processed_dir
    output_root = ROOT / args.output_root

    for seed in args.seeds:
        seed_root = output_root / f"seed_{seed}"

        for model in NEURAL_BASELINES:
            out = seed_root / "baselines"
            metric = out / model / "metrics.json"
            if args.skip_existing and metric.exists():
                print(f"Skipping existing: {metric}", flush=True)
            else:
                run(
                    [
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
                )

        for model in ["tfidf_word_char_qos"]:
            out = seed_root / "sklearn"
            metric = out / model / "metrics.json"
            if args.skip_existing and metric.exists():
                print(f"Skipping existing: {metric}", flush=True)
            else:
                run(
                    [
                        args.python,
                        str(ROOT / "scripts" / "train_sklearn_text_baseline.py"),
                        "--processed-dir",
                        str(processed),
                        "--model",
                        model,
                        "--output-dir",
                        str(out),
                        "--seed",
                        str(seed),
                    ]
                )

        supervised_out = seed_root / "kcgt_supervised"
        supervised_metric = supervised_out / "kcgt" / "metrics.json"
        if args.skip_existing and supervised_metric.exists():
            print(f"Skipping existing: {supervised_metric}", flush=True)
        else:
            run(
                [
                    args.python,
                    str(ROOT / "src" / "experiments" / "train_baseline_classification.py"),
                    "--model",
                    "kcgt",
                    "--processed-dir",
                    str(processed),
                    "--output-dir",
                    str(supervised_out),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(seed),
                ]
            )

        contrastive_out = seed_root / "kcgt_contrastive"
        contrastive_metric = contrastive_out / "metrics.json"
        if args.skip_existing and contrastive_metric.exists():
            print(f"Skipping existing: {contrastive_metric}", flush=True)
        else:
            run(
                [
                    args.python,
                    str(ROOT / "scripts" / "train_kcgt_contrastive.py"),
                    "--processed-dir",
                    str(processed),
                    "--output-dir",
                    str(contrastive_out),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(seed),
                    "--positive-mode",
                    "group",
                ]
            )

    if args.run_pretrained:
        seed_root = output_root / "seed_1"
        pretrained = [
            ("hf", "google/bert_uncased_L-2_H-128_A-2", "bert_mini"),
            ("sbert", "sentence-transformers/all-MiniLM-L6-v2", "sbert_minilm"),
        ]
        for encoder, model_name, name in pretrained:
            out = seed_root / "pretrained"
            metric = out / name / "metrics.json"
            if args.skip_existing and metric.exists():
                print(f"Skipping existing: {metric}", flush=True)
                continue
            run(
                [
                    args.python,
                    str(ROOT / "scripts" / "train_pretrained_text_baseline.py"),
                    "--processed-dir",
                    str(processed),
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
            )


if __name__ == "__main__":
    main()
