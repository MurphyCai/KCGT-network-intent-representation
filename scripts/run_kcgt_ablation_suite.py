from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "full_group",
        "description": "Full KCGT with group-level contrastive positives.",
        "extra": ["--positive-mode", "group", "--contrastive-weight", "0.5"],
    },
    {
        "name": "positive_class",
        "description": "Full KCGT with class-level contrastive positives.",
        "extra": ["--positive-mode", "class", "--contrastive-weight", "0.5"],
    },
    {
        "name": "positive_both",
        "description": "Full KCGT with group-or-class positives.",
        "extra": ["--positive-mode", "both", "--contrastive-weight", "0.5"],
    },
    {
        "name": "no_contrastive",
        "description": "Full KCGT trained with classification loss only.",
        "extra": ["--positive-mode", "group", "--contrastive-weight", "0.0"],
    },
    {
        "name": "no_graph",
        "description": "Remove graph encoder and use zero graph representation.",
        "extra": ["--positive-mode", "group", "--contrastive-weight", "0.5", "--disable-graph"],
    },
    {
        "name": "no_qos",
        "description": "Remove QoS encoder and use zero QoS representation.",
        "extra": ["--positive-mode", "group", "--contrastive-weight", "0.5", "--disable-qos"],
    },
    {
        "name": "no_gated_fusion",
        "description": "Replace gated fusion with non-gated concatenation fusion.",
        "extra": ["--positive-mode", "group", "--contrastive-weight", "0.5", "--fusion-type", "concat"],
    },
]


def run_command(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def train_experiment(exp: dict[str, object], args: argparse.Namespace) -> Path:
    output_dir = ROOT / args.output_root / str(exp["name"])
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        return output_dir
    cmd = [
        sys.executable,
        "scripts/train_kcgt_contrastive.py",
        "--processed-dir",
        args.processed_dir,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ] + list(exp["extra"])
    run_command(cmd, ROOT)
    return output_dir


def evaluate_experiment(exp_name: str, run_dir: Path, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for embedding_type, flag in [("projection", []), ("fused", ["--use-embedding"])]:
        out_dir = ROOT / args.retrieval_root / exp_name / embedding_type
        metrics_path = out_dir / "retrieval_metrics.json"
        if not metrics_path.exists() or args.force:
            cmd = [
                sys.executable,
                "scripts/evaluate_kcgt_retrieval.py",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--gallery",
                args.gallery,
                "--query",
                args.query,
                "--relevance",
                args.relevance,
                "--output-dir",
                str(out_dir),
                "--batch-size",
                str(args.eval_batch_size),
                "--device",
                args.device,
            ] + flag
            if args.tsne and embedding_type == "projection":
                cmd.append("--tsne")
            run_command(cmd, ROOT)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        results[embedding_type] = payload["metrics"]
    return results


def load_classification_metrics(run_dir: Path) -> dict[str, float]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    test = payload["test"]
    return {
        "accuracy": test["accuracy"],
        "f1_macro": test["f1_macro"],
        "loss": test["loss"],
        "inference_time_ms_per_sample": test["inference_time_ms_per_sample"],
        "num_parameters": payload["num_parameters"],
    }


def write_reports(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    report_dir = ROOT / "outputs" / "reports"
    figure_dir = ROOT / "outputs" / "figures" / "ablations"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "kcgt_ablation_results.csv"
    fields = [
        "experiment",
        "description",
        "accuracy",
        "f1_macro",
        "loss",
        "embedding_type",
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@5",
        "valid_queries",
        "inference_time_ms_per_sample",
        "num_parameters",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = report_dir / "kcgt_ablation_results.md"
    lines = [
        "# KCGT Systematic Ablation Results",
        "",
        "| Experiment | Emb. | Acc. | F1 | R@1 | R@5 | R@10 | MRR | NDCG@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['embedding_type']} | {float(row['accuracy']):.4f} | "
            f"{float(row['f1_macro']):.4f} | {float(row['recall@1']):.4f} | "
            f"{float(row['recall@5']):.4f} | {float(row['recall@10']):.4f} | "
            f"{float(row['mrr']):.4f} | {float(row['ndcg@5']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Experimental Setting",
            "",
            f"- Epochs per ablation run: `{args.epochs}`.",
            f"- Retrieval relevance: `{args.relevance}`.",
            "- Each trained checkpoint is evaluated with both the contrastive projection head and the fused backbone embedding.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    projection_rows = [row for row in rows if row["embedding_type"] == "projection"]
    labels = [str(row["experiment"]) for row in projection_rows]
    x = list(range(len(labels)))
    plt.figure(figsize=(max(8, len(labels) * 1.25), 4.6))
    plt.bar([i - 0.2 for i in x], [float(row["recall@1"]) for row in projection_rows], 0.4, label="Recall@1")
    plt.bar([i + 0.2 for i in x], [float(row["mrr"]) for row in projection_rows], 0.4, label="MRR")
    plt.ylim(0.0, 1.05)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("KCGT Ablation: Projection Retrieval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "kcgt_ablation_projection_retrieval.png", dpi=220)
    plt.close()

    full_rows = [row for row in rows if row["experiment"] == "full_group"]
    plt.figure(figsize=(5.6, 4.2))
    plt.bar([str(row["embedding_type"]) for row in full_rows], [float(row["recall@1"]) for row in full_rows], color=["#4C78A8", "#F58518"])
    plt.ylim(0.0, 1.05)
    plt.ylabel("Recall@1")
    plt.title("Fused vs Projection Embedding")
    plt.tight_layout()
    plt.savefig(figure_dir / "kcgt_fused_vs_projection.png", dpi=220)
    plt.close()

    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run systematic KCGT ablation experiments.")
    parser.add_argument("--processed-dir", default="data/processed/hard")
    parser.add_argument("--output-root", default="outputs/ablations/kcgt")
    parser.add_argument("--retrieval-root", default="outputs/retrieval/ablations")
    parser.add_argument("--query", default="data/processed/hard/test.jsonl")
    parser.add_argument("--gallery", default="data/processed/hard/test.jsonl")
    parser.add_argument("--relevance", choices=["group", "class", "group_or_class"], default="group")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for exp in EXPERIMENTS:
        run_dir = train_experiment(exp, args)
        cls = load_classification_metrics(run_dir)
        retrieval = evaluate_experiment(str(exp["name"]), run_dir, args)
        for embedding_type, metrics in retrieval.items():
            rows.append(
                {
                    "experiment": exp["name"],
                    "description": exp["description"],
                    **cls,
                    "embedding_type": embedding_type,
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@10": metrics["recall@10"],
                    "mrr": metrics["mrr"],
                    "ndcg@5": metrics["ndcg@5"],
                    "valid_queries": metrics["valid_queries"],
                }
            )
    write_reports(rows, args)


if __name__ == "__main__":
    main()
