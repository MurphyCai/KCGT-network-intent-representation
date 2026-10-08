from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import matplotlib.pyplot as plt


DEFAULT_RUNS = [
    ("KCGT-v1", "test-group", Path("outputs/retrieval/kcgt_v1_hard/test_group/retrieval_metrics.json")),
    (
        "KCGT-CL-both",
        "test-group",
        Path("outputs/retrieval/kcgt_contrastive_hard/test_group/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-both-fused",
        "test-group",
        Path("outputs/retrieval/kcgt_contrastive_hard/test_group_fused/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-group",
        "test-group",
        Path("outputs/retrieval/kcgt_contrastive_group_hard/test_group/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-both",
        "few-shot-k5",
        Path("outputs/retrieval/kcgt_contrastive_hard/fewshot_k5/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-group",
        "few-shot-k5",
        Path("outputs/retrieval/kcgt_contrastive_group_hard/fewshot_k5/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-both",
        "unseen-template",
        Path("outputs/retrieval/kcgt_contrastive_hard/unseen_template/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-group",
        "unseen-template",
        Path("outputs/retrieval/kcgt_contrastive_group_hard/unseen_template/retrieval_metrics.json"),
    ),
    (
        "KCGT-CL-group",
        "unseen-application",
        Path("outputs/retrieval/kcgt_contrastive_group_hard/unseen_application/retrieval_metrics.json"),
    ),
]


def load_rows(runs: list[tuple[str, str, Path]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, scenario, path in runs:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "relevance": payload.get("relevance", ""),
                "embedding_type": payload.get("embedding_type", ""),
                "recall@1": metrics.get("recall@1", 0.0),
                "recall@5": metrics.get("recall@5", 0.0),
                "recall@10": metrics.get("recall@10", 0.0),
                "precision@1": metrics.get("precision@1", 0.0),
                "precision@5": metrics.get("precision@5", 0.0),
                "precision@10": metrics.get("precision@10", 0.0),
                "mrr": metrics.get("mrr", 0.0),
                "ndcg@5": metrics.get("ndcg@5", 0.0),
                "valid_queries": metrics.get("valid_queries", 0),
                "source": str(path),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "scenario",
        "relevance",
        "embedding_type",
        "recall@1",
        "recall@5",
        "recall@10",
        "precision@1",
        "precision@5",
        "precision@10",
        "mrr",
        "ndcg@5",
        "valid_queries",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Model",
        "Scenario",
        "Rel.",
        "Emb.",
        "R@1",
        "R@5",
        "R@10",
        "MRR",
        "NDCG@5",
        "Queries",
    ]
    lines = [
        "# Retrieval and Representation Evaluation",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["scenario"]),
                    str(row["relevance"]),
                    str(row["embedding_type"]),
                    f"{float(row['recall@1']):.4f}",
                    f"{float(row['recall@5']):.4f}",
                    f"{float(row['recall@10']):.4f}",
                    f"{float(row['mrr']):.4f}",
                    f"{float(row['ndcg@5']):.4f}",
                    str(row["valid_queries"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Key Reading",
            "",
            "- `test-group` evaluates whether paraphrases of the same underlying network intent are nearest neighbors.",
            "- `few-shot-k5` evaluates class-level retrieval when only five labeled gallery examples are available per intent class.",
            "- `unseen-template` and `unseen-application` probe generalization to held-out expression templates and held-out application groups.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_group(rows: list[dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({str(row["scenario"]) for row in rows})
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        labels = [str(row["model"]) for row in scenario_rows]
        x = list(range(len(labels)))
        width = 0.35

        plt.figure(figsize=(max(7, len(labels) * 1.6), 4.2))
        plt.bar([i - width / 2 for i in x], [float(row["recall@1"]) for row in scenario_rows], width, label="Recall@1")
        plt.bar([i + width / 2 for i in x], [float(row["mrr"]) for row in scenario_rows], width, label="MRR")
        plt.ylim(0.0, 1.05)
        plt.ylabel("Score")
        plt.title(f"Retrieval Quality: {scenario}")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"retrieval_{scenario.replace('-', '_')}.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KCGT retrieval evaluation results.")
    parser.add_argument("--output-csv", default="outputs/reports/retrieval_results.csv")
    parser.add_argument("--output-md", default="outputs/reports/retrieval_results.md")
    parser.add_argument("--figure-dir", default="outputs/figures/retrieval")
    args = parser.parse_args()

    rows = load_rows(DEFAULT_RUNS)
    if not rows:
        raise SystemExit("No retrieval metric files found.")

    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    plot_group(rows, Path(args.figure_dir))

    print(json.dumps({"rows": len(rows), "csv": args.output_csv, "markdown": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
