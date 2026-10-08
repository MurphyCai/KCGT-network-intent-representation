"""Run method-specific ablations for the upgraded KCGT objective and graph encoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:  # pragma: no cover - the report remains usable without SciPy.
    ttest_rel = wilcoxon = None


ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "full_structured_relgt": [
        "--graph-model", "relation_transformer",
        "--contrastive-objective", "structured",
        "--relation-mode", "typed",
    ],
    "plain_supcon_relgt": [
        "--graph-model", "relation_transformer",
        "--contrastive-objective", "supcon",
        "--relation-mode", "typed",
    ],
    "structured_rgcn": [
        "--graph-model", "rgcn",
        "--contrastive-objective", "structured",
        "--relation-mode", "typed",
    ],
    "structured_relgt_shuffled": [
        "--graph-model", "relation_transformer",
        "--contrastive-objective", "structured",
        "--relation-mode", "shuffled",
    ],
}


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(f"Command failed ({completed.returncode}): {' '.join(command)}")


def train(args: argparse.Namespace, seed: int, variant: str, extra: list[str]) -> Path:
    out = ROOT / args.output_root / f"seed_{seed}" / variant
    metric = out / "metrics.json"
    if not (args.skip_existing and metric.exists()):
        run(
            [
                args.python,
                str(ROOT / "scripts" / "train_kcgt_contrastive.py"),
                "--processed-dir", str(ROOT / args.processed_dir),
                "--output-dir", str(out),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(seed),
                "--positive-mode", "group",
                "--contrastive-weight", str(args.contrastive_weight),
            ]
            + extra
        )
    return out / "best.pt"


def evaluate(
    args: argparse.Namespace,
    seed: int,
    variant: str,
    checkpoint: Path,
    scenario: str,
    query: Path,
    gallery: Path,
    relevance: str,
) -> None:
    out = ROOT / args.output_root / f"seed_{seed}" / variant / scenario
    metric = out / "retrieval_metrics.json"
    if args.skip_existing and metric.exists():
        return
    run(
        [
            args.python,
            str(ROOT / "scripts" / "evaluate_kcgt_retrieval.py"),
            "--checkpoint", str(checkpoint),
            "--query", str(query),
            "--gallery", str(gallery),
            "--output-dir", str(out),
            "--relevance", relevance,
            "--batch-size", str(args.batch_size),
        ]
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def paired_statistics(reference: list[float], comparator: list[float], bootstrap_samples: int = 20_000) -> dict[str, float | None]:
    differences = [left - right for left, right in zip(reference, comparator)]
    mean_difference = statistics.mean(differences)
    std_difference = statistics.stdev(differences) if len(differences) > 1 else 0.0
    effect_size = mean_difference / std_difference if std_difference > 0 else math.inf

    rng = random.Random(20260831)
    bootstrap_means = []
    for _ in range(bootstrap_samples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        bootstrap_means.append(statistics.mean(sample))
    bootstrap_means.sort()
    lower = bootstrap_means[int(0.025 * bootstrap_samples)]
    upper = bootstrap_means[int(0.975 * bootstrap_samples) - 1]

    t_p = None
    wilcoxon_p = None
    if ttest_rel is not None:
        t_p = float(ttest_rel(reference, comparator).pvalue)
        try:
            wilcoxon_p = float(wilcoxon(reference, comparator, alternative="two-sided").pvalue)
        except ValueError:
            wilcoxon_p = 1.0 if all(value == 0 for value in differences) else None
    return {
        "mean_difference": mean_difference,
        "bootstrap_95ci_low": lower,
        "bootstrap_95ci_high": upper,
        "paired_t_p": t_p,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": effect_size,
    }


def summarize(args: argparse.Namespace) -> None:
    rows = []
    all_values: dict[str, dict[str, list[float]]] = {}
    for variant in VARIANTS:
        collected: dict[str, list[float]] = {
            "macro_f1": [],
            "group_recall@1": [],
            "group_mrr": [],
            "class_recall@1": [],
            "class_mrr": [],
            "unseen_recall@1": [],
            "unseen_mrr": [],
            "hard_recall@1": [],
            "hard_mrr": [],
            "hard_precision@5": [],
            "hard_ndcg@5": [],
        }
        for seed in args.seeds:
            root = ROOT / args.output_root / f"seed_{seed}" / variant
            classification = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            group = json.loads((root / "test_group" / "retrieval_metrics.json").read_text(encoding="utf-8"))["metrics"]
            class_metrics = json.loads((root / "test_class" / "retrieval_metrics.json").read_text(encoding="utf-8"))["metrics"]
            unseen = json.loads((root / "unseen_composition" / "retrieval_metrics.json").read_text(encoding="utf-8"))["metrics"]
            hard = json.loads((root / "pairwise_hard_negative" / "retrieval_metrics.json").read_text(encoding="utf-8"))["metrics"]
            collected["macro_f1"].append(float(classification["test"]["f1_macro"]))
            collected["group_recall@1"].append(float(group["recall@1"]))
            collected["group_mrr"].append(float(group["mrr"]))
            collected["class_recall@1"].append(float(class_metrics["recall@1"]))
            collected["class_mrr"].append(float(class_metrics["mrr"]))
            collected["unseen_recall@1"].append(float(unseen["recall@1"]))
            collected["unseen_mrr"].append(float(unseen["mrr"]))
            collected["hard_recall@1"].append(float(hard["recall@1"]))
            collected["hard_mrr"].append(float(hard["mrr"]))
            collected["hard_precision@5"].append(float(hard["precision@5"]))
            collected["hard_ndcg@5"].append(float(hard["ndcg@5"]))
        all_values[variant] = collected
        row = {"variant": variant, "seeds": len(args.seeds)}
        for metric, values in collected.items():
            mean, std = mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)

    out = ROOT / args.output_root
    csv_path = out / "method_novelty_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# KCGT Method Novelty Ablation",
        "",
        "| Variant | Macro-F1 | Class R@1 | Group R@1 | Group MRR | Unseen R@1 | Hard P@5 | Hard NDCG@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f} | "
            f"{row['class_recall@1_mean']:.4f} +/- {row['class_recall@1_std']:.4f} | "
            f"{row['group_recall@1_mean']:.4f} +/- {row['group_recall@1_std']:.4f} | "
            f"{row['group_mrr_mean']:.4f} +/- {row['group_mrr_std']:.4f} | "
            f"{row['unseen_recall@1_mean']:.4f} +/- {row['unseen_recall@1_std']:.4f} | "
            f"{row['hard_precision@5_mean']:.4f} +/- {row['hard_precision@5_std']:.4f} | "
            f"{row['hard_ndcg@5_mean']:.4f} +/- {row['hard_ndcg@5_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "- `plain_supcon_relgt` isolates the structured operational objective.",
            "- `structured_rgcn` isolates relation-aware Transformer attention from relation-aware message passing.",
            "- `structured_relgt_shuffled` tests whether typed relations, rather than graph density alone, explain the gain.",
        ]
    )
    (out / "method_novelty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparisons = {
        "structured objective vs. plain SupCon": "plain_supcon_relgt",
        "relation-aware Graph Transformer vs. R-GCN": "structured_rgcn",
        "typed relations vs. shuffled relations": "structured_relgt_shuffled",
    }
    significance_rows = []
    for comparison, comparator in comparisons.items():
        for metric in ("macro_f1", "group_recall@1", "group_mrr", "hard_precision@5", "hard_ndcg@5"):
            stats = paired_statistics(
                all_values["full_structured_relgt"][metric],
                all_values[comparator][metric],
            )
            significance_rows.append({"comparison": comparison, "metric": metric, **stats})

    significance_csv = out / "method_novelty_significance.csv"
    with significance_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(significance_rows[0]))
        writer.writeheader()
        writer.writerows(significance_rows)
    significance_lines = [
        "# Paired Significance Analysis",
        "",
        "Differences are computed as full KCGT minus the comparator over identical seeds.",
        "The bootstrap interval is a paired 95% confidence interval with 20,000 resamples.",
        "",
        "| Comparison | Metric | Mean diff. | Bootstrap 95% CI | Paired t-test p | Wilcoxon p | Cohen dz |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in significance_rows:
        t_p = "n/a" if row["paired_t_p"] is None else f"{row['paired_t_p']:.4f}"
        w_p = "n/a" if row["wilcoxon_p"] is None else f"{row['wilcoxon_p']:.4f}"
        dz = row["cohen_dz"]
        dz_text = "inf" if math.isinf(float(dz)) else f"{dz:.2f}"
        significance_lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_difference']:+.4f} | "
            f"[{row['bootstrap_95ci_low']:+.4f}, {row['bootstrap_95ci_high']:+.4f}] | "
            f"{t_p} | {w_p} | {dz_text} |"
        )
    (out / "method_novelty_significance.md").write_text("\n".join(significance_lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {significance_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--output-root", default="outputs/method_novelty_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    processed = ROOT / args.processed_dir
    scenarios = [
        ("test_group", processed / "test.jsonl", processed / "test.jsonl", "group"),
        ("test_class", processed / "test.jsonl", processed / "test.jsonl", "class"),
        (
            "unseen_composition",
            processed / "unseen_composition_test.jsonl",
            processed / "gallery_standard.jsonl",
            "class",
        ),
        (
            "pairwise_hard_negative",
            processed / "test.jsonl",
            processed / "pairwise_hard_negative_gallery.jsonl",
            "group",
        ),
    ]
    for seed in args.seeds:
        for variant, extra in VARIANTS.items():
            checkpoint = train(args, seed, variant, extra)
            for scenario, query, gallery, relevance in scenarios:
                evaluate(args, seed, variant, checkpoint, scenario, query, gallery, relevance)
    summarize(args)


if __name__ == "__main__":
    main()
