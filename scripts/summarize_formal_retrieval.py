"""Summarize formal retrieval metrics across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["recall@1", "recall@5", "recall@10", "precision@1", "precision@5", "precision@10", "mrr", "ndcg@5"]


def fmt(vals):
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f} ± {stdev(vals):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize formal retrieval results.")
    parser.add_argument("--root", default="outputs/formal_v2_retrieval")
    parser.add_argument("--output-prefix", default="formal_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.glob("seed_*/**/retrieval_metrics.json")):
        scenario = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        for metric in METRICS:
            grouped[scenario][metric].append(float(metrics[metric]))

    rows = []
    for scenario, metrics in grouped.items():
        row = {"scenario": scenario, "runs": len(next(iter(metrics.values())))}
        for metric in METRICS:
            row[metric] = fmt(metrics[metric])
        rows.append(row)
    order = {
        "test_group": 0,
        "fewshot_k1": 1,
        "fewshot_k5": 2,
        "fewshot_k10": 3,
        "fewshot_k20": 4,
        "unseen_template": 5,
        "unseen_application": 6,
        "unseen_scenario": 7,
        "unseen_slice": 8,
        "unseen_qos_range": 9,
        "hard_negative_gallery": 10,
        "pairwise_hard_negative_gallery": 11,
    }
    rows.sort(key=lambda r: order.get(r["scenario"], 99))

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "runs", *METRICS]
    csv_path = out_dir / f"{args.output_prefix}_retrieval_mean_std.csv"
    md_path = out_dir / f"{args.output_prefix}_retrieval_mean_std.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Formal Retrieval Results (mean ± std)", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "scenarios": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
