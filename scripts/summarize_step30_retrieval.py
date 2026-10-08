"""Summarize Step 30 retrieval ablations across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["recall@1", "recall@5", "recall@10", "precision@1", "precision@5", "precision@10", "mrr", "ndcg@5"]


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--output-prefix", default="step30_neutral_kg")
    args = parser.parse_args()

    root = Path(args.root)
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.glob("seed_*/*/*/retrieval_metrics.json")):
        rel = path.relative_to(root)
        ablation = rel.parts[1]
        scenario_dir = rel.parts[2]
        payload = json.loads(path.read_text(encoding="utf-8"))
        embedding = payload.get("embedding_type", "projection")
        key = (ablation, scenario_dir, embedding)
        for metric in METRICS:
            grouped[key][metric].append(float(payload["metrics"][metric]))

    rows = []
    for (ablation, scenario, embedding), values in grouped.items():
        row = {
            "ablation": ablation,
            "scenario": scenario,
            "embedding": embedding,
            "runs": len(next(iter(values.values()))) if values else 0,
        }
        for metric in METRICS:
            row[metric] = fmt(values.get(metric, [])) if values.get(metric) else ""
        rows.append(row)
    rows.sort(key=lambda row: (row["ablation"], row["scenario"], row["embedding"]))

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["ablation", "scenario", "embedding", "runs", *METRICS]
    csv_path = out_dir / f"{args.output_prefix}_retrieval_ablation_mean_std.csv"
    md_path = out_dir / f"{args.output_prefix}_retrieval_ablation_mean_std.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Step 30 Retrieval Ablation Results", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
