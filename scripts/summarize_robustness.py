"""Summarize KCGT robustness retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["recall@1", "recall@5", "recall@10", "mrr"]


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/formal_v2_robustness")
    parser.add_argument("--output-prefix", default="formal_v2")
    args = parser.parse_args()

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for metrics_path in Path(args.root).glob("seed_*/*/retrieval_metrics.json"):
        mode = metrics_path.parent.name
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        for metric in METRICS:
            buckets[mode][metric].append(float(metrics[metric]))

    rows = []
    for mode in sorted(buckets):
        row = {"mode": mode, "runs": len(next(iter(buckets[mode].values())))}
        for metric in METRICS:
            values = buckets[mode].get(metric, [])
            row[f"{metric}_mean"] = mean(values) if values else 0.0
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            row[metric] = fmt(values) if values else "n/a"
        rows.append(row)

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.output_prefix}_robustness_retrieval.csv"
    md_path = out_dir / f"{args.output_prefix}_robustness_retrieval.md"
    fields = ["mode", "runs"] + [f"{metric}_mean" for metric in METRICS] + [f"{metric}_std" for metric in METRICS]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    md_fields = ["mode", "runs"] + METRICS
    lines = ["# Robustness Retrieval Summary", "", "| " + " | ".join(md_fields) + " |", "| " + " | ".join("---" for _ in md_fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in md_fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "modes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
