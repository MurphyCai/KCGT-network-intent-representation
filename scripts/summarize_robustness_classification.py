"""Summarize classification robustness results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["accuracy", "f1_macro", "inference_time_ms_per_sample"]


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/formal_v2_robustness_classification")
    parser.add_argument("--output-prefix", default="formal_v2")
    args = parser.parse_args()

    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in Path(args.root).glob("seed_*/*/*/metrics.json"):
        model = path.parent.parent.name
        mode = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for metric in METRICS:
            buckets[(model, mode)][metric].append(float(payload["test"][metric]))

    rows = []
    for (model, mode), values in sorted(buckets.items()):
        row = {"model": model, "mode": mode, "runs": len(values["f1_macro"])}
        for metric in METRICS:
            row[f"{metric}_mean"] = mean(values[metric])
            row[f"{metric}_std"] = stdev(values[metric]) if len(values[metric]) > 1 else 0.0
            row[metric] = fmt(values[metric])
        rows.append(row)

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.output_prefix}_robustness_classification.csv"
    md_path = out_dir / f"{args.output_prefix}_robustness_classification.md"
    csv_fields = ["model", "mode", "runs"] + [f"{metric}_mean" for metric in METRICS] + [f"{metric}_std" for metric in METRICS]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    md_fields = ["model", "mode", "runs"] + METRICS
    lines = ["# Classification Robustness Summary", "", "| " + " | ".join(md_fields) + " |", "| " + " | ".join("---" for _ in md_fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in md_fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
