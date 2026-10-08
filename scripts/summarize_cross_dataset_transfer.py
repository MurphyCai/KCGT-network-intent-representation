"""Summarize cross-dataset transfer classification results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "loss", "inference_time_ms_per_sample"]


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/cross_dataset_transfer")
    parser.add_argument("--output-prefix", default="cross_dataset_transfer")
    args = parser.parse_args()

    root = Path(args.root)
    grouped = defaultdict(lambda: defaultdict(list))
    source_seen = defaultdict(lambda: defaultdict(list))
    params = defaultdict(list)
    for path in sorted(root.glob("seed_*/*/*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload["source"]
        target = payload["target"]
        model = payload["model"]
        key = (source, target, model)
        for metric in METRICS:
            grouped[key][metric].append(float(payload["target_test"][metric]))
            source_seen[key][metric].append(float(payload["source_test"][metric]))
        params[key].append(int(payload.get("num_parameters", 0)))

    rows = []
    for (source, target, model), values in grouped.items():
        row = {
            "source": source,
            "target": target,
            "model": model,
            "runs": len(values.get("f1_macro", [])),
            "source_f1": fmt(source_seen[(source, target, model)]["f1_macro"]),
        }
        for metric in METRICS:
            row[f"target_{metric}"] = fmt(values.get(metric, [])) if values.get(metric) else ""
        row["num_parameters"] = int(mean(params[(source, target, model)])) if params[(source, target, model)] else ""
        rows.append(row)
    model_order = {"kcgt_contrastive": 0, "kcgt": 1, "text_gat": 2, "text_rgcn": 3, "transformer": 4, "textcnn": 5}
    rows.sort(key=lambda r: (r["source"], r["target"], model_order.get(r["model"], 99)))

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["source", "target", "model", "runs", "source_f1", *[f"target_{m}" for m in METRICS], "num_parameters"]
    csv_path = out_dir / f"{args.output_prefix}_classification_mean_std.csv"
    md_path = out_dir / f"{args.output_prefix}_classification_mean_std.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Cross-Dataset Transfer Classification Results", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
