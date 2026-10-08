"""Summarize formal multi-seed classification results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "loss", "inference_time_ms_per_sample"]


def find_metrics(root: Path) -> List[Path]:
    return sorted(root.glob("seed_*/**/metrics.json"))


def model_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3:
        family = parts[1]
        model = parts[-2]
        if family in {"kcgt_contrastive", "kcgt_supervised"}:
            return family
        return model
    return path.parent.name


def fmt(values: List[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize formal experiment metrics.")
    parser.add_argument("--root", default="outputs/formal_v2")
    parser.add_argument("--output-prefix", default="formal_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    param_counts: Dict[str, List[int]] = defaultdict(list)
    for path in find_metrics(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = model_name(path, root)
        test = payload.get("test", {})
        for metric in METRICS:
            if metric in test:
                grouped[name][metric].append(float(test[metric]))
        if "num_parameters" in payload:
            param_counts[name].append(int(payload["num_parameters"]))

    rows = []
    for name, values in grouped.items():
        row = {"model": name}
        for metric in METRICS:
            row[metric] = fmt(values.get(metric, [])) if values.get(metric) else ""
        row["num_parameters"] = int(mean(param_counts[name])) if param_counts[name] else ""
        row["runs"] = len(values.get("f1_macro", []))
        rows.append(row)
    rows.sort(key=lambda r: float(str(r["f1_macro"]).split(" ± ")[0] or 0), reverse=True)

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.output_prefix}_classification_mean_std.csv"
    md_path = out_dir / f"{args.output_prefix}_classification_mean_std.md"
    fields = ["model", "runs", *METRICS, "num_parameters"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Formal Classification Results (mean ± std)", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "models": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
