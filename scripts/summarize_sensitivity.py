"""Summarize training-size sensitivity results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("outputs/formal_v2_sensitivity")
    rows = []
    for frac_dir in sorted(root.glob("frac_*")):
        fraction = int(frac_dir.name.split("_")[1]) / 100
        for metrics_path in frac_dir.glob("**/metrics.json"):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            model = payload["model"]
            if model == "kcgt_contrastive":
                name = "kcgt_contrastive"
            else:
                name = metrics_path.parent.name
            test = payload["test"]
            rows.append(
                {
                    "fraction": fraction,
                    "model": name,
                    "accuracy": test["accuracy"],
                    "f1_macro": test["f1_macro"],
                    "loss": test["loss"],
                    "inference_time_ms_per_sample": test["inference_time_ms_per_sample"],
                }
            )
    rows.sort(key=lambda r: (r["fraction"], r["model"]))
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "formal_v2_sensitivity.csv"
    md_path = out_dir / "formal_v2_sensitivity.md"
    fields = ["fraction", "model", "accuracy", "f1_macro", "loss", "inference_time_ms_per_sample"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Training-Size Sensitivity", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
