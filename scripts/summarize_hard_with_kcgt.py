"""Summarize hard-set baselines together with KCGT v1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "qos_mlp": ROOT / "outputs" / "baselines_hard" / "qos_mlp" / "metrics.json",
    "textcnn": ROOT / "outputs" / "baselines_hard" / "textcnn" / "metrics.json",
    "lstm": ROOT / "outputs" / "baselines_hard" / "lstm" / "metrics.json",
    "transformer": ROOT / "outputs" / "baselines_hard" / "transformer" / "metrics.json",
    "gcn": ROOT / "outputs" / "baselines_graph_hard" / "gcn" / "metrics.json",
    "gat": ROOT / "outputs" / "baselines_graph_hard" / "gat" / "metrics.json",
    "rgcn": ROOT / "outputs" / "baselines_graph_hard" / "rgcn" / "metrics.json",
    "text_gcn": ROOT / "outputs" / "baselines_graph_hard" / "text_gcn" / "metrics.json",
    "text_gat": ROOT / "outputs" / "baselines_graph_hard" / "text_gat" / "metrics.json",
    "text_rgcn": ROOT / "outputs" / "baselines_graph_hard" / "text_rgcn" / "metrics.json",
    "kcgt_v1": ROOT / "outputs" / "kcgt_v1_hard" / "kcgt" / "metrics.json",
}


def md_table(rows):
    lines = [
        "| Model | Accuracy | Precision | Recall | Macro-F1 | Loss | ms/sample | Params |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision_macro']:.4f} | "
            f"{row['recall_macro']:.4f} | {row['f1_macro']:.4f} | {row['loss']:.4f} | "
            f"{row['inference_time_ms_per_sample']:.4f} | {row['num_parameters']} |"
        )
    return "\n".join(lines)


def main():
    rows = []
    for model, path in SOURCES.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload["test"]
        rows.append(
            {
                "model": model,
                "accuracy": float(test["accuracy"]),
                "precision_macro": float(test["precision_macro"]),
                "recall_macro": float(test["recall_macro"]),
                "f1_macro": float(test["f1_macro"]),
                "loss": float(test["loss"]),
                "inference_time_ms_per_sample": float(test["inference_time_ms_per_sample"]),
                "num_parameters": int(payload["num_parameters"]),
            }
        )
    rows.sort(key=lambda item: item["f1_macro"], reverse=True)
    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hard_all_baselines_with_kcgt.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "hard_all_baselines_with_kcgt.md"
    md_path.write_text("# Hard Dataset Baselines with KCGT v1\n\n" + md_table(rows) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
