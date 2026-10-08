"""Run baseline classification experiments and summarize results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt


DEFAULT_MODELS = ["qos_mlp", "textcnn", "lstm", "transformer"]


def run_command(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def read_metrics(output_dir: Path, model: str) -> Dict[str, object]:
    path = output_dir / model / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "loss",
        "inference_time_ms_per_sample",
        "best_val_f1_macro",
        "num_parameters",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: List[Dict[str, object]], epochs: int, figure_prefix: str) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["model"],
                f"{row['accuracy']:.4f}",
                f"{row['precision_macro']:.4f}",
                f"{row['recall_macro']:.4f}",
                f"{row['f1_macro']:.4f}",
                f"{row['loss']:.4f}",
                f"{row['inference_time_ms_per_sample']:.4f}",
                row["num_parameters"],
            ]
        )
    content = [
        "# Baseline Classification Results",
        "",
        f"Epochs: {epochs}",
        "",
        markdown_table(
            ["Model", "Accuracy", "Precision", "Recall", "Macro-F1", "Loss", "ms/sample", "Params"],
            table_rows,
        ),
        "",
        "Generated figures:",
        "",
        f"- [Accuracy comparison](../figures/baselines/{figure_prefix}accuracy.png)",
        f"- [Macro-F1 comparison](../figures/baselines/{figure_prefix}f1.png)",
        f"- [Loss curves](../figures/baselines/{figure_prefix}loss_curves.png)",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def save_bar(path: Path, rows: List[Dict[str, object]], metric: str, title: str, ylabel: str) -> None:
    models = [row["model"] for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.figure(figsize=(8.5, 5))
    bars = plt.bar(models, values, color=["#5C677D", "#3B82C4", "#4BAA7D", "#E07A5F"], edgecolor="#1F2937", linewidth=0.6)
    plt.ylim(0, max(1.0, max(values) * 1.12))
    plt.title(title, fontsize=13)
    plt.ylabel(ylabel)
    plt.xlabel("Baseline Model")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_loss_curves(path: Path, output_dir: Path, models: List[str]) -> None:
    plt.figure(figsize=(9, 5.5))
    for model in models:
        metrics = read_metrics(output_dir, model)
        history = metrics["history"]
        epochs = [item["epoch"] for item in history]
        train_loss = [item["train_loss"] for item in history]
        val_loss = [item["val"]["loss"] for item in history]
        plt.plot(epochs, train_loss, marker="o", linewidth=1.5, label=f"{model} train")
        plt.plot(epochs, val_loss, marker="s", linewidth=1.2, linestyle="--", label=f"{model} val")
    plt.title("Baseline Loss Curves", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def aggregate(output_dir: Path, models: List[str], epochs: int) -> None:
    report_dir = ROOT / "outputs" / "reports"
    fig_dir = ROOT / "outputs" / "figures" / "baselines"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in models:
        metrics = read_metrics(output_dir, model)
        test = metrics["test"]
        rows.append(
            {
                "model": model,
                "accuracy": float(test["accuracy"]),
                "precision_macro": float(test["precision_macro"]),
                "recall_macro": float(test["recall_macro"]),
                "f1_macro": float(test["f1_macro"]),
                "loss": float(test["loss"]),
                "inference_time_ms_per_sample": float(test["inference_time_ms_per_sample"]),
                "best_val_f1_macro": float(metrics["best_val_f1_macro"]),
                "num_parameters": int(metrics["num_parameters"]),
            }
        )
    rows.sort(key=lambda row: row["f1_macro"], reverse=True)
    suffix = output_dir.name.replace("baselines_", "")
    if output_dir.name == "baselines":
        suffix = ""
    name_prefix = f"baseline_{suffix}_" if suffix else "baseline_"
    title_suffix = f" ({suffix})" if suffix else ""
    write_csv(report_dir / f"{name_prefix}results.csv", rows)
    write_markdown(report_dir / f"{name_prefix}results.md", rows, epochs, name_prefix)
    save_bar(fig_dir / f"{name_prefix}accuracy.png", rows, "accuracy", f"Baseline Accuracy Comparison{title_suffix}", "Accuracy")
    save_bar(fig_dir / f"{name_prefix}f1.png", rows, "f1_macro", f"Baseline Macro-F1 Comparison{title_suffix}", "Macro-F1")
    save_loss_curves(fig_dir / f"{name_prefix}loss_curves.png", output_dir, models)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KNI-Intent baseline suite.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "baselines"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_script = ROOT / "scripts" / "train_baseline_classification.py"

    if not args.skip_train:
        for model in args.models:
            cmd = [
                args.python,
                str(train_script),
                "--model",
                model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--processed-dir",
                str(Path(args.processed_dir)),
                "--output-dir",
                str(output_dir),
            ]
            run_command(cmd)

    aggregate(output_dir, args.models, args.epochs)
    suffix = output_dir.name.replace("baselines_", "")
    if output_dir.name == "baselines":
        suffix = ""
    name_prefix = f"baseline_{suffix}_" if suffix else "baseline_"
    print(
        json.dumps(
            {
                "results_csv": str(ROOT / "outputs" / "reports" / f"{name_prefix}results.csv"),
                "results_md": str(ROOT / "outputs" / "reports" / f"{name_prefix}results.md"),
                "figures_dir": str(ROOT / "outputs" / "figures" / "baselines"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
