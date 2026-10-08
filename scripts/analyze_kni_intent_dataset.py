"""Analyze KNI-Intent dataset quality and generate visualizations.

Outputs:
- outputs/reports/kni_intent_pilot_report.md
- outputs/reports/kni_intent_pilot_stats.json
- outputs/figures/dataset/*.png
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt


REPORT_DIR = ROOT / "outputs" / "reports"

QOS_FIELDS = [
    "qos_latency_ms",
    "qos_bandwidth_mbps",
    "qos_reliability",
    "qos_jitter_ms",
    "qos_packet_loss",
    "qos_security_level",
    "qos_mobility",
    "qos_energy_sensitivity",
]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def counter(records: Sequence[Dict[str, object]], field: str) -> Counter:
    return Counter(str(r[field]) for r in records)


def nested_counter(records: Sequence[Dict[str, object]], outer: str, inner: str) -> Dict[str, Counter]:
    result: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        result[str(record[outer])][str(record[inner])] += 1
    return dict(result)


def numeric_values(records: Sequence[Dict[str, object]], field: str) -> List[float]:
    return [float(r[field]) for r in records]


def describe(values: Sequence[float]) -> Dict[str, float]:
    sorted_values = sorted(values)
    n = len(sorted_values)
    if not n:
        return {}
    return {
        "min": round(sorted_values[0], 8),
        "max": round(sorted_values[-1], 8),
        "mean": round(mean(sorted_values), 8),
        "median": round(median(sorted_values), 8),
        "p25": round(sorted_values[int((n - 1) * 0.25)], 8),
        "p75": round(sorted_values[int((n - 1) * 0.75)], 8),
    }


def list_lengths(records: Sequence[Dict[str, object]], field: str) -> List[int]:
    return [len(r[field]) for r in records]


def save_bar(
    counts: Counter,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    rotate: int = 35,
    color: str = "#3B82C4",
) -> None:
    labels, values = zip(*counts.most_common()) if counts else ([], [])
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(labels, values, color=color, edgecolor="#1F2937", linewidth=0.6)
    plt.title(title, fontsize=13)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_hist(values: Sequence[float], title: str, xlabel: str, path: Path, bins: int = 24, log_x: bool = False) -> None:
    plt.figure(figsize=(8, 5))
    plot_values = [math.log10(v) for v in values if v > 0] if log_x else list(values)
    plt.hist(plot_values, bins=bins, color="#4BAA7D", edgecolor="#1F2937", alpha=0.88)
    plt.title(title, fontsize=13)
    plt.xlabel(f"log10({xlabel})" if log_x else xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_grouped_bar(nested: Dict[str, Counter], title: str, path: Path) -> None:
    outer_labels = sorted(nested.keys())
    inner_labels = sorted({inner for counts in nested.values() for inner in counts})
    x = list(range(len(outer_labels)))
    width = 0.78 / max(1, len(inner_labels))
    plt.figure(figsize=(11, 5.5))
    palette = ["#3B82C4", "#E07A5F", "#4BAA7D", "#F2CC8F", "#8D6AB8", "#5C677D", "#D62828"]
    for idx, inner in enumerate(inner_labels):
        values = [nested[outer][inner] for outer in outer_labels]
        offsets = [v + (idx - (len(inner_labels) - 1) / 2) * width for v in x]
        plt.bar(offsets, values, width=width, label=inner, color=palette[idx % len(palette)], edgecolor="#1F2937", linewidth=0.4)
    plt.title(title, fontsize=13)
    plt.xlabel("Intent Class")
    plt.ylabel("Samples")
    plt.xticks(x, outer_labels, rotation=35, ha="right")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_qos_boxplot(records: Sequence[Dict[str, object]], path: Path) -> None:
    fields = [
        "qos_latency_ms",
        "qos_bandwidth_mbps",
        "qos_jitter_ms",
        "qos_security_level",
        "qos_mobility",
        "qos_energy_sensitivity",
    ]
    data = [numeric_values(records, field) for field in fields]
    labels = [field.replace("qos_", "").replace("_", " ") for field in fields]
    plt.figure(figsize=(11, 5.5))
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.title("QoS Feature Distribution", fontsize=13)
    plt.ylabel("Raw Value")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_kg_degree(edges: Sequence[Dict[str, str]], path: Path) -> None:
    degree = Counter()
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    top = Counter(dict(degree.most_common(20)))
    save_bar(top, "Top-20 KG Node Degrees", "KG Node", "Degree", path, rotate=60, color="#8D6AB8")


def make_stats(records: Sequence[Dict[str, object]], splits: Dict[str, List[Dict[str, object]]], kg_nodes, kg_edges) -> Dict[str, object]:
    group_to_split = {}
    leakage = []
    for split_name, split_records in splits.items():
        for record in split_records:
            group = str(record["positive_group_id"])
            prev = group_to_split.setdefault(group, split_name)
            if prev != split_name:
                leakage.append(group)

    stats = {
        "num_samples": len(records),
        "num_classes": len(counter(records, "intent_class")),
        "class_distribution": dict(counter(records, "intent_class")),
        "slice_distribution": dict(counter(records, "slice_type")),
        "split_distribution": {name: len(items) for name, items in splits.items()},
        "groups_per_split": {
            name: len({str(r["positive_group_id"]) for r in items}) for name, items in splits.items()
        },
        "positive_group_count": len({str(r["positive_group_id"]) for r in records}),
        "positive_group_leakage_count": len(set(leakage)),
        "qos_summary": {field: describe(numeric_values(records, field)) for field in QOS_FIELDS},
        "intent_text_words": describe([len(str(r["intent_text"]).split()) for r in records]),
        "network_function_count": describe(list_lengths(records, "network_functions")),
        "resource_count": describe(list_lengths(records, "resources")),
        "sample_kg_node_count": describe(list_lengths(records, "kg_nodes")),
        "sample_kg_edge_count": describe(list_lengths(records, "kg_edges")),
        "kg": {
            "num_nodes": len(kg_nodes),
            "num_edges": len(kg_edges),
            "node_type_distribution": dict(Counter(row["node_type"] for row in kg_nodes)),
            "relation_distribution": dict(Counter(row["relation"] for row in kg_edges)),
        },
    }
    return stats


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def write_report(stats: Dict[str, object], figure_paths: Dict[str, Path], report_path: Path, figure_link_prefix: str) -> None:
    qos_rows = []
    for field, desc in stats["qos_summary"].items():
        qos_rows.append([field, desc["min"], desc["max"], desc["mean"], desc["median"], desc["p25"], desc["p75"]])

    report = []
    report.append(f"# {stats['dataset_name']} Dataset Quality Report\n")
    report.append(f"This report summarizes the generated {stats['dataset_name']} dataset for KCGT experiments.\n")
    report.append("## 1. Overall Statistics\n")
    report.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Samples", stats["num_samples"]],
                ["Intent classes", stats["num_classes"]],
                ["Positive groups", stats["positive_group_count"]],
                ["Positive group leakage", stats["positive_group_leakage_count"]],
                ["KG nodes", stats["kg"]["num_nodes"]],
                ["KG edges", stats["kg"]["num_edges"]],
            ],
        )
    )
    report.append("\n## 2. Split Distribution\n")
    report.append(markdown_table(["Split", "Samples", "Positive Groups"], [[k, v, stats["groups_per_split"][k]] for k, v in stats["split_distribution"].items()]))
    report.append("\n## 3. Intent Class Distribution\n")
    report.append(markdown_table(["Intent Class", "Samples"], stats["class_distribution"].items()))
    report.append("\n## 4. Slice Distribution\n")
    report.append(markdown_table(["Slice Type", "Samples"], stats["slice_distribution"].items()))
    report.append("\n## 5. QoS Summary\n")
    report.append(markdown_table(["QoS Field", "Min", "Max", "Mean", "Median", "P25", "P75"], qos_rows))
    report.append("\n## 6. Sample-Level Structure\n")
    sample_rows = [
        ["Intent text length words", stats["intent_text_words"]],
        ["Network functions per sample", stats["network_function_count"]],
        ["Resources per sample", stats["resource_count"]],
        ["KG nodes per sample", stats["sample_kg_node_count"]],
        ["KG edges per sample", stats["sample_kg_edge_count"]],
    ]
    report.append(markdown_table(["Metric", "Description"], [[name, json.dumps(desc)] for name, desc in sample_rows]))
    report.append("\n## 7. KG Statistics\n")
    report.append("### Node Type Distribution\n")
    report.append(markdown_table(["Node Type", "Count"], stats["kg"]["node_type_distribution"].items()))
    report.append("\n### Relation Distribution\n")
    report.append(markdown_table(["Relation", "Count"], stats["kg"]["relation_distribution"].items()))
    report.append("\n## 8. Generated Figures\n")
    for name, path in figure_paths.items():
        report.append(f"- {name}: [{path.name}]({figure_link_prefix}/{path.name})")
    report.append("\n## 9. Quality Conclusion\n")
    if stats["positive_group_leakage_count"] == 0:
        report.append("The group-disjoint split check passed: no positive group leaks across train, validation, and test splits.")
    else:
        report.append("Warning: positive group leakage was detected and should be fixed before model training.")
    report.append("\nThe dataset is suitable for the next step: implementing dataset loaders and baseline models.")
    report_path.write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KNI-Intent dataset quality.")
    parser.add_argument("--profile", choices=["pilot", "hard", "hard_v2"], default="pilot")
    return parser.parse_args()


def resolve_paths(profile: str) -> Dict[str, Path]:
    if profile == "hard":
        return {
            "raw": ROOT / "data" / "raw" / "kni_intent_hard_raw.jsonl",
            "train": ROOT / "data" / "processed" / "hard" / "train.jsonl",
            "val": ROOT / "data" / "processed" / "hard" / "val.jsonl",
            "test": ROOT / "data" / "processed" / "hard" / "test.jsonl",
            "kg_nodes": ROOT / "data" / "kg" / "hard" / "kg_nodes.csv",
            "kg_edges": ROOT / "data" / "kg" / "hard" / "kg_edges.csv",
            "fig_dir": ROOT / "outputs" / "figures" / "dataset_hard",
            "report": ROOT / "outputs" / "reports" / "kni_intent_hard_report.md",
            "stats": ROOT / "outputs" / "reports" / "kni_intent_hard_stats.json",
            "dataset_name": "KNI-Intent-Hard",
            "figure_link_prefix": "../figures/dataset_hard",
        }
    if profile == "hard_v2":
        return {
            "raw": ROOT / "data" / "raw" / "kni_intent_hard_v2_raw.jsonl",
            "train": ROOT / "data" / "processed" / "hard_v2" / "train.jsonl",
            "val": ROOT / "data" / "processed" / "hard_v2" / "val.jsonl",
            "test": ROOT / "data" / "processed" / "hard_v2" / "test.jsonl",
            "kg_nodes": ROOT / "data" / "kg" / "hard_v2" / "kg_nodes.csv",
            "kg_edges": ROOT / "data" / "kg" / "hard_v2" / "kg_edges.csv",
            "fig_dir": ROOT / "outputs" / "figures" / "dataset_hard_v2",
            "report": ROOT / "outputs" / "reports" / "kni_intent_hard_v2_report.md",
            "stats": ROOT / "outputs" / "reports" / "kni_intent_hard_v2_stats.json",
            "dataset_name": "KNI-Intent-Hard-v2",
            "figure_link_prefix": "../figures/dataset_hard_v2",
        }
    return {
        "raw": ROOT / "data" / "raw" / "kni_intent_raw.jsonl",
        "train": ROOT / "data" / "processed" / "train.jsonl",
        "val": ROOT / "data" / "processed" / "val.jsonl",
        "test": ROOT / "data" / "processed" / "test.jsonl",
        "kg_nodes": ROOT / "data" / "kg" / "kg_nodes.csv",
        "kg_edges": ROOT / "data" / "kg" / "kg_edges.csv",
        "fig_dir": ROOT / "outputs" / "figures" / "dataset",
        "report": ROOT / "outputs" / "reports" / "kni_intent_pilot_report.md",
        "stats": ROOT / "outputs" / "reports" / "kni_intent_pilot_stats.json",
        "dataset_name": "KNI-Intent-Pilot",
        "figure_link_prefix": "../figures/dataset",
    }


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.profile)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = paths["fig_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(paths["raw"])
    splits = {
        "train": read_jsonl(paths["train"]),
        "val": read_jsonl(paths["val"]),
        "test": read_jsonl(paths["test"]),
    }
    kg_nodes = read_csv(paths["kg_nodes"])
    kg_edges = read_csv(paths["kg_edges"])

    stats = make_stats(records, splits, kg_nodes, kg_edges)
    stats["dataset_name"] = paths["dataset_name"]
    paths["stats"].write_text(json.dumps(stats, indent=2), encoding="utf-8")

    figure_paths = {
        "Intent class distribution": fig_dir / "class_distribution.png",
        "Slice type distribution": fig_dir / "slice_distribution.png",
        "Class-slice matrix": fig_dir / "class_slice_distribution.png",
        "QoS boxplot": fig_dir / "qos_boxplot.png",
        "Latency histogram": fig_dir / "qos_latency_hist.png",
        "Bandwidth histogram": fig_dir / "qos_bandwidth_hist.png",
        "Packet loss histogram": fig_dir / "qos_packet_loss_log_hist.png",
        "KG node type distribution": fig_dir / "kg_node_type_distribution.png",
        "KG relation distribution": fig_dir / "kg_relation_distribution.png",
        "KG top node degree": fig_dir / "kg_top_degree.png",
    }

    save_bar(counter(records, "intent_class"), "Intent Class Distribution", "Intent Class", "Samples", figure_paths["Intent class distribution"])
    save_bar(counter(records, "slice_type"), "Slice Type Distribution", "Slice Type", "Samples", figure_paths["Slice type distribution"], rotate=0, color="#E07A5F")
    save_grouped_bar(nested_counter(records, "intent_class", "slice_type"), "Class-Slice Distribution", figure_paths["Class-slice matrix"])
    save_qos_boxplot(records, figure_paths["QoS boxplot"])
    save_hist(numeric_values(records, "qos_latency_ms"), "Latency Requirement Distribution", "latency_ms", figure_paths["Latency histogram"])
    save_hist(numeric_values(records, "qos_bandwidth_mbps"), "Bandwidth Requirement Distribution", "bandwidth_mbps", figure_paths["Bandwidth histogram"], log_x=True)
    save_hist(numeric_values(records, "qos_packet_loss"), "Packet Loss Requirement Distribution", "packet_loss", figure_paths["Packet loss histogram"], log_x=True)
    save_bar(Counter(stats["kg"]["node_type_distribution"]), "KG Node Type Distribution", "Node Type", "Count", figure_paths["KG node type distribution"], rotate=30, color="#F2CC8F")
    save_bar(Counter(stats["kg"]["relation_distribution"]), "KG Relation Distribution", "Relation", "Count", figure_paths["KG relation distribution"], rotate=35, color="#5C677D")
    save_kg_degree(kg_edges, figure_paths["KG top node degree"])

    write_report(stats, figure_paths, paths["report"], paths["figure_link_prefix"])
    print(json.dumps({
        "report": str(paths["report"]),
        "stats": str(paths["stats"]),
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "positive_group_leakage_count": stats["positive_group_leakage_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
