"""Create KNI-Intent-Challenge-KG-Masked splits.

This protocol weakens KG shortcuts by removing slice nodes and replacing exact
application/function/resource nodes with coarse semantic families.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CONTEXT_BY_CLASS = {
    "Industrial URLLC": "cyber_physical_control",
    "Video Streaming eMBB": "broadband_media",
    "AR/VR Immersive Service": "immersive_interaction",
    "Massive IoT mMTC": "dense_sensing",
    "Autonomous Driving / V2X": "mobile_coordination",
    "Telemedicine": "critical_remote_operation",
    "Cloud Gaming": "interactive_media",
    "Security-Critical Enterprise": "protected_enterprise",
}

FUNCTION_FAMILY = {
    "latency_optimization": "latency_control",
    "priority_queueing": "latency_control",
    "qos_scheduling": "traffic_engineering",
    "packet_loss_control": "reliability_control",
    "reliability_assurance": "reliability_control",
    "network_slicing": "traffic_engineering",
    "traffic_classification": "traffic_engineering",
    "traffic_engineering": "traffic_engineering",
    "load_balancing": "traffic_engineering",
    "edge_computing": "compute_placement",
    "edge_caching": "compute_placement",
    "mobility_management": "mobility_support",
    "handover_optimization": "mobility_support",
    "device_management": "device_scale_management",
    "massive_access_control": "device_scale_management",
    "energy_saving": "device_scale_management",
    "security_monitoring": "security_assurance",
    "encryption": "security_assurance",
    "access_control": "security_assurance",
    "isolation": "security_assurance",
}

RESOURCE_FAMILY = {
    "compute_resource": "compute",
    "edge_resource": "compute",
    "radio_resource": "radio",
    "bandwidth_resource": "radio",
    "routing_resource": "routing",
    "queue_resource": "queueing",
    "storage_resource": "storage",
    "security_resource": "security",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def map_node(node: str, intent_class: str) -> str | None:
    if node.startswith("slice:"):
        return None
    if node.startswith("intent_class:"):
        return None
    if node.startswith("application:") or node.startswith("scenario:"):
        return f"context:{CONTEXT_BY_CLASS.get(intent_class, 'general_service')}"
    if node.startswith("function:"):
        name = node.split(":", 1)[1]
        return f"capability:{FUNCTION_FAMILY.get(name, 'service_control')}"
    if node.startswith("resource:"):
        name = node.split(":", 1)[1]
        return f"resource_family:{RESOURCE_FAMILY.get(name, 'shared_resource')}"
    return node


def map_relation(relation: str) -> str:
    if relation in {"belongs_to", "deployed_in"}:
        return "has_context"
    if relation in {"supported_by", "has_function"}:
        return "uses_capability"
    if relation == "has_resource":
        return "uses_resource_family"
    return relation


def transform_record(record: dict[str, Any]) -> dict[str, Any]:
    item = dict(record)
    cls = str(item["intent_class"])
    mapped_nodes = []
    for node in item.get("kg_nodes", []):
        mapped = map_node(str(node), cls)
        if mapped is not None:
            mapped_nodes.append(mapped)
    mapped_nodes = sorted(set(mapped_nodes))

    mapped_edges = set()
    for src, relation, dst in item.get("kg_edges", []):
        mapped_src = map_node(str(src), cls)
        mapped_dst = map_node(str(dst), cls)
        if mapped_src is None or mapped_dst is None or mapped_src == mapped_dst:
            continue
        mapped_edges.add((mapped_src, map_relation(str(relation)), mapped_dst))

    item["kg_nodes"] = mapped_nodes
    item["kg_edges"] = [list(edge) for edge in sorted(mapped_edges)]
    item["application"] = f"context_{CONTEXT_BY_CLASS.get(cls, 'general_service')}"
    item["scenario"] = f"context_{CONTEXT_BY_CLASS.get(cls, 'general_service')}"
    item["network_functions"] = sorted({FUNCTION_FAMILY.get(fn, "service_control") for fn in item.get("network_functions", [])})
    item["resources"] = sorted({RESOURCE_FAMILY.get(res, "shared_resource") for res in item.get("resources", [])})
    item["challenge_type"] = sorted(set(item.get("challenge_type", [])) | {"kg_masked", "slice_removed", "application_generalized", "function_family_masked"})
    return item


def make_gallery(processed: Path) -> None:
    lines = []
    for split in ["train", "val", "test"]:
        lines.extend((processed / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
    (processed / "gallery_standard.jsonl").write_text("\n".join(line for line in lines if line.strip()) + "\n", encoding="utf-8")


def make_hard_gallery(processed: Path) -> None:
    all_records = {}
    for split in ["train", "val", "test", "unseen_composition_test"]:
        for record in load_jsonl(processed / f"{split}.jsonl"):
            all_records[record["intent_id"]] = record
    wanted = set()
    for record in load_jsonl(processed / "test.jsonl"):
        wanted.update(record.get("positive_ids", []))
        wanted.update(record.get("hard_negative_ids", []))
    gallery = [all_records[intent_id] for intent_id in sorted(wanted) if intent_id in all_records]
    write_jsonl(processed / "hard_negative_gallery.jsonl", gallery)


def report(records_by_split: dict[str, list[dict[str, Any]]], report_path: Path) -> None:
    records = [record for rows in records_by_split.values() for record in rows]
    node_counts = [len(record["kg_nodes"]) for record in records]
    edge_counts = [len(record["kg_edges"]) for record in records]
    node_types = Counter(node.split(":", 1)[0] for record in records for node in record["kg_nodes"])
    class_counts = Counter(record["intent_class"] for record in records_by_split["train"] + records_by_split["val"] + records_by_split["test"])
    slice_nodes = sum(1 for record in records for node in record["kg_nodes"] if node.startswith("slice:"))
    app_nodes = sum(1 for record in records for node in record["kg_nodes"] if node.startswith("application:"))
    function_nodes = sum(1 for record in records for node in record["kg_nodes"] if node.startswith("function:"))
    lines = [
        "# KNI-Intent-Challenge-KG-Masked Report",
        "",
        "## Scale",
        "",
        f"- Train/val/test: {len(records_by_split['train'])} / {len(records_by_split['val'])} / {len(records_by_split['test'])}",
        f"- Unseen composition: {len(records_by_split['unseen_composition_test'])}",
        "",
        "## Masking Checks",
        "",
        f"- Remaining slice nodes: {slice_nodes}",
        f"- Remaining exact application nodes: {app_nodes}",
        f"- Remaining exact function nodes: {function_nodes}",
        f"- Average KG nodes per sample: {mean(node_counts):.2f}",
        f"- Average KG edges per sample: {mean(edge_counts):.2f}",
        f"- Node type distribution: {dict(sorted(node_types.items()))}",
        "",
        "## Standard Split Class Distribution",
        "",
    ]
    for cls, count in sorted(class_counts.items()):
        lines.append(f"- {cls}: {count}")
    lines.extend(
        [
            "",
            "## Intended Use",
            "",
            "This split is intended to evaluate whether graph-fusion and KCGT models remain effective when exact slice, application, function, and resource identifiers are masked or generalized.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/processed/challenge")
    parser.add_argument("--output-dir", default="data/processed/challenge_kg_masked")
    parser.add_argument("--report", default="outputs/reports/kni_intent_challenge_kg_masked_report.md")
    args = parser.parse_args()

    source = ROOT / args.source_dir
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test", "unseen_composition_test"]
    records_by_split = {}
    for split in splits:
        transformed = [transform_record(record) for record in load_jsonl(source / f"{split}.jsonl")]
        records_by_split[split] = transformed
        write_jsonl(output / f"{split}.jsonl", transformed)
    make_gallery(output)
    make_hard_gallery(output)
    metadata = {
        "dataset": "KNI-Intent-Challenge-KG-Masked",
        "source": str(source),
        "splits": {split: len(rows) for split, rows in records_by_split.items()},
    }
    (output / "challenge_kg_masked_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report(records_by_split, ROOT / args.report)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
