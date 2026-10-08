"""Create KNI-Intent-Challenge-NeutralKG splits.

This protocol removes class-specific KG context, coarsens QoS concepts, shares
capability nodes across confusable classes, and builds a stricter pairwise
hard-negative retrieval gallery.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

NEUTRAL_CONTEXT_NODES = [
    "context:service_request",
    "context:performance_management",
    "context:resource_coordination",
]

MEDIA_CLASSES = {
    "Cloud Gaming",
    "AR/VR Immersive Service",
    "Video Streaming eMBB",
}

PAIR_CAPABILITIES = {
    frozenset({"Industrial URLLC", "Autonomous Driving / V2X"}): [
        "capability:closed_loop_control",
        "capability:mobility_aware_assurance",
        "capability:reliable_delivery",
    ],
    frozenset(MEDIA_CLASSES): [
        "capability:interactive_delivery",
        "capability:adaptive_throughput",
        "capability:edge_assisted_experience",
    ],
    frozenset({"Telemedicine", "Security-Critical Enterprise"}): [
        "capability:trusted_service_assurance",
        "capability:policy_enforcement",
        "capability:resilient_session_control",
    ],
    frozenset({"Massive IoT mMTC"}): [
        "capability:massive_device_coordination",
        "capability:efficient_access_control",
        "capability:aggregate_traffic_handling",
    ],
}

RESOURCE_FAMILY = {
    "compute_resource": "resource_family:shared_compute",
    "edge_resource": "resource_family:shared_compute",
    "radio_resource": "resource_family:shared_radio",
    "bandwidth_resource": "resource_family:shared_radio",
    "routing_resource": "resource_family:shared_routing",
    "queue_resource": "resource_family:shared_queueing",
    "storage_resource": "resource_family:shared_storage",
    "security_resource": "resource_family:shared_security",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def class_family(intent_class: str, confusable_with: str | None = None) -> frozenset[str]:
    candidates = {intent_class}
    if confusable_with:
        candidates.add(confusable_with)
    if candidates & MEDIA_CLASSES:
        return frozenset(MEDIA_CLASSES)
    for family in PAIR_CAPABILITIES:
        if candidates <= family or candidates & family:
            return family
    return frozenset({intent_class})


def capability_nodes(intent_class: str, confusable_with: str | None) -> list[str]:
    family = class_family(intent_class, confusable_with)
    return list(PAIR_CAPABILITIES.get(family, ["capability:generic_service_assurance", "capability:resource_coordination"]))


def qos_family_from_name(name: str) -> str:
    value = name.lower()
    if "latency" in value or "delay" in value:
        return "qos_family:latency_constraint"
    if "bandwidth" in value or "throughput" in value or "embb" in value:
        return "qos_family:throughput_constraint"
    if "reliability" in value or "loss" in value or "reliable" in value:
        return "qos_family:reliability_constraint"
    if "jitter" in value or "stable" in value or "stability" in value:
        return "qos_family:stability_constraint"
    if "security" in value or "secure" in value or "privacy" in value:
        return "qos_family:security_constraint"
    if "mobility" in value or "handover" in value or "coverage" in value:
        return "qos_family:mobility_constraint"
    if "energy" in value or "battery" in value:
        return "qos_family:energy_constraint"
    return "qos_family:service_constraint"


def qos_nodes(record: dict[str, Any]) -> list[str]:
    nodes = {
        qos_family_from_name(str(node).split(":", 1)[1])
        for node in record.get("kg_nodes", [])
        if str(node).startswith("qos:")
    }
    # Add coarse numeric constraints only when they are broad and shared.
    if float(record.get("qos_latency_ms", 0.0)) <= 40:
        nodes.add("qos_family:latency_constraint")
    if float(record.get("qos_bandwidth_mbps", 0.0)) >= 80:
        nodes.add("qos_family:throughput_constraint")
    if float(record.get("qos_reliability", 0.0)) >= 0.995:
        nodes.add("qos_family:reliability_constraint")
    if int(record.get("qos_security_level", 0)) >= 4:
        nodes.add("qos_family:security_constraint")
    if int(record.get("qos_mobility", 0)) >= 4:
        nodes.add("qos_family:mobility_constraint")
    return sorted(nodes) or ["qos_family:service_constraint"]


def resource_nodes(record: dict[str, Any]) -> list[str]:
    nodes = {
        RESOURCE_FAMILY.get(str(resource), "resource_family:shared_resource")
        for resource in record.get("resources", [])
    }
    if not nodes:
        nodes.add("resource_family:shared_resource")
    return sorted(nodes)


def transform_record(record: dict[str, Any]) -> dict[str, Any]:
    item = dict(record)
    cls = str(item["intent_class"])
    confusable = item.get("confusable_with")
    caps = capability_nodes(cls, str(confusable) if confusable else None)
    qnodes = qos_nodes(item)
    rnodes = resource_nodes(item)

    nodes = sorted(set(NEUTRAL_CONTEXT_NODES + caps + qnodes + rnodes))
    edges: set[tuple[str, str, str]] = set()
    for ctx in NEUTRAL_CONTEXT_NODES:
        for qos in qnodes:
            edges.add((ctx, "has_constraint", qos))
        for cap in caps:
            edges.add((ctx, "uses_capability", cap))
    for cap in caps:
        for res in rnodes:
            edges.add((cap, "uses_resource_family", res))

    item["kg_nodes"] = nodes
    item["kg_edges"] = [list(edge) for edge in sorted(edges)]
    item["application"] = "neutral_service_request"
    item["scenario"] = "neutral_operational_context"
    item["network_functions"] = sorted({cap.split(":", 1)[1] for cap in caps})
    item["resources"] = sorted({res.split(":", 1)[1] for res in rnodes})
    item["challenge_type"] = sorted(
        set(item.get("challenge_type", []))
        | {
            "neutral_kg",
            "class_context_removed",
            "qos_coarsened",
            "confusable_capability_shared",
            "pairwise_hard_negative_ready",
        }
    )
    return item


def make_gallery(processed: Path) -> None:
    lines = []
    for split in ["train", "val", "test"]:
        lines.extend((processed / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
    (processed / "gallery_standard.jsonl").write_text("\n".join(line for line in lines if line.strip()) + "\n", encoding="utf-8")


def make_pairwise_hard_gallery(processed: Path) -> None:
    all_records = {}
    for split in ["train", "val", "test", "unseen_composition_test"]:
        for record in load_jsonl(processed / f"{split}.jsonl"):
            all_records[str(record["intent_id"])] = record

    wanted: set[str] = set()
    per_query_rows = []
    for record in load_jsonl(processed / "test.jsonl"):
        query_id = str(record["intent_id"])
        positives = [str(x) for x in record.get("positive_ids", []) if str(x) != query_id]
        negatives = [str(x) for x in record.get("hard_negative_ids", []) if str(x) != query_id]
        selected_pos = [pid for pid in positives if pid in all_records][:5]
        selected_neg = [nid for nid in negatives if nid in all_records][:8]
        wanted.update(selected_pos)
        wanted.update(selected_neg)
        per_query_rows.append(
            {
                "query_id": query_id,
                "positive_ids": selected_pos,
                "hard_negative_ids": selected_neg,
                "num_positives": len(selected_pos),
                "num_hard_negatives": len(selected_neg),
            }
        )
    gallery = [all_records[intent_id] for intent_id in sorted(wanted) if intent_id in all_records]
    write_jsonl(processed / "pairwise_hard_negative_gallery.jsonl", gallery)
    write_jsonl(processed / "pairwise_hard_negative_index.jsonl", per_query_rows)
    # Keep compatibility with the existing retrieval runner.
    write_jsonl(processed / "hard_negative_gallery.jsonl", gallery)


def report(records_by_split: dict[str, list[dict[str, Any]]], report_path: Path, output: Path) -> None:
    records = [record for rows in records_by_split.values() for record in rows]
    node_counts = [len(record["kg_nodes"]) for record in records]
    edge_counts = [len(record["kg_edges"]) for record in records]
    node_types = Counter(node.split(":", 1)[0] for record in records for node in record["kg_nodes"])
    class_counts = Counter(record["intent_class"] for record in records_by_split["train"] + records_by_split["val"] + records_by_split["test"])
    forbidden_prefixes = ("slice:", "intent_class:", "application:", "scenario:", "function:", "resource:")
    forbidden_counts = Counter(
        node.split(":", 1)[0]
        for record in records
        for node in record["kg_nodes"]
        if node.startswith(forbidden_prefixes)
    )
    context_counts = Counter(node for record in records for node in record["kg_nodes"] if node.startswith("context:"))
    cap_by_class: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for record in records:
        caps = tuple(sorted(node for node in record["kg_nodes"] if node.startswith("capability:")))
        cap_by_class[str(record["intent_class"])].add(caps)
    pair_index = load_jsonl(output / "pairwise_hard_negative_index.jsonl")
    pos_counts = [int(row["num_positives"]) for row in pair_index]
    neg_counts = [int(row["num_hard_negatives"]) for row in pair_index]

    lines = [
        "# KNI-Intent-Challenge-NeutralKG Report",
        "",
        "## Scale",
        "",
        f"- Train/val/test: {len(records_by_split['train'])} / {len(records_by_split['val'])} / {len(records_by_split['test'])}",
        f"- Unseen composition: {len(records_by_split['unseen_composition_test'])}",
        "",
        "## Neutralization Checks",
        "",
        f"- Forbidden direct KG nodes: {dict(sorted(forbidden_counts.items()))}",
        f"- Average KG nodes per sample: {mean(node_counts):.2f}",
        f"- Average KG edges per sample: {mean(edge_counts):.2f}",
        f"- Node type distribution: {dict(sorted(node_types.items()))}",
        f"- Context nodes: {dict(sorted(context_counts.items()))}",
        "",
        "## Capability Sharing",
        "",
    ]
    for cls, bundles in sorted(cap_by_class.items()):
        lines.append(f"- {cls}: {len(bundles)} unique neutral capability bundle(s)")
    lines.extend(
        [
            "",
            "## Pairwise Hard-Negative Retrieval Gallery",
            "",
            f"- Query count: {len(pair_index)}",
            f"- Average positives per query: {mean(pos_counts):.2f}",
            f"- Average hard negatives per query: {mean(neg_counts):.2f}",
            f"- Gallery size: {sum(1 for _ in (output / 'pairwise_hard_negative_gallery.jsonl').open(encoding='utf-8'))}",
            "",
            "## Standard Split Class Distribution",
            "",
        ]
    )
    for cls, count in sorted(class_counts.items()):
        lines.append(f"- {cls}: {count}")
    lines.extend(
        [
            "",
            "## Intended Use",
            "",
            "This split evaluates whether KCGT and graph-fusion baselines remain robust when KG nodes are neutralized beyond direct identifier masking. It is especially useful for pairwise hard-negative retrieval analysis.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/processed/challenge")
    parser.add_argument("--output-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--report", default="outputs/reports/kni_intent_challenge_neutral_kg_report.md")
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
    make_pairwise_hard_gallery(output)
    metadata = {
        "dataset": "KNI-Intent-Challenge-NeutralKG",
        "source": str(source),
        "splits": {split: len(rows) for split, rows in records_by_split.items()},
        "neutralization": [
            "class_context_removed",
            "qos_coarsened",
            "confusable_capability_shared",
            "pairwise_hard_negative_gallery",
        ],
    }
    (output / "challenge_neutral_kg_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report(records_by_split, ROOT / args.report, output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
