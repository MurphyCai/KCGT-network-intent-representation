"""Generate the KNI-Intent pilot dataset.

The generator is intentionally deterministic and dependency-free. It creates
the first pilot version described in docs/kni_intent_dataset_spec.md:

- 8 intent classes
- 20 base intents per class
- 3 paraphrases per base intent
- about 480 total samples
- group-disjoint train/validation/test split
- global KG node/edge files
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_KG = ROOT / "data" / "kg"

SEED = 20260710
BASE_INTENTS_PER_CLASS = 20
PARAPHRASES_PER_BASE = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


@dataclass(frozen=True)
class IntentClassSpec:
    class_id: str
    name: str
    primary_slice: str
    auxiliary_slices: Tuple[str, ...]
    applications: Tuple[str, ...]
    scenarios: Tuple[str, ...]
    qos_ranges: Dict[str, Tuple[float, float]]
    functions: Tuple[str, ...]
    resources: Tuple[str, ...]
    qos_nodes: Tuple[str, ...]


INTENT_CLASSES: Tuple[IntentClassSpec, ...] = (
    IntentClassSpec(
        "C1",
        "Industrial URLLC",
        "URLLC",
        ("Edge",),
        ("industrial_robot", "motion_control", "factory_automation", "remote_plc_control", "deterministic_video_inspection"),
        ("smart_factory", "industrial_workshop", "robotic_assembly_line", "automated_quality_inspection"),
        {
            "latency_ms": (1, 10),
            "bandwidth_mbps": (5, 100),
            "reliability": (0.9990, 0.999999),
            "jitter_ms": (0.1, 5),
            "packet_loss": (1e-7, 1e-4),
            "security_level": (3, 5),
            "mobility": (1, 3),
            "energy_sensitivity": (2, 4),
        },
        ("traffic_classification", "qos_scheduling", "priority_queueing", "network_slicing", "edge_computing", "latency_optimization", "reliability_control", "packet_loss_control"),
        ("radio_resource", "queue_resource", "compute_resource", "routing_resource", "bandwidth_resource"),
        ("ultra_low_latency", "high_reliability", "low_packet_loss", "deterministic_jitter"),
    ),
    IntentClassSpec(
        "C2",
        "Video Streaming eMBB",
        "eMBB",
        ("Edge",),
        ("video_streaming", "live_broadcast", "4k_video_delivery", "online_video_service", "surveillance_video_upload"),
        ("urban_mobile_broadband", "stadium_streaming", "campus_video_service", "home_entertainment"),
        {
            "latency_ms": (30, 150),
            "bandwidth_mbps": (20, 1000),
            "reliability": (0.9500, 0.9990),
            "jitter_ms": (5, 40),
            "packet_loss": (1e-4, 1e-2),
            "security_level": (1, 3),
            "mobility": (1, 4),
            "energy_sensitivity": (1, 3),
        },
        ("traffic_classification", "qos_scheduling", "network_slicing", "load_balancing", "bandwidth_reservation", "packet_loss_control"),
        ("bandwidth_resource", "storage_resource", "radio_resource", "routing_resource"),
        ("high_bandwidth", "stable_throughput", "moderate_latency"),
    ),
    IntentClassSpec(
        "C3",
        "AR/VR Immersive Service",
        "eMBB",
        ("URLLC", "Edge"),
        ("ar_assistance", "vr_conference", "immersive_training", "metaverse_interaction", "remote_visualization"),
        ("remote_training", "immersive_education", "virtual_collaboration", "industrial_ar_maintenance"),
        {
            "latency_ms": (5, 30),
            "bandwidth_mbps": (100, 2000),
            "reliability": (0.9900, 0.9999),
            "jitter_ms": (1, 10),
            "packet_loss": (1e-5, 1e-3),
            "security_level": (2, 4),
            "mobility": (1, 4),
            "energy_sensitivity": (1, 3),
        },
        ("traffic_classification", "qos_scheduling", "network_slicing", "edge_computing", "bandwidth_reservation", "latency_optimization", "jitter_control"),
        ("bandwidth_resource", "compute_resource", "queue_resource", "radio_resource"),
        ("high_bandwidth", "low_latency", "low_jitter", "edge_assisted"),
    ),
    IntentClassSpec(
        "C4",
        "Massive IoT mMTC",
        "mMTC",
        tuple(),
        ("smart_metering", "environmental_sensing", "agricultural_monitoring", "wearable_sensors", "city_sensor_network"),
        ("smart_city", "smart_agriculture", "utility_metering", "large_scale_monitoring"),
        {
            "latency_ms": (100, 1000),
            "bandwidth_mbps": (0.01, 5),
            "reliability": (0.9000, 0.9900),
            "jitter_ms": (20, 200),
            "packet_loss": (1e-3, 5e-2),
            "security_level": (1, 3),
            "mobility": (1, 2),
            "energy_sensitivity": (4, 5),
        },
        ("traffic_classification", "network_slicing", "energy_optimization", "massive_access_control", "qos_scheduling"),
        ("energy_resource", "radio_resource", "spectrum_resource", "routing_resource"),
        ("massive_connectivity", "energy_saving", "delay_tolerant"),
    ),
    IntentClassSpec(
        "C5",
        "Autonomous Driving / V2X",
        "V2X",
        ("URLLC",),
        ("vehicle_platooning", "cooperative_driving", "roadside_safety_alert", "autonomous_vehicle_control", "traffic_coordination"),
        ("urban_traffic", "highway_mobility", "intersection_safety", "connected_vehicle_network"),
        {
            "latency_ms": (1, 20),
            "bandwidth_mbps": (5, 200),
            "reliability": (0.9990, 0.999999),
            "jitter_ms": (0.1, 10),
            "packet_loss": (1e-7, 1e-4),
            "security_level": (3, 5),
            "mobility": (4, 5),
            "energy_sensitivity": (2, 4),
        },
        ("traffic_classification", "qos_scheduling", "priority_queueing", "network_slicing", "latency_optimization", "reliability_control", "mobility_management", "packet_loss_control"),
        ("radio_resource", "queue_resource", "routing_resource", "spectrum_resource", "bandwidth_resource"),
        ("ultra_low_latency", "high_mobility", "high_reliability", "low_packet_loss"),
    ),
    IntentClassSpec(
        "C6",
        "Telemedicine",
        "URLLC",
        ("eMBB", "Edge"),
        ("remote_surgery", "medical_video_consultation", "patient_monitoring", "ambulance_telemetry", "medical_image_transfer"),
        ("remote_hospital", "emergency_care", "mobile_healthcare", "specialist_consultation"),
        {
            "latency_ms": (1, 50),
            "bandwidth_mbps": (10, 1000),
            "reliability": (0.9990, 0.999999),
            "jitter_ms": (0.1, 20),
            "packet_loss": (1e-7, 1e-4),
            "security_level": (4, 5),
            "mobility": (1, 4),
            "energy_sensitivity": (1, 3),
        },
        ("traffic_classification", "qos_scheduling", "priority_queueing", "network_slicing", "edge_computing", "latency_optimization", "reliability_control", "encryption", "jitter_control"),
        ("bandwidth_resource", "compute_resource", "security_resource", "queue_resource", "routing_resource"),
        ("low_latency", "high_reliability", "high_security", "low_jitter"),
    ),
    IntentClassSpec(
        "C7",
        "Cloud Gaming",
        "eMBB",
        ("URLLC", "Edge"),
        ("cloud_game_streaming", "interactive_game_service", "mobile_cloud_gaming", "multiplayer_low_latency_game"),
        ("home_gaming", "mobile_gaming", "edge_game_rendering", "interactive_entertainment"),
        {
            "latency_ms": (5, 40),
            "bandwidth_mbps": (50, 1000),
            "reliability": (0.9900, 0.9999),
            "jitter_ms": (1, 15),
            "packet_loss": (1e-5, 1e-3),
            "security_level": (2, 4),
            "mobility": (1, 4),
            "energy_sensitivity": (1, 3),
        },
        ("traffic_classification", "qos_scheduling", "network_slicing", "edge_computing", "load_balancing", "bandwidth_reservation", "latency_optimization", "jitter_control"),
        ("bandwidth_resource", "compute_resource", "queue_resource", "radio_resource"),
        ("low_latency", "high_bandwidth", "low_jitter", "edge_assisted"),
    ),
    IntentClassSpec(
        "C8",
        "Security-Critical Enterprise",
        "Enterprise",
        ("Security",),
        ("enterprise_vpn", "secure_database_access", "financial_transaction_service", "confidential_file_transfer", "zero_trust_access"),
        ("enterprise_branch", "financial_network", "private_cloud_access", "secure_remote_work"),
        {
            "latency_ms": (20, 200),
            "bandwidth_mbps": (5, 500),
            "reliability": (0.9900, 0.9999),
            "jitter_ms": (5, 50),
            "packet_loss": (1e-5, 1e-3),
            "security_level": (5, 5),
            "mobility": (1, 3),
            "energy_sensitivity": (1, 3),
        },
        ("traffic_classification", "network_slicing", "load_balancing", "security_firewall", "intrusion_detection", "encryption", "access_control"),
        ("security_resource", "routing_resource", "bandwidth_resource", "compute_resource"),
        ("high_security", "traffic_isolation", "policy_enforcement"),
    ),
)


TEMPLATES: Tuple[str, ...] = (
    "{application} requires {qos_phrase} communication in {scenario}.",
    "The user needs {slice_phrase} service for {application} with {qos_phrase}.",
    "Provide {function_phrase} for {application} under {qos_phrase} constraints.",
    "In {scenario}, {application} should maintain {qos_phrase} and {resource_phrase}.",
    "Ensure {application} traffic is handled with {slice_type} slicing and {function_phrase}.",
    "{scenario} requests {slice_phrase} connectivity for {application} requiring {qos_phrase}.",
    "The network should support {application} using {function_phrase} while satisfying {qos_phrase}.",
    "Allocate {resource_phrase} to {application} in {scenario} for {qos_phrase} service.",
)


QOS_PHRASES: Dict[str, Tuple[str, ...]] = {
    "ultra_low_latency": ("ultra-low latency", "millisecond-level response", "minimal end-to-end delay"),
    "low_latency": ("low latency", "short response time", "delay-sensitive transmission"),
    "high_reliability": ("high reliability", "ultra-reliable delivery", "dependable packet delivery"),
    "low_packet_loss": ("low packet loss", "loss-sensitive delivery", "high packet delivery ratio"),
    "deterministic_jitter": ("deterministic jitter", "bounded delay variation", "predictable packet timing"),
    "high_bandwidth": ("high bandwidth", "high throughput", "large data-rate capacity"),
    "stable_throughput": ("stable throughput", "consistent streaming capacity", "smooth service bandwidth"),
    "moderate_latency": ("moderate latency", "acceptable service delay", "non-critical delay"),
    "low_jitter": ("low jitter", "stable delay variation", "smooth interactive timing"),
    "edge_assisted": ("edge-assisted processing", "near-user edge support", "edge computing support"),
    "massive_connectivity": ("massive connectivity", "large-scale device access", "dense sensor connectivity"),
    "energy_saving": ("energy-efficient connectivity", "low-power communication", "energy-saving transmission"),
    "delay_tolerant": ("delay-tolerant transmission", "non-real-time delivery", "relaxed latency"),
    "high_mobility": ("high mobility support", "fast handover capability", "mobility-aware continuity"),
    "high_security": ("high security", "strong security protection", "secure and isolated access"),
    "traffic_isolation": ("traffic isolation", "isolated enterprise traffic", "private service separation"),
    "policy_enforcement": ("policy enforcement", "strict access control", "security policy compliance"),
}


def text_name(value: str) -> str:
    return value.replace("_", " ")


def choose_many(rng: random.Random, values: Sequence[str], minimum: int, maximum: int) -> List[str]:
    maximum = min(maximum, len(values))
    count = rng.randint(minimum, maximum)
    return sorted(rng.sample(list(values), count))


def sample_float(rng: random.Random, lo: float, hi: float, digits: int = 6) -> float:
    return round(rng.uniform(lo, hi), digits)


def sample_int(rng: random.Random, lo: float, hi: float) -> int:
    return int(round(rng.uniform(lo, hi)))


def format_qos_phrase(rng: random.Random, qos_nodes: Sequence[str]) -> str:
    picked = rng.sample(list(qos_nodes), k=min(rng.randint(2, 3), len(qos_nodes)))
    phrases = [rng.choice(QOS_PHRASES[node]) for node in picked]
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def build_edges(
    spec: IntentClassSpec,
    application: str,
    scenario: str,
    functions: Sequence[str],
    resources: Sequence[str],
    qos_nodes: Sequence[str],
) -> List[Tuple[str, str, str]]:
    intent_node = f"intent_class:{spec.name}"
    app_node = f"application:{application}"
    scenario_node = f"scenario:{scenario}"
    slice_node = f"slice:{spec.primary_slice}"
    edges: List[Tuple[str, str, str]] = [
        (app_node, "deployed_in", scenario_node),
        (app_node, "belongs_to", slice_node),
        (intent_node, "belongs_to", slice_node),
    ]
    for aux_slice in spec.auxiliary_slices:
        edges.append((intent_node, "supported_by", f"slice:{aux_slice}"))
    for qos in qos_nodes:
        qos_node = f"qos:{qos}"
        edges.append((app_node, "requires", qos_node))
        edges.append((intent_node, "requires", qos_node))
    for function in functions:
        function_node = f"function:{function}"
        edges.append((intent_node, "has_function", function_node))
        edges.append((slice_node, "supported_by", function_node))
    for resource in resources:
        resource_node = f"resource:{resource}"
        edges.append((intent_node, "has_resource", resource_node))
    for function in functions:
        resource = resources[hash((function, spec.class_id)) % len(resources)]
        edges.append((f"function:{function}", "consumes", f"resource:{resource}"))
    return sorted(set(edges))


def edge_nodes(edges: Iterable[Tuple[str, str, str]]) -> List[str]:
    nodes = set()
    for src, _, dst in edges:
        nodes.add(src)
        nodes.add(dst)
    return sorted(nodes)


def make_record(
    rng: random.Random,
    spec: IntentClassSpec,
    base_idx: int,
    para_idx: int,
) -> Dict[str, object]:
    application = rng.choice(spec.applications)
    scenario = rng.choice(spec.scenarios)
    functions = choose_many(rng, spec.functions, 3, 5)
    resources = choose_many(rng, spec.resources, 1, 4)
    qos_nodes = choose_many(rng, spec.qos_nodes, 2, min(4, len(spec.qos_nodes)))
    template_id = f"T{rng.randint(1, len(TEMPLATES)):02d}"
    template = TEMPLATES[int(template_id[1:]) - 1]
    qos_phrase = format_qos_phrase(rng, qos_nodes)
    function_phrase = ", ".join(text_name(v) for v in functions[:2])
    resource_phrase = ", ".join(text_name(v) for v in resources[:2])
    slice_phrase = f"{spec.primary_slice} slice"
    intent_text = template.format(
        application=text_name(application),
        scenario=text_name(scenario),
        qos_phrase=qos_phrase,
        function_phrase=function_phrase,
        resource_phrase=resource_phrase,
        slice_type=spec.primary_slice,
        slice_phrase=slice_phrase,
    )

    qos = {}
    for key, (lo, hi) in spec.qos_ranges.items():
        if key in {"security_level", "mobility", "energy_sensitivity"}:
            qos[key] = sample_int(rng, lo, hi)
        elif key in {"packet_loss", "reliability"}:
            qos[key] = sample_float(rng, lo, hi, digits=8)
        else:
            qos[key] = sample_float(rng, lo, hi, digits=3)

    edges = build_edges(spec, application, scenario, functions, resources, qos_nodes)
    positive_group_id = f"{spec.class_id}_B{base_idx:03d}"
    intent_id = f"{positive_group_id}_P{para_idx:02d}"
    return {
        "intent_id": intent_id,
        "intent_text": intent_text,
        "intent_class": spec.name,
        "class_id": spec.class_id,
        "slice_type": spec.primary_slice,
        "auxiliary_slice_type": list(spec.auxiliary_slices),
        "application": application,
        "scenario": scenario,
        "qos_latency_ms": qos["latency_ms"],
        "qos_bandwidth_mbps": qos["bandwidth_mbps"],
        "qos_reliability": qos["reliability"],
        "qos_jitter_ms": qos["jitter_ms"],
        "qos_packet_loss": qos["packet_loss"],
        "qos_security_level": qos["security_level"],
        "qos_mobility": qos["mobility"],
        "qos_energy_sensitivity": qos["energy_sensitivity"],
        "network_functions": functions,
        "resources": resources,
        "kg_nodes": edge_nodes(edges),
        "kg_edges": edges,
        "positive_group_id": positive_group_id,
        "source_template_id": template_id,
        "split_tag": "",
    }


def generate_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in INTENT_CLASSES:
        for base_idx in range(1, BASE_INTENTS_PER_CLASS + 1):
            # A group-level seed makes positive groups reproducible and related
            # while still allowing paraphrases to differ.
            base_seed = f"{SEED}-{spec.class_id}-{base_idx}"
            for para_idx in range(1, PARAPHRASES_PER_BASE + 1):
                rng = random.Random(f"{base_seed}-{para_idx}")
                records.append(make_record(rng, spec, base_idx, para_idx))
    return records


def split_records(records: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    groups = sorted({str(r["positive_group_id"]) for r in records})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    train_end = int(len(groups) * TRAIN_RATIO)
    val_end = int(len(groups) * (TRAIN_RATIO + VAL_RATIO))
    split_by_group = {}
    for group in groups[:train_end]:
        split_by_group[group] = "train"
    for group in groups[train_end:val_end]:
        split_by_group[group] = "val"
    for group in groups[val_end:]:
        split_by_group[group] = "test"

    splits = {"train": [], "val": [], "test": []}
    for record in records:
        split = split_by_group[str(record["positive_group_id"])]
        record["split_tag"] = split
        splits[split].append(record)
    return splits


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def flatten_for_csv(record: Dict[str, object]) -> Dict[str, object]:
    flat = dict(record)
    for key in ("network_functions", "resources", "kg_nodes", "kg_edges", "auxiliary_slice_type"):
        flat[key] = json.dumps(flat[key], ensure_ascii=False)
    return flat


def write_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    fieldnames = list(flatten_for_csv(records[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_for_csv(record))


def collect_global_kg(records: Sequence[Dict[str, object]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    nodes = {}
    edges = set()
    for record in records:
        for node in record["kg_nodes"]:
            node_str = str(node)
            node_type, _, node_name = node_str.partition(":")
            nodes[node_str] = node_type
        for src, rel, dst in record["kg_edges"]:
            edges.add((src, rel, dst))

    conflict_pairs = [
        ("qos:ultra_low_latency", "conflicts_with", "qos:energy_saving"),
        ("qos:high_bandwidth", "conflicts_with", "qos:energy_saving"),
        ("qos:high_security", "conflicts_with", "qos:low_latency"),
        ("qos:traffic_isolation", "conflicts_with", "qos:stable_throughput"),
    ]
    for src, rel, dst in conflict_pairs:
        nodes[src] = src.split(":", 1)[0]
        nodes[dst] = dst.split(":", 1)[0]
        edges.add((src, rel, dst))

    return sorted(nodes.items()), sorted(edges)


def write_kg(records: Sequence[Dict[str, object]]) -> None:
    nodes, edges = collect_global_kg(records)
    with (DATA_KG / "kg_nodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "node_type", "node_name"])
        writer.writeheader()
        for node_id, node_type in nodes:
            writer.writerow({"node_id": node_id, "node_type": node_type, "node_name": node_id.split(":", 1)[1]})

    with (DATA_KG / "kg_edges.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "relation", "target"])
        writer.writeheader()
        for src, rel, dst in edges:
            writer.writerow({"source": src, "relation": rel, "target": dst})

    relation_types = sorted({rel for _, rel, _ in edges})
    node_types = sorted({node_type for _, node_type in nodes})
    (DATA_KG / "relation_types.json").write_text(json.dumps(relation_types, indent=2), encoding="utf-8")
    (DATA_KG / "node_types.json").write_text(json.dumps(node_types, indent=2), encoding="utf-8")


def write_generation_config() -> None:
    config = {
        "dataset": "KNI-Intent-Pilot",
        "seed": SEED,
        "classes": len(INTENT_CLASSES),
        "base_intents_per_class": BASE_INTENTS_PER_CLASS,
        "paraphrases_per_base": PARAPHRASES_PER_BASE,
        "expected_samples": len(INTENT_CLASSES) * BASE_INTENTS_PER_CLASS * PARAPHRASES_PER_BASE,
        "split": {
            "mode": "group-disjoint",
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": round(1 - TRAIN_RATIO - VAL_RATIO, 2),
        },
    }
    (DATA_RAW / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def summarize(records: Sequence[Dict[str, object]], splits: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    classes = sorted({str(r["intent_class"]) for r in records})
    slices = sorted({str(r["slice_type"]) for r in records})
    groups = sorted({str(r["positive_group_id"]) for r in records})
    nodes, edges = collect_global_kg(records)
    summary = {
        "num_samples": len(records),
        "num_classes": len(classes),
        "classes": classes,
        "num_slice_types": len(slices),
        "slice_types": slices,
        "num_positive_groups": len(groups),
        "samples_per_group": PARAPHRASES_PER_BASE,
        "splits": {name: len(items) for name, items in splits.items()},
        "groups_per_split": {
            name: len({str(r["positive_group_id"]) for r in items}) for name, items in splits.items()
        },
        "num_kg_nodes": len(nodes),
        "num_kg_edges": len(edges),
        "avg_kg_nodes_per_sample": round(sum(len(r["kg_nodes"]) for r in records) / len(records), 2),
        "avg_kg_edges_per_sample": round(sum(len(r["kg_edges"]) for r in records) / len(records), 2),
        "avg_intent_length_words": round(sum(len(str(r["intent_text"]).split()) for r in records) / len(records), 2),
    }
    return summary


def validate(records: Sequence[Dict[str, object]], splits: Dict[str, List[Dict[str, object]]]) -> None:
    assert records, "No records generated."
    ids = [r["intent_id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate intent IDs."
    for record in records:
        assert record["intent_text"], "Empty intent text."
        assert record["intent_class"], "Missing class."
        assert record["slice_type"], "Missing slice."
        assert len(record["network_functions"]) >= 2, "Too few network functions."
        assert len(record["resources"]) >= 1, "Missing resources."
        assert record["kg_nodes"], "Empty KG nodes."
        assert record["kg_edges"], "Empty KG edges."

    group_to_split = {}
    for split_name, split_records_value in splits.items():
        for record in split_records_value:
            group = record["positive_group_id"]
            previous = group_to_split.setdefault(group, split_name)
            assert previous == split_name, "Positive group leaked across splits."


def main() -> None:
    for directory in (DATA_RAW, DATA_PROCESSED, DATA_KG):
        directory.mkdir(parents=True, exist_ok=True)

    records = generate_records()
    splits = split_records(records)
    validate(records, splits)

    write_generation_config()
    write_jsonl(DATA_RAW / "kni_intent_raw.jsonl", records)
    write_csv(DATA_RAW / "kni_intent_raw.csv", records)

    for split_name, split_items in splits.items():
        write_jsonl(DATA_PROCESSED / f"{split_name}.jsonl", split_items)
        write_csv(DATA_PROCESSED / f"{split_name}.csv", split_items)

    write_kg(records)
    summary = summarize(records, splits)
    (DATA_RAW / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
