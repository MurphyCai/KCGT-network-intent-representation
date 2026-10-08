"""Generate KNI-Intent-Challenge from KNI-Intent-Hard-v2.

This generator creates a stress-test split by rewriting text, moving QoS values
toward class boundaries, removing explicit intent-class KG nodes, and adding an
unseen-composition evaluation split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
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

CONFUSABLE_CLASS = {
    "Industrial URLLC": "Autonomous Driving / V2X",
    "Video Streaming eMBB": "Cloud Gaming",
    "AR/VR Immersive Service": "Cloud Gaming",
    "Massive IoT mMTC": "Industrial URLLC",
    "Autonomous Driving / V2X": "Industrial URLLC",
    "Telemedicine": "Industrial URLLC",
    "Cloud Gaming": "AR/VR Immersive Service",
    "Security-Critical Enterprise": "Telemedicine",
}

BOUNDARY_QOS = {
    "Industrial URLLC": {
        "latency": (8.0, 18.0),
        "bandwidth": (20.0, 140.0),
        "reliability": (0.995, 0.9994),
        "jitter": (4.0, 14.0),
        "loss": (0.00005, 0.0015),
    },
    "Video Streaming eMBB": {
        "latency": (55.0, 170.0),
        "bandwidth": (120.0, 760.0),
        "reliability": (0.965, 0.993),
        "jitter": (12.0, 45.0),
        "loss": (0.0005, 0.006),
    },
    "AR/VR Immersive Service": {
        "latency": (18.0, 42.0),
        "bandwidth": (220.0, 980.0),
        "reliability": (0.985, 0.998),
        "jitter": (5.0, 22.0),
        "loss": (0.0002, 0.003),
    },
    "Massive IoT mMTC": {
        "latency": (220.0, 900.0),
        "bandwidth": (0.8, 12.0),
        "reliability": (0.92, 0.985),
        "jitter": (35.0, 180.0),
        "loss": (0.002, 0.035),
    },
    "Autonomous Driving / V2X": {
        "latency": (10.0, 28.0),
        "bandwidth": (35.0, 210.0),
        "reliability": (0.996, 0.9996),
        "jitter": (3.0, 18.0),
        "loss": (0.00003, 0.001),
    },
    "Telemedicine": {
        "latency": (22.0, 70.0),
        "bandwidth": (45.0, 420.0),
        "reliability": (0.992, 0.9992),
        "jitter": (5.0, 24.0),
        "loss": (0.00004, 0.0016),
    },
    "Cloud Gaming": {
        "latency": (22.0, 55.0),
        "bandwidth": (80.0, 520.0),
        "reliability": (0.982, 0.997),
        "jitter": (5.0, 26.0),
        "loss": (0.0002, 0.004),
    },
    "Security-Critical Enterprise": {
        "latency": (45.0, 190.0),
        "bandwidth": (45.0, 360.0),
        "reliability": (0.985, 0.9985),
        "jitter": (12.0, 60.0),
        "loss": (0.0002, 0.005),
    },
}

GENERIC_SERVICE_ALIASES = {
    "Industrial URLLC": ["closed-loop control task", "precision automation flow", "deterministic machine operation"],
    "Video Streaming eMBB": ["high-volume media delivery", "continuous visual feed", "broadband content session"],
    "AR/VR Immersive Service": ["immersive interaction session", "spatial collaboration flow", "real-time visual workspace"],
    "Massive IoT mMTC": ["large-scale sensing campaign", "dense device reporting task", "periodic telemetry collection"],
    "Autonomous Driving / V2X": ["mobile coordination exchange", "roadside cooperative flow", "moving asset safety update"],
    "Telemedicine": ["remote expert operation", "critical assisted service", "interactive care workflow"],
    "Cloud Gaming": ["interactive rendered session", "latency-sensitive media control", "remote graphics interaction"],
    "Security-Critical Enterprise": ["protected business transaction", "isolated enterprise workflow", "policy-sensitive access flow"],
}

ADVERSARIAL_TEMPLATES = [
    "Keep the {service} usable when demand fluctuates; prioritize {qos_phrase} while relying on {function_phrase}.",
    "The request describes a {service} that should stay stable under load with {qos_phrase} and controlled resource use.",
    "For this {service}, the management plane should balance {qos_phrase}, continuity, and {function_phrase}.",
    "Maintain an acceptable experience for a {service}; the main constraints are {qos_phrase} plus predictable handling.",
    "A previously unseen service composition asks for {qos_phrase} without exposing the exact service category.",
    "The controller should infer the right treatment for this {service} from performance targets and operational dependencies.",
]

LEAKAGE_PATTERNS = [
    r"\bURLLC\b",
    r"\beMBB\b",
    r"\bmMTC\b",
    r"\bV2X\b",
    r"\bslice\b",
    r"\bindustrial\b",
    r"\bgaming\b",
    r"\btelemedicine\b",
    r"\bsurgery\b",
    r"\bvehicle\b",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def text_name(value: str) -> str:
    return value.replace("_", " ").replace("/", " ").strip()


def extract_qos_nodes(record: dict[str, Any]) -> list[str]:
    return [node.split(":", 1)[1] for node in record.get("kg_nodes", []) if str(node).startswith("qos:")]


def clean_leakage(text: str) -> str:
    cleaned = text
    replacements = {
        "URLLC": "strict-service",
        "eMBB": "broadband-service",
        "mMTC": "dense-device-service",
        "V2X": "mobile-service",
        "slice": "service profile",
        "industrial": "production",
        "gaming": "interactive media",
        "telemedicine": "remote assistance",
        "surgery": "critical operation",
        "vehicle": "mobile unit",
    }
    for old, new in replacements.items():
        cleaned = re.sub(old, new, cleaned, flags=re.IGNORECASE)
    return cleaned


def sample_boundary_qos(record: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    bounds = BOUNDARY_QOS[record["intent_class"]]
    item = dict(record)
    item["qos_latency_ms"] = round(rng.uniform(*bounds["latency"]), 3)
    item["qos_bandwidth_mbps"] = round(rng.uniform(*bounds["bandwidth"]), 3)
    item["qos_reliability"] = round(rng.uniform(*bounds["reliability"]), 8)
    item["qos_jitter_ms"] = round(rng.uniform(*bounds["jitter"]), 3)
    item["qos_packet_loss"] = round(rng.uniform(*bounds["loss"]), 8)
    if rng.random() < 0.35:
        item["qos_security_level"] = max(1, min(5, int(record.get("qos_security_level", 3)) + rng.choice([-1, 0, 1])))
    if rng.random() < 0.35:
        item["qos_mobility"] = max(1, min(5, int(record.get("qos_mobility", 2)) + rng.choice([-1, 0, 1])))
    return item


def qos_phrase(record: dict[str, Any], rng: random.Random) -> str:
    parts = []
    latency = float(record["qos_latency_ms"])
    bandwidth = float(record["qos_bandwidth_mbps"])
    reliability = float(record["qos_reliability"])
    if latency < 25:
        parts.append(rng.choice(["bounded response time", "tight delay control", "short reaction window"]))
    elif latency < 80:
        parts.append(rng.choice(["moderate response time", "controlled delay", "interactive responsiveness"]))
    else:
        parts.append(rng.choice(["delay-tolerant delivery", "relaxed response timing", "scheduled delivery"]))
    if bandwidth > 200:
        parts.append(rng.choice(["substantial throughput", "large traffic volume", "high data rate"]))
    elif bandwidth < 20:
        parts.append(rng.choice(["small payload exchange", "narrow data demand", "lightweight reporting"]))
    if reliability > 0.995:
        parts.append(rng.choice(["consistent delivery", "low failure exposure", "dependable continuity"]))
    return ", ".join(parts[:3])


def rewrite_text(record: dict[str, Any], rng: random.Random) -> str:
    service = rng.choice(GENERIC_SERVICE_ALIASES[record["intent_class"]])
    functions = [text_name(v) for v in record.get("network_functions", [])]
    rng.shuffle(functions)
    function_phrase = ", ".join(functions[:2]) if functions else "adaptive handling"
    template = rng.choice(ADVERSARIAL_TEMPLATES)
    text = template.format(service=service, qos_phrase=qos_phrase(record, rng), function_phrase=function_phrase)
    return clean_leakage(text)


def remove_intent_class_kg(record: dict[str, Any]) -> None:
    record["kg_nodes"] = [node for node in record.get("kg_nodes", []) if not str(node).startswith("intent_class:")]
    record["kg_edges"] = [
        edge for edge in record.get("kg_edges", [])
        if not str(edge[0]).startswith("intent_class:") and not str(edge[2]).startswith("intent_class:")
    ]


def composition_key(record: dict[str, Any]) -> str:
    qos_nodes = extract_qos_nodes(record)
    qos = sorted(qos_nodes)[0] if qos_nodes else "qos_unknown"
    functions = sorted(record.get("network_functions", []))
    function = functions[0] if functions else "function_unknown"
    return f"{record['intent_class']}|{record['application']}|{qos}|{function}"


def transform_record(record: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}-{record['intent_id']}")
    item = sample_boundary_qos(record, rng)
    item["challenge_source_id"] = record["intent_id"]
    item["intent_id"] = f"CHAL_{record['intent_id']}"
    item["positive_group_id"] = f"CHAL_{record['positive_group_id']}"
    item["intent_text"] = rewrite_text(item, rng)
    item["confusable_with"] = CONFUSABLE_CLASS[item["intent_class"]]
    item["composition_key"] = composition_key(item)
    item["challenge_type"] = [
        "keyword_debiased_text",
        "boundary_qos",
        "confusable_neighbor",
        "label_node_removed_kg",
        "adversarial_paraphrase",
    ]
    remove_intent_class_kg(item)
    item["positive_ids"] = []
    item["hard_negative_ids"] = []
    item["easy_negative_ids"] = []
    return item


def qos_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    total = 0.0
    for field in QOS_FIELDS:
        av = float(a.get(field, 0.0))
        bv = float(b.get(field, 0.0))
        scale = max(abs(av), abs(bv), 1.0)
        total += ((av - bv) / scale) ** 2
    return math.sqrt(total)


def attach_contrastive_metadata(records: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["positive_group_id"]].append(record)
        by_class[record["intent_class"]].append(record)
        by_slice[record["slice_type"]].append(record)

    for record in records:
        positives = [r["intent_id"] for r in by_group[record["positive_group_id"]] if r["intent_id"] != record["intent_id"]]
        record["positive_ids"] = positives[:5]

        candidates = []
        candidates.extend(by_class.get(record["confusable_with"], []))
        candidates.extend(by_slice.get(record["slice_type"], []))
        rng = random.Random(f"neg-{record['intent_id']}")
        candidates.extend(rng.sample(records, min(600, len(records))))
        seen = set()
        scored_hard = []
        scored_easy = []
        funcs = set(record.get("network_functions", []))
        for candidate in candidates:
            cid = candidate["intent_id"]
            if cid in seen or cid == record["intent_id"] or candidate["positive_group_id"] == record["positive_group_id"]:
                continue
            seen.add(cid)
            if candidate["intent_class"] == record["intent_class"]:
                continue
            overlap = len(funcs & set(candidate.get("network_functions", [])))
            same_slice = candidate["slice_type"] == record["slice_type"]
            confusable = candidate["intent_class"] == record["confusable_with"]
            dist = qos_distance(record, candidate)
            score = (2.0 if confusable else 0.0) + (1.0 if same_slice else 0.0) + 0.3 * overlap - dist
            if confusable or same_slice or overlap >= 2 or dist < 0.8:
                scored_hard.append((score, cid))
            else:
                scored_easy.append((-score, cid))
        scored_hard.sort(reverse=True)
        scored_easy.sort(reverse=True)
        record["hard_negative_ids"] = [cid for _, cid in scored_hard[:8]]
        record["easy_negative_ids"] = [cid for _, cid in scored_easy[:10]]


def select_unseen_compositions(records: list[dict[str, Any]], per_class: int, seed: int) -> set[str]:
    by_class_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class_group[record["intent_class"]][record["positive_group_id"]].append(record)

    heldout_groups = set()
    for cls, groups in by_class_group.items():
        component_count = Counter()
        group_key = {}
        for group_id, items in groups.items():
            representative = items[0]
            key = representative["composition_key"]
            group_key[group_id] = key
            _, app, qos, function = key.split("|", 3)
            component_count[("app", app)] += 1
            component_count[("qos", qos)] += 1
            component_count[("function", function)] += 1
        candidates = []
        for group_id, key in group_key.items():
            _, app, qos, function = key.split("|", 3)
            if component_count[("app", app)] > 1 and component_count[("qos", qos)] > 1 and component_count[("function", function)] > 1:
                candidates.append(group_id)
        rng = random.Random(f"{seed}-composition-{cls}")
        rng.shuffle(candidates)
        heldout_groups.update(candidates[:per_class])
    return heldout_groups


def split_standard(records: list[dict[str, Any]], heldout_groups: set[str], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups = sorted({record["positive_group_id"] for record in records if record["positive_group_id"] not in heldout_groups})
    rng = random.Random(seed)
    rng.shuffle(groups)
    train_end = int(len(groups) * 0.70)
    val_end = int(len(groups) * 0.85)
    group_to_split = {group: "train" for group in groups[:train_end]}
    group_to_split.update({group: "val" for group in groups[train_end:val_end]})
    group_to_split.update({group: "test" for group in groups[val_end:]})
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        group = record["positive_group_id"]
        if group in heldout_groups:
            continue
        split = group_to_split[group]
        item = dict(record)
        item["split_tag"] = split
        splits[split].append(item)
    return splits


def leakage_count(records: list[dict[str, Any]]) -> int:
    joined = "\n".join(record["intent_text"] for record in records)
    count = 0
    for pattern in LEAKAGE_PATTERNS:
        count += len(re.findall(pattern, joined, flags=re.IGNORECASE))
    return count


def build_report(
    records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    unseen: list[dict[str, Any]],
    output_path: Path,
) -> None:
    all_split_records = splits["train"] + splits["val"] + splits["test"]
    class_counts = Counter(record["intent_class"] for record in records)
    split_counts = {name: len(items) for name, items in splits.items()}
    group_counts = {name: len({r["positive_group_id"] for r in items}) for name, items in splits.items()}
    kg_node_counts = [len(record["kg_nodes"]) for record in records]
    kg_edge_counts = [len(record["kg_edges"]) for record in records]
    hard_counts = [len(record["hard_negative_ids"]) for record in records]
    leak_terms = leakage_count(records)
    train_groups = {r["positive_group_id"] for r in splits["train"]}
    val_groups = {r["positive_group_id"] for r in splits["val"]}
    test_groups = {r["positive_group_id"] for r in splits["test"]}
    group_leakage = len((train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups))

    lines = [
        "# KNI-Intent-Challenge Report",
        "",
        "KNI-Intent-Challenge is a stress-test split derived from KNI-Intent-Hard-v2.",
        "",
        "## Scale",
        "",
        f"- Total challenge records: {len(records)}",
        f"- Standard split records: {len(all_split_records)}",
        f"- Unseen-composition records: {len(unseen)}",
        f"- Standard split sizes: {split_counts}",
        f"- Groups per standard split: {group_counts}",
        f"- Group leakage across train/val/test: {group_leakage}",
        "",
        "## Class Distribution",
        "",
    ]
    for cls, count in sorted(class_counts.items()):
        lines.append(f"- {cls}: {count}")
    lines.extend(
        [
            "",
            "## Challenge Controls",
            "",
            f"- Explicit leakage keyword matches after rewriting: {leak_terms}",
            "- Sample-level `intent_class:*` KG nodes are removed.",
            "- QoS values are moved toward class-boundary or overlapping regions.",
            "- Hard negatives prioritize confusable classes, same-slice samples, overlapping functions, and close QoS profiles.",
            "",
            "## KG Statistics",
            "",
            f"- Average KG nodes per sample: {mean(kg_node_counts):.2f}",
            f"- Average KG edges per sample: {mean(kg_edge_counts):.2f}",
            f"- Average hard negatives per sample: {mean(hard_counts):.2f}",
            "",
            "## Recommended Experiments",
            "",
            "1. Classification on `test.jsonl`.",
            "2. Retrieval on `test.jsonl` with group-level relevance.",
            "3. Unseen-composition retrieval using `unseen_composition_test.jsonl` as query and train/val/test as gallery.",
            "4. Error analysis over confusable class pairs.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/processed/hard_v2")
    parser.add_argument("--output-dir", default="data/processed/challenge")
    parser.add_argument("--report", default="outputs/reports/kni_intent_challenge_report.md")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--heldout-compositions-per-class", type=int, default=20)
    args = parser.parse_args()

    source = ROOT / args.source_dir
    output = ROOT / args.output_dir
    records = []
    for split in ["train", "val", "test"]:
        records.extend(load_jsonl(source / f"{split}.jsonl"))

    challenge_records = [transform_record(record, args.seed) for record in records]
    attach_contrastive_metadata(challenge_records)
    heldout_groups = select_unseen_compositions(challenge_records, args.heldout_compositions_per_class, args.seed)
    unseen = [dict(record, split_tag="unseen_composition") for record in challenge_records if record["positive_group_id"] in heldout_groups]
    splits = split_standard(challenge_records, heldout_groups, args.seed)

    output.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        write_jsonl(output / f"{split}.jsonl", items)
    write_jsonl(output / "unseen_composition_test.jsonl", unseen)

    metadata = {
        "dataset": "KNI-Intent-Challenge",
        "source": str(source),
        "num_records_total": len(challenge_records),
        "standard_splits": {name: len(items) for name, items in splits.items()},
        "unseen_composition": len(unseen),
        "heldout_composition_groups": len(heldout_groups),
        "seed": args.seed,
    }
    (output / "challenge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    build_report(challenge_records, splits, unseen, ROOT / args.report)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
