"""Generate a harder KNI-Intent dataset split.

Compared with the pilot generator, this version adds:

- true paraphrase groups sharing one base intent configuration;
- more paraphrases per base intent;
- hard negative metadata for contrastive learning;
- group-disjoint standard split;
- unseen-application and unseen-template splits;
- few-shot train splits.

Outputs are kept separate from the pilot dataset:

- data/raw/kni_intent_hard_raw.jsonl
- data/processed/hard/*.jsonl
- data/kg/hard/*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_kni_intent_dataset import (  # noqa: E402
    INTENT_CLASSES,
    QOS_PHRASES,
    build_edges,
    collect_global_kg,
    edge_nodes,
    flatten_for_csv,
    format_qos_phrase,
    sample_float,
    sample_int,
    text_name,
    write_csv,
    write_jsonl,
)


SEED = 20260711
BASE_INTENTS_PER_CLASS = 60
PARAPHRASES_PER_BASE = 6
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HELDOUT_TEMPLATE_IDS = {"HT07", "HT08"}
FEW_SHOT_K = [1, 5, 10, 20]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / "hard"
KG_DIR = ROOT / "data" / "kg" / "hard"


HARD_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    ("HT01", "{application} needs {qos_phrase} communication in {scenario}."),
    ("HT02", "The service should keep {qos_phrase} for {application} traffic."),
    ("HT03", "Support {application} in {scenario} with {function_phrase} and {resource_phrase}."),
    ("HT04", "The network must satisfy {qos_phrase} while serving {application}."),
    ("HT05", "Provide a managed network service for {application} requiring {qos_phrase}."),
    ("HT06", "{scenario} requests reliable service continuity for {application} using {function_phrase}."),
    ("HT07", "For {application}, allocate resources to meet {qos_phrase} under changing network load."),
    ("HT08", "Guarantee service assurance for {application} by combining {function_phrase} with {resource_phrase}."),
    ("HT09", "The intent is to maintain {qos_phrase} for {application} without manual configuration."),
    ("HT10", "A network slice should be selected for {application} according to {qos_phrase}."),
    ("HT11", "Optimize service delivery for {application} in {scenario} with {resource_phrase}."),
    ("HT12", "Ensure application traffic receives {qos_phrase} and suitable management functions."),
)


CONFLICT_CLASS_PAIRS = {
    ("Industrial URLLC", "Massive IoT mMTC"),
    ("Autonomous Driving / V2X", "Massive IoT mMTC"),
    ("Security-Critical Enterprise", "Video Streaming eMBB"),
    ("Telemedicine", "Video Streaming eMBB"),
    ("Telemedicine", "Cloud Gaming"),
    ("AR/VR Immersive Service", "Cloud Gaming"),
}


def choose_many(rng: random.Random, values: Sequence[str], minimum: int, maximum: int) -> List[str]:
    maximum = min(maximum, len(values))
    return sorted(rng.sample(list(values), rng.randint(minimum, maximum)))


def sample_base(rng: random.Random, spec) -> Dict[str, object]:
    qos = {}
    for key, (lo, hi) in spec.qos_ranges.items():
        if key in {"security_level", "mobility", "energy_sensitivity"}:
            qos[key] = sample_int(rng, lo, hi)
        elif key in {"packet_loss", "reliability"}:
            qos[key] = sample_float(rng, lo, hi, digits=8)
        else:
            qos[key] = sample_float(rng, lo, hi, digits=3)
    return {
        "application": rng.choice(spec.applications),
        "scenario": rng.choice(spec.scenarios),
        "functions": choose_many(rng, spec.functions, 3, 6),
        "resources": choose_many(rng, spec.resources, 2, 4),
        "qos_nodes": choose_many(rng, spec.qos_nodes, 2, min(4, len(spec.qos_nodes))),
        "qos": qos,
    }


def make_text(rng: random.Random, template_id: str, template: str, spec, base: Mapping[str, object]) -> str:
    qos_phrase = format_qos_phrase(rng, base["qos_nodes"])
    functions = list(base["functions"])
    resources = list(base["resources"])
    return template.format(
        application=text_name(str(base["application"])),
        scenario=text_name(str(base["scenario"])),
        qos_phrase=qos_phrase,
        function_phrase=", ".join(text_name(v) for v in functions[:2]),
        resource_phrase=", ".join(text_name(v) for v in resources[:2]),
        slice_type=spec.primary_slice,
        slice_phrase=f"{spec.primary_slice} slice",
        template_id=template_id,
    )


def make_record(spec, base: Mapping[str, object], base_idx: int, para_idx: int, template_id: str, template: str, seed: int = SEED) -> Dict[str, object]:
    rng = random.Random(f"{seed}-{spec.class_id}-{base_idx}-{para_idx}-{template_id}")
    edges = build_edges(
        spec,
        str(base["application"]),
        str(base["scenario"]),
        list(base["functions"]),
        list(base["resources"]),
        list(base["qos_nodes"]),
    )
    positive_group_id = f"HARD_{spec.class_id}_B{base_idx:03d}"
    record = {
        "intent_id": f"{positive_group_id}_P{para_idx:02d}",
        "intent_text": make_text(rng, template_id, template, spec, base),
        "intent_class": spec.name,
        "class_id": spec.class_id,
        "slice_type": spec.primary_slice,
        "auxiliary_slice_type": list(spec.auxiliary_slices),
        "application": base["application"],
        "scenario": base["scenario"],
        "qos_latency_ms": base["qos"]["latency_ms"],
        "qos_bandwidth_mbps": base["qos"]["bandwidth_mbps"],
        "qos_reliability": base["qos"]["reliability"],
        "qos_jitter_ms": base["qos"]["jitter_ms"],
        "qos_packet_loss": base["qos"]["packet_loss"],
        "qos_security_level": base["qos"]["security_level"],
        "qos_mobility": base["qos"]["mobility"],
        "qos_energy_sensitivity": base["qos"]["energy_sensitivity"],
        "network_functions": list(base["functions"]),
        "resources": list(base["resources"]),
        "kg_nodes": edge_nodes(edges),
        "kg_edges": edges,
        "positive_group_id": positive_group_id,
        "source_template_id": template_id,
        "split_tag": "",
        "positive_ids": [],
        "hard_negative_ids": [],
        "easy_negative_ids": [],
    }
    return record


def generate_records(base_intents_per_class: int = BASE_INTENTS_PER_CLASS, paraphrases_per_base: int = PARAPHRASES_PER_BASE, seed: int = SEED) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in INTENT_CLASSES:
        for base_idx in range(1, base_intents_per_class + 1):
            base_rng = random.Random(f"{seed}-{spec.class_id}-{base_idx}")
            base = sample_base(base_rng, spec)
            template_rng = random.Random(f"{seed}-templates-{spec.class_id}-{base_idx}")
            selected_templates = template_rng.sample(list(HARD_TEMPLATES), paraphrases_per_base)
            for para_idx, (template_id, template) in enumerate(selected_templates, start=1):
                records.append(make_record(spec, base, base_idx, para_idx, template_id, template, seed=seed))
    attach_contrastive_metadata(records)
    return records


def class_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def function_overlap(a: Mapping[str, object], b: Mapping[str, object]) -> int:
    return len(set(a["network_functions"]) & set(b["network_functions"]))


def attach_contrastive_metadata(records: List[Dict[str, object]]) -> None:
    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_class: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_slice: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_group[str(record["positive_group_id"])].append(record)
        by_class[str(record["intent_class"])].append(record)
        by_slice[str(record["slice_type"])].append(record)

    conflict_lookup = {class_pair(*p) for p in CONFLICT_CLASS_PAIRS}

    for record in records:
        positives = [r["intent_id"] for r in by_group[str(record["positive_group_id"])] if r["intent_id"] != record["intent_id"]]
        record["positive_ids"] = positives[:5]

        hard = []
        easy = []
        rng = random.Random(f"candidate-{record['intent_id']}")
        candidate_pool = []
        for cls, cls_records in by_class.items():
            if class_pair(str(record["intent_class"]), cls) in conflict_lookup:
                candidate_pool.extend(cls_records)
        candidate_pool.extend(by_slice[str(record["slice_type"])])
        candidate_pool.extend(rng.sample(records, min(800, len(records))))

        seen = set()
        unique_candidates = []
        for candidate in candidate_pool:
            cid = str(candidate["intent_id"])
            if cid not in seen:
                seen.add(cid)
                unique_candidates.append(candidate)

        for candidate in unique_candidates:
            if candidate["intent_id"] == record["intent_id"]:
                continue
            same_class = candidate["intent_class"] == record["intent_class"]
            same_group = candidate["positive_group_id"] == record["positive_group_id"]
            if same_class or same_group:
                continue
            conflict = class_pair(str(record["intent_class"]), str(candidate["intent_class"])) in conflict_lookup
            same_slice = candidate["slice_type"] == record["slice_type"]
            overlap = function_overlap(record, candidate)
            if conflict or (same_slice and overlap >= 2):
                hard.append(str(candidate["intent_id"]))
            elif overlap == 0 or candidate["slice_type"] != record["slice_type"]:
                easy.append(str(candidate["intent_id"]))
            if len(hard) >= 40 and len(easy) >= 60:
                break
        hard_rng = random.Random(f"hard-{record['intent_id']}")
        easy_rng = random.Random(f"easy-{record['intent_id']}")
        hard_rng.shuffle(hard)
        easy_rng.shuffle(easy)
        record["hard_negative_ids"] = hard[:5]
        record["easy_negative_ids"] = easy[:10]


def split_groups(records: List[Dict[str, object]], seed: int = SEED) -> Dict[str, List[Dict[str, object]]]:
    groups = sorted({str(r["positive_group_id"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    train_end = int(len(groups) * TRAIN_RATIO)
    val_end = int(len(groups) * (TRAIN_RATIO + VAL_RATIO))
    group_split = {g: "train" for g in groups[:train_end]}
    group_split.update({g: "val" for g in groups[train_end:val_end]})
    group_split.update({g: "test" for g in groups[val_end:]})
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        split = group_split[str(record["positive_group_id"])]
        item = dict(record)
        item["split_tag"] = split
        splits[split].append(item)
    return splits


def write_kg(records: Sequence[Dict[str, object]], kg_dir: Path = KG_DIR) -> None:
    nodes, edges = collect_global_kg(records)
    kg_dir.mkdir(parents=True, exist_ok=True)
    with (kg_dir / "kg_nodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "node_type", "node_name"])
        writer.writeheader()
        for node_id, ntype in nodes:
            writer.writerow({"node_id": node_id, "node_type": ntype, "node_name": node_id.split(":", 1)[1]})
    with (kg_dir / "kg_edges.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "relation", "target"])
        writer.writeheader()
        for src, rel, dst in edges:
            writer.writerow({"source": src, "relation": rel, "target": dst})
    (kg_dir / "relation_types.json").write_text(json.dumps(sorted({rel for _, rel, _ in edges}), indent=2), encoding="utf-8")
    (kg_dir / "node_types.json").write_text(json.dumps(sorted({ntype for _, ntype in nodes}), indent=2), encoding="utf-8")


def write_unseen_splits(records: Sequence[Dict[str, object]], processed_dir: Path = PROCESSED_DIR) -> None:
    unseen_dir = processed_dir / "unseen"
    unseen_dir.mkdir(parents=True, exist_ok=True)

    heldout_apps = {"remote_surgery", "vehicle_platooning", "ar_assistance", "enterprise_vpn", "smart_metering", "video_streaming", "cloud_game_streaming", "industrial_robot"}
    app_train = [r for r in records if r["application"] not in heldout_apps]
    app_test = [r for r in records if r["application"] in heldout_apps]
    write_jsonl(unseen_dir / "application_train.jsonl", app_train)
    write_jsonl(unseen_dir / "application_test.jsonl", app_test)

    template_train = [r for r in records if r["source_template_id"] not in HELDOUT_TEMPLATE_IDS]
    template_test = [r for r in records if r["source_template_id"] in HELDOUT_TEMPLATE_IDS]
    write_jsonl(unseen_dir / "template_train.jsonl", template_train)
    write_jsonl(unseen_dir / "template_test.jsonl", template_test)


def write_few_shot_splits(records: Sequence[Dict[str, object]], processed_dir: Path = PROCESSED_DIR, seed: int = SEED) -> None:
    few_dir = processed_dir / "few_shot"
    few_dir.mkdir(parents=True, exist_ok=True)
    by_class_group: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class_group[str(record["intent_class"])][str(record["positive_group_id"])].append(record)

    for k in FEW_SHOT_K:
        sampled = []
        for cls, groups in by_class_group.items():
            group_ids = sorted(groups)
            rng = random.Random(f"{seed}-few-{k}-{cls}")
            rng.shuffle(group_ids)
            count = 0
            for group_id in group_ids:
                if count >= k:
                    break
                sampled.append(groups[group_id][0])
                count += 1
        write_jsonl(few_dir / f"k{k}_train.jsonl", sampled)


def summarize(records: Sequence[Dict[str, object]], splits: Mapping[str, Sequence[Dict[str, object]]], dataset_name: str = "KNI-Intent-Hard", paraphrases_per_base: int = PARAPHRASES_PER_BASE) -> Dict[str, object]:
    nodes, edges = collect_global_kg(records)
    return {
        "dataset": dataset_name,
        "num_samples": len(records),
        "num_classes": len({r["intent_class"] for r in records}),
        "num_positive_groups": len({r["positive_group_id"] for r in records}),
        "paraphrases_per_base": paraphrases_per_base,
        "splits": {name: len(items) for name, items in splits.items()},
        "groups_per_split": {name: len({r["positive_group_id"] for r in items}) for name, items in splits.items()},
        "heldout_template_ids": sorted(HELDOUT_TEMPLATE_IDS),
        "avg_positive_ids": round(sum(len(r["positive_ids"]) for r in records) / len(records), 2),
        "avg_hard_negative_ids": round(sum(len(r["hard_negative_ids"]) for r in records) / len(records), 2),
        "avg_easy_negative_ids": round(sum(len(r["easy_negative_ids"]) for r in records) / len(records), 2),
        "num_kg_nodes": len(nodes),
        "num_kg_edges": len(edges),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KNI-Intent-Hard or a larger formal variant.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--base-intents-per-class", type=int, default=BASE_INTENTS_PER_CLASS)
    parser.add_argument("--paraphrases-per-base", type=int, default=PARAPHRASES_PER_BASE)
    parser.add_argument("--name", default="KNI-Intent-Hard")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--kg-dir", default=str(KG_DIR))
    parser.add_argument("--raw-prefix", default="kni_intent_hard")
    parser.add_argument("--config-suffix", default="hard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    kg_dir = Path(args.kg_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    records = generate_records(args.base_intents_per_class, args.paraphrases_per_base, args.seed)
    splits = split_groups(records, args.seed)

    write_jsonl(raw_dir / f"{args.raw_prefix}_raw.jsonl", records)
    write_csv(raw_dir / f"{args.raw_prefix}_raw.csv", records)
    for name, items in splits.items():
        write_jsonl(processed_dir / f"{name}.jsonl", items)
        write_csv(processed_dir / f"{name}.csv", items)
    write_unseen_splits(records, processed_dir)
    write_few_shot_splits(records, processed_dir, args.seed)
    write_kg(records, kg_dir)

    config = {
        "dataset": args.name,
        "seed": args.seed,
        "base_intents_per_class": args.base_intents_per_class,
        "paraphrases_per_base": args.paraphrases_per_base,
        "templates": [tid for tid, _ in HARD_TEMPLATES],
        "heldout_template_ids": sorted(HELDOUT_TEMPLATE_IDS),
        "split_mode": "group-disjoint",
        "unseen_modes": ["application", "template"],
        "few_shot_k": FEW_SHOT_K,
    }
    (raw_dir / f"generation_config_{args.config_suffix}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = summarize(records, splits, args.name, args.paraphrases_per_base)
    (raw_dir / f"dataset_summary_{args.config_suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
