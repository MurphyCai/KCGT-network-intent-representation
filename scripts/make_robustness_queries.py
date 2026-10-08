"""Create perturbed query sets for robustness evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


QOS_FIELDS = ["latency_ms", "reliability", "bandwidth_mbps", "mobility_kmh", "jitter_ms"]
RELATION_FIELDS = ["requires", "belongs_to", "supported_by", "consumes"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clone(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record))


def text_dropout(records: list[dict[str, Any]], rng: random.Random, drop_prob: float) -> list[dict[str, Any]]:
    perturbed = []
    for record in records:
        item = clone(record)
        words = item["intent_text"].split()
        if len(words) > 4:
            kept = [word for word in words if rng.random() > drop_prob]
            if len(kept) < 3:
                kept = words[:3]
            item["intent_text"] = " ".join(kept)
        item["robustness_mode"] = "text_dropout"
        perturbed.append(item)
    return perturbed


def qos_noise(records: list[dict[str, Any]], rng: random.Random, noise: float) -> list[dict[str, Any]]:
    perturbed = []
    for record in records:
        item = clone(record)
        qos = item.get("qos", {})
        for field in QOS_FIELDS:
            if field in qos:
                factor = 1.0 + rng.uniform(-noise, noise)
                qos[field] = max(0.0, float(qos[field]) * factor)
        item["robustness_mode"] = "qos_noise"
        perturbed.append(item)
    return perturbed


def missing_qos(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    medians = {
        field: median(float(record.get("qos", {}).get(field, 0.0)) for record in records)
        for field in QOS_FIELDS
    }
    perturbed = []
    for record in records:
        item = clone(record)
        item["qos"] = dict(item.get("qos", {}))
        for field, value in medians.items():
            item["qos"][field] = value
        item["robustness_mode"] = "missing_qos"
        perturbed.append(item)
    return perturbed


def kg_edge_drop(records: list[dict[str, Any]], rng: random.Random, drop_prob: float) -> list[dict[str, Any]]:
    perturbed = []
    for record in records:
        item = clone(record)
        kg = item.get("kg_entities", {})
        for relation in RELATION_FIELDS:
            values = list(kg.get(relation, []))
            if len(values) > 1:
                kept = [value for value in values if rng.random() > drop_prob]
                kg[relation] = kept if kept else values[:1]
        item["robustness_mode"] = "kg_edge_drop"
        perturbed.append(item)
    return perturbed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/hard_v2/test.jsonl")
    parser.add_argument("--output-dir", default="data/processed/hard_v2_robustness")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--text-drop-prob", type=float, default=0.25)
    parser.add_argument("--qos-noise", type=float, default=0.15)
    parser.add_argument("--kg-drop-prob", type=float, default=0.30)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(Path(args.input))
    out_dir = Path(args.output_dir)

    outputs = {
        "text_dropout": text_dropout(records, rng, args.text_drop_prob),
        "qos_noise": qos_noise(records, rng, args.qos_noise),
        "missing_qos": missing_qos(records),
        "kg_edge_drop": kg_edge_drop(records, rng, args.kg_drop_prob),
    }
    for name, payload in outputs.items():
        write_jsonl(out_dir / f"{name}.jsonl", payload)

    print(json.dumps({"output_dir": str(out_dir), "modes": sorted(outputs), "records_per_mode": len(records)}, indent=2))


if __name__ == "__main__":
    main()
