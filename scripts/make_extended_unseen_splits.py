"""Create extended unseen splits for scenario, slice-stratified, and QoS-range evaluation."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import quantiles
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scenario_split(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    by_class: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_class[record["intent_class"]].add(record["scenario"])
    heldout = {cls: sorted(values)[:1] for cls, values in by_class.items()}
    heldout_pairs = {(cls, scenario) for cls, scenarios in heldout.items() for scenario in scenarios}
    test = [record for record in records if (record["intent_class"], record["scenario"]) in heldout_pairs]
    train = [record for record in records if (record["intent_class"], record["scenario"]) not in heldout_pairs]
    return train, test, heldout


def slice_stratified_split(records: list[dict[str, Any]], seed: int, ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    by_slice_group: dict[str, set[str]] = defaultdict(set)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_slice_group[record["slice_type"]].add(record["positive_group_id"])
        by_group[record["positive_group_id"]].append(record)

    heldout_groups = set()
    counts = {}
    for slice_type, groups in sorted(by_slice_group.items()):
        group_list = sorted(groups)
        rng = random.Random(f"{seed}-{slice_type}")
        rng.shuffle(group_list)
        n_test = max(1, int(len(group_list) * ratio))
        chosen = set(group_list[:n_test])
        heldout_groups.update(chosen)
        counts[slice_type] = len(chosen)

    test = [record for group in sorted(heldout_groups) for record in by_group[group]]
    train = [record for record in records if record["positive_group_id"] not in heldout_groups]
    return train, test, counts


def qos_range_split(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    thresholds = {}
    heldout = []
    train = []
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[record["intent_class"]].append(record)

    for cls, items in by_class.items():
        latency_values = sorted(float(item["qos_latency_ms"]) for item in items)
        bandwidth_values = sorted(float(item["qos_bandwidth_mbps"]) for item in items)
        lat_q = quantiles(latency_values, n=10)
        bw_q = quantiles(bandwidth_values, n=10)
        lat_hi = lat_q[8]
        bw_hi = bw_q[8]
        thresholds[cls] = {"latency_p90": lat_hi, "bandwidth_p90": bw_hi}
        for item in items:
            if float(item["qos_latency_ms"]) >= lat_hi or float(item["qos_bandwidth_mbps"]) >= bw_hi:
                heldout.append(item)
            else:
                train.append(item)
    return train, heldout, thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/hard_v2")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--slice-ratio", type=float, default=0.2)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    records = []
    for split in ["train", "val", "test"]:
        records.extend(load_jsonl(processed / f"{split}.jsonl"))

    unseen_dir = processed / "unseen"
    scenario_train, scenario_test, heldout_scenarios = scenario_split(records)
    slice_train, slice_test, heldout_slice_counts = slice_stratified_split(records, args.seed, args.slice_ratio)
    qos_train, qos_test, qos_thresholds = qos_range_split(records)

    write_jsonl(unseen_dir / "scenario_train.jsonl", scenario_train)
    write_jsonl(unseen_dir / "scenario_test.jsonl", scenario_test)
    write_jsonl(unseen_dir / "slice_train.jsonl", slice_train)
    write_jsonl(unseen_dir / "slice_test.jsonl", slice_test)
    write_jsonl(unseen_dir / "qos_range_train.jsonl", qos_train)
    write_jsonl(unseen_dir / "qos_range_test.jsonl", qos_test)

    summary = {
        "scenario": {"train": len(scenario_train), "test": len(scenario_test), "heldout": heldout_scenarios},
        "slice": {"train": len(slice_train), "test": len(slice_test), "heldout_groups_by_slice": heldout_slice_counts},
        "qos_range": {"train": len(qos_train), "test": len(qos_test), "thresholds": qos_thresholds},
    }
    (unseen_dir / "extended_unseen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
