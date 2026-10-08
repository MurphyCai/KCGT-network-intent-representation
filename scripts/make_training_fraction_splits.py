"""Create group-preserving training-fraction splits from a processed KNI dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build training-fraction splits.")
    parser.add_argument("--processed-dir", default="data/processed/hard_v2")
    parser.add_argument("--output-root", default="data/processed/hard_v2_fractions")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.processed_dir)
    out_root = Path(args.output_root)
    train = read_jsonl(src / "train.jsonl")
    groups = defaultdict(list)
    for record in train:
        groups[str(record["positive_group_id"])].append(record)
    group_ids = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)

    summary = []
    for frac in args.fractions:
        tag = f"frac_{int(frac * 100):03d}"
        out = out_root / tag
        n_groups = max(1, int(round(len(group_ids) * frac)))
        selected = set(group_ids[:n_groups])
        subset = [record for gid in group_ids if gid in selected for record in groups[gid]]
        write_jsonl(out / "train.jsonl", subset)
        shutil.copyfile(src / "val.jsonl", out / "val.jsonl")
        shutil.copyfile(src / "test.jsonl", out / "test.jsonl")
        for subdir in ["few_shot", "unseen"]:
            if (src / subdir).exists() and not (out / subdir).exists():
                shutil.copytree(src / subdir, out / subdir)
        summary.append({"fraction": frac, "groups": n_groups, "samples": len(subset), "dir": str(out)})
    (out_root / "fraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
