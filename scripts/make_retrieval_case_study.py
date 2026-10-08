"""Create qualitative retrieval examples from saved embedding CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            records[item["intent_id"]] = item
    return records


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    ids = []
    vectors = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        z_fields = [field for field in reader.fieldnames or [] if field.startswith("z")]
        z_fields.sort(key=lambda name: int(name[1:]))
        for row in reader:
            ids.append(row["intent_id"])
            vectors.append([float(row[field]) for field in z_fields])
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return ids, matrix


def compact(text: str, limit: int = 118) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/processed/hard_v2/test.jsonl")
    parser.add_argument("--embeddings", default="outputs/formal_v2_retrieval/seed_1/test_group/query_embeddings.csv")
    parser.add_argument("--output", default="outputs/reports/formal_v2_retrieval_case_study.md")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    records = load_jsonl(Path(args.records))
    ids, vectors = load_embeddings(Path(args.embeddings))
    id_to_index = {intent_id: i for i, intent_id in enumerate(ids)}

    selected = []
    seen_classes = set()
    for intent_id in ids:
        item = records[intent_id]
        if item["intent_class"] not in seen_classes:
            selected.append(intent_id)
            seen_classes.add(item["intent_class"])
        if len(selected) >= args.queries:
            break

    lines = [
        "# Retrieval Case Study",
        "",
        "Embedding source: `outputs/formal_v2_retrieval/seed_1/test_group/query_embeddings.csv`.",
        "",
    ]

    for query_id in selected:
        query = records[query_id]
        q_index = id_to_index[query_id]
        scores = vectors @ vectors[q_index]
        ranking = np.argsort(-scores)
        ranking = [idx for idx in ranking if ids[idx] != query_id][: args.topk]

        lines.extend(
            [
                f"## Query: {query_id}",
                "",
                f"- Class: {query['intent_class']}",
                f"- Positive group: `{query['positive_group_id']}`",
                f"- Text: {compact(query['intent_text'])}",
                "",
                "| Rank | Intent ID | Similarity | Same Group | Same Class | Retrieved Text |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for rank, idx in enumerate(ranking, start=1):
            item = records[ids[idx]]
            same_group = item["positive_group_id"] == query["positive_group_id"]
            same_class = item["intent_class"] == query["intent_class"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        item["intent_id"],
                        f"{scores[idx]:.4f}",
                        "yes" if same_group else "no",
                        "yes" if same_class else "no",
                        compact(item["intent_text"]).replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.append("")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "queries": len(selected), "topk": args.topk}, indent=2))


if __name__ == "__main__":
    main()
