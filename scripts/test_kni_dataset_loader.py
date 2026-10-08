"""Smoke test for the KNI-Intent PyTorch dataset loader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from datasets import KNIIntentDataset, build_artifacts, kni_collate_fn


def main() -> None:
    processed = ROOT / "data" / "processed"
    artifact_path = processed / "dataset_artifacts.json"
    paths = {
        "train": processed / "train.jsonl",
        "val": processed / "val.jsonl",
        "test": processed / "test.jsonl",
    }
    artifacts = build_artifacts(
        train_path=paths["train"],
        all_paths=[paths["train"], paths["val"], paths["test"]],
        max_text_len=40,
        min_token_freq=1,
    )
    artifacts.save(artifact_path)

    datasets = {name: KNIIntentDataset(path, artifacts) for name, path in paths.items()}
    loader = DataLoader(datasets["train"], batch_size=8, shuffle=False, collate_fn=kni_collate_fn)
    batch = next(iter(loader))
    summary = {
        "artifact_path": str(artifact_path),
        "num_train": len(datasets["train"]),
        "num_val": len(datasets["val"]),
        "num_test": len(datasets["test"]),
        "vocab_size": len(artifacts.token_to_id),
        "num_classes": len(artifacts.class_to_id),
        "num_slices": len(artifacts.slice_to_id),
        "num_kg_nodes": len(artifacts.node_to_id),
        "num_relations": len(artifacts.relation_to_id),
        "num_node_types": len(artifacts.node_type_to_id),
        "batch_text_ids_shape": list(batch["text_ids"].shape),
        "batch_attention_mask_shape": list(batch["attention_mask"].shape),
        "batch_qos_shape": list(batch["qos"].shape),
        "batch_y_shape": list(batch["y"].shape),
        "batch_slice_y_shape": list(batch["slice_y"].shape),
        "batch_graph_nodes": int(batch["graph"].num_nodes),
        "batch_graph_edges": int(batch["graph"].edge_index.shape[1]),
        "first_intent_id": batch["intent_id"][0],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
