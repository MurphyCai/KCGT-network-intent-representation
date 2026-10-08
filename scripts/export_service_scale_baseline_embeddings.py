"""Export test-set embeddings for service-template scale baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import KNIIntentArtifacts, KNIIntentDataset, kni_collate_fn
from models import build_baseline_model
from models.baselines import GraphOnlyClassifier, TextGraphFusionClassifier
from train_pretrained_text_baseline import encode_sbert, load_jsonl


BASELINE_DISPLAY = {
    "text_gat": "Text+GAT",
    "text_graph_transformer": "Text+GraphTransformer",
    "graph_transformer": "GraphTransformer",
}


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if key == "graph":
            moved[key] = value.to(device)
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    artifacts = KNIIntentArtifacts.from_dict(checkpoint["artifacts"])
    args = checkpoint["args"]
    model = build_baseline_model(
        model_name=args["model"],
        vocab_size=len(artifacts.token_to_id),
        num_classes=len(artifacts.class_to_id),
        pad_idx=artifacts.token_to_id["<pad>"],
        qos_dim=len(artifacts.qos_stats),
        embed_dim=int(args.get("embed_dim", 64)),
        hidden_dim=int(args.get("hidden_dim", 96)),
        dropout=float(args.get("dropout", 0.2)),
        max_text_len=int(args.get("max_text_len", artifacts.max_text_len)),
        node_feat_dim=len(artifacts.node_type_to_id),
        num_relations=len(artifacts.relation_to_id),
        num_kg_nodes=len(artifacts.node_to_id),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, artifacts


def baseline_features(model, batch: dict[str, object]) -> torch.Tensor:
    inner = model.model
    if isinstance(inner, TextGraphFusionClassifier):
        text_z = inner.text_encoder(batch["text_ids"], batch["attention_mask"])
        graph_z = inner.graph_encoder(batch["graph"])
        fused = torch.cat([text_z, graph_z], dim=-1)
        return torch.relu(inner.classifier[0](fused))
    if isinstance(inner, GraphOnlyClassifier):
        return inner.encoder(batch["graph"])
    raise TypeError(f"Unsupported service-scale baseline: {inner.__class__.__name__}")


def write_embeddings(path: Path, records: list[dict], embedding: np.ndarray, class_to_id: dict[str, int]) -> None:
    payload: dict[str, object] = {
        "intent_id": [str(r["intent_id"]) for r in records],
        "label": [int(class_to_id[str(r["intent_class"])]) for r in records],
        "positive_group_id": [str(r["positive_group_id"]) for r in records],
    }
    for index in range(embedding.shape[1]):
        payload[f"z{index}"] = embedding[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload).to_csv(path, index=False)


def export_neural(
    processed_dir: Path,
    checkpoint_root: Path,
    output_root: Path,
    models: list[str],
    seeds: list[int],
    batch_size: int,
    device: torch.device,
) -> None:
    records = load_jsonl(processed_dir / "test.jsonl")
    for seed in seeds:
        for model_name in models:
            checkpoint_path = checkpoint_root / f"seed_{seed}" / "baselines" / model_name / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            model, artifacts = checkpoint_model(checkpoint_path, device)
            dataset = KNIIntentDataset(processed_dir / "test.jsonl", artifacts)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
            chunks = []
            with torch.no_grad():
                for batch in loader:
                    batch = move_batch(batch, device)
                    chunks.append(baseline_features(model, batch).cpu().numpy())
            embedding = normalize(np.concatenate(chunks, axis=0).astype(np.float32))
            target = output_root / f"seed_{seed}" / model_name / "test_group_projection" / "query_embeddings.csv"
            write_embeddings(target, records, embedding, artifacts.class_to_id)
            print(f"[exported] {BASELINE_DISPLAY[model_name]} seed={seed} shape={embedding.shape} -> {target}")


def export_sbert(
    processed_dir: Path,
    output_root: Path,
    seeds: list[int],
    batch_size: int,
    model_name: str,
) -> None:
    records = load_jsonl(processed_dir / "test.jsonl")
    texts = [str(r["intent_text"]) for r in records]
    embedding = normalize(encode_sbert(texts, model_name, batch_size))
    labels = sorted({str(r["intent_class"]) for r in records})
    class_to_id = {label: index for index, label in enumerate(labels)}
    for seed in seeds:
        target = output_root / f"seed_{seed}" / "sbert_minilm" / "test_group_projection" / "query_embeddings.csv"
        write_embeddings(target, records, embedding, class_to_id)
        print(f"[exported] SBERT-MiniLM seed={seed} shape={embedding.shape} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--checkpoint-root", default="outputs/step30_neutral_kg_baselines")
    parser.add_argument("--output-root", default="outputs/service_template_scale_baseline_embeddings")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["text_gat", "text_graph_transformer", "graph_transformer"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_root = Path(args.output_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not args.skip_neural:
        export_neural(
            processed_dir,
            Path(args.checkpoint_root),
            output_root,
            args.models,
            args.seeds,
            args.batch_size,
            device,
        )
    if not args.skip_sbert:
        export_sbert(processed_dir, output_root, args.seeds, args.batch_size, args.sbert_model)

    manifest_path = output_root / "manifest.json"
    previous_models: list[str] = []
    if manifest_path.exists():
        previous_models = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("models", []))
    exported_models = ([] if args.skip_neural else args.models) + ([] if args.skip_sbert else ["sbert_minilm"])
    manifest = {
        "processed_dir": str(processed_dir),
        "checkpoint_root": args.checkpoint_root,
        "models": sorted(set(previous_models + exported_models)),
        "seeds": args.seeds,
        "embedding_definition": {
            "text_graph": "ReLU output of the first fusion-classifier linear layer",
            "graph_transformer": "pooled graph encoder output",
            "sbert": "L2-normalized all-MiniLM-L6-v2 sentence embedding",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
