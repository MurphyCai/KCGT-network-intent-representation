"""Evaluate KCGT embedding quality with retrieval metrics and t-SNE."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import KNIIntentArtifacts, KNIIntentDataset, kni_collate_fn
from experiments.train_baseline_classification import move_batch
from models import build_kcgt_model


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    artifacts = KNIIntentArtifacts.from_dict(checkpoint["artifacts"])
    args = checkpoint.get("args", {})
    model = build_kcgt_model(
        vocab_size=len(artifacts.token_to_id),
        num_classes=len(artifacts.class_to_id),
        pad_idx=artifacts.token_to_id["<pad>"],
        qos_dim=len(artifacts.qos_stats),
        node_feat_dim=len(artifacts.node_type_to_id),
        num_relations=len(artifacts.relation_to_id),
        num_kg_nodes=len(artifacts.node_to_id),
        graph_model=args.get("graph_model", args.get("kcgt_graph_model", "rgcn")),
        graph_heads=int(args.get("graph_heads", 4)),
        relation_dim=int(args.get("relation_dim", 32)),
        text_dim=int(args.get("embed_dim", 64)),
        graph_dim=int(args.get("hidden_dim", 96)),
        fusion_dim=int(args.get("fusion_dim", max(int(args.get("hidden_dim", 96)), int(args.get("embed_dim", 64))) * 2)),
        projection_dim=int(args.get("projection_dim", 64)),
        dropout=float(args.get("dropout", 0.2)),
        use_graph=not bool(args.get("disable_graph", False)),
        use_qos=not bool(args.get("disable_qos", False)),
        fusion_type=args.get("fusion_type", "gated"),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, artifacts, args


def extract_embeddings(
    model,
    artifacts,
    jsonl_path: Path,
    batch_size: int,
    device: torch.device,
    use_projection: bool,
    relation_mode: str = "typed",
    relation_seed: int = 0,
):
    dataset = KNIIntentDataset(jsonl_path, artifacts, relation_mode=relation_mode, relation_seed=relation_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    embeddings = []
    labels = []
    groups = []
    ids = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            encoded = model.encode(batch)
            z = encoded["projection"] if use_projection else encoded["embedding"]
            z = torch.nn.functional.normalize(z, dim=-1)
            embeddings.append(z.cpu())
            labels.extend(batch["y"].cpu().tolist())
            groups.extend(batch["positive_group_id"])
            ids.extend(batch["intent_id"])
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "labels": labels,
        "groups": groups,
        "ids": ids,
    }


def dcg(relevances: List[int]) -> float:
    return sum(rel / torch.log2(torch.tensor(idx + 2.0)).item() for idx, rel in enumerate(relevances))


def retrieval_metrics(query, gallery, relevance: str, ks=(1, 5, 10)) -> Dict[str, float]:
    q = query["embeddings"]
    g = gallery["embeddings"]
    scores = q @ g.t()
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"precision@{k}": 0.0 for k in ks})
    mrr = 0.0
    ndcg5 = 0.0
    valid = 0
    same_set = query["ids"] == gallery["ids"]
    for i in range(q.size(0)):
        order = torch.argsort(scores[i], descending=True).tolist()
        relevant = []
        for j in order:
            if same_set and query["ids"][i] == gallery["ids"][j]:
                continue
            if relevance == "group":
                is_rel = query["groups"][i] == gallery["groups"][j]
            elif relevance == "class":
                is_rel = query["labels"][i] == gallery["labels"][j]
            else:
                is_rel = (query["groups"][i] == gallery["groups"][j]) or (query["labels"][i] == gallery["labels"][j])
            relevant.append(1 if is_rel else 0)
        if not any(relevant):
            continue
        valid += 1
        for k in ks:
            topk = relevant[:k]
            metrics[f"recall@{k}"] += 1.0 if any(topk) else 0.0
            metrics[f"precision@{k}"] += sum(topk) / k
        first = next(idx for idx, rel in enumerate(relevant) if rel)
        mrr += 1.0 / (first + 1)
        ideal_count = min(5, sum(relevant))
        ndcg5 += dcg(relevant[:5]) / max(1e-12, dcg([1] * ideal_count))
    if valid == 0:
        return {**metrics, "mrr": 0.0, "ndcg@5": 0.0, "valid_queries": 0}
    for key in list(metrics):
        metrics[key] = round(metrics[key] / valid, 6)
    metrics["mrr"] = round(mrr / valid, 6)
    metrics["ndcg@5"] = round(ndcg5 / valid, 6)
    metrics["valid_queries"] = valid
    return metrics


def save_tsne(data, output_path: Path, title: str) -> None:
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        print(f"Skipping t-SNE because sklearn is unavailable: {exc}")
        return
    x = data["embeddings"].numpy()
    labels = data["labels"]
    if len(x) > 600:
        x = x[:600]
        labels = labels[:600]
    perplexity = min(30, max(5, len(x) // 10))
    coords = TSNE(n_components=2, random_state=20260710, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(x)
    plt.figure(figsize=(7.5, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.85)
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.grid(alpha=0.2)
    plt.colorbar(scatter, label="Intent class id")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_embedding_csv(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        dim = data["embeddings"].shape[1]
        writer.writerow(["intent_id", "label", "positive_group_id"] + [f"z{i}" for i in range(dim)])
        for intent_id, label, group, emb in zip(data["ids"], data["labels"], data["groups"], data["embeddings"].tolist()):
            writer.writerow([intent_id, label, group] + emb)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KCGT retrieval quality.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery", default=str(ROOT / "data" / "processed" / "hard" / "test.jsonl"))
    parser.add_argument("--query", default=str(ROOT / "data" / "processed" / "hard" / "test.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "retrieval" / "kcgt_contrastive_hard"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--relevance", choices=["group", "class", "group_or_class"], default="group")
    parser.add_argument("--use-embedding", action="store_true", help="Use fused embedding instead of projection head.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tsne", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, artifacts, train_args = load_model(Path(args.checkpoint), device)
    relation_mode = train_args.get("relation_mode", "typed")
    relation_seed = int(train_args.get("seed", 0))
    query = extract_embeddings(model, artifacts, Path(args.query), args.batch_size, device, not args.use_embedding, relation_mode, relation_seed)
    gallery = extract_embeddings(model, artifacts, Path(args.gallery), args.batch_size, device, not args.use_embedding, relation_mode, relation_seed)
    metrics = retrieval_metrics(query, gallery, args.relevance)
    payload = {
        "checkpoint": str(args.checkpoint),
        "query": str(args.query),
        "gallery": str(args.gallery),
        "relevance": args.relevance,
        "embedding_type": "fused" if args.use_embedding else "projection",
        "metrics": metrics,
    }
    (output_dir / "retrieval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_embedding_csv(query, output_dir / "query_embeddings.csv")
    if args.tsne:
        save_tsne(query, output_dir / "tsne_query_embeddings.png", "KCGT Intent Embedding t-SNE")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
