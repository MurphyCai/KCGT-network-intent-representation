"""Train CPU-friendly baseline classifiers on KNI-Intent."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datasets import KNIIntentArtifacts, KNIIntentDataset, build_artifacts, kni_collate_fn
from models import build_baseline_model, build_kcgt_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if key == "graph":
            moved[key] = value.to(device)
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def macro_scores(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
    return {
        "accuracy": round(accuracy, 6),
        "precision_macro": round(sum(precisions) / num_classes, 6),
        "recall_macro": round(sum(recalls) / num_classes, 6),
        "f1_macro": round(sum(f1s) / num_classes, 6),
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["y"])
            total_loss += float(loss.item()) * int(batch["y"].numel())
            total_count += int(batch["y"].numel())
            pred = logits.argmax(dim=-1)
            y_true.extend(batch["y"].detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
    elapsed = time.perf_counter() - start
    scores = macro_scores(y_true, y_pred, num_classes)
    scores["loss"] = round(total_loss / max(1, total_count), 6)
    scores["inference_time_ms_per_sample"] = round(elapsed * 1000 / max(1, total_count), 6)
    return scores


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch["y"].numel())
        total_count += int(batch["y"].numel())
    return total_loss / max(1, total_count)


def load_or_build_artifacts(processed_dir: Path, max_text_len: int) -> KNIIntentArtifacts:
    artifact_path = processed_dir / "dataset_artifacts.json"
    if artifact_path.exists():
        artifacts = KNIIntentArtifacts.load(artifact_path)
        if artifacts.max_text_len == max_text_len:
            return artifacts
    artifacts = build_artifacts(
        train_path=processed_dir / "train.jsonl",
        all_paths=[processed_dir / "train.jsonl", processed_dir / "val.jsonl", processed_dir / "test.jsonl"],
        max_text_len=max_text_len,
    )
    artifacts.save(artifact_path)
    return artifacts


def make_loaders(processed_dir: Path, artifacts: KNIIntentArtifacts, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = KNIIntentDataset(processed_dir / "train.jsonl", artifacts)
    val_ds = KNIIntentDataset(processed_dir / "val.jsonl", artifacts)
    test_ds = KNIIntentDataset(processed_dir / "test.jsonl", artifacts)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=kni_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KNI-Intent baseline classifier.")
    parser.add_argument(
        "--model",
        choices=[
            "textcnn",
            "lstm",
            "transformer",
            "qos_mlp",
            "gcn",
            "gat",
            "rgcn",
            "graph_transformer",
            "text_gcn",
            "text_gat",
            "text_rgcn",
            "text_graph_transformer",
            "hypergraph",
            "text_hypergraph",
            "kcgt",
        ],
        default="textcnn",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-text-len", type=int, default=40)
    parser.add_argument(
        "--kcgt-graph-model",
        choices=["relation_transformer", "gcn", "gat", "rgcn"],
        default="relation_transformer",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "baselines"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_or_build_artifacts(processed_dir, args.max_text_len)
    train_loader, val_loader, test_loader = make_loaders(processed_dir, artifacts, args.batch_size)
    if args.model == "kcgt":
        model = build_kcgt_model(
            vocab_size=len(artifacts.token_to_id),
            num_classes=len(artifacts.class_to_id),
            pad_idx=artifacts.token_to_id["<pad>"],
            qos_dim=len(artifacts.qos_stats),
            node_feat_dim=len(artifacts.node_type_to_id),
            num_relations=len(artifacts.relation_to_id),
            num_kg_nodes=len(artifacts.node_to_id),
            graph_model=args.kcgt_graph_model,
            text_dim=args.embed_dim,
            graph_dim=args.hidden_dim,
            fusion_dim=max(args.hidden_dim, args.embed_dim) * 2,
            dropout=args.dropout,
        ).to(device)
    else:
        model = build_baseline_model(
            model_name=args.model,
            vocab_size=len(artifacts.token_to_id),
            num_classes=len(artifacts.class_to_id),
            pad_idx=artifacts.token_to_id["<pad>"],
            qos_dim=len(artifacts.qos_stats),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            max_text_len=args.max_text_len,
            node_feat_dim=len(artifacts.node_type_to_id),
            num_relations=len(artifacts.relation_to_id),
            num_kg_nodes=len(artifacts.node_to_id),
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_val_f1 = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, len(artifacts.class_to_id))
        row = {"epoch": epoch, "train_loss": round(train_loss, 6), "val": val_metrics}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "artifacts": artifacts.to_dict(), "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, criterion, device, len(artifacts.class_to_id))
    result = {
        "model": args.model,
        "best_val_f1_macro": best_val_f1,
        "test": test_metrics,
        "history": history,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "class_to_id": artifacts.class_to_id,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output_dir / "metrics.json"), "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
