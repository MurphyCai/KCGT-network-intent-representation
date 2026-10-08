"""Train on one KNI-Intent dataset and evaluate on another dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import KNIIntentArtifacts, KNIIntentDataset, build_artifacts, kni_collate_fn
from experiments.train_baseline_classification import evaluate, move_batch
from experiments.train_kcgt_contrastive import train_one_epoch as train_kcgt_epoch
from models import build_baseline_model, build_kcgt_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def split_paths(processed_dir: Path) -> list[Path]:
    paths = [processed_dir / name for name in ["train.jsonl", "val.jsonl", "test.jsonl", "unseen_composition_test.jsonl"]]
    return [path for path in paths if path.exists()]


def build_transfer_artifacts(source_dir: Path, target_dir: Path, max_text_len: int) -> KNIIntentArtifacts:
    # Text vocabulary and QoS stats are fitted on source train only by build_artifacts.
    # Node/relation/class mappings include target splits so target graphs can be encoded.
    return build_artifacts(
        train_path=source_dir / "train.jsonl",
        all_paths=split_paths(source_dir) + split_paths(target_dir),
        max_text_len=max_text_len,
    )


def make_loaders(source_dir: Path, target_dir: Path, artifacts: KNIIntentArtifacts, batch_size: int, relation_mode: str, seed: int):
    dataset_args = {"relation_mode": relation_mode, "relation_seed": seed}
    train_ds = KNIIntentDataset(source_dir / "train.jsonl", artifacts, **dataset_args)
    val_ds = KNIIntentDataset(source_dir / "val.jsonl", artifacts, **dataset_args)
    source_test_ds = KNIIntentDataset(source_dir / "test.jsonl", artifacts, **dataset_args)
    target_test_ds = KNIIntentDataset(target_dir / "test.jsonl", artifacts, **dataset_args)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=kni_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    source_test_loader = DataLoader(source_test_ds, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    target_test_loader = DataLoader(target_test_ds, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    return train_loader, val_loader, source_test_loader, target_test_loader


def train_supervised_epoch(model, loader, criterion, optimizer, device) -> float:
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
        count = int(batch["y"].numel())
        total_loss += float(loss.item()) * count
        total_count += count
    return total_loss / max(1, total_count)


def build_model(args: argparse.Namespace, artifacts: KNIIntentArtifacts, device: torch.device):
    if args.model in {"kcgt", "kcgt_contrastive"}:
        return build_kcgt_model(
            vocab_size=len(artifacts.token_to_id),
            num_classes=len(artifacts.class_to_id),
            pad_idx=artifacts.token_to_id["<pad>"],
            qos_dim=len(artifacts.qos_stats),
            node_feat_dim=len(artifacts.node_type_to_id),
            num_relations=len(artifacts.relation_to_id),
            num_kg_nodes=len(artifacts.node_to_id),
            graph_model=args.graph_model,
            graph_heads=args.graph_heads,
            relation_dim=args.relation_dim,
            text_dim=args.embed_dim,
            graph_dim=args.hidden_dim,
            fusion_dim=args.fusion_dim,
            projection_dim=args.projection_dim,
            dropout=args.dropout,
            fusion_type=args.fusion_type,
        ).to(device)
    return build_baseline_model(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--model", choices=["textcnn", "transformer", "text_gat", "text_rgcn", "kcgt", "kcgt_contrastive"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--fusion-dim", type=int, default=192)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--positive-mode", choices=["group", "class", "both"], default="group")
    parser.add_argument("--graph-model", choices=["relation_transformer", "gcn", "gat", "rgcn"], default="relation_transformer")
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--relation-dim", type=int, default=32)
    parser.add_argument("--relation-mode", choices=["typed", "shuffled", "uniform"], default="typed")
    parser.add_argument("--contrastive-objective", choices=["structured", "supcon"], default="structured")
    parser.add_argument("--kg-weight", type=float, default=0.40)
    parser.add_argument("--qos-weight", type=float, default=0.35)
    parser.add_argument("--slice-weight", type=float, default=0.25)
    parser.add_argument("--hard-negative-boost", type=float, default=2.0)
    parser.add_argument("--near-negative-boost", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--base-margin", type=float, default=0.10)
    parser.add_argument("--margin-scale", type=float, default=0.20)
    parser.add_argument("--fusion-type", choices=["gated", "concat"], default="gated")
    parser.add_argument("--max-text-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    source_dir = ROOT / args.source_dir
    target_dir = ROOT / args.target_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_transfer_artifacts(source_dir, target_dir, args.max_text_len)
    train_loader, val_loader, source_test_loader, target_test_loader = make_loaders(
        source_dir, target_dir, artifacts, args.batch_size, args.relation_mode, args.seed
    )
    model = build_model(args, artifacts, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_f1 = -1.0
    best_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        if args.model == "kcgt_contrastive":
            train_metrics = train_kcgt_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.contrastive_weight,
                args.temperature,
                args.positive_mode,
                args.contrastive_objective,
                args.kg_weight,
                args.qos_weight,
                args.slice_weight,
                args.hard_negative_boost,
                args.near_negative_boost,
                args.margin_weight,
                args.base_margin,
                args.margin_scale,
            )
        else:
            train_loss = train_supervised_epoch(model, train_loader, criterion, optimizer, device)
            train_metrics = {"loss": train_loss}
        val_metrics = evaluate(model, val_loader, criterion, device, len(artifacts.class_to_id))
        row = {"epoch": epoch, "train": {k: round(v, 6) for k, v in train_metrics.items()}, "val": val_metrics, "seconds": round(time.perf_counter() - start, 3)}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "artifacts": artifacts.to_dict(), "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    source_metrics = evaluate(model, source_test_loader, criterion, device, len(artifacts.class_to_id))
    target_metrics = evaluate(model, target_test_loader, criterion, device, len(artifacts.class_to_id))
    result = {
        "model": args.model,
        "source": args.source_name,
        "target": args.target_name,
        "best_val_f1_macro": best_val_f1,
        "source_test": source_metrics,
        "target_test": target_metrics,
        "test": target_metrics,
        "history": history,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "class_to_id": artifacts.class_to_id,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output_dir / "metrics.json"), "source_test": source_metrics, "target_test": target_metrics}, indent=2))


if __name__ == "__main__":
    main()
