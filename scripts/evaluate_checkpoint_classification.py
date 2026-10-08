"""Evaluate a saved neural checkpoint on an arbitrary JSONL split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import KNIIntentArtifacts, KNIIntentDataset, kni_collate_fn
from experiments.train_baseline_classification import evaluate, move_batch
from models import build_baseline_model, build_kcgt_model


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    artifacts = KNIIntentArtifacts.from_dict(checkpoint["artifacts"])
    args = checkpoint.get("args", {})
    model_name = args.get("model", "kcgt_contrastive")
    if model_name == "kcgt" or "positive_mode" in args:
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
        )
    else:
        model = build_baseline_model(
            model_name=model_name,
            vocab_size=len(artifacts.token_to_id),
            num_classes=len(artifacts.class_to_id),
            pad_idx=artifacts.token_to_id["<pad>"],
            qos_dim=len(artifacts.qos_stats),
            embed_dim=int(args.get("embed_dim", 64)),
            hidden_dim=int(args.get("hidden_dim", 96)),
            dropout=float(args.get("dropout", 0.2)),
            max_text_len=int(args.get("max_text_len", 40)),
            node_feat_dim=len(artifacts.node_type_to_id),
            num_relations=len(artifacts.relation_to_id),
            num_kg_nodes=len(artifacts.node_to_id),
        )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    return model, artifacts, model_name, args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, artifacts, model_name, train_args = load_checkpoint(Path(args.checkpoint), device)
    dataset = KNIIntentDataset(
        Path(args.jsonl),
        artifacts,
        relation_mode=train_args.get("relation_mode", "typed"),
        relation_seed=int(train_args.get("seed", 0)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=kni_collate_fn)
    result = evaluate(model, loader, nn.CrossEntropyLoss(), device, len(artifacts.class_to_id))
    payload = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "jsonl": args.jsonl,
        "test": result,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
