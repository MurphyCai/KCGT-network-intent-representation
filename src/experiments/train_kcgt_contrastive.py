"""Train KCGT with classification and supervised contrastive learning."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datasets import KNIIntentArtifacts, KNIIntentDataset, build_artifacts, kni_collate_fn
from experiments.train_baseline_classification import evaluate, move_batch
from models import build_kcgt_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _masked_supcon_loss(features: torch.Tensor, pos_mask: torch.Tensor, self_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    features = torch.nn.functional.normalize(features, dim=-1)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return features.new_tensor(0.0)
    loss = -(log_prob * pos_mask.float()).sum(dim=1)[valid] / pos_count[valid]
    return loss.mean()


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, group_ids: List[str], temperature: float, positive_mode: str) -> torch.Tensor:
    """Supervised contrastive loss using same class, same group, or a dual objective."""
    n = features.size(0)
    device = features.device
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    group_mask = torch.tensor(
        [[group_ids[i] == group_ids[j] for j in range(n)] for i in range(n)],
        dtype=torch.bool,
        device=device,
    ) & ~self_mask
    if positive_mode == "group":
        return _masked_supcon_loss(features, group_mask, self_mask, temperature)
    if positive_mode == "class":
        return _masked_supcon_loss(features, label_mask, self_mask, temperature)
    group_loss = _masked_supcon_loss(features, group_mask, self_mask, temperature)
    class_loss = _masked_supcon_loss(features, label_mask, self_mask, temperature)
    return 0.5 * (group_loss + class_loss)


def _positive_mask(labels: torch.Tensor, group_ids: Sequence[str], positive_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    n = labels.numel()
    device = labels.device
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    group_mask = torch.tensor(
        [[group_ids[i] == group_ids[j] for j in range(n)] for i in range(n)],
        dtype=torch.bool,
        device=device,
    ) & ~self_mask
    if positive_mode == "group":
        return group_mask, self_mask
    if positive_mode == "class":
        return label_mask, self_mask
    return label_mask | group_mask, self_mask


def _jaccard_matrix(signatures: Sequence[Set[tuple | int]], device: torch.device) -> torch.Tensor:
    n = len(signatures)
    result = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i in range(n):
        result[i, i] = 1.0
        for j in range(i + 1, n):
            union = signatures[i] | signatures[j]
            value = len(signatures[i] & signatures[j]) / len(union) if union else 1.0
            result[i, j] = value
            result[j, i] = value
    return result


def graph_operational_similarity(graph, batch_size: int, device: torch.device) -> torch.Tensor:
    """Pairwise KG similarity combining global node and typed-triple Jaccard scores."""
    node_sets: List[Set[int]] = [set() for _ in range(batch_size)]
    triple_sets: List[Set[tuple]] = [set() for _ in range(batch_size)]
    node_ids = graph.node_ids.detach().cpu().tolist()
    node_batch = graph.batch.detach().cpu().tolist()
    for node_id, graph_id in zip(node_ids, node_batch):
        node_sets[int(graph_id)].add(int(node_id))
    if graph.edge_type.numel():
        edge_index = graph.edge_index.detach().cpu()
        edge_types = graph.edge_type.detach().cpu().tolist()
        for edge_id, relation_id in enumerate(edge_types):
            src = int(edge_index[0, edge_id])
            dst = int(edge_index[1, edge_id])
            graph_id = int(node_batch[src])
            triple_sets[graph_id].add((int(node_ids[src]), int(relation_id), int(node_ids[dst])))
    return 0.5 * _jaccard_matrix(node_sets, device) + 0.5 * _jaccard_matrix(triple_sets, device)


def hard_negative_mask(intent_ids: Sequence[str], hard_negative_ids: Sequence[Sequence[str]], device: torch.device) -> torch.Tensor:
    hard_sets = [set(values) for values in hard_negative_ids]
    n = len(intent_ids)
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i in range(n):
        for j in range(i + 1, n):
            if intent_ids[j] in hard_sets[i] or intent_ids[i] in hard_sets[j]:
                mask[i, j] = True
                mask[j, i] = True
    return mask


def structured_contrastive_loss(
    features: torch.Tensor,
    batch: Dict[str, object],
    temperature: float,
    positive_mode: str,
    kg_weight: float,
    qos_weight: float,
    slice_weight: float,
    hard_negative_boost: float,
    near_negative_boost: float,
    margin_weight: float,
    base_margin: float,
    margin_scale: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Operationally structured contrastive loss for network intent embeddings.

    Positive contributions are weighted by KG, QoS, and slice compatibility.
    Explicit and in-batch near hard negatives receive larger denominator weights
    and an adaptive ranking margin.
    """
    features = torch.nn.functional.normalize(features, dim=-1)
    labels = batch["y"]
    positive, self_mask = _positive_mask(labels, batch["positive_group_id"], positive_mode)
    n = features.size(0)
    kg_sim = graph_operational_similarity(batch["graph"], n, features.device)
    qos_sim = 1.0 - torch.cdist(batch["qos"], batch["qos"], p=1) / max(1, batch["qos"].size(1))
    qos_sim = qos_sim.clamp(0.0, 1.0)
    slice_sim = batch["slice_y"].unsqueeze(0).eq(batch["slice_y"].unsqueeze(1)).float()
    weight_sum = max(1e-12, kg_weight + qos_weight + slice_weight)
    operational_sim = (kg_weight * kg_sim + qos_weight * qos_sim + slice_weight * slice_sim) / weight_sum

    explicit_hard = hard_negative_mask(batch["intent_id"], batch["hard_negative_ids"], features.device)
    negative = ~positive & ~self_mask
    near_hard = negative & (operational_sim >= 0.65)
    effective_hard = explicit_hard | near_hard

    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_weight = torch.ones_like(logits)
    denominator_weight = denominator_weight + hard_negative_boost * explicit_hard.float()
    denominator_weight = denominator_weight + near_negative_boost * operational_sim * near_hard.float()
    exp_logits = torch.exp(logits) * (~self_mask).float() * denominator_weight
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_weight = positive.float() * (0.25 + operational_sim)
    positive_mass = positive_weight.sum(dim=1)
    valid = positive_mass > 0
    if valid.any():
        info_nce = -(log_prob * positive_weight).sum(dim=1)[valid] / positive_mass[valid]
        info_nce = info_nce.mean()
        positive_similarity = ((features @ features.t()) * positive_weight).sum(dim=1) / positive_mass.clamp_min(1e-12)
    else:
        info_nce = features.new_tensor(0.0)
        positive_similarity = features.new_zeros(n)

    cosine = features @ features.t()
    adaptive_margin = base_margin + margin_scale * operational_sim
    ranking = torch.relu(cosine - positive_similarity.unsqueeze(1) + adaptive_margin)
    hard_count = effective_hard.sum()
    margin_loss = (ranking * effective_hard.float()).sum() / hard_count.clamp_min(1)
    total = info_nce + margin_weight * margin_loss
    return total, {
        "structured_info_nce": info_nce.detach(),
        "structured_margin": margin_loss.detach(),
        "hard_pairs": hard_count.detach().float(),
        "mean_operational_similarity": operational_sim[~self_mask].mean().detach(),
    }


def train_one_epoch(
    model,
    loader,
    ce_loss,
    optimizer,
    device,
    con_weight: float,
    temperature: float,
    positive_mode: str,
    contrastive_objective: str = "structured",
    kg_weight: float = 0.40,
    qos_weight: float = 0.35,
    slice_weight: float = 0.25,
    hard_negative_boost: float = 2.0,
    near_negative_boost: float = 1.0,
    margin_weight: float = 0.25,
    base_margin: float = 0.10,
    margin_scale: float = 0.20,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_con = 0.0
    total_info_nce = 0.0
    total_margin = 0.0
    total_hard_pairs = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encode(batch)
        logits = model.classifier(encoded["embedding"])
        ce = ce_loss(logits, batch["y"])
        if contrastive_objective == "structured":
            con, con_parts = structured_contrastive_loss(
                encoded["projection"],
                batch,
                temperature,
                positive_mode,
                kg_weight,
                qos_weight,
                slice_weight,
                hard_negative_boost,
                near_negative_boost,
                margin_weight,
                base_margin,
                margin_scale,
            )
        else:
            con = supcon_loss(encoded["projection"], batch["y"], batch["positive_group_id"], temperature, positive_mode)
            con_parts = {
                "structured_info_nce": con.detach(),
                "structured_margin": con.detach().new_tensor(0.0),
                "hard_pairs": con.detach().new_tensor(0.0),
            }
        loss = ce + con_weight * con
        loss.backward()
        optimizer.step()
        count = int(batch["y"].numel())
        total_loss += float(loss.item()) * count
        total_ce += float(ce.item()) * count
        total_con += float(con.item()) * count
        total_info_nce += float(con_parts["structured_info_nce"].item()) * count
        total_margin += float(con_parts["structured_margin"].item()) * count
        total_hard_pairs += float(con_parts["hard_pairs"].item())
        total_count += count
    return {
        "loss": total_loss / max(1, total_count),
        "ce_loss": total_ce / max(1, total_count),
        "contrastive_loss": total_con / max(1, total_count),
        "structured_info_nce": total_info_nce / max(1, total_count),
        "structured_margin": total_margin / max(1, total_count),
        "hard_pairs_per_batch": total_hard_pairs / max(1, len(loader)),
    }


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train KCGT with supervised contrastive learning.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "hard"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "kcgt_contrastive_hard"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--fusion-dim", type=int, default=192)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--positive-mode", choices=["group", "class", "both"], default="both")
    parser.add_argument("--contrastive-objective", choices=["supcon", "structured"], default="structured")
    parser.add_argument("--kg-weight", type=float, default=0.40)
    parser.add_argument("--qos-weight", type=float, default=0.35)
    parser.add_argument("--slice-weight", type=float, default=0.25)
    parser.add_argument("--hard-negative-boost", type=float, default=2.0)
    parser.add_argument("--near-negative-boost", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--base-margin", type=float, default=0.10)
    parser.add_argument("--margin-scale", type=float, default=0.20)
    parser.add_argument("--graph-model", choices=["gcn", "gat", "rgcn", "relation_transformer"], default="relation_transformer")
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--relation-dim", type=int, default=32)
    parser.add_argument("--relation-mode", choices=["typed", "shuffled", "uniform"], default="typed")
    parser.add_argument("--disable-graph", action="store_true", help="Ablation: replace graph encoder output with zeros.")
    parser.add_argument("--disable-qos", action="store_true", help="Ablation: replace QoS encoder output with zeros.")
    parser.add_argument("--fusion-type", choices=["gated", "concat"], default="gated", help="Use gated fusion or non-gated concatenation fusion.")
    parser.add_argument("--max-text-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_or_build_artifacts(processed_dir, args.max_text_len)
    train_ds = KNIIntentDataset(processed_dir / "train.jsonl", artifacts, relation_mode=args.relation_mode, relation_seed=args.seed)
    val_ds = KNIIntentDataset(processed_dir / "val.jsonl", artifacts, relation_mode=args.relation_mode, relation_seed=args.seed)
    test_ds = KNIIntentDataset(processed_dir / "test.jsonl", artifacts, relation_mode=args.relation_mode, relation_seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=kni_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=kni_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=kni_collate_fn)

    model = build_kcgt_model(
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
        use_graph=not args.disable_graph,
        use_qos=not args.disable_qos,
        fusion_type=args.fusion_type,
    ).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_f1 = -1.0
    best_path = output_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            ce_loss,
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
        val_metrics = evaluate(model, val_loader, ce_loss, device, len(artifacts.class_to_id))
        row = {"epoch": epoch, "train": {k: round(v, 6) for k, v in train_metrics.items()}, "val": val_metrics, "seconds": round(time.perf_counter() - start, 3)}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "artifacts": artifacts.to_dict(), "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, ce_loss, device, len(artifacts.class_to_id))
    result = {
        "model": "kcgt_contrastive",
        "ablation": {
            "contrastive_weight": args.contrastive_weight,
            "positive_mode": args.positive_mode,
            "contrastive_objective": args.contrastive_objective,
            "kg_weight": args.kg_weight,
            "qos_weight": args.qos_weight,
            "slice_weight": args.slice_weight,
            "hard_negative_boost": args.hard_negative_boost,
            "near_negative_boost": args.near_negative_boost,
            "margin_weight": args.margin_weight,
            "base_margin": args.base_margin,
            "margin_scale": args.margin_scale,
            "relation_mode": args.relation_mode,
            "graph_model": args.graph_model,
            "use_graph": not args.disable_graph,
            "use_qos": not args.disable_qos,
            "fusion_type": args.fusion_type,
        },
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
