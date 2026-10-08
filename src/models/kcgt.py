"""KCGT backbone with relation-aware graph attention and multi-modal fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch_geometric.nn import TransformerConv, global_mean_pool

from .baselines import GraphEncoder, TextMeanEncoder


@dataclass(frozen=True)
class KCGTConfig:
    vocab_size: int
    num_classes: int
    pad_idx: int = 0
    qos_dim: int = 8
    text_dim: int = 64
    graph_dim: int = 96
    qos_hidden_dim: int = 64
    fusion_dim: int = 128
    projection_dim: int = 64
    dropout: float = 0.2
    graph_model: str = "relation_transformer"
    graph_heads: int = 4
    relation_dim: int = 32
    node_feat_dim: int = 7
    num_relations: int = 7
    num_kg_nodes: int = 1
    node_embed_dim: int = 32
    use_graph: bool = True
    use_qos: bool = True
    fusion_type: str = "gated"


class QoSEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, qos: torch.Tensor) -> torch.Tensor:
        return self.net(qos)


class RelationAwareGraphTransformerLayer(nn.Module):
    """Graph Transformer block whose attention keys and values include edge relations."""

    def __init__(self, input_dim: int, output_dim: int, relation_dim: int, heads: int, dropout: float):
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError(f"output_dim={output_dim} must be divisible by heads={heads}")
        self.attention = TransformerConv(
            in_channels=input_dim,
            out_channels=output_dim // heads,
            heads=heads,
            concat=True,
            beta=True,
            dropout=dropout,
            edge_dim=relation_dim,
            root_weight=True,
        )
        self.residual = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)
        self.norm1 = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(self.residual(x) + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


class RelationAwareGraphTransformerEncoder(nn.Module):
    """Encode typed intent subgraphs with relation-conditioned multi-head attention."""

    def __init__(
        self,
        node_feat_dim: int,
        num_kg_nodes: int,
        node_embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        relation_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.node_embedding = nn.Embedding(num_kg_nodes, node_embed_dim)
        self.relation_embedding = nn.Embedding(num_relations, relation_dim)
        input_dim = node_feat_dim + node_embed_dim
        self.layer1 = RelationAwareGraphTransformerLayer(input_dim, hidden_dim, relation_dim, heads, dropout)
        self.layer2 = RelationAwareGraphTransformerLayer(hidden_dim, out_dim, relation_dim, heads, dropout)

    def forward(self, graph) -> torch.Tensor:
        x = torch.cat([graph.x, self.node_embedding(graph.node_ids)], dim=-1)
        edge_attr = self.relation_embedding(graph.edge_type)
        x = self.layer1(x, graph.edge_index, edge_attr)
        x = self.layer2(x, graph.edge_index, edge_attr)
        return global_mean_pool(x, graph.batch)


class GatedFusion(nn.Module):
    """Fuse text, graph, and QoS embeddings with sample-wise gates."""

    def __init__(self, text_dim: int, graph_dim: int, qos_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.qos_proj = nn.Linear(qos_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(text_dim + graph_dim + qos_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 3),
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_z: torch.Tensor, graph_z: torch.Tensor, qos_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_logits = self.gate(torch.cat([text_z, graph_z, qos_z], dim=-1))
        gates = torch.softmax(gate_logits, dim=-1)
        parts = torch.stack(
            [
                self.text_proj(text_z),
                self.graph_proj(graph_z),
                self.qos_proj(qos_z),
            ],
            dim=1,
        )
        fused = (gates.unsqueeze(-1) * parts).sum(dim=1)
        fused = self.dropout(self.norm(fused))
        return {"embedding": fused, "gates": gates}


class ConcatFusion(nn.Module):
    """Fuse modality embeddings without learned sample-wise gates."""

    def __init__(self, text_dim: int, graph_dim: int, qos_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + graph_dim + qos_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, text_z: torch.Tensor, graph_z: torch.Tensor, qos_z: torch.Tensor) -> Dict[str, torch.Tensor]:
        fused = self.net(torch.cat([text_z, graph_z, qos_z], dim=-1))
        gates = torch.full((text_z.size(0), 3), 1.0 / 3.0, device=text_z.device)
        return {"embedding": fused, "gates": gates}


class KCGTClassifier(nn.Module):
    """Knowledge-Guided Contrastive Graph Transformer representation backbone."""

    def __init__(self, cfg: KCGTConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextMeanEncoder(cfg.vocab_size, cfg.text_dim, cfg.pad_idx)
        if not cfg.use_graph:
            self.graph_encoder = None
        elif cfg.graph_model in {"relation_transformer", "rel_transformer", "kcgt"}:
            self.graph_encoder = RelationAwareGraphTransformerEncoder(
                node_feat_dim=cfg.node_feat_dim,
                num_kg_nodes=cfg.num_kg_nodes,
                node_embed_dim=cfg.node_embed_dim,
                hidden_dim=cfg.graph_dim,
                out_dim=cfg.graph_dim,
                num_relations=cfg.num_relations,
                relation_dim=cfg.relation_dim,
                heads=cfg.graph_heads,
                dropout=cfg.dropout,
            )
        else:
            self.graph_encoder = GraphEncoder(
                graph_model=cfg.graph_model,
                node_feat_dim=cfg.node_feat_dim,
                num_kg_nodes=cfg.num_kg_nodes,
                node_embed_dim=cfg.node_embed_dim,
                hidden_dim=cfg.graph_dim,
                out_dim=cfg.graph_dim,
                num_relations=cfg.num_relations,
                dropout=cfg.dropout,
            )
        self.qos_encoder = QoSEncoder(cfg.qos_dim, cfg.qos_hidden_dim, cfg.qos_hidden_dim, cfg.dropout) if cfg.use_qos else None
        fusion_cls = GatedFusion if cfg.fusion_type == "gated" else ConcatFusion
        self.fusion = fusion_cls(
            text_dim=cfg.text_dim,
            graph_dim=cfg.graph_dim,
            qos_dim=cfg.qos_hidden_dim,
            fusion_dim=cfg.fusion_dim,
            dropout=cfg.dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.num_classes),
        )
        self.projector = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.ReLU(),
            nn.Linear(cfg.fusion_dim, cfg.projection_dim),
        )

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_z = self.text_encoder(batch["text_ids"], batch["attention_mask"])
        if self.graph_encoder is None:
            graph_z = text_z.new_zeros((text_z.size(0), self.cfg.graph_dim))
        else:
            graph_z = self.graph_encoder(batch["graph"])
        if self.qos_encoder is None:
            qos_z = text_z.new_zeros((text_z.size(0), self.cfg.qos_hidden_dim))
        else:
            qos_z = self.qos_encoder(batch["qos"])
        fused = self.fusion(text_z, graph_z, qos_z)
        return {
            "embedding": fused["embedding"],
            "projection": torch.nn.functional.normalize(self.projector(fused["embedding"]), dim=-1),
            "gates": fused["gates"],
            "text_z": text_z,
            "graph_z": graph_z,
            "qos_z": qos_z,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encode(batch)
        return self.classifier(encoded["embedding"])


def build_kcgt_model(
    vocab_size: int,
    num_classes: int,
    pad_idx: int,
    qos_dim: int,
    node_feat_dim: int,
    num_relations: int,
    num_kg_nodes: int,
    graph_model: str = "relation_transformer",
    graph_heads: int = 4,
    relation_dim: int = 32,
    text_dim: int = 64,
    graph_dim: int = 96,
    fusion_dim: int = 128,
    projection_dim: int = 64,
    dropout: float = 0.2,
    use_graph: bool = True,
    use_qos: bool = True,
    fusion_type: str = "gated",
) -> KCGTClassifier:
    cfg = KCGTConfig(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
        qos_dim=qos_dim,
        text_dim=text_dim,
        graph_dim=graph_dim,
        fusion_dim=fusion_dim,
        projection_dim=projection_dim,
        dropout=dropout,
        graph_model=graph_model,
        graph_heads=graph_heads,
        relation_dim=relation_dim,
        node_feat_dim=node_feat_dim,
        num_relations=num_relations,
        num_kg_nodes=num_kg_nodes,
        use_graph=use_graph,
        use_qos=use_qos,
        fusion_type=fusion_type,
    )
    return KCGTClassifier(cfg)
