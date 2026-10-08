"""CPU-friendly baseline classifiers for KNI-Intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, RGCNConv, TransformerConv, global_mean_pool


@dataclass(frozen=True)
class BaselineConfig:
    model_name: str
    vocab_size: int
    num_classes: int
    pad_idx: int = 0
    qos_dim: int = 8
    embed_dim: int = 64
    hidden_dim: int = 96
    dropout: float = 0.2
    transformer_heads: int = 4
    transformer_layers: int = 2
    max_text_len: int = 40
    node_feat_dim: int = 7
    num_relations: int = 7
    num_kg_nodes: int = 1
    node_embed_dim: int = 32


class TextCNNClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(cfg.embed_dim, cfg.hidden_dim, kernel_size=k, padding=0)
            for k in (2, 3, 4)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim * len(self.convs), cfg.num_classes)

    def forward(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, qos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(text_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            pooled.append(torch.max(h, dim=-1).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=-1)))


class LSTMClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim * 2, cfg.num_classes)

    def forward(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, qos: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(text_ids)
        output, _ = self.lstm(emb)
        if attention_mask is None:
            pooled = output.mean(dim=1)
        else:
            mask = attention_mask.float().unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.positional = PositionalEncoding(cfg.embed_dim, max_len=max(256, cfg.max_text_len + 4))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.hidden_dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.embed_dim, cfg.num_classes)

    def forward(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, qos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.positional(self.embedding(text_ids))
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            mask = attention_mask.float().unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


class QoSMLPClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.qos_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, qos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if qos is None:
            raise ValueError("QoSMLPClassifier requires qos tensor.")
        return self.net(qos)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        graph_model: str,
        node_feat_dim: int,
        num_kg_nodes: int,
        node_embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float,
    ):
        super().__init__()
        self.graph_model = graph_model.lower()
        self.dropout = nn.Dropout(dropout)
        self.node_embedding = nn.Embedding(num_kg_nodes, node_embed_dim)
        input_dim = node_feat_dim + node_embed_dim
        if self.graph_model == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif self.graph_model == "gat":
            self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * 2, out_dim, heads=1, concat=False, dropout=dropout)
        elif self.graph_model in {"rgcn", "r-gcn"}:
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations)
        elif self.graph_model in {"graph_transformer", "transformer"}:
            self.conv1 = TransformerConv(input_dim, hidden_dim, heads=2, concat=True, dropout=dropout)
            self.conv2 = TransformerConv(hidden_dim * 2, out_dim, heads=1, concat=False, dropout=dropout)
        else:
            raise ValueError(f"Unknown graph model: {graph_model}")

    def forward(self, graph) -> torch.Tensor:
        node_emb = self.node_embedding(graph.node_ids)
        x, edge_index = torch.cat([graph.x, node_emb], dim=-1), graph.edge_index
        if self.graph_model in {"rgcn", "r-gcn"}:
            edge_type = graph.edge_type
            x = self.conv1(x, edge_index, edge_type)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index, edge_type)
        else:
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return global_mean_pool(x, graph.batch)


class GraphOnlyClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig, graph_model: str, node_feat_dim: int, num_relations: int):
        super().__init__()
        self.encoder = GraphEncoder(
            graph_model=graph_model,
            node_feat_dim=node_feat_dim,
            num_kg_nodes=cfg.num_kg_nodes,
            node_embed_dim=cfg.node_embed_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            num_relations=num_relations,
            dropout=cfg.dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, graph) -> torch.Tensor:
        return self.classifier(self.encoder(graph))


class TextMeanEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, text_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(text_ids)
        mask = attention_mask.float().unsqueeze(-1)
        return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class TextGraphFusionClassifier(nn.Module):
    def __init__(
        self,
        cfg: BaselineConfig,
        graph_model: str,
        node_feat_dim: int,
        num_relations: int,
        use_qos: bool = False,
    ):
        super().__init__()
        self.use_qos = use_qos
        self.text_encoder = TextMeanEncoder(cfg.vocab_size, cfg.embed_dim, cfg.pad_idx)
        self.graph_encoder = GraphEncoder(
            graph_model=graph_model,
            node_feat_dim=node_feat_dim,
            num_kg_nodes=cfg.num_kg_nodes,
            node_embed_dim=cfg.node_embed_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.hidden_dim,
            num_relations=num_relations,
            dropout=cfg.dropout,
        )
        in_dim = cfg.embed_dim + cfg.hidden_dim + (cfg.qos_dim if use_qos else 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, text_ids: torch.Tensor, attention_mask: torch.Tensor, graph, qos: Optional[torch.Tensor] = None) -> torch.Tensor:
        text_z = self.text_encoder(text_ids, attention_mask)
        graph_z = self.graph_encoder(graph)
        features = [text_z, graph_z]
        if self.use_qos:
            if qos is None:
                raise ValueError("TextGraphFusionClassifier with use_qos=True requires qos tensor.")
            features.append(qos)
        return self.classifier(torch.cat(features, dim=-1))


class HypergraphApproxEncoder(nn.Module):
    """Lightweight hypergraph-style encoder using node and relation incidence sets."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.node_embedding = nn.Embedding(cfg.num_kg_nodes, cfg.node_embed_dim)
        self.relation_embedding = nn.Embedding(cfg.num_relations, cfg.hidden_dim)
        node_input_dim = cfg.node_feat_dim + cfg.node_embed_dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, graph) -> torch.Tensor:
        node_x = torch.cat([graph.x, self.node_embedding(graph.node_ids)], dim=-1)
        node_h = self.node_proj(node_x)
        node_pool = global_mean_pool(node_h, graph.batch)

        batch_size = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        rel_pool = node_pool.new_zeros((batch_size, node_pool.size(-1)))
        rel_count = node_pool.new_zeros((batch_size, 1))
        if graph.edge_type.numel():
            edge_batch = graph.batch[graph.edge_index[0]]
            rel_h = self.relation_embedding(graph.edge_type)
            rel_pool.index_add_(0, edge_batch, rel_h)
            rel_count.index_add_(0, edge_batch, torch.ones((rel_h.size(0), 1), device=rel_h.device, dtype=rel_h.dtype))
            rel_pool = rel_pool / rel_count.clamp_min(1.0)
        return self.out_proj(torch.cat([node_pool, rel_pool], dim=-1))


class HypergraphApproxClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.encoder = HypergraphApproxEncoder(cfg)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, graph) -> torch.Tensor:
        return self.classifier(self.encoder(graph))


class TextHypergraphFusionClassifier(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.text_encoder = TextMeanEncoder(cfg.vocab_size, cfg.embed_dim, cfg.pad_idx)
        self.hyper_encoder = HypergraphApproxEncoder(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, text_ids: torch.Tensor, attention_mask: torch.Tensor, graph) -> torch.Tensor:
        return self.classifier(torch.cat([self.text_encoder(text_ids, attention_mask), self.hyper_encoder(graph)], dim=-1))


class BaselineClassifier(nn.Module):
    """Uniform wrapper exposing a baseline classifier."""

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        name = cfg.model_name.lower()
        self.graph_required = name in {
            "gcn",
            "gat",
            "rgcn",
            "text_gcn",
            "text_gat",
            "text_rgcn",
            "text_graph_gcn",
            "text_graph_gat",
            "text_graph_rgcn",
            "graph_transformer",
            "text_graph_transformer",
            "hypergraph",
            "text_hypergraph",
        }
        if name == "textcnn":
            self.model = TextCNNClassifier(cfg)
        elif name == "lstm":
            self.model = LSTMClassifier(cfg)
        elif name == "transformer":
            self.model = TransformerClassifier(cfg)
        elif name in {"qos_mlp", "qos-mlp", "qosmlp"}:
            self.model = QoSMLPClassifier(cfg)
        elif name == "gcn":
            self.model = GraphOnlyClassifier(cfg, "gcn", cfg.node_feat_dim, cfg.num_relations)
        elif name == "gat":
            self.model = GraphOnlyClassifier(cfg, "gat", cfg.node_feat_dim, cfg.num_relations)
        elif name == "rgcn":
            self.model = GraphOnlyClassifier(cfg, "rgcn", cfg.node_feat_dim, cfg.num_relations)
        elif name == "graph_transformer":
            self.model = GraphOnlyClassifier(cfg, "graph_transformer", cfg.node_feat_dim, cfg.num_relations)
        elif name in {"text_gcn", "text_graph_gcn"}:
            self.model = TextGraphFusionClassifier(cfg, "gcn", cfg.node_feat_dim, cfg.num_relations)
        elif name in {"text_gat", "text_graph_gat"}:
            self.model = TextGraphFusionClassifier(cfg, "gat", cfg.node_feat_dim, cfg.num_relations)
        elif name in {"text_rgcn", "text_graph_rgcn"}:
            self.model = TextGraphFusionClassifier(cfg, "rgcn", cfg.node_feat_dim, cfg.num_relations)
        elif name == "text_graph_transformer":
            self.model = TextGraphFusionClassifier(cfg, "graph_transformer", cfg.node_feat_dim, cfg.num_relations)
        elif name == "hypergraph":
            self.model = HypergraphApproxClassifier(cfg)
        elif name == "text_hypergraph":
            self.model = TextHypergraphFusionClassifier(cfg)
        else:
            raise ValueError(f"Unknown baseline model: {cfg.model_name}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.graph_required:
            name = self.model.__class__.__name__
            if isinstance(self.model, (GraphOnlyClassifier, HypergraphApproxClassifier)):
                return self.model(batch["graph"])
            if isinstance(self.model, TextHypergraphFusionClassifier):
                return self.model(
                    text_ids=batch["text_ids"],
                    attention_mask=batch["attention_mask"],
                    graph=batch["graph"],
                )
            return self.model(
                text_ids=batch["text_ids"],
                attention_mask=batch["attention_mask"],
                graph=batch["graph"],
                qos=batch.get("qos"),
            )
        return self.model(text_ids=batch["text_ids"], attention_mask=batch.get("attention_mask"), qos=batch.get("qos"))


def build_baseline_model(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    pad_idx: int = 0,
    qos_dim: int = 8,
    embed_dim: int = 64,
    hidden_dim: int = 96,
    dropout: float = 0.2,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    max_text_len: int = 40,
    node_feat_dim: int = 7,
    num_relations: int = 7,
    num_kg_nodes: int = 1,
    node_embed_dim: int = 32,
) -> BaselineClassifier:
    cfg = BaselineConfig(
        model_name=model_name,
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
        qos_dim=qos_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        max_text_len=max_text_len,
        node_feat_dim=node_feat_dim,
        num_relations=num_relations,
        num_kg_nodes=num_kg_nodes,
        node_embed_dim=node_embed_dim,
    )
    return BaselineClassifier(cfg)
