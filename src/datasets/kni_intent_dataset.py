"""PyTorch dataset loader for KNI-Intent.

The loader converts KNI-Intent JSONL records into model-ready fields:

- text token ids and attention mask
- normalized QoS tensor
- PyTorch Geometric graph tensor for the sample-specific KG subgraph
- intent class label and slice label

The design is deliberately lightweight and CPU-friendly for the pilot stage.
"""

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

QOS_FIELDS = [
    "qos_latency_ms",
    "qos_bandwidth_mbps",
    "qos_reliability",
    "qos_jitter_ms",
    "qos_packet_loss",
    "qos_security_level",
    "qos_mobility",
    "qos_energy_sensitivity",
]

SPECIAL_TOKENS = ["<pad>", "<unk>"]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def node_type(node_id: str) -> str:
    return node_id.split(":", 1)[0] if ":" in node_id else "unknown"


def compute_qos_stats(records: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for field in QOS_FIELDS:
        values = [float(record[field]) for record in records]
        lo = min(values)
        hi = max(values)
        stats[field] = {"min": lo, "max": hi}
    return stats


def normalize_qos(record: Mapping[str, object], stats: Mapping[str, Mapping[str, float]]) -> List[float]:
    values: List[float] = []
    for field in QOS_FIELDS:
        raw = float(record[field])
        lo = float(stats[field]["min"])
        hi = float(stats[field]["max"])
        if hi == lo:
            values.append(0.0)
        else:
            values.append((raw - lo) / (hi - lo))
    return values


@dataclass
class KNIIntentArtifacts:
    token_to_id: Dict[str, int]
    class_to_id: Dict[str, int]
    slice_to_id: Dict[str, int]
    node_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    node_type_to_id: Dict[str, int]
    qos_stats: Dict[str, Dict[str, float]]
    max_text_len: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "token_to_id": self.token_to_id,
            "class_to_id": self.class_to_id,
            "slice_to_id": self.slice_to_id,
            "node_to_id": self.node_to_id,
            "relation_to_id": self.relation_to_id,
            "node_type_to_id": self.node_type_to_id,
            "qos_stats": self.qos_stats,
            "max_text_len": self.max_text_len,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "KNIIntentArtifacts":
        return cls(
            token_to_id=dict(payload["token_to_id"]),
            class_to_id=dict(payload["class_to_id"]),
            slice_to_id=dict(payload["slice_to_id"]),
            node_to_id=dict(payload["node_to_id"]),
            relation_to_id=dict(payload["relation_to_id"]),
            node_type_to_id=dict(payload["node_type_to_id"]),
            qos_stats=dict(payload["qos_stats"]),
            max_text_len=int(payload["max_text_len"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KNIIntentArtifacts":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_vocab(records: Sequence[Mapping[str, object]], min_freq: int = 1) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(tokenize(str(record["intent_text"])))
    token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, count in sorted(counts.items()):
        if count >= min_freq and token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return token_to_id


def collect_nodes(records: Sequence[Mapping[str, object]]) -> List[str]:
    nodes = set()
    for record in records:
        nodes.update(str(node) for node in record["kg_nodes"])
    return sorted(nodes)


def collect_relations(records: Sequence[Mapping[str, object]]) -> List[str]:
    relations = set()
    for record in records:
        for _, relation, _ in record["kg_edges"]:
            relations.add(str(relation))
    return sorted(relations)


def build_artifacts(
    train_path: Path,
    all_paths: Optional[Sequence[Path]] = None,
    max_text_len: int = 40,
    min_token_freq: int = 1,
) -> KNIIntentArtifacts:
    """Build reusable dataset artifacts from train records.

    Text vocabulary and QoS normalization are fitted on the train split.
    Label/node/relation mappings are built from all provided paths so that
    validation/test subgraphs can be encoded without unknown KG nodes.
    """

    train_records = read_jsonl(train_path)
    mapping_records = list(train_records)
    if all_paths:
        mapping_records = []
        for path in all_paths:
            mapping_records.extend(read_jsonl(path))

    token_to_id = build_vocab(train_records, min_freq=min_token_freq)
    class_to_id = {label: idx for idx, label in enumerate(sorted({str(r["intent_class"]) for r in mapping_records}))}
    slice_to_id = {label: idx for idx, label in enumerate(sorted({str(r["slice_type"]) for r in mapping_records}))}
    node_to_id = {node: idx for idx, node in enumerate(collect_nodes(mapping_records))}
    relation_to_id = {rel: idx for idx, rel in enumerate(collect_relations(mapping_records))}
    node_type_to_id = {typ: idx for idx, typ in enumerate(sorted({node_type(node) for node in node_to_id}))}
    return KNIIntentArtifacts(
        token_to_id=token_to_id,
        class_to_id=class_to_id,
        slice_to_id=slice_to_id,
        node_to_id=node_to_id,
        relation_to_id=relation_to_id,
        node_type_to_id=node_type_to_id,
        qos_stats=compute_qos_stats(train_records),
        max_text_len=max_text_len,
    )


class KNIIntentDataset(Dataset):
    """KNI-Intent split dataset."""

    def __init__(
        self,
        jsonl_path: Path,
        artifacts: KNIIntentArtifacts,
        drop_label_nodes: bool = True,
        relation_mode: str = "typed",
        relation_seed: int = 0,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.records = read_jsonl(self.jsonl_path)
        self.artifacts = artifacts
        self.drop_label_nodes = drop_label_nodes
        if relation_mode not in {"typed", "shuffled", "uniform"}:
            raise ValueError(f"Unknown relation_mode: {relation_mode}")
        self.relation_mode = relation_mode
        self.relation_seed = relation_seed

    def __len__(self) -> int:
        return len(self.records)

    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        unk_id = self.artifacts.token_to_id["<unk>"]
        pad_id = self.artifacts.token_to_id["<pad>"]
        token_ids = [self.artifacts.token_to_id.get(token, unk_id) for token in tokenize(text)]
        token_ids = token_ids[: self.artifacts.max_text_len]
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < self.artifacts.max_text_len:
            token_ids.append(pad_id)
            attention_mask.append(0)
        return {
            "text_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }

    def encode_graph(self, record: Mapping[str, object]) -> Data:
        sample_nodes = [str(node) for node in record["kg_nodes"]]
        if self.drop_label_nodes:
            sample_nodes = [node for node in sample_nodes if not node.startswith("intent_class:")]
        local_node_to_idx = {node: idx for idx, node in enumerate(sample_nodes)}
        type_ids = [self.artifacts.node_type_to_id[node_type(node)] for node in sample_nodes]
        global_node_ids = [self.artifacts.node_to_id[node] for node in sample_nodes]

        edges = []
        edge_types = []
        for src, relation, dst in record["kg_edges"]:
            src = str(src)
            dst = str(dst)
            relation = str(relation)
            if src in local_node_to_idx and dst in local_node_to_idx:
                edges.append([local_node_to_idx[src], local_node_to_idx[dst]])
                edge_types.append(self.artifacts.relation_to_id[relation])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            if self.relation_mode == "uniform":
                edge_type = torch.zeros_like(edge_type)
            elif self.relation_mode == "shuffled" and edge_type.numel() > 1:
                key = f"{record.get('intent_id', '')}:{self.relation_seed}".encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)
                generator = torch.Generator().manual_seed(seed)
                edge_type = edge_type[torch.randperm(edge_type.numel(), generator=generator)]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        x = torch.nn.functional.one_hot(
            torch.tensor(type_ids, dtype=torch.long),
            num_classes=len(self.artifacts.node_type_to_id),
        ).float()
        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            node_ids=torch.tensor(global_node_ids, dtype=torch.long),
            num_nodes=len(sample_nodes),
        )

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        text = self.encode_text(str(record["intent_text"]))
        qos = torch.tensor(normalize_qos(record, self.artifacts.qos_stats), dtype=torch.float32)
        graph = self.encode_graph(record)
        y = torch.tensor(self.artifacts.class_to_id[str(record["intent_class"])], dtype=torch.long)
        slice_y = torch.tensor(self.artifacts.slice_to_id[str(record["slice_type"])], dtype=torch.long)
        return {
            "intent_id": str(record["intent_id"]),
            "positive_group_id": str(record["positive_group_id"]),
            "hard_negative_ids": [str(value) for value in record.get("hard_negative_ids", [])],
            "text_ids": text["text_ids"],
            "attention_mask": text["attention_mask"],
            "qos": qos,
            "graph": graph,
            "y": y,
            "slice_y": slice_y,
        }


def kni_collate_fn(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "intent_id": [str(item["intent_id"]) for item in batch],
        "positive_group_id": [str(item["positive_group_id"]) for item in batch],
        "hard_negative_ids": [list(item["hard_negative_ids"]) for item in batch],
        "text_ids": torch.stack([item["text_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "qos": torch.stack([item["qos"] for item in batch]),
        "graph": Batch.from_data_list([item["graph"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "slice_y": torch.stack([item["slice_y"] for item in batch]),
    }
