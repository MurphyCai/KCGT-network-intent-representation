"""Train frozen BERT/SBERT text baselines with a lightweight classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_sbert(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True),
        dtype=np.float32,
    )


def encode_hf(texts: list[str], model_name: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer, BertTokenizer

    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def qos_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(record.get(field, 0.0)) for field in QOS_FIELDS] for record in records], dtype=np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, elapsed: float) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "loss": 0.0,
        "inference_time_ms_per_sample": round(elapsed * 1000 / max(1, len(y_true)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=["hf", "sbert"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--include-qos", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    train = load_jsonl(processed / "train.jsonl")
    val = load_jsonl(processed / "val.jsonl")
    test = load_jsonl(processed / "test.jsonl")
    labels = sorted({record["intent_class"] for record in train + val + test})
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    texts = {
        "train": [record["intent_text"] for record in train],
        "val": [record["intent_text"] for record in val],
        "test": [record["intent_text"] for record in test],
    }
    encoder = encode_sbert if args.encoder == "sbert" else encode_hf
    if args.encoder == "sbert":
        x_train = encoder(texts["train"], args.model_name, args.batch_size)
        x_val = encoder(texts["val"], args.model_name, args.batch_size)
        start = time.perf_counter()
        x_test = encoder(texts["test"], args.model_name, args.batch_size)
    else:
        x_train = encoder(texts["train"], args.model_name, args.batch_size, args.max_length)
        x_val = encoder(texts["val"], args.model_name, args.batch_size, args.max_length)
        start = time.perf_counter()
        x_test = encoder(texts["test"], args.model_name, args.batch_size, args.max_length)
    encode_elapsed = time.perf_counter() - start

    if args.include_qos:
        scaler = StandardScaler()
        q_train = scaler.fit_transform(qos_matrix(train))
        q_val = scaler.transform(qos_matrix(val))
        q_test = scaler.transform(qos_matrix(test))
        x_train = np.concatenate([x_train, q_train], axis=1)
        x_val = np.concatenate([x_val, q_val], axis=1)
        x_test = np.concatenate([x_test, q_test], axis=1)

    y_train = np.asarray([label_to_id[record["intent_class"]] for record in train])
    y_val = np.asarray([label_to_id[record["intent_class"]] for record in val])
    y_test = np.asarray([label_to_id[record["intent_class"]] for record in test])

    clf = LogisticRegression(max_iter=3000, random_state=args.seed, solver="lbfgs", multi_class="auto")
    clf.fit(np.concatenate([x_train, x_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    pred = clf.predict(x_test)
    test_metrics = metrics(y_test, pred, encode_elapsed)

    result = {
        "model": args.name,
        "encoder": args.encoder,
        "model_name": args.model_name,
        "include_qos": args.include_qos,
        "test": test_metrics,
        "num_parameters": int(np.prod(clf.coef_.shape) + clf.intercept_.shape[0]),
        "class_to_id": label_to_id,
    }
    out = Path(args.output_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out / "metrics.json"), "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
