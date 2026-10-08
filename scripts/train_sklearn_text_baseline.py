"""Train strong CPU-friendly sklearn text baselines for KNI-Intent."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def labels(records: List[Dict[str, object]], encoder: LabelEncoder | None = None):
    y = [str(r["intent_class"]) for r in records]
    if encoder is None:
        encoder = LabelEncoder()
        return encoder.fit_transform(y), encoder
    return encoder.transform(y), encoder


def qos_matrix(records: List[Dict[str, object]]) -> np.ndarray:
    return np.array([[float(r[f]) for f in QOS_FIELDS] for r in records], dtype=np.float32)


def macro_metrics(y_true, y_pred, loss: float, infer_ms: float) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision), 6),
        "recall_macro": round(float(recall), 6),
        "f1_macro": round(float(f1), 6),
        "loss": round(float(loss), 6),
        "inference_time_ms_per_sample": round(float(infer_ms), 6),
    }


def build_features(kind: str):
    if kind == "tfidf_word":
        return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=20000)
    if kind == "tfidf_char":
        return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=25000)
    if kind == "tfidf_word_char":
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=16000)),
                ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=16000)),
            ]
        )
    raise ValueError(f"Unknown sklearn text baseline: {kind}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train sklearn TF-IDF baseline.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--model", choices=["tfidf_word", "tfidf_char", "tfidf_word_char", "tfidf_word_char_qos"], default="tfidf_word_char")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(processed / "train.jsonl")
    val = read_jsonl(processed / "val.jsonl")
    test = read_jsonl(processed / "test.jsonl")
    y_train, encoder = labels(train)
    y_val, _ = labels(val, encoder)
    y_test, _ = labels(test, encoder)

    text_model = args.model.replace("_qos", "")
    vectorizer = build_features(text_model)
    x_train = vectorizer.fit_transform([str(r["intent_text"]) for r in train])
    x_val = vectorizer.transform([str(r["intent_text"]) for r in val])
    x_test = vectorizer.transform([str(r["intent_text"]) for r in test])

    if args.model.endswith("_qos"):
        scaler = MinMaxScaler()
        q_train = scaler.fit_transform(qos_matrix(train))
        q_val = scaler.transform(qos_matrix(val))
        q_test = scaler.transform(qos_matrix(test))
        x_train = sparse.hstack([x_train, sparse.csr_matrix(q_train)])
        x_val = sparse.hstack([x_val, sparse.csr_matrix(q_val)])
        x_test = sparse.hstack([x_test, sparse.csr_matrix(q_test)])

    clf = LogisticRegression(max_iter=args.max_iter, random_state=args.seed, n_jobs=1, solver="liblinear")
    start_train = time.perf_counter()
    clf.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start_train

    start = time.perf_counter()
    val_pred = clf.predict(x_val)
    val_infer = (time.perf_counter() - start) * 1000 / max(1, len(y_val))
    start = time.perf_counter()
    test_pred = clf.predict(x_test)
    test_infer = (time.perf_counter() - start) * 1000 / max(1, len(y_test))

    val_score = macro_metrics(y_val, val_pred, 0.0, val_infer)
    test_score = macro_metrics(y_test, test_pred, 0.0, test_infer)
    result = {
        "model": args.model,
        "best_val_f1_macro": val_score["f1_macro"],
        "test": test_score,
        "history": [{"epoch": 1, "train_loss": 0.0, "val": val_score}],
        "num_parameters": int(x_train.shape[1] * len(encoder.classes_)),
        "class_to_id": {label: int(idx) for idx, label in enumerate(encoder.classes_)},
        "train_seconds": round(train_seconds, 3),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output_dir / "metrics.json"), "test": test_score}, indent=2))


if __name__ == "__main__":
    main()
