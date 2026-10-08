"""Evaluate frozen KCGT representations on an external 3GPP 5QI/QoS protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.evaluate_kcgt_retrieval import extract_embeddings, load_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def standardized_qos(records: list[dict]) -> np.ndarray:
    rows = []
    for x in records:
        rows.append([
            np.log1p(float(x["packet_delay_budget_ms"])) / np.log1p(300.0),
            (np.log10(float(x["packet_error_rate"])) + 6.0) / 4.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def score_qos(query: list[dict], gallery: list[dict]) -> np.ndarray:
    q, g = standardized_qos(query), standardized_qos(gallery)
    # Priority and resource type are profile outputs, not query-side selector inputs.
    weights = np.asarray([0.5, 0.5], dtype=np.float32)
    return -np.sqrt(np.sum(((q[:, None, :] - g[None, :, :]) ** 2) * weights, axis=2))


def rerank_with_constraints(cosine: np.ndarray, query: list[dict], gallery: list[dict], topk: int = 10) -> np.ndarray:
    qos = score_qos(query, gallery)
    result = np.full_like(cosine, -1e9)
    for i, request in enumerate(query):
        shortlist = np.argsort(-cosine[i])[:topk]
        for j in shortlist:
            profile = gallery[j]
            feasible = (
                float(profile["packet_delay_budget_ms"]) <= float(request["packet_delay_budget_ms"])
                and float(profile["packet_error_rate"]) <= float(request["packet_error_rate"])
            )
            result[i, j] = cosine[i, j] + 0.20 * qos[i, j] + 0.06 * float(feasible)
    return result


def evaluate_scores(name: str, scores: np.ndarray, query: list[dict], gallery: list[dict], elapsed_ms: float, seed: int) -> dict:
    order = np.argsort(-scores, axis=1)
    q_target = np.asarray([int(x["fiveqi"]) for x in query])
    g_target = np.asarray([int(x["fiveqi"]) for x in gallery])
    top1 = order[:, 0]
    exact1 = g_target[top1] == q_target
    exact5 = np.asarray([target in g_target[idx[:5]] for target, idx in zip(q_target, order)])
    family = np.asarray([gallery[j]["service_family"] == query[i]["service_family"] for i, j in enumerate(top1)])
    resource = np.asarray([gallery[j]["resource_type"] == query[i]["resource_type"] for i, j in enumerate(top1)])
    feasible = np.asarray([
        float(gallery[j]["packet_delay_budget_ms"]) <= float(query[i]["packet_delay_budget_ms"])
        and float(gallery[j]["packet_error_rate"]) <= float(query[i]["packet_error_rate"])
        for i, j in enumerate(top1)
    ])
    ranks = []
    for i, target in enumerate(q_target):
        ranks.append(int(np.where(g_target[order[i]] == target)[0][0]) + 1)
    return {
        "model": name, "seed": seed, "queries": len(query), "profiles": len(gallery),
        "fiveqi_success@1": float(exact1.mean()), "fiveqi_success@5": float(exact5.mean()),
        "fiveqi_mrr": float(np.mean(1.0 / np.asarray(ranks))), "service_family@1": float(family.mean()),
        "resource_type@1": float(resource.mean()), "standardized_sla_fulfillment@1": float(feasible.mean()),
        "standardized_sla_violation@1": float(1.0 - feasible.mean()),
        "decision_time_ms_per_query": elapsed_ms / len(query),
    }


def bootstrap_ci(scores: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    means = [scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(n)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/external/5qi")
    parser.add_argument("--checkpoint-root", default="outputs/step30_neutral_kg_ablation")
    parser.add_argument(
        "--checkpoint-pattern",
        default="seed_{seed}/full_group/best.pt",
        help="Checkpoint path relative to --checkpoint-root; {seed} is substituted.",
    )
    parser.add_argument("--output-dir", default="outputs/external_5qi_protocol")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path, gallery_path = data_dir / "model_queries.jsonl", data_dir / "model_gallery.jsonl"
    query, gallery = read_jsonl(query_path), read_jsonl(gallery_path)
    device = torch.device(args.device)
    rows, detailed = [], []

    for seed in args.seeds:
        checkpoint = Path(args.checkpoint_root) / args.checkpoint_pattern.format(seed=seed)
        model, artifacts, _ = load_model(checkpoint, device)
        q = extract_embeddings(model, artifacts, query_path, args.batch_size, device, True)["embeddings"].numpy()
        g = extract_embeddings(model, artifacts, gallery_path, args.batch_size, device, True)["embeddings"].numpy()
        start = time.perf_counter()
        cosine = q @ g.T
        elapsed = (time.perf_counter() - start) * 1000.0
        rows.append(evaluate_scores("KCGT", cosine, query, gallery, elapsed, seed))
        start = time.perf_counter()
        reranked = rerank_with_constraints(cosine, query, gallery)
        elapsed = (time.perf_counter() - start) * 1000.0
        rows.append(evaluate_scores("KCGT + standardized-QoS reranking", reranked, query, gallery, elapsed, seed))

    start = time.perf_counter()
    qos_scores = score_qos(query, gallery)
    elapsed = (time.perf_counter() - start) * 1000.0
    rows.append(evaluate_scores("Standardized QoS-only", qos_scores, query, gallery, elapsed, 0))

    try:
        from sentence_transformers import SentenceTransformer
        cache_root = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        snapshots = sorted(cache_root.glob("*"))
        model_ref = str(snapshots[-1]) if snapshots else "sentence-transformers/all-MiniLM-L6-v2"
        encoder = SentenceTransformer(model_ref, device=str(device), local_files_only=bool(snapshots))
        q_text = encoder.encode([x["intent_text"] for x in query], normalize_embeddings=True, show_progress_bar=False)
        g_text = encoder.encode([x["intent_text"] for x in gallery], normalize_embeddings=True, show_progress_bar=False)
        start = time.perf_counter()
        sbert_scores = q_text @ g_text.T
        elapsed = (time.perf_counter() - start) * 1000.0
        rows.append(evaluate_scores("SBERT-MiniLM", sbert_scores, query, gallery, elapsed, 0))
    except Exception as exc:
        print(f"[warning] SBERT skipped: {exc}")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(output_dir / "external_5qi_per_run.csv", index=False)
    summary = per_run.groupby("model", sort=False).agg(
        runs=("seed", "count"),
        fiveqi_success_at_1=("fiveqi_success@1", "mean"), fiveqi_success_at_5=("fiveqi_success@5", "mean"),
        fiveqi_success_at_1_sd=("fiveqi_success@1", "std"), fiveqi_success_at_5_sd=("fiveqi_success@5", "std"),
        fiveqi_mrr=("fiveqi_mrr", "mean"), service_family_at_1=("service_family@1", "mean"),
        fiveqi_mrr_sd=("fiveqi_mrr", "std"), service_family_at_1_sd=("service_family@1", "std"),
        resource_type_at_1=("resource_type@1", "mean"), sla_fulfillment_at_1=("standardized_sla_fulfillment@1", "mean"),
        resource_type_at_1_sd=("resource_type@1", "std"), sla_fulfillment_at_1_sd=("standardized_sla_fulfillment@1", "std"),
        decision_time_ms=("decision_time_ms_per_query", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "external_5qi_summary.csv", index=False)

    lines = [
        "# External 3GPP 5QI/QoS Test Protocol", "",
        "## Scope", "",
        "This test-only protocol uses 20 standardized 5QI profiles and 160 independently phrased requests derived from the example-service categories in 3GPP TS 23.501 Table 5.7.4-1. It is disjoint from KNI-Intent generation templates and retains a separate raw KG schema. An explicit type-level adapter is used only because the evaluated KCGT checkpoints are frozen.", "",
        "Query text and adapted query graphs do not contain the target 5QI identifier. Core SLA fulfillment uses only standardized packet-delay-budget and packet-error-rate fields; authored auxiliary KCGT fields are excluded.", "",
        "## Results", "",
        "| Method | 5QI@1 | 5QI@5 | MRR | Family@1 | Resource@1 | SLA fulfill@1 | ms/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, x in summary.iterrows():
        def value(mean: float, sd: float) -> str:
            return f"{mean:.4f} +/- {sd:.4f}" if pd.notna(sd) else f"{mean:.4f}"
        lines.append(f"| {x.model} | {value(x.fiveqi_success_at_1, x.fiveqi_success_at_1_sd)} | {value(x.fiveqi_success_at_5, x.fiveqi_success_at_5_sd)} | {value(x.fiveqi_mrr, x.fiveqi_mrr_sd)} | {value(x.service_family_at_1, x.service_family_at_1_sd)} | {value(x.resource_type_at_1, x.resource_type_at_1_sd)} | {value(x.sla_fulfillment_at_1, x.sla_fulfillment_at_1_sd)} | {x.decision_time_ms:.4f} |")
    lines.extend(["", "## Interpretation", "", "This is an external standards-profile mapping test, not a live 5G core deployment. It measures whether a frozen representation can select and validate standardized QoS candidates under vocabulary and schema shift. Reported decision time covers candidate scoring and reranking after embeddings have been computed; it is not end-to-end encoder latency."])
    report = "\n".join(lines) + "\n"
    (output_dir / "external_5qi_report.md").write_text(report, encoding="utf-8")
    reports = Path("outputs/reports"); reports.mkdir(parents=True, exist_ok=True)
    (reports / "external_5qi_protocol_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
