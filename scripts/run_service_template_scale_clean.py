from __future__ import annotations

import argparse
import hashlib
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from run_service_template_scale_evaluation import (
    VARIANTS,
    build_indices,
    fmt_mean_std,
    jaccard,
    load_embeddings,
    normalize,
    qos_fulfillment,
    read_jsonl,
    service_profile,
)


TARGET_RATIOS = {"hard_negative": 0.45, "same_slice": 0.30, "random": 0.25}


def sample_by_id(samples: list[dict]) -> dict[str, dict]:
    return {sample["intent_id"]: sample for sample in samples}


def select_queries(samples: list[dict], ids: list[str], seed: int, pool_size: int, limit: int) -> list[dict]:
    samples_by_id = sample_by_id(samples)
    ordered = [samples_by_id[x] for x in ids if x in samples_by_id]
    if len(ordered) <= limit:
        return ordered
    rng = np.random.default_rng(seed + pool_size * 1009)
    by_class: dict[str, list[dict]] = {}
    for sample in ordered:
        by_class.setdefault(sample.get("intent_class", sample.get("class_id", "")), []).append(sample)
    selected: list[dict] = []
    per_class = max(1, limit // max(len(by_class), 1))
    for cls in sorted(by_class):
        pool = by_class[cls]
        idx = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        selected.extend(pool[int(i)] for i in idx)
    if len(selected) < limit:
        selected_ids = {sample["intent_id"] for sample in selected}
        remaining = [sample for sample in ordered if sample["intent_id"] not in selected_ids]
        idx = rng.choice(len(remaining), size=min(limit - len(selected), len(remaining)), replace=False)
        selected.extend(remaining[int(i)] for i in idx)
    return selected


def clean_source_pool(
    query: dict,
    pool_size: int,
    seed: int,
    ids: list[str],
    samples_by_id: dict[str, dict],
    by_group: dict[str, list[str]],
    by_slice: dict[str, list[str]],
    by_class: dict[str, list[str]],
) -> tuple[str, list[str], list[str], list[str]]:
    ids_set = set(ids)
    query_code = zlib.crc32(query["intent_id"].encode("utf-8"))
    rng = np.random.default_rng(seed * 1_000_003 + pool_size * 1009 + query_code)
    positive_options = [
        x
        for x in query.get("positive_ids", [])
        if x in ids_set and x != query["intent_id"] and samples_by_id[x].get("intent_text") != query.get("intent_text")
    ]
    if not positive_options:
        positive_options = [
            x
            for x in by_group.get(query["positive_group_id"], [])
            if x in ids_set and x != query["intent_id"] and samples_by_id[x].get("intent_text") != query.get("intent_text")
        ]
    if not positive_options:
        raise RuntimeError(f"No clean positive for query {query['intent_id']}")
    positive_id = str(rng.choice(positive_options))

    same_group = set(by_group.get(query["positive_group_id"], []))
    exact_text = {
        x for x in ids if samples_by_id[x].get("intent_text", "") == query.get("intent_text", "")
    }
    forbidden = same_group | exact_text | {query["intent_id"], positive_id}
    explicit_hard = {
        x for x in query.get("hard_negative_ids", []) if x in ids_set and x not in forbidden
    }
    confusable_class = str(query.get("confusable_with", ""))
    semantic_hard = {
        x for x in by_class.get(confusable_class, []) if x in ids_set and x not in forbidden
    }
    hard_pool = sorted(explicit_hard | semantic_hard)
    same_slice_pool = sorted(
        x
        for x in by_slice.get(query.get("slice_type", ""), [])
        if x in ids_set and x not in forbidden and x not in set(hard_pool)
    )
    random_pool = sorted(
        x
        for x in ids
        if x not in forbidden and x not in set(hard_pool) and x not in set(same_slice_pool)
    )
    if not hard_pool or not same_slice_pool or not random_pool:
        raise RuntimeError(
            f"Clean category unavailable for {query['intent_id']}: "
            f"hard={len(hard_pool)}, same_slice={len(same_slice_pool)}, random={len(random_pool)}"
        )

    n = pool_size - 1
    hard_count = int(round(n * TARGET_RATIOS["hard_negative"]))
    same_slice_count = int(round(n * TARGET_RATIOS["same_slice"]))
    random_count = n - hard_count - same_slice_count
    source_ids = [positive_id]
    categories = ["positive"]
    for category, source_pool, count in [
        ("hard_negative", hard_pool, hard_count),
        ("same_slice", same_slice_pool, same_slice_count),
        ("random", random_pool, random_count),
    ]:
        source_ids.extend(str(x) for x in rng.choice(source_pool, size=count, replace=True))
        categories.extend([category] * count)
    permutation = rng.permutation(len(source_ids))
    source_ids = [source_ids[int(i)] for i in permutation]
    categories = [categories[int(i)] for i in permutation]
    variant_ids = [f"{source_id}::clean-s{seed}-p{pool_size}-q{query_code}-v{rank}" for rank, source_id in enumerate(source_ids)]
    return positive_id, source_ids, categories, variant_ids


def evaluate_clean(
    samples: list[dict],
    ids: list[str],
    embeddings: np.ndarray,
    pool_size: int,
    seed: int,
    noise_std: float,
    max_queries: int,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float | str | bool]]:
    samples_by_id = sample_by_id(samples)
    by_group, by_slice = build_indices(samples)
    by_class: dict[str, list[str]] = {}
    for sample in samples:
        by_class.setdefault(str(sample.get("intent_class", "")), []).append(sample["intent_id"])
    id_to_idx = {intent_id: i for i, intent_id in enumerate(ids)}
    queries = select_queries(samples, ids, seed, pool_size, max_queries)
    stores: dict[str, list[float]] = {
        key: []
        for key in [
            "template_success@1",
            "template_success@5",
            "profile_success@1",
            "slice_match@1",
            "function_jaccard@1",
            "resource_jaccard@1",
            "fulfillment@1",
            "sla_violation@1",
            "avg_qos_gap@1",
            "overprovision@1",
            "fulfillment@5",
            "sla_violation@5",
            "avg_qos_gap@5",
            "overprovision@5",
            "decision_time_ms",
        ]
    }
    category_counts = Counter()
    total_unique_sources = 0
    exact_text_leaks = 0
    same_group_leaks = 0
    digest = hashlib.sha256()

    for query in queries:
        positive_id, source_ids, categories, variant_ids = clean_source_pool(
            query, pool_size, seed, ids, samples_by_id, by_group, by_slice, by_class
        )
        category_counts.update(categories)
        total_unique_sources += len(set(source_ids))
        digest.update("\n".join(variant_ids).encode("utf-8"))
        candidates = [samples_by_id[x] for x in source_ids]
        exact_text_leaks += sum(
            candidate.get("intent_text", "") == query.get("intent_text", "")
            for candidate, category in zip(candidates, categories)
            if category != "positive"
        )
        same_group_leaks += sum(
            candidate.get("positive_group_id") == query.get("positive_group_id")
            for candidate, category in zip(candidates, categories)
            if category != "positive"
        )
        candidate_matrix = embeddings[[id_to_idx[x] for x in source_ids]].copy()
        query_code = zlib.crc32(query["intent_id"].encode("utf-8"))
        noise_rng = np.random.default_rng(seed * 7_000_001 + pool_size * 9173 + query_code)
        candidate_matrix = normalize(candidate_matrix + noise_rng.normal(0.0, noise_std, candidate_matrix.shape))
        query_embedding = embeddings[id_to_idx[query["intent_id"]]]
        positive_mask = np.asarray([x == positive_id and c == "positive" for x, c in zip(source_ids, categories)])

        elapsed = 0.0
        topk = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            scores = candidate_matrix @ query_embedding
            k = min(5, len(scores))
            topk = np.argpartition(-scores, kth=k - 1)[:k]
            topk = topk[np.argsort(-scores[topk])]
            elapsed += time.perf_counter() - t0
        assert topk is not None
        stores["decision_time_ms"].append(elapsed / repeats * 1000.0)
        top = int(topk[0])
        selected = candidates[top]
        top_candidates = [candidates[int(i)] for i in topk]
        stores["template_success@1"].append(float(positive_mask[top]))
        stores["template_success@5"].append(float(positive_mask[topk].any()))
        stores["profile_success@1"].append(float(service_profile(query) == service_profile(selected)))
        stores["slice_match@1"].append(float(query.get("slice_type") == selected.get("slice_type")))
        stores["function_jaccard@1"].append(jaccard(query.get("network_functions", []), selected.get("network_functions", [])))
        stores["resource_jaccard@1"].append(jaccard(query.get("resources", []), selected.get("resources", [])))
        f_ratio, all_met, gap, over = qos_fulfillment(query, selected)
        stores["fulfillment@1"].append(f_ratio)
        stores["sla_violation@1"].append(float(not all_met))
        stores["avg_qos_gap@1"].append(gap)
        stores["overprovision@1"].append(over)
        best = sorted((qos_fulfillment(query, candidate) for candidate in top_candidates), key=lambda x: (-x[0], x[2], x[3]))[0]
        stores["fulfillment@5"].append(best[0])
        stores["sla_violation@5"].append(float(not best[1]))
        stores["avg_qos_gap@5"].append(best[2])
        stores["overprovision@5"].append(best[3])

    metrics = {key: float(np.mean(values)) for key, values in stores.items()}
    metrics["decision_time_p95_ms"] = float(np.percentile(stores["decision_time_ms"], 95))
    metrics["throughput_qps"] = 1000.0 / max(metrics["decision_time_ms"], 1e-12)
    metrics["index_memory_mib_float32"] = pool_size * embeddings.shape[1] * 4 / (1024**2)
    metrics["num_queries"] = float(len(queries))
    distractors = len(queries) * (pool_size - 1)
    audit: dict[str, float | str | bool] = {
        "seed": seed,
        "pool_size": pool_size,
        "num_queries": len(queries),
        "hard_negative_fraction": category_counts["hard_negative"] / distractors,
        "same_slice_fraction": category_counts["same_slice"] / distractors,
        "random_fraction": category_counts["random"] / distractors,
        "unique_variant_id_fraction": 1.0,
        "unique_source_id_fraction": total_unique_sources / (len(queries) * pool_size),
        "same_group_leak_count": same_group_leaks,
        "exact_text_leak_count": exact_text_leaks,
        "candidate_variant_digest": digest.hexdigest(),
        "clean_checks_passed": same_group_leaks == 0 and exact_text_leaks == 0,
    }
    return metrics, audit


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "variant", "pool_size"]
    metric_cols = [c for c in per_seed.columns if c not in {*keys, "seed"}]
    bounded_metrics = {
        "template_success@1",
        "template_success@5",
        "profile_success@1",
        "slice_match@1",
        "function_jaccard@1",
        "resource_jaccard@1",
        "fulfillment@1",
        "sla_violation@1",
        "fulfillment@5",
        "sla_violation@5",
    }
    rows = []
    for group_keys, group in per_seed.groupby(keys, sort=False):
        row = dict(zip(keys, group_keys))
        row["runs"] = len(group)
        critical = float(t.ppf(0.975, len(group) - 1))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            half = critical * sd / len(values) ** 0.5
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            low, high = mean - half, mean + half
            if metric in bounded_metrics:
                low, high = max(0.0, low), min(1.0, high)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, audit: pd.DataFrame, path: Path) -> None:
    rows = summary.sort_values("template_success@5_mean", ascending=False)
    clean_ok = bool(audit["clean_checks_passed"].all())
    lines = [
        "# Clean Service-Template Candidate Scale Evaluation",
        "",
        "## Protocol",
        "",
        "- Test-only source gallery: 1,080 Challenge-NeutralKG held-out intents.",
        "- Scale: 10,000 uniquely identified candidate variants per query; source profiles may repeat and are reported separately.",
        "- Queries: exactly 512 stratified queries for each model and seed.",
        "- Distractor allocation: 45% hard negative, 30% same-slice, and 25% random.",
        "- Clean exclusions: query ID, every same-positive-group source, and every exact-text duplicate are excluded from distractors.",
        "- Explicit hard-negative IDs are supplemented only by the dataset's predeclared confusable class.",
        "- Candidate variant IDs and source selections are fixed before model-specific embedding perturbation.",
        "",
        "## Audit",
        "",
        f"- All clean checks passed: `{clean_ok}`.",
        f"- Realized hard/same-slice/random fractions: {audit['hard_negative_fraction'].mean():.4f}/{audit['same_slice_fraction'].mean():.4f}/{audit['random_fraction'].mean():.4f}.",
        f"- Same-group leaks: {int(audit['same_group_leak_count'].sum())}; exact-text leaks: {int(audit['exact_text_leak_count'].sum())}.",
        f"- Unique variant-ID fraction: {audit['unique_variant_id_fraction'].mean():.4f}; unique source-ID fraction: {audit['unique_source_id_fraction'].mean():.4f}.",
        "",
        "## Results at 10,000 candidates",
        "",
        "| Model | Template@1 (95% CI) | Template@5 (95% CI) | Fulfill@5 | Mean ms | P95 ms | QPS | Index MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['model']} | {row['template_success@1_mean']:.4f} "
            f"[{row['template_success@1_ci95_low']:.4f}, {row['template_success@1_ci95_high']:.4f}] | "
            f"{row['template_success@5_mean']:.4f} "
            f"[{row['template_success@5_ci95_low']:.4f}, {row['template_success@5_ci95_high']:.4f}] | "
            f"{row['fulfillment@5_mean']:.4f} | {row['decision_time_ms_mean']:.3f} | "
            f"{row['decision_time_p95_ms_mean']:.3f} | {row['throughput_qps_mean']:.1f} | "
            f"{row['index_memory_mib_float32_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This protocol removes group-level and exact-text contamination while preserving a controlled 10,000-variant repository. It supports a candidate-variant scalability claim, not a claim that 10,000 independent operator templates were collected.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--retrieval-root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--baseline-root", default="outputs/service_template_scale_baseline_embeddings")
    parser.add_argument("--output-dir", default="outputs/service_template_scale_clean")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--embedding-split", default="test_group_projection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(Path(args.processed_dir) / "test.jsonl")
    sample_ids = {sample["intent_id"] for sample in samples}
    baseline_variants = {
        "SBERT-MiniLM": "sbert_minilm",
        "Text+GAT": "text_gat",
        "Text+GraphTransformer": "text_graph_transformer",
        "GraphTransformer": "graph_transformer",
    }
    variants = [(name, variant, Path(args.retrieval_root)) for name, variant in VARIANTS.items()]
    variants.extend((name, variant, Path(args.baseline_root)) for name, variant in baseline_variants.items())
    rows = []
    audit_rows = []
    reference_digests: dict[int, str] = {}
    for seed in args.seeds:
        for model, variant, root in variants:
            path = root / f"seed_{seed}" / variant / args.embedding_split / "query_embeddings.csv"
            if not path.exists():
                continue
            ids, embeddings = load_embeddings(path)
            keep = [i for i, intent_id in enumerate(ids) if intent_id in sample_ids]
            ids = [ids[i] for i in keep]
            embeddings = embeddings[keep]
            print(f"[clean-scale] model={model} seed={seed} pool={args.pool_size}", flush=True)
            metrics, audit = evaluate_clean(
                samples, ids, embeddings, args.pool_size, seed, args.noise_std, args.max_queries, args.repeats
            )
            digest = str(audit["candidate_variant_digest"])
            if seed in reference_digests and reference_digests[seed] != digest:
                raise RuntimeError(f"Candidate pool mismatch for seed {seed}, model {model}")
            reference_digests.setdefault(seed, digest)
            rows.append({"model": model, "variant": variant, "seed": seed, "pool_size": args.pool_size, **metrics})
            if model == "KCGT":
                audit_rows.append(audit)
            print(
                f"[done] {model} seed={seed} T@1={metrics['template_success@1']:.4f} "
                f"T@5={metrics['template_success@5']:.4f} mean={metrics['decision_time_ms']:.3f}ms",
                flush=True,
            )
    per_seed = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    summary = summarize(per_seed)
    per_seed.to_csv(output_dir / "clean_scale_per_seed.csv", index=False)
    summary.to_csv(output_dir / "clean_scale_summary.csv", index=False)
    audit.to_csv(output_dir / "clean_candidate_pool_audit.csv", index=False)
    write_report(summary, audit, output_dir / "clean_scale_report.md")
    reports = Path("outputs/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "service_template_scale_clean_report.md").write_text(
        (output_dir / "clean_scale_report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print((output_dir / "clean_scale_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
