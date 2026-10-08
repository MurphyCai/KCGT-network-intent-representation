from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from run_service_template_scale_evaluation import (
    build_indices,
    choose_distractors,
    choose_positive,
    read_jsonl,
)


def select_queries(samples: list[dict], ids: list[str], seed: int, pool_size: int, limit: int) -> list[dict]:
    samples_by_id = {sample["intent_id"]: sample for sample in samples}
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


def candidate_ids_for_query(
    query: dict,
    pool_size: int,
    seed: int,
    ids: list[str],
    by_group: dict[str, list[str]],
    by_slice: dict[str, list[str]],
) -> tuple[str, list[str]]:
    ids_set = set(ids)
    query_code = zlib.crc32(query["intent_id"].encode("utf-8"))
    rng = np.random.default_rng(seed * 1_000_003 + pool_size * 1009 + query_code)
    positive_id = choose_positive(query, ids_set, by_group, rng)
    distractors = choose_distractors(
        query,
        pool_size - 1,
        ids,
        ids_set,
        by_slice,
        rng,
        forbidden={positive_id},
    )
    candidate_ids = [positive_id, *distractors]
    rng.shuffle(candidate_ids)
    return positive_id, candidate_ids


def digest_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def confidence_intervals(per_seed: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["model", "variant", "pool_size"]
    metric_cols = [c for c in per_seed.columns if c not in {*key_cols, "seed"}]
    rows: list[dict] = []
    for keys, group in per_seed.groupby(key_cols, sort=False):
        n = len(group)
        critical = float(t.ppf(0.975, n - 1)) if n > 1 else math.nan
        row = dict(zip(key_cols, keys))
        row["runs"] = n
        row["t_critical_95"] = critical
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            half = critical * sd / math.sqrt(len(values)) if len(values) > 1 else math.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd_sample"] = sd
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def audit_pools(samples: list[dict], ids: list[str], seeds: list[int], pool_sizes: list[int], max_queries: int) -> pd.DataFrame:
    samples_by_id = {sample["intent_id"]: sample for sample in samples}
    by_group, by_slice = build_indices(samples)
    ids_set = set(ids)
    rows: list[dict] = []
    for seed in seeds:
        for pool_size in pool_sizes:
            queries = select_queries(samples, ids, seed, pool_size, max_queries)
            category = Counter()
            total_unique = 0
            total_false_group = 0
            total_exact_text = 0
            total_same_source_template = 0
            queries_with_false_group = 0
            queries_with_exact_text = 0
            deterministic = True
            positive_count_ok = True
            aggregate_hash = hashlib.sha256()
            for query in queries:
                positive_id, candidate_ids = candidate_ids_for_query(query, pool_size, seed, ids, by_group, by_slice)
                positive_id_2, candidate_ids_2 = candidate_ids_for_query(query, pool_size, seed, ids, by_group, by_slice)
                deterministic &= positive_id == positive_id_2 and digest_ids(candidate_ids) == digest_ids(candidate_ids_2)
                aggregate_hash.update(digest_ids(candidate_ids).encode("ascii"))
                positive_count_ok &= candidate_ids.count(positive_id) == 1
                total_unique += len(set(candidate_ids))

                hard_set = {
                    x for x in query.get("hard_negative_ids", []) if x in ids_set and x not in {query["intent_id"], positive_id}
                }
                same_slice_set = {
                    x
                    for x in by_slice.get(query.get("slice_type", ""), [])
                    if x in ids_set and x not in {query["intent_id"], positive_id} and x not in hard_set
                }
                false_group = 0
                exact_text = 0
                same_source_template = 0
                for candidate_id in candidate_ids:
                    if candidate_id == positive_id:
                        category["positive"] += 1
                        continue
                    candidate = samples_by_id[candidate_id]
                    if candidate_id in hard_set:
                        category["hard_negative"] += 1
                    elif candidate_id in same_slice_set:
                        category["same_slice"] += 1
                    else:
                        category["random_or_fallback"] += 1
                    false_group += int(candidate.get("positive_group_id") == query.get("positive_group_id"))
                    exact_text += int(candidate.get("intent_text", "") == query.get("intent_text", ""))
                    same_source_template += int(
                        candidate.get("source_template_id") == query.get("source_template_id")
                    )
                total_false_group += false_group
                total_exact_text += exact_text
                total_same_source_template += same_source_template
                queries_with_false_group += int(false_group > 0)
                queries_with_exact_text += int(exact_text > 0)

            total_candidates = len(queries) * pool_size
            total_distractors = len(queries) * (pool_size - 1)
            rows.append(
                {
                    "seed": seed,
                    "pool_size": pool_size,
                    "num_queries": len(queries),
                    "candidate_pool_digest": aggregate_hash.hexdigest(),
                    "deterministic_regeneration": deterministic,
                    "exactly_one_selected_positive": positive_count_ok,
                    "hard_negative_fraction": category["hard_negative"] / total_distractors,
                    "same_slice_fraction": category["same_slice"] / total_distractors,
                    "random_or_fallback_fraction": category["random_or_fallback"] / total_distractors,
                    "unique_source_id_fraction": total_unique / total_candidates,
                    "source_id_duplicate_fraction": 1.0 - total_unique / total_candidates,
                    "false_group_distractor_fraction": total_false_group / total_distractors,
                    "queries_with_false_group_distractor_fraction": queries_with_false_group / len(queries),
                    "exact_query_text_distractor_fraction": total_exact_text / total_distractors,
                    "queries_with_exact_text_distractor_fraction": queries_with_exact_text / len(queries),
                    "same_source_template_distractor_fraction": total_same_source_template / total_distractors,
                }
            )
    return pd.DataFrame(rows)


def embedding_order_audit(project: Path, seeds: list[int], split: str) -> pd.DataFrame:
    roots = {
        "KCGT": project / "outputs/step30_neutral_kg_retrieval",
        "SBERT-MiniLM": project / "outputs/service_template_scale_baseline_embeddings",
        "Text+GAT": project / "outputs/service_template_scale_baseline_embeddings",
        "Text+GraphTransformer": project / "outputs/service_template_scale_baseline_embeddings",
        "GraphTransformer": project / "outputs/service_template_scale_baseline_embeddings",
    }
    variants = {
        "KCGT": "full_group",
        "SBERT-MiniLM": "sbert_minilm",
        "Text+GAT": "text_gat",
        "Text+GraphTransformer": "text_graph_transformer",
        "GraphTransformer": "graph_transformer",
    }
    rows = []
    for seed in seeds:
        reference: list[str] | None = None
        for model, root in roots.items():
            path = root / f"seed_{seed}" / variants[model] / split / "query_embeddings.csv"
            frame = pd.read_csv(path, usecols=["intent_id"])
            ids = frame["intent_id"].astype(str).tolist()
            if reference is None:
                reference = ids
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "num_ids": len(ids),
                    "same_id_set_as_kcgt": set(ids) == set(reference),
                    "same_id_order_as_kcgt": ids == reference,
                    "id_order_digest": digest_ids(ids),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    ci: pd.DataFrame,
    audit: pd.DataFrame,
    order: pd.DataFrame,
    output: Path,
    noise_summary: pd.DataFrame | None = None,
) -> None:
    kcgt = ci[(ci["model"] == "KCGT") & (ci["pool_size"] == ci["pool_size"].max())].iloc[0]
    largest = audit[audit["pool_size"] == audit["pool_size"].max()]
    lines = [
        "# Service-Template Scale Statistical Credibility Audit",
        "",
        "## Statistical protocol",
        "",
        "- Results use three matched seeds. Two-sided 95% confidence intervals are computed as mean +/- t(0.975, 2) * sample-SD / sqrt(3); t(0.975, 2)=4.3027.",
        f"- Every model/scale/seed row contains exactly 512 stratified queries: `{bool((ci['num_queries_mean'] == 512).all())}`.",
        f"- Embedding files share the same intent-ID set and order across audited models: `{bool(order['same_id_set_as_kcgt'].all() and order['same_id_order_as_kcgt'].all())}`.",
        "- Candidate IDs are generated from query ID, seed, and pool size before embedding perturbation; model dimensionality therefore cannot alter candidate identities.",
        "",
        "## Key 10,000-candidate confidence intervals",
        "",
        f"- KCGT Template@1: {kcgt['template_success@1_mean']:.4f} [{kcgt['template_success@1_ci95_low']:.4f}, {kcgt['template_success@1_ci95_high']:.4f}].",
        f"- KCGT Template@5: {kcgt['template_success@5_mean']:.4f} [{kcgt['template_success@5_ci95_low']:.4f}, {kcgt['template_success@5_ci95_high']:.4f}].",
        f"- KCGT mean decision latency: {kcgt['decision_time_ms_mean']:.4f} [{kcgt['decision_time_ms_ci95_low']:.4f}, {kcgt['decision_time_ms_ci95_high']:.4f}] ms.",
        "",
        "## Candidate-pool audit",
        "",
        "- The sampler requests 45% hard negatives, 30% same-slice distractors, and 25% random distractors. The CSV reports realized semantic categories after fallback.",
        f"- At 10,000 candidates, realized hard-negative fraction is {largest['hard_negative_fraction'].mean():.4f}; same-slice is {largest['same_slice_fraction'].mean():.4f}; random/fallback is {largest['random_or_fallback_fraction'].mean():.4f}.",
        f"- Deterministic regeneration passed for every audited pool: `{bool(audit['deterministic_regeneration'].all())}`; exactly one selected positive ID was present: `{bool(audit['exactly_one_selected_positive'].all())}`.",
        f"- At 10,000 candidates, only {largest['unique_source_id_fraction'].mean():.4f} of candidate occurrences have unique source intent IDs. The remaining occurrences are perturbation variants drawn from repeated source profiles, not independent operator templates.",
        f"- Same-group distractors occur in {largest['queries_with_false_group_distractor_fraction'].mean():.4f} of queries and account for {largest['false_group_distractor_fraction'].mean():.4f} of distractor occurrences. These are potential false negatives under a group-level relevance definition.",
        f"- Exact query-text distractors account for {largest['exact_query_text_distractor_fraction'].mean():.6f} of distractors; same source-template distractors account for {largest['same_source_template_distractor_fraction'].mean():.4f}.",
        "",
        "## Interpretation",
        "",
        "The three-seed confidence intervals quantify run-level uncertainty but remain wide because n=3. Candidate pools are reproducible and model-aligned, but the 10,000-item repository is a controlled template-variant stress test built from 1,080 held-out source intents. It must not be described as 10,000 independently collected real service templates. Source-ID reuse and same-group false-negative contamination are explicitly reported and should be corrected or treated as limitations before a strong scalability claim.",
    ]
    if noise_summary is not None and not noise_summary.empty:
        sensitivity = noise_summary[noise_summary["pool_size"] == noise_summary["pool_size"].max()].sort_values(
            "noise_std"
        )
        lines.extend(
            [
                "",
                "## Perturbation sensitivity at 10,000 candidates",
                "",
                "| Noise std | Template@1 | Template@5 | Fulfill@5 | SLA violation@5 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sensitivity.iterrows():
            lines.append(
                f"| {row['noise_std']:.3f} | {row['template_success@1_mean']:.4f} | "
                f"{row['template_success@5_mean']:.4f} | {row['fulfillment@5_mean']:.4f} | "
                f"{row['sla_violation@5_mean']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Retrieval accuracy decreases as perturbation grows, while top-five QoS fulfillment increases because the candidate variants become more diverse. The adopted sigma=0.015 is a middle sensitivity point and must be reported as a simulation parameter.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/service_template_scale_audit/service_scale_confidence_intervals.csv`",
            "- `outputs/service_template_scale_audit/candidate_pool_audit.csv`",
            "- `outputs/service_template_scale_audit/embedding_order_audit.csv`",
            "- `outputs/service_template_scale_noise_sensitivity/noise_sensitivity_per_seed.csv`",
            "- `outputs/service_template_scale_noise_sensitivity/noise_sensitivity_summary.csv`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--per-seed", default="outputs/service_template_scale/service_template_scale_per_seed.csv")
    parser.add_argument("--output-dir", default="outputs/service_template_scale_audit")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--embedding-split", default="test_group_projection")
    args = parser.parse_args()

    project = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(Path(args.processed_dir) / "test.jsonl")
    id_frame = pd.read_csv(
        project / "outputs/step30_neutral_kg_retrieval/seed_1/full_group" / args.embedding_split / "query_embeddings.csv",
        usecols=["intent_id"],
    )
    ids = id_frame["intent_id"].astype(str).tolist()

    ci = confidence_intervals(pd.read_csv(args.per_seed))
    audit = audit_pools(samples, ids, args.seeds, args.pool_sizes, args.max_queries)
    order = embedding_order_audit(project, args.seeds, args.embedding_split)
    ci.to_csv(output_dir / "service_scale_confidence_intervals.csv", index=False)
    audit.to_csv(output_dir / "candidate_pool_audit.csv", index=False)
    order.to_csv(output_dir / "embedding_order_audit.csv", index=False)
    noise_path = Path("outputs/service_template_scale_noise_sensitivity/noise_sensitivity_summary.csv")
    noise_summary = pd.read_csv(noise_path) if noise_path.exists() else None
    write_report(ci, audit, order, output_dir / "statistical_credibility_report.md", noise_summary)
    reports = Path("outputs/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "service_scale_statistical_credibility.md").write_text(
        (output_dir / "statistical_credibility_report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print((output_dir / "statistical_credibility_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
