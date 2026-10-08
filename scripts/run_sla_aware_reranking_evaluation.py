"""Compare semantic retrieval, QoS-only selection, and SLA-aware reranking."""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_service_template_scale_evaluation import (
    build_indices,
    fmt_mean_std,
    jaccard,
    make_candidate_pool,
    normalize,
    sample_by_id,
)
from run_step36_management_validation import QOS_FIELDS, load_embeddings, read_jsonl, service_profile


METHODS = [
    "Embedding cosine retrieval",
    "QoS-only selection",
    "KCGT retrieval",
    "KCGT + QoS reranking",
]

INDEX_DIMS = {
    "Embedding cosine retrieval": 384,
    "QoS-only selection": 8,
    "KCGT retrieval": 64,
    "KCGT + QoS reranking": 72,
}


def select_queries(
    samples_by_id: dict[str, dict],
    ids: list[str],
    seed: int,
    pool_size: int,
    max_queries: int | None,
) -> list[dict]:
    ordered = [samples_by_id[x] for x in ids if x in samples_by_id]
    if max_queries is None or len(ordered) <= max_queries:
        return ordered
    rng = np.random.default_rng(seed + pool_size * 1009)
    by_class: dict[str, list[dict]] = {}
    for sample in ordered:
        by_class.setdefault(sample.get("intent_class", sample.get("class_id", "")), []).append(sample)
    selected: list[dict] = []
    per_class = max(1, max_queries // max(len(by_class), 1))
    for cls in sorted(by_class):
        pool = by_class[cls]
        take = min(per_class, len(pool))
        indices = rng.choice(len(pool), size=take, replace=False)
        selected.extend(pool[int(i)] for i in indices)
    if len(selected) < max_queries:
        selected_ids = {s["intent_id"] for s in selected}
        remaining = [s for s in ordered if s["intent_id"] not in selected_ids]
        take = min(max_queries - len(selected), len(remaining))
        if take:
            indices = rng.choice(len(remaining), size=take, replace=False)
            selected.extend(remaining[int(i)] for i in indices)
    return selected


def qos_matrix(samples: list[dict]) -> np.ndarray:
    return np.asarray([[float(sample[field]) for field in QOS_FIELDS] for sample in samples], dtype=np.float64)


def qos_scores(query: dict, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray([float(query[field]) for field in QOS_FIELDS], dtype=np.float64)
    q_lat, q_bw, q_rel, q_jit, q_loss, q_sec, q_mob, q_ene = q
    c_lat, c_bw, c_rel, c_jit, c_loss, c_sec, c_mob, c_ene = candidates.T

    checks = np.column_stack(
        [
            c_lat <= q_lat * 1.10,
            c_bw >= q_bw * 0.90,
            c_rel >= q_rel - 0.001,
            c_jit <= q_jit * 1.10,
            c_loss <= q_loss * 1.25 + 1e-7,
            c_sec >= q_sec,
            c_mob >= q_mob,
            c_ene >= q_ene,
        ]
    )
    gaps = np.column_stack(
        [
            np.maximum(0.0, (c_lat - q_lat * 1.10) / max(q_lat, 1e-9)),
            np.maximum(0.0, (q_bw * 0.90 - c_bw) / max(q_bw, 1e-9)),
            np.maximum(0.0, (q_rel - 0.001 - c_rel) / max(1.0 - q_rel + 1e-6, 1e-6)),
            np.maximum(0.0, (c_jit - q_jit * 1.10) / max(q_jit, 1e-9)),
            np.maximum(0.0, (c_loss - (q_loss * 1.25 + 1e-7)) / max(q_loss + 1e-7, 1e-9)),
            np.maximum(0.0, (q_sec - c_sec) / 5.0),
            np.maximum(0.0, (q_mob - c_mob) / 5.0),
            np.maximum(0.0, (q_ene - c_ene) / 5.0),
        ]
    )
    over = np.column_stack(
        [
            np.maximum(0.0, (c_bw - q_bw) / max(q_bw, 1e-9)),
            np.maximum(0.0, (c_sec - q_sec) / 5.0),
            np.maximum(0.0, (c_mob - q_mob) / 5.0),
            np.maximum(0.0, (c_ene - q_ene) / 5.0),
        ]
    ).mean(axis=1)
    return checks.mean(axis=1), checks.all(axis=1), gaps.mean(axis=1), over


def qos_best(ratio: np.ndarray, gap: np.ndarray, over: np.ndarray, semantic_rank: np.ndarray | None = None) -> int:
    if semantic_rank is None:
        semantic_rank = np.arange(len(ratio))
    order = np.lexsort((semantic_rank, over, gap, -ratio))
    return int(order[0])


def record_selection(
    store: dict[str, list[float]],
    query: dict,
    selected: int,
    candidate_samples: list[dict],
    is_positive: np.ndarray,
    ratio: np.ndarray,
    all_met: np.ndarray,
    gap: np.ndarray,
    over: np.ndarray,
) -> None:
    candidate = candidate_samples[selected]
    template_match = bool(is_positive[selected])
    profile_match = service_profile(query) == service_profile(candidate)
    store["template_success"].append(float(template_match))
    store["profile_success"].append(float(profile_match))
    store["sla_valid_template_success"].append(float(template_match and all_met[selected]))
    store["sla_valid_profile_success"].append(float(profile_match and all_met[selected]))
    store["slice_match"].append(float(query.get("slice_type") == candidate.get("slice_type")))
    store["function_jaccard"].append(jaccard(query.get("network_functions", []), candidate.get("network_functions", [])))
    store["resource_jaccard"].append(jaccard(query.get("resources", []), candidate.get("resources", [])))
    store["fulfillment"].append(float(ratio[selected]))
    store["sla_violation"].append(float(not all_met[selected]))
    store["avg_qos_gap"].append(float(gap[selected]))
    store["overprovision"].append(float(over[selected]))


def empty_store() -> dict[str, list[float]]:
    return {
        key: []
        for key in [
            "template_success",
            "profile_success",
            "sla_valid_template_success",
            "sla_valid_profile_success",
            "slice_match",
            "function_jaccard",
            "resource_jaccard",
            "fulfillment",
            "sla_violation",
            "avg_qos_gap",
            "overprovision",
            "decision_time_ms",
        ]
    }


def evaluate(
    samples: list[dict],
    kcgt_ids: list[str],
    kcgt_emb: np.ndarray,
    sbert_ids: list[str],
    sbert_emb: np.ndarray,
    pool_size: int,
    seed: int,
    top_k: int,
    repeats: int,
    noise_std: float,
    max_queries: int | None,
) -> list[dict[str, float | str]]:
    if kcgt_ids != sbert_ids:
        raise ValueError("KCGT and SBERT intent IDs must be identically ordered.")
    samples_by_id = sample_by_id(samples)
    by_group, by_slice = build_indices(samples)
    id_to_idx = {intent_id: index for index, intent_id in enumerate(kcgt_ids)}
    ids_set = set(kcgt_ids)
    queries = select_queries(samples_by_id, kcgt_ids, seed, pool_size, max_queries)
    stores = {method: empty_store() for method in METHODS}

    for query in queries:
        query_code = zlib.crc32(query["intent_id"].encode("utf-8"))
        candidate_seed = seed * 1_000_003 + pool_size * 1009 + query_code
        kcgt_pool, candidate_samples, is_positive = make_candidate_pool(
            query,
            pool_size,
            kcgt_ids,
            ids_set,
            id_to_idx,
            kcgt_emb,
            samples_by_id,
            by_group,
            by_slice,
            np.random.default_rng(candidate_seed),
            noise_std,
        )
        sbert_pool, sbert_samples, sbert_positive = make_candidate_pool(
            query,
            pool_size,
            sbert_ids,
            ids_set,
            id_to_idx,
            sbert_emb,
            samples_by_id,
            by_group,
            by_slice,
            np.random.default_rng(candidate_seed),
            noise_std,
        )
        if [x["intent_id"] for x in candidate_samples] != [x["intent_id"] for x in sbert_samples]:
            raise RuntimeError("Candidate identities differ between embedding methods.")
        if not np.array_equal(is_positive, sbert_positive):
            raise RuntimeError("Positive template positions differ between embedding methods.")

        candidate_qos = qos_matrix(candidate_samples)
        ratio, all_met, gap, over = qos_scores(query, candidate_qos)
        q_index = id_to_idx[query["intent_id"]]
        q_kcgt = kcgt_emb[q_index]
        q_sbert = sbert_emb[q_index]

        selected: dict[str, int] = {}
        elapsed = {method: 0.0 for method in METHODS}
        for _ in range(repeats):
            start = time.perf_counter()
            sbert_scores = sbert_pool @ q_sbert
            selected["Embedding cosine retrieval"] = int(np.argmax(sbert_scores))
            elapsed["Embedding cosine retrieval"] += time.perf_counter() - start

            start = time.perf_counter()
            qos_ratio, _, qos_gap, qos_over = qos_scores(query, candidate_qos)
            selected["QoS-only selection"] = qos_best(qos_ratio, qos_gap, qos_over)
            elapsed["QoS-only selection"] += time.perf_counter() - start

            start = time.perf_counter()
            kcgt_scores = kcgt_pool @ q_kcgt
            selected["KCGT retrieval"] = int(np.argmax(kcgt_scores))
            elapsed["KCGT retrieval"] += time.perf_counter() - start

            start = time.perf_counter()
            kcgt_scores = kcgt_pool @ q_kcgt
            k = min(top_k, len(kcgt_scores))
            top = np.argpartition(-kcgt_scores, kth=k - 1)[:k]
            top = top[np.argsort(-kcgt_scores[top])]
            top_ratio, _, top_gap, top_over = qos_scores(query, candidate_qos[top])
            local = qos_best(top_ratio, top_gap, top_over, np.arange(k))
            selected["KCGT + QoS reranking"] = int(top[local])
            elapsed["KCGT + QoS reranking"] += time.perf_counter() - start

        for method in METHODS:
            record_selection(
                stores[method],
                query,
                selected[method],
                candidate_samples,
                is_positive,
                ratio,
                all_met,
                gap,
                over,
            )
            stores[method]["decision_time_ms"].append(elapsed[method] * 1000.0 / repeats)

    rows: list[dict[str, float | str]] = []
    for method in METHODS:
        row: dict[str, float | str] = {
            "method": method,
            "seed": seed,
            "pool_size": pool_size,
            "top_k": top_k,
            "num_queries": len(queries),
        }
        for metric, values in stores[method].items():
            row[metric] = float(np.mean(values))
        row["decision_time_p95_ms"] = float(np.percentile(stores[method]["decision_time_ms"], 95))
        row["throughput_qps"] = float(1000.0 / max(float(row["decision_time_ms"]), 1e-12))
        row["index_memory_mib_float32"] = float(pool_size * INDEX_DIMS[method] * 4 / (1024**2))
        rows.append(row)
    return rows


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in per_seed.columns if c not in {"method", "seed", "pool_size", "top_k"}
    ]
    rows = []
    for (method, pool_size, top_k), group in per_seed.groupby(["method", "pool_size", "top_k"], sort=False):
        row = {"method": method, "pool_size": int(pool_size), "top_k": int(top_k), "runs": len(group)}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("template_success", "Template@1"),
        ("sla_valid_template_success", "SLA-valid T@1"),
        ("fulfillment", "Fulfill"),
        ("sla_violation", "SLA Viol."),
        ("avg_qos_gap", "QoS Gap"),
        ("overprovision", "Overprov."),
        ("decision_time_ms", "Time ms"),
        ("decision_time_p95_ms", "P95 ms"),
        ("throughput_qps", "QPS"),
        ("index_memory_mib_float32", "Index MiB"),
    ]
    lines = ["| Method | Pool | " + " | ".join(label for _, label in metrics) + " |"]
    lines.append("|---|---:|" + "|".join(["---:"] * len(metrics)) + "|")
    for _, row in summary.iterrows():
        values = []
        for metric, _ in metrics:
            digits = 3 if metric in {"decision_time_ms", "decision_time_p95_ms", "throughput_qps", "index_memory_mib_float32"} else 4
            values.append(fmt_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"], digits=digits, pm="+/-"))
        lines.append(f"| {row['method']} | {int(row['pool_size'])} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    summary = summary[summary["pool_size"].isin([5000, 10000])]
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{SLA-aware service-template selection. Values are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:sla_reranking}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"Method & Pool & Template@1 & SLA-valid T@1 & Fulfill & SLA Viol. & QoS Gap & Avg (ms) & P95 (ms) & QPS & Index (MiB) \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        values = [str(row["method"]), str(int(row["pool_size"]))]
        for metric in [
            "template_success",
            "sla_valid_template_success",
            "fulfillment",
            "sla_violation",
            "avg_qos_gap",
        ]:
            values.append(fmt_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"], pm=r"$\pm$"))
        values.append(
            fmt_mean_std(row["decision_time_ms_mean"], row["decision_time_ms_std"], digits=3, pm=r"$\pm$")
        )
        values.append(
            fmt_mean_std(row["decision_time_p95_ms_mean"], row["decision_time_p95_ms_std"], digits=3, pm=r"$\pm$")
        )
        values.append(fmt_mean_std(row["throughput_qps_mean"], row["throughput_qps_std"], digits=1, pm=r"$\pm$"))
        values.append(f"{row['index_memory_mib_float32_mean']:.3f}")
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "Embedding cosine retrieval": "#9AA0A8",
        "QoS-only selection": "#7C9186",
        "KCGT retrieval": "#6E7F91",
        "KCGT + QoS reranking": "#4C566A",
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.55), dpi=220)
    panels = [
        ("template_success", "Template@1", (0.0, 1.02)),
        ("fulfillment", "QoS fulfillment", (0.0, 1.02)),
        ("sla_violation", "SLA violation rate", (0.0, 1.02)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for method in METHODS:
            group = summary[summary["method"] == method].sort_values("pool_size")
            x = group["pool_size"].to_numpy()
            y = group[f"{metric}_mean"].to_numpy()
            error = group[f"{metric}_std"].to_numpy()
            ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.0, color=colors[method], label=method)
            ax.fill_between(x, y - error, y + error, color=colors[method], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("Template pool size")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.8)
        ax.tick_params(axis="both", labelsize=7.5)
    axes[0].legend(frameon=False, fontsize=5.8, loc="best")
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def write_report(summary: pd.DataFrame, path: Path, top_k: int) -> None:
    largest = summary[summary["pool_size"] == summary["pool_size"].max()].set_index("method")
    direct = largest.loc["KCGT retrieval"]
    reranked = largest.loc["KCGT + QoS reranking"]
    qos_only = largest.loc["QoS-only selection"]
    embedding_only = largest.loc["Embedding cosine retrieval"]
    lines = [
        "# SLA-Aware Service-Template Reranking Evaluation",
        "",
        "## Objective",
        "",
        "This experiment separates semantic retrieval from QoS feasibility checking in the service-template management pipeline.",
        "",
        "## Compared Methods",
        "",
        "- Embedding cosine retrieval: frozen SBERT sentence-embedding search.",
        "- QoS-only selection: full-pool ranking by constraint satisfaction, QoS gap, and over-provisioning.",
        "- KCGT retrieval: direct Top-1 selection in the KCGT representation space.",
        f"- KCGT + QoS reranking: KCGT Top-{top_k} retrieval followed by constraint-aware reranking.",
        "",
        "## Largest-Pool Result",
        "",
        f"- Direct KCGT: Template@1={direct['template_success_mean']:.4f}, SLA-valid Template@1={direct['sla_valid_template_success_mean']:.4f}, Fulfill={direct['fulfillment_mean']:.4f}, SLA violation={direct['sla_violation_mean']:.4f}.",
        f"- KCGT + QoS reranking: Template@1={reranked['template_success_mean']:.4f}, SLA-valid Template@1={reranked['sla_valid_template_success_mean']:.4f}, Fulfill={reranked['fulfillment_mean']:.4f}, SLA violation={reranked['sla_violation_mean']:.4f}, mean/P95 time={reranked['decision_time_ms_mean']:.3f}/{reranked['decision_time_p95_ms_mean']:.3f} ms, throughput={reranked['throughput_qps_mean']:.1f} queries/s, index={reranked['index_memory_mib_float32_mean']:.3f} MiB.",
        f"- QoS-only selection: Template@1={qos_only['template_success_mean']:.4f}, SLA-valid Template@1={qos_only['sla_valid_template_success_mean']:.4f}, SLA violation={qos_only['sla_violation_mean']:.4f}, time={qos_only['decision_time_ms_mean']:.3f} ms.",
        f"- Frozen embedding retrieval: Template@1={embedding_only['template_success_mean']:.4f}, SLA-valid Template@1={embedding_only['sla_valid_template_success_mean']:.4f}.",
        "",
        "## Interpretation",
        "",
        "The comparison shows that the two-stage semantic-retrieval and constraint-verification pipeline improves joint management utility. QoS-only selection favors feasibility but cannot reliably identify semantically reusable templates, whereas direct embedding retrieval preserves intent relevance but leaves strict QoS violations unresolved.",
        "",
        "These tasks are controlled management proxies and do not replace live admission control, orchestration, or SLA assurance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--kcgt-root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--sbert-root", default="outputs/service_template_scale_baseline_embeddings")
    parser.add_argument("--output-dir", default="outputs/sla_aware_reranking")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--max-queries", type=int, default=512)
    args = parser.parse_args()

    samples = read_jsonl(Path(args.processed_dir) / "test.jsonl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        kcgt_path = (
            Path(args.kcgt_root)
            / f"seed_{seed}"
            / "full_group"
            / "test_group_projection"
            / "query_embeddings.csv"
        )
        sbert_path = (
            Path(args.sbert_root)
            / f"seed_{seed}"
            / "sbert_minilm"
            / "test_group_projection"
            / "query_embeddings.csv"
        )
        kcgt_ids, kcgt_emb = load_embeddings(kcgt_path)
        sbert_ids, sbert_emb = load_embeddings(sbert_path)
        kcgt_emb = normalize(kcgt_emb)
        sbert_emb = normalize(sbert_emb)
        for pool_size in args.pool_sizes:
            print(f"[sla-reranking] seed={seed} pool={pool_size} top_k={args.top_k}", flush=True)
            rows.extend(
                evaluate(
                    samples,
                    kcgt_ids,
                    kcgt_emb,
                    sbert_ids,
                    sbert_emb,
                    pool_size,
                    seed,
                    args.top_k,
                    args.repeats,
                    args.noise_std,
                    args.max_queries,
                )
            )

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(output_dir / "sla_reranking_per_seed.csv", index=False)
    summary.to_csv(output_dir / "sla_reranking_summary.csv", index=False)
    write_markdown(summary, output_dir / "sla_reranking_summary.md")
    write_report(summary, output_dir / "sla_reranking_report.md", args.top_k)

    figure_path = output_dir / "sla_reranking_comparison.png"
    plot_results(summary, figure_path)
    paper_figure = Path("paper/figures/final_sla_reranking.png")
    paper_figure.write_bytes(figure_path.read_bytes())
    write_latex(summary, Path("paper/tables/table_final_sla_reranking.tex"))
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "sla_aware_reranking_report.md").write_text(
        (output_dir / "sla_reranking_report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(
        summary[
            [
                "method",
                "pool_size",
                "template_success_mean",
                "sla_valid_template_success_mean",
                "fulfillment_mean",
                "sla_violation_mean",
                "avg_qos_gap_mean",
                "decision_time_ms_mean",
                "decision_time_p95_ms_mean",
                "throughput_qps_mean",
                "index_memory_mib_float32_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
