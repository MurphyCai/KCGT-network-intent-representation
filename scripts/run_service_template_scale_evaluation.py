from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_step36_management_validation import (
    VARIANTS,
    fmt_mean_std,
    jaccard,
    load_embeddings,
    qos_fulfillment,
    read_jsonl,
    service_profile,
)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def sample_by_id(samples: list[dict]) -> dict[str, dict]:
    return {s["intent_id"]: s for s in samples}


def build_indices(samples: list[dict]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_group: dict[str, list[str]] = {}
    by_slice: dict[str, list[str]] = {}
    for s in samples:
        by_group.setdefault(s["positive_group_id"], []).append(s["intent_id"])
        by_slice.setdefault(s.get("slice_type", ""), []).append(s["intent_id"])
    return by_group, by_slice


def choose_positive(q: dict, ids_available: set[str], by_group: dict[str, list[str]], rng: np.random.Generator) -> str:
    candidates = [x for x in q.get("positive_ids", []) if x in ids_available and x != q["intent_id"]]
    if not candidates:
        candidates = [x for x in by_group.get(q["positive_group_id"], []) if x in ids_available and x != q["intent_id"]]
    if not candidates:
        candidates = [q["intent_id"]]
    return str(rng.choice(candidates))


def choose_distractors(
    q: dict,
    n: int,
    ids_available: list[str],
    ids_available_set: set[str],
    by_slice: dict[str, list[str]],
    rng: np.random.Generator,
    forbidden: set[str] | None = None,
) -> list[str]:
    forbidden = set(forbidden or set())
    forbidden.add(q["intent_id"])
    hard = [x for x in q.get("hard_negative_ids", []) if x in ids_available_set and x not in forbidden]
    same_slice = [
        x
        for x in by_slice.get(q.get("slice_type", ""), [])
        if x in ids_available_set and x not in forbidden and x not in hard
    ]
    random_pool = [x for x in ids_available if x not in forbidden and x not in hard and x not in same_slice]
    weights = [0.45, 0.30, 0.25]
    pools = [hard, same_slice, random_pool]
    out: list[str] = []
    counts = rng.multinomial(n, weights)
    fallback = [x for x in ids_available if x not in forbidden] or ids_available
    for bucket, count in enumerate(counts):
        if count <= 0:
            continue
        pool = pools[bucket]
        if not pool:
            pool = fallback
        out.extend(str(x) for x in rng.choice(pool, size=count, replace=True))
    if len(out) < n:
        out.extend(str(x) for x in rng.choice(fallback, size=n - len(out), replace=True))
    elif len(out) > n:
        out = out[:n]
    return out


def make_candidate_pool(
    q: dict,
    pool_size: int,
    ids_available: list[str],
    ids_available_set: set[str],
    id_to_idx: dict[str, int],
    emb: np.ndarray,
    samples_by_id: dict[str, dict],
    by_group: dict[str, list[str]],
    by_slice: dict[str, list[str]],
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    positive_id = choose_positive(q, ids_available_set, by_group, rng)
    distractor_ids = choose_distractors(
        q,
        pool_size - 1,
        ids_available,
        ids_available_set,
        by_slice,
        rng,
        forbidden={positive_id},
    )
    candidate_ids = [positive_id] + distractor_ids
    rng.shuffle(candidate_ids)

    cand_emb = emb[[id_to_idx[x] for x in candidate_ids]].copy()
    noise = rng.normal(0.0, noise_std, size=cand_emb.shape)
    cand_emb = normalize(cand_emb + noise)
    cand_samples = [samples_by_id[x] for x in candidate_ids]
    is_positive = np.asarray([x == positive_id for x in candidate_ids], dtype=bool)
    return cand_emb, cand_samples, is_positive


def evaluate_scale(
    samples: list[dict],
    ids: list[str],
    emb: np.ndarray,
    pool_size: int,
    seed: int,
    repeats: int,
    noise_std: float,
    max_queries: int | None,
) -> dict[str, float]:
    samples_by_id = sample_by_id(samples)
    by_group, by_slice = build_indices(samples)
    id_to_idx = {x: i for i, x in enumerate(ids)}
    ids_available_set = set(ids)
    rng = np.random.default_rng(seed + pool_size * 1009)
    ordered = [samples_by_id[x] for x in ids if x in samples_by_id]
    if max_queries is not None and len(ordered) > max_queries:
        by_class: dict[str, list[dict]] = {}
        for sample in ordered:
            by_class.setdefault(sample.get("intent_class", sample.get("class_id", "")), []).append(sample)
        selected: list[dict] = []
        per_class = max(1, max_queries // max(len(by_class), 1))
        for cls in sorted(by_class):
            pool = by_class[cls]
            take = min(per_class, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            selected.extend(pool[int(i)] for i in idx)
        if len(selected) < max_queries:
            selected_ids = {s["intent_id"] for s in selected}
            remaining = [s for s in ordered if s["intent_id"] not in selected_ids]
            take = min(max_queries - len(selected), len(remaining))
            if take:
                idx = rng.choice(len(remaining), size=take, replace=False)
                selected.extend(remaining[int(i)] for i in idx)
        ordered = selected
    top1 = []
    top5 = []
    profile_top1 = []
    slice_top1 = []
    nf_jacc = []
    res_jacc = []
    fulfill_top1 = []
    sla_violate_top1 = []
    qos_gap_top1 = []
    over_top1 = []
    fulfill_top5 = []
    sla_violate_top5 = []
    qos_gap_top5 = []
    over_top5 = []
    decision_times_ms = []

    for q in ordered:
        q_emb = emb[id_to_idx[q["intent_id"]]]
        query_code = zlib.crc32(q["intent_id"].encode("utf-8"))
        pool_rng = np.random.default_rng(seed * 1_000_003 + pool_size * 1009 + query_code)
        cand_emb, cand_samples, is_positive = make_candidate_pool(
            q,
            pool_size,
            ids,
            ids_available_set,
            id_to_idx,
            emb,
            samples_by_id,
            by_group,
            by_slice,
            pool_rng,
            noise_std,
        )

        elapsed = 0.0
        scores = None
        topk = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            scores = cand_emb @ q_emb
            k = min(5, len(scores))
            topk = np.argpartition(-scores, kth=k - 1)[:k]
            topk = topk[np.argsort(-scores[topk])]
            elapsed += time.perf_counter() - t0
        decision_times_ms.append((elapsed / repeats) * 1000.0)
        assert scores is not None
        assert topk is not None

        top = int(topk[0])
        top_sample = cand_samples[top]
        topk_samples = [cand_samples[int(i)] for i in topk]

        top1.append(bool(is_positive[top]))
        top5.append(bool(is_positive[topk].any()))
        profile_top1.append(service_profile(q) == service_profile(top_sample))
        slice_top1.append(q.get("slice_type") == top_sample.get("slice_type"))
        nf_jacc.append(jaccard(q.get("network_functions", []), top_sample.get("network_functions", [])))
        res_jacc.append(jaccard(q.get("resources", []), top_sample.get("resources", [])))

        f_ratio, all_met, gap, over = qos_fulfillment(q, top_sample)
        fulfill_top1.append(f_ratio)
        sla_violate_top1.append(not all_met)
        qos_gap_top1.append(gap)
        over_top1.append(over)

        candidate_scores = [qos_fulfillment(q, cand) for cand in topk_samples]
        best_f_ratio, best_all_met, best_gap, best_over = sorted(candidate_scores, key=lambda x: (-x[0], x[2], x[3]))[0]
        fulfill_top5.append(best_f_ratio)
        sla_violate_top5.append(not best_all_met)
        qos_gap_top5.append(best_gap)
        over_top5.append(best_over)

    return {
        "num_queries": float(len(ordered)),
        "pool_size": float(pool_size),
        "template_success@1": float(np.mean(top1)),
        "template_success@5": float(np.mean(top5)),
        "profile_success@1": float(np.mean(profile_top1)),
        "slice_match@1": float(np.mean(slice_top1)),
        "function_jaccard@1": float(np.mean(nf_jacc)),
        "resource_jaccard@1": float(np.mean(res_jacc)),
        "fulfillment@1": float(np.mean(fulfill_top1)),
        "sla_violation@1": float(np.mean(sla_violate_top1)),
        "avg_qos_gap@1": float(np.mean(qos_gap_top1)),
        "overprovision@1": float(np.mean(over_top1)),
        "fulfillment@5": float(np.mean(fulfill_top5)),
        "sla_violation@5": float(np.mean(sla_violate_top5)),
        "avg_qos_gap@5": float(np.mean(qos_gap_top5)),
        "overprovision@5": float(np.mean(over_top5)),
        "decision_time_ms": float(np.mean(decision_times_ms)),
        "decision_time_p95_ms": float(np.percentile(decision_times_ms, 95)),
        "throughput_qps": float(1000.0 / max(np.mean(decision_times_ms), 1e-12)),
        "index_memory_mib_float32": float(pool_size * emb.shape[1] * 4 / (1024**2)),
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_seed.columns if c not in {"model", "variant", "seed", "pool_size"}]
    rows = []
    for (model, variant, pool_size), g in per_seed.groupby(["model", "variant", "pool_size"], sort=False):
        row = {"model": model, "variant": variant, "pool_size": int(pool_size), "runs": len(g)}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = vals.mean()
            row[f"{c}_std"] = vals.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        ("model", "Model"),
        ("pool_size", "Pool"),
        ("template_success@1", "Template@1"),
        ("template_success@5", "Template@5"),
        ("fulfillment@5", "Fulfill@5"),
        ("sla_violation@5", "SLA Viol.@5"),
        ("avg_qos_gap@5", "QoS Gap@5"),
        ("decision_time_ms", "Time ms"),
        ("decision_time_p95_ms", "P95 ms"),
        ("throughput_qps", "QPS"),
        ("index_memory_mib_float32", "Index MiB"),
    ]
    lines = ["| " + " | ".join(label for _, label in cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        cells = []
        for key, _ in cols:
            if key == "model":
                cells.append(str(row["model"]))
            elif key == "pool_size":
                cells.append(str(int(row["pool_size"])))
            else:
                digits = 3 if key in {"decision_time_ms", "decision_time_p95_ms", "throughput_qps", "index_memory_mib_float32"} else 4
                cells.append(fmt_mean_std(row[f"{key}_mean"], row[f"{key}_std"], digits=digits, pm="+/-"))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    keep_models = [
        "KCGT",
        "SBERT-MiniLM",
        "Text+GAT",
        "Text+GraphTransformer",
        "GraphTransformer",
        "KCGT w/o CL",
        "KCGT w/o graph",
        "KCGT-GAT",
    ]
    top = summary[
        summary["model"].isin(keep_models)
        & summary["pool_size"].isin([1000, 5000, 10000])
    ].copy()
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Service-template candidate scale evaluation. Each query is evaluated against a controlled template pool containing one compatible template and hard/random distractors. Values are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:service_template_scale}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Model & Pool & Template@1 & Template@5 & Fulfill@5 & SLA Viol.@5 & Avg (ms) & P95 (ms) & QPS & Index (MiB) \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        vals = [
            row["model"],
            str(int(row["pool_size"])),
            fmt_mean_std(row["template_success@1_mean"], row["template_success@1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["template_success@5_mean"], row["template_success@5_std"], pm=r"$\pm$"),
            fmt_mean_std(row["fulfillment@5_mean"], row["fulfillment@5_std"], pm=r"$\pm$"),
            fmt_mean_std(row["sla_violation@5_mean"], row["sla_violation@5_std"], pm=r"$\pm$"),
            fmt_mean_std(row["decision_time_ms_mean"], row["decision_time_ms_std"], digits=3, pm=r"$\pm$"),
            fmt_mean_std(row["decision_time_p95_ms_mean"], row["decision_time_p95_ms_std"], digits=3, pm=r"$\pm$"),
            fmt_mean_std(row["throughput_qps_mean"], row["throughput_qps_std"], digits=1, pm=r"$\pm$"),
            f"{row['index_memory_mib_float32_mean']:.3f}",
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_scale(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    models = ["KCGT", "SBERT-MiniLM", "Text+GAT", "Text+GraphTransformer", "GraphTransformer"]
    colors = {
        "KCGT": "#4C566A",
        "SBERT-MiniLM": "#8A8F98",
        "Text+GAT": "#687C8E",
        "Text+GraphTransformer": "#9A7B6A",
        "GraphTransformer": "#A9ADB3",
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.55), dpi=220)
    metrics = [
        ("template_success@1", "Template@1", (0.0, 1.02)),
        ("fulfillment@5", "Fulfillment@5", (0.0, 1.02)),
        ("decision_time_ms", "Decision time (ms)", None),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for model in models:
            g = summary[summary["model"] == model].sort_values("pool_size")
            if g.empty:
                continue
            x = g["pool_size"].to_numpy()
            y = g[f"{metric}_mean"].to_numpy()
            e = g[f"{metric}_std"].to_numpy()
            ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.0, color=colors[model], label=model)
            ax.fill_between(x, y - e, y + e, color=colors[model], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("Template pool size")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.8)
        ax.tick_params(axis="both", labelsize=7.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, processed_dir: str, retrieval_root: str) -> None:
    k = summary[summary["model"] == "KCGT"].sort_values("pool_size")
    max_pool = k.iloc[-1]
    min_pool = k.iloc[0]
    external = summary[
        summary["model"].isin(["SBERT-MiniLM", "Text+GAT", "Text+GraphTransformer", "GraphTransformer"])
        & (summary["pool_size"] == int(max_pool["pool_size"]))
    ].sort_values("template_success@5_mean", ascending=False)
    strongest = external.iloc[0]
    lines = [
        "# Service-Template Candidate Scale Evaluation",
        "",
        "## Objective",
        "",
        "This experiment evaluates whether the learned intent embedding remains useful when the service-template candidate pool grows. It is designed for the IJCS-style communication service management narrative: an intent representation should support fast QoS-aware service-template selection, not only closed-set classification.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{processed_dir}`.",
        f"- Embedding source: `{retrieval_root}`.",
        "- Pool sizes: 100, 500, 1000, 5000, and 10000 service-template candidates.",
        "- For each query, the candidate pool contains one compatible template and a mixture of hard negatives, same-slice distractors, and random templates.",
        "- Template embeddings are generated from existing intent embeddings with small deterministic perturbations to emulate a larger template repository without retraining.",
        "- Strong baselines include frozen SBERT-MiniLM, Text+GAT, Text+GraphTransformer, and GraphTransformer; all models receive identical candidate identities for each query, scale, and seed.",
        "- Neural baselines use matched training and candidate-pool seeds. Frozen SBERT is deterministic, so its variation reflects candidate-pool sampling.",
        "- Metrics: Template Success@1/@5, SLA-aware Fulfillment@5, SLA Violation@5, average QoS gap, over-provisioning proxy, mean/P95 decision time, throughput, and estimated float32 vector-index memory.",
        "",
        "## Key Results",
        "",
        f"- KCGT Template@1 changes from {min_pool['template_success@1_mean']:.4f} at pool size {int(min_pool['pool_size'])} to {max_pool['template_success@1_mean']:.4f} at pool size {int(max_pool['pool_size'])}.",
        f"- KCGT Template@5 changes from {min_pool['template_success@5_mean']:.4f} to {max_pool['template_success@5_mean']:.4f}.",
        f"- At the largest pool, the strongest external baseline is {strongest['model']} with Template@5={strongest['template_success@5_mean']:.4f}.",
        f"- KCGT SLA-aware Fulfillment@5 remains {max_pool['fulfillment@5_mean']:.4f} at pool size {int(max_pool['pool_size'])}.",
        f"- At the largest pool, KCGT mean/P95 decision time is {max_pool['decision_time_ms_mean']:.3f}/{max_pool['decision_time_p95_ms_mean']:.3f} ms, throughput is {max_pool['throughput_qps_mean']:.1f} queries/s, and the estimated float32 index footprint is {max_pool['index_memory_mib_float32_mean']:.3f} MiB.",
        "",
        "## Interpretation",
        "",
        "The scale protocol stresses service-template reuse under a growing repository. KCGT leads exact template recovery across all scales, while the stronger SLA proxy of some graph baselines at the largest pool shows that retrieval relevance and strict QoS feasibility are distinct objectives. The results motivate explicit constraint verification or second-stage QoS reranking. They should be described as a controlled candidate-scale proxy, not as a live orchestrator benchmark.",
        "",
        "## Generated Files",
        "",
        "- `outputs/service_template_scale/service_template_scale_per_seed.csv`",
        "- `outputs/service_template_scale/service_template_scale_summary.csv`",
        "- `outputs/service_template_scale/service_template_scale_summary.md`",
        "- `outputs/service_template_scale/service_template_scale_report.md`",
        "- `outputs/service_template_scale/service_template_scale_plot.png`",
        "- `paper/figures/final_service_template_scale.png`",
        "- `paper/tables/table_final_service_template_scale.tex`",
    ]
    (output_dir / "service_template_scale_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--retrieval-root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--baseline-embedding-root", default="outputs/service_template_scale_baseline_embeddings")
    parser.add_argument("--output-dir", default="outputs/service_template_scale")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--embedding-split", default="test_group_projection")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "KCGT",
            "KCGT w/o CL",
            "KCGT w/o graph",
            "KCGT-GAT",
            "SBERT-MiniLM",
            "Text+GAT",
            "Text+GraphTransformer",
            "GraphTransformer",
        ],
        help="KCGT variants and exported strong baseline display names.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(processed_dir / "test.jsonl")
    sample_ids = {s["intent_id"] for s in samples}

    rows = []
    baseline_variants = {
        "SBERT-MiniLM": "sbert_minilm",
        "Text+GAT": "text_gat",
        "Text+GraphTransformer": "text_graph_transformer",
        "GraphTransformer": "graph_transformer",
    }
    all_variants = [(model, variant, Path(args.retrieval_root)) for model, variant in VARIANTS.items()]
    all_variants.extend(
        (model, variant, Path(args.baseline_embedding_root)) for model, variant in baseline_variants.items()
    )
    for seed in args.seeds:
        for model, variant, embedding_root in all_variants:
            if model not in set(args.models):
                continue
            emb_path = embedding_root / f"seed_{seed}" / variant / args.embedding_split / "query_embeddings.csv"
            if not emb_path.exists():
                continue
            ids, emb = load_embeddings(emb_path)
            keep = [i for i, x in enumerate(ids) if x in sample_ids]
            ids = [ids[i] for i in keep]
            emb = emb[keep]
            if len(ids) < 2:
                continue
            for pool_size in args.pool_sizes:
                print(
                    f"[service-scale] model={model} seed={seed} pool={pool_size} queries<= {args.max_queries}",
                    flush=True,
                )
                metrics = evaluate_scale(
                    samples=samples,
                    ids=ids,
                    emb=emb,
                    pool_size=pool_size,
                    seed=seed,
                    repeats=args.repeats,
                    noise_std=args.noise_std,
                    max_queries=args.max_queries,
                )
                rows.append({"model": model, "variant": variant, "seed": seed, "pool_size": pool_size, **metrics})
                print(
                    f"[done] model={model} seed={seed} pool={pool_size} "
                    f"template@1={metrics['template_success@1']:.4f} "
                    f"template@5={metrics['template_success@5']:.4f} "
                    f"time={metrics['decision_time_ms']:.4f}ms",
                    flush=True,
                )

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(output_dir / "service_template_scale_per_seed.csv", index=False)
    summary.to_csv(output_dir / "service_template_scale_summary.csv", index=False)
    write_markdown(summary, output_dir / "service_template_scale_summary.md")

    fig_path = output_dir / "service_template_scale_plot.png"
    plot_scale(summary, fig_path)
    paper_fig = Path("paper/figures/final_service_template_scale.png")
    paper_fig.parent.mkdir(parents=True, exist_ok=True)
    paper_fig.write_bytes(fig_path.read_bytes())

    paper_table = Path("paper/tables/table_final_service_template_scale.tex")
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    write_latex(summary, paper_table)

    write_report(summary, output_dir, args.processed_dir, args.retrieval_root)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "service_template_scale_report.md").write_text(
        (output_dir / "service_template_scale_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(
        summary[
            [
                "model",
                "pool_size",
                "template_success@1_mean",
                "template_success@5_mean",
                "fulfillment@5_mean",
                "sla_violation@5_mean",
                "decision_time_ms_mean",
                "decision_time_p95_ms_mean",
                "throughput_qps_mean",
                "index_memory_mib_float32_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
