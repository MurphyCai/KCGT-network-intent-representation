"""One-factor-at-a-time sensitivity analysis for the end-to-end simulation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_end_to_end_management_simulation import (
    deployed_profile,
    disturbance,
    normalize,
    resource_demand,
    select_queries,
)
from run_sla_aware_reranking_evaluation import qos_best, qos_matrix, qos_scores
from run_step36_management_validation import load_embeddings, read_jsonl


DEFAULTS = {"capacity": 0.45, "sla_count": 7, "gap": 0.08, "top_k": 10}


def configurations() -> list[dict]:
    configs = [{"factor": "reference", "value": "default", **DEFAULTS}]
    sweeps = {
        "capacity": [0.35, 0.45, 0.55, 0.65],
        "sla_count": [6, 7, 8],
        "gap": [0.05, 0.08, 0.10],
        "top_k": [5, 10, 20],
    }
    for factor, values in sweeps.items():
        for value in values:
            if value == DEFAULTS[factor]:
                continue
            cfg = {"factor": factor, "value": str(value), **DEFAULTS}
            cfg[factor] = value
            configs.append(cfg)
    return configs


def capacity(seed: int, window: int, size: int, coefficient: float) -> np.ndarray:
    rng = np.random.default_rng(seed * 1_000_003 + window * 1009)
    profile = np.asarray([1.03, 0.99, 1.01, 0.97], dtype=np.float64)
    return coefficient * size * profile * rng.uniform(0.92, 1.08, size=4)


def select_kcgt_qos(
    qi: int,
    candidates: np.ndarray,
    embeddings: np.ndarray,
    samples: list[dict],
    top_k: int,
) -> int:
    scores = embeddings[candidates] @ embeddings[qi]
    k = min(top_k, len(candidates))
    top = np.argpartition(-scores, kth=k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    ratio, _, gap, over = qos_scores(samples[qi], qos_matrix([samples[int(candidates[i])] for i in top]))
    local = qos_best(ratio, gap, over, np.arange(k))
    return int(candidates[int(top[local])])


def evaluate_config(
    seed: int,
    config: dict,
    samples: list[dict],
    ids: list[str],
    embeddings: np.ndarray,
    query_limit: int,
    window_size: int,
) -> dict:
    by_id = {x["intent_id"]: x for x in samples}
    ordered = [by_id[x] for x in ids]
    id_to_index = {x: i for i, x in enumerate(ids)}
    queries = select_queries(ordered, ids, seed, query_limit)
    all_indices = np.arange(len(ordered), dtype=int)
    mapping, admitted, violations, utilization = [], [], [], []
    ratio_threshold = float(config["sla_count"]) / 8.0
    gap_threshold = float(config["gap"])

    for start in range(0, len(queries), window_size):
        window = start // window_size
        cap = capacity(seed, window, window_size, float(config["capacity"]))
        used = np.zeros(4, dtype=np.float64)
        for query in queries[start : start + window_size]:
            qi = id_to_index[query["intent_id"]]
            candidates = all_indices[all_indices != qi]
            selected = select_kcgt_qos(qi, candidates, embeddings, ordered, int(config["top_k"]))
            candidate = ordered[selected]
            demand = resource_demand(candidate)
            pre_ratio, _, pre_gap, _ = qos_scores(query, qos_matrix([candidate]))
            qos_valid = bool(pre_ratio[0] >= ratio_threshold and pre_gap[0] <= gap_threshold)
            capacity_valid = bool(np.all(used + demand <= cap))
            accept = qos_valid and capacity_valid
            mapping.append(float(candidate["positive_group_id"] == query["positive_group_id"]))
            admitted.append(float(accept))
            if accept:
                used += demand
                load = float(np.max(used / cap))
                deployed = deployed_profile(candidate, load, disturbance(seed, query["intent_id"]))
                outcome_ratio, _, outcome_gap, _ = qos_scores(query, qos_matrix([deployed]))
                sla_met = bool(outcome_ratio[0] >= ratio_threshold and outcome_gap[0] <= gap_threshold)
                violations.append(float(not sla_met))
        utilization.append(float(np.mean(used / cap)))

    return {
        "factor": config["factor"],
        "value": config["value"],
        "capacity": float(config["capacity"]),
        "sla_count": int(config["sla_count"]),
        "gap": float(config["gap"]),
        "top_k": int(config["top_k"]),
        "seed": seed,
        "mapping_success": float(np.mean(mapping)),
        "admission_ratio": float(np.mean(admitted)),
        "sla_violation": float(np.mean(violations)) if violations else 1.0,
        "resource_utilization": float(np.mean(utilization)),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["factor", "value", "capacity", "sla_count", "gap", "top_k"]
    metrics = ["mapping_success", "admission_ratio", "sla_violation", "resource_utilization"]
    rows = []
    for group_key, group in frame.groupby(keys, sort=False):
        row = dict(zip(keys, group_key))
        row["runs"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def label(row: pd.Series) -> str:
    names = {"reference": "Reference", "capacity": "Capacity", "sla_count": "SLA count", "gap": "QoS gap", "top_k": "Top-K"}
    return names[row["factor"]] + ("" if row["factor"] == "reference" else f"={row['value']}")


def write_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{One-factor sensitivity of the end-to-end KCGT--QoS--admission pipeline. Values are mean $\pm$ standard deviation over five seeds.}",
        r"\label{tab:supp_end_to_end_sensitivity}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & Mapping success & Admission ratio & SLA violation & Resource utilization \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        values = [label(row)]
        for metric in ["mapping_success", "admission_ratio", "sla_violation", "resource_utilization"]:
            values.append(f"{row[f'{metric}_mean']:.4f} $\\pm$ {row[f'{metric}_std']:.4f}")
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    colors = {"reference": "#2F2F2F", "capacity": "#6E7F91", "sla_count": "#8A7F72", "gap": "#7C9186", "top_k": "#A08B9B"}
    markers = {"reference": "*", "capacity": "o", "sla_count": "s", "gap": "^", "top_k": "D"}
    fig, ax = plt.subplots(figsize=(4.2, 3.1), dpi=240)
    for factor, group in summary.groupby("factor", sort=False):
        ax.errorbar(
            group["admission_ratio_mean"],
            group["sla_violation_mean"],
            xerr=group["admission_ratio_std"],
            yerr=group["sla_violation_std"],
            marker=markers[factor],
            markersize=5 if factor != "reference" else 8,
            linewidth=0.9,
            capsize=2,
            color=colors[factor],
            label={"reference": "Reference", "capacity": "Capacity", "sla_count": "SLA count", "gap": "QoS gap", "top_k": "Top-K"}[factor],
        )
        for _, row in group.iterrows():
            if factor != "reference":
                ax.annotate(str(row["value"]), (row["admission_ratio_mean"], row["sla_violation_mean"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Admission ratio")
    ax.set_ylabel("SLA violation rate")
    ax.set_xlim(0.18, 0.72)
    ax.set_ylim(0.15, 0.36)
    ax.grid(color="#D9D9D9", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(pad=0.35)
    fig.savefig(output / "admission_sla_tradeoff.png", bbox_inches="tight")
    fig.savefig(Path("paper") / "figures" / "supp_admission_sla_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--kcgt-root", default="outputs/method_novelty_v1")
    parser.add_argument("--output-dir", default="outputs/end_to_end_sensitivity")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()

    samples = read_jsonl(Path(args.processed_dir) / "test.jsonl")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        path = Path(args.kcgt_root) / f"seed_{seed}" / "full_structured_relgt" / "test_group" / "query_embeddings.csv"
        ids, embeddings = load_embeddings(path)
        embeddings = normalize(embeddings)
        for config in configurations():
            rows.append(evaluate_config(seed, config, samples, ids, embeddings, args.queries, args.window_size))
        print(f"[sensitivity] seed={seed} complete", flush=True)

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(output / "sensitivity_per_seed.csv", index=False)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    write_table(summary, Path("paper") / "tables" / "table_supp_end_to_end_sensitivity.tex")
    plot_tradeoff(summary, output)

    lines = [
        "# End-to-End Simulation Sensitivity",
        "",
        "One factor is changed at a time around capacity=0.45, SLA count=7/8, QoS gap=0.08, and Top-K=10. All settings use five matched seeds and 512 requests per seed.",
        "",
        "| Configuration | Mapping | Admission | SLA violation | Utilization |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {label(row)} | {row['mapping_success_mean']:.4f} +/- {row['mapping_success_std']:.4f} | {row['admission_ratio_mean']:.4f} +/- {row['admission_ratio_std']:.4f} | {row['sla_violation_mean']:.4f} +/- {row['sla_violation_std']:.4f} | {row['resource_utilization_mean']:.4f} +/- {row['resource_utilization_std']:.4f} |")
    (output / "sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
