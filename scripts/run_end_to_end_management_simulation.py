"""Lightweight intent-to-SLA management-loop simulation.

Pipeline: intent -> template retrieval -> QoS verification -> admission -> SLA outcome.
The experiment is a controlled management-plane simulation, not a packet-level emulator.
"""

from __future__ import annotations

import argparse
import csv
import time
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_sla_aware_reranking_evaluation import qos_best, qos_matrix, qos_scores
from run_step36_management_validation import QOS_FIELDS, load_embeddings, read_jsonl


METHODS = ["SBERT cosine", "QoS-only", "KCGT retrieval", "KCGT + QoS + admission"]
QOS_GUARD = {"QoS-only", "KCGT + QoS + admission"}
RESOURCE_NAMES = ["compute", "radio", "transport", "edge"]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def select_queries(samples: list[dict], ids: list[str], seed: int, limit: int) -> list[dict]:
    available = {x["intent_id"]: x for x in samples if x["intent_id"] in set(ids)}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for sample in available.values():
        by_class[sample["intent_class"]].append(sample)
    rng = np.random.default_rng(seed * 104729)
    selected: list[dict] = []
    per_class = max(1, limit // len(by_class))
    for label in sorted(by_class):
        group = by_class[label]
        order = rng.permutation(len(group))[: min(per_class, len(group))]
        selected.extend(group[int(i)] for i in order)
    if len(selected) < limit:
        chosen = {x["intent_id"] for x in selected}
        remainder = [x for x in available.values() if x["intent_id"] not in chosen]
        order = rng.permutation(len(remainder))[: limit - len(selected)]
        selected.extend(remainder[int(i)] for i in order)
    rng.shuffle(selected)
    return selected[:limit]


def resource_demand(sample: dict) -> np.ndarray:
    bw = np.clip(np.log1p(float(sample["qos_bandwidth_mbps"])) / np.log1p(1000.0), 0.0, 1.0)
    latency = np.clip(1.0 - float(sample["qos_latency_ms"]) / 120.0, 0.0, 1.0)
    reliability = np.clip((float(sample["qos_reliability"]) - 0.95) / 0.05, 0.0, 1.0)
    mobility = float(sample["qos_mobility"]) / 5.0
    security = float(sample["qos_security_level"]) / 5.0
    functions = min(len(sample.get("network_functions", [])) / 4.0, 1.0)
    resources = set(sample.get("resources", []))
    return np.asarray(
        [
            0.16 + 0.24 * functions + 0.14 * security + 0.10 * ("shared_compute" in resources),
            0.14 + 0.30 * bw + 0.18 * mobility + 0.10 * ("shared_radio" in resources),
            0.14 + 0.28 * bw + 0.18 * reliability + 0.10 * ("shared_routing" in resources),
            0.12 + 0.28 * latency + 0.16 * functions + 0.10 * ("shared_edge" in resources),
        ],
        dtype=np.float64,
    )


def capacity_for_window(seed: int, window: int, window_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1_000_003 + window * 1009)
    base = np.asarray([0.48, 0.46, 0.47, 0.45]) * window_size
    return base * rng.uniform(0.92, 1.08, size=4)


def disturbance(seed: int, intent_id: str) -> np.ndarray:
    code = zlib.crc32(intent_id.encode("utf-8"))
    rng = np.random.default_rng(seed * 10_000_019 + code)
    return rng.normal(0.0, 1.0, size=5)


def deployed_profile(candidate: dict, load: float, noise: np.ndarray) -> dict:
    deployed = dict(candidate)
    deployed["qos_latency_ms"] = float(candidate["qos_latency_ms"]) * (1.0 + 0.18 * load + 0.025 * abs(noise[0]))
    deployed["qos_bandwidth_mbps"] = float(candidate["qos_bandwidth_mbps"]) * max(0.55, 1.0 - 0.16 * load - 0.020 * abs(noise[1]))
    deployed["qos_reliability"] = max(0.0, float(candidate["qos_reliability"]) - 0.00045 * load - 0.00008 * abs(noise[2]))
    deployed["qos_jitter_ms"] = float(candidate["qos_jitter_ms"]) * (1.0 + 0.20 * load + 0.030 * abs(noise[3]))
    deployed["qos_packet_loss"] = float(candidate["qos_packet_loss"]) * (1.0 + 0.28 * load + 0.040 * abs(noise[4]))
    return deployed


def choose_candidate(
    method: str,
    query_index: int,
    candidate_indices: np.ndarray,
    kcgt: np.ndarray,
    sbert: np.ndarray,
    samples: list[dict],
    top_k: int,
) -> int:
    candidates = [samples[int(i)] for i in candidate_indices]
    c_qos = qos_matrix(candidates)
    ratio, _, gap, over = qos_scores(samples[query_index], c_qos)
    if method == "SBERT cosine":
        return int(candidate_indices[int(np.argmax(sbert[candidate_indices] @ sbert[query_index]))])
    if method == "QoS-only":
        return int(candidate_indices[qos_best(ratio, gap, over)])
    scores = kcgt[candidate_indices] @ kcgt[query_index]
    if method == "KCGT retrieval":
        return int(candidate_indices[int(np.argmax(scores))])
    k = min(top_k, len(candidate_indices))
    top = np.argpartition(-scores, kth=k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    local = qos_best(ratio[top], gap[top], over[top], np.arange(k))
    return int(candidate_indices[int(top[local])])


def evaluate_seed(
    seed: int,
    samples: list[dict],
    ids: list[str],
    kcgt: np.ndarray,
    sbert: np.ndarray,
    query_limit: int,
    top_k: int,
    window_size: int,
    timing_repeats: int,
) -> list[dict]:
    by_id = {x["intent_id"]: x for x in samples}
    ordered = [by_id[x] for x in ids]
    id_to_index = {x: i for i, x in enumerate(ids)}
    queries = select_queries(ordered, ids, seed, query_limit)
    all_indices = np.arange(len(ordered), dtype=int)
    stores = {
        method: defaultdict(list)
        for method in METHODS
    }
    utilization = {method: [] for method in METHODS}

    for window_start in range(0, len(queries), window_size):
        window = window_start // window_size
        capacity = capacity_for_window(seed, window, window_size)
        used = {method: np.zeros(4, dtype=np.float64) for method in METHODS}
        for query in queries[window_start : window_start + window_size]:
            qi = id_to_index[query["intent_id"]]
            candidates = all_indices[all_indices != qi]
            for method in METHODS:
                selected = -1
                candidate = None
                demand = None
                qos_valid = False
                capacity_valid = False
                start = time.perf_counter()
                for _ in range(timing_repeats):
                    selected = choose_candidate(method, qi, candidates, kcgt, sbert, ordered, top_k)
                    candidate = ordered[selected]
                    demand = resource_demand(candidate)
                    ratio, _, gap, _ = qos_scores(query, qos_matrix([candidate]))
                    qos_valid = bool(ratio[0] >= 0.875 and gap[0] <= 0.08)
                    capacity_valid = bool(np.all(used[method] + demand <= capacity))
                elapsed_ms = (time.perf_counter() - start) * 1000.0 / timing_repeats
                assert candidate is not None and demand is not None
                admitted = capacity_valid and (qos_valid if method in QOS_GUARD else True)
                if admitted:
                    used[method] += demand
                    load = float(np.max(used[method] / capacity))
                    deployed = deployed_profile(candidate, load, disturbance(seed, query["intent_id"]))
                    outcome_ratio, _, outcome_gap, _ = qos_scores(query, qos_matrix([deployed]))
                    # Composite SLA: at least seven of eight normalized service
                    # requirements must hold and the residual mean gap must stay
                    # below the same threshold used by the admission verifier.
                    sla_met = bool(outcome_ratio[0] >= 0.875 and outcome_gap[0] <= 0.08)
                    stores[method]["sla_violation"].append(float(not sla_met))
                stores[method]["mapping_success"].append(float(candidate["positive_group_id"] == query["positive_group_id"]))
                stores[method]["admitted"].append(float(admitted))
                stores[method]["decision_time_ms"].append(elapsed_ms)
        for method in METHODS:
            utilization[method].append(float(np.mean(used[method] / capacity)))

    rows = []
    for method in METHODS:
        sla = stores[method]["sla_violation"]
        times = stores[method]["decision_time_ms"]
        rows.append(
            {
                "method": method,
                "seed": seed,
                "queries": len(queries),
                "mapping_success": float(np.mean(stores[method]["mapping_success"])),
                "admission_ratio": float(np.mean(stores[method]["admitted"])),
                "sla_violation": float(np.mean(sla)) if sla else 1.0,
                "resource_utilization": float(np.mean(utilization[method])),
                "decision_time_ms": float(np.mean(times)),
                "decision_time_p95_ms": float(np.percentile(times, 95)),
                "admitted_requests": int(sum(stores[method]["admitted"])),
            }
        )
    return rows


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mapping_success", "admission_ratio", "sla_violation", "resource_utilization", "decision_time_ms", "decision_time_p95_ms"]
    rows = []
    for method, group in per_seed.groupby("method", sort=False):
        row = {"method": method, "runs": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(row: pd.Series, key: str, digits: int = 4) -> str:
    return f"{row[f'{key}_mean']:.{digits}f} $\\pm$ {row[f'{key}_std']:.{digits}f}"


def write_outputs(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "| Method | Mapping success | Admission ratio | SLA violation | Resource utilization | Avg latency (ms) | P95 latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        cells = [row["method"]]
        for key in ["mapping_success", "admission_ratio", "sla_violation", "resource_utilization", "decision_time_ms", "decision_time_p95_ms"]:
            cells.append(f"{row[f'{key}_mean']:.4f} +/- {row[f'{key}_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    (output / "end_to_end_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Lightweight end-to-end intent management simulation. Values are mean $\pm$ standard deviation over five seeds.}",
        r"\label{tab:end_to_end_management}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Mapping success & Admission ratio & SLA violation & Resource util. & Avg (ms) & P95 (ms) \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        latex.append(" & ".join([str(row["method"]), fmt(row, "mapping_success"), fmt(row, "admission_ratio"), fmt(row, "sla_violation"), fmt(row, "resource_utilization"), fmt(row, "decision_time_ms", 3), fmt(row, "decision_time_p95_ms", 3)]) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    (Path("paper") / "tables" / "table_end_to_end_management.tex").write_text("\n".join(latex), encoding="utf-8")

    fig, axes_grid = plt.subplots(2, 2, figsize=(6.8, 4.4), dpi=220)
    axes = axes_grid.ravel()
    metrics = [("mapping_success", "Mapping success"), ("sla_violation", "SLA violation"), ("resource_utilization", "Resource utilization"), ("decision_time_ms", "Decision time (ms)")]
    colors = ["#9AA0A8", "#7C9186", "#6E7F91", "#4C566A"]
    x = np.arange(len(summary))
    labels = ["SBERT", "QoS-only", "KCGT", "KCGT+QoS"]
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.bar(x, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], capsize=2, color=colors, edgecolor="#333333", linewidth=0.45)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        if metric != "decision_time_ms":
            ax.set_ylim(0, 1.02)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.98, wspace=0.38, hspace=0.52)
    fig.savefig(output / "end_to_end_management.png", bbox_inches="tight")
    fig.savefig(Path("paper") / "figures" / "final_end_to_end_management.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--kcgt-root", default="outputs/method_novelty_v1")
    parser.add_argument("--sbert-root", default="outputs/service_template_scale_baseline_embeddings")
    parser.add_argument("--output-dir", default="outputs/end_to_end_management_simulation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()

    samples = read_jsonl(Path(args.processed_dir) / "test.jsonl")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        kcgt_path = Path(args.kcgt_root) / f"seed_{seed}" / "full_structured_relgt" / "test_group" / "query_embeddings.csv"
        sbert_path = Path(args.sbert_root) / f"seed_{seed}" / "sbert_minilm" / "test_group_projection" / "query_embeddings.csv"
        if not sbert_path.exists():
            # Frozen SBERT is deterministic; seeds 4--5 vary the simulation state,
            # while reusing the audited seed-1 embedding export.
            sbert_path = Path(args.sbert_root) / "seed_1" / "sbert_minilm" / "test_group_projection" / "query_embeddings.csv"
        kcgt_ids, kcgt = load_embeddings(kcgt_path)
        sbert_ids, sbert = load_embeddings(sbert_path)
        common = [x for x in kcgt_ids if x in set(sbert_ids)]
        kcgt_pos = {x: i for i, x in enumerate(kcgt_ids)}
        sbert_pos = {x: i for i, x in enumerate(sbert_ids)}
        kcgt = normalize(kcgt[[kcgt_pos[x] for x in common]])
        sbert = normalize(sbert[[sbert_pos[x] for x in common]])
        rows.extend(evaluate_seed(seed, samples, common, kcgt, sbert, args.queries, args.top_k, args.window_size, args.timing_repeats))
        print(f"[end-to-end] seed={seed} complete", flush=True)

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(output / "end_to_end_per_seed.csv", index=False)
    summary.to_csv(output / "end_to_end_summary.csv", index=False)
    write_outputs(summary, output)
    report = [
        "# Lightweight End-to-End Intent Management Simulation",
        "",
        "Pipeline: intent -> template retrieval -> QoS verification -> capacity-aware admission -> perturbed SLA outcome.",
        "",
        "The protocol uses five matched seeds, 512 requests per seed, windows of 32 concurrent requests, identical capacity trajectories, and identical exogenous disturbances for all methods. The composite SLA is satisfied when at least seven of eight normalized requirements hold and the mean residual QoS gap is at most 0.08. SLA violation is measured only among admitted requests; admission ratio is reported to prevent rejection-all policies from appearing favorable.",
        "Frozen SBERT embeddings are deterministic. The audited seed-1 export is reused for simulation seeds 4 and 5, while request order, resource capacity, and network disturbance remain seed-specific.",
        "",
        (output / "end_to_end_summary.md").read_text(encoding="utf-8"),
        "This is a controlled management-plane simulation rather than packet-level or live-operator orchestration. Resource demand, capacity, and service degradation are deterministic functions of auditable QoS and service-profile fields plus seeded perturbations.",
    ]
    (output / "end_to_end_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
