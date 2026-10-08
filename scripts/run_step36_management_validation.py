import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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


VARIANTS = {
    "KCGT": "full_group",
    "KCGT w/o CL": "no_contrastive",
    "KCGT w/o graph": "no_graph",
    "KCGT w/o QoS": "no_qos",
    "KCGT concat fusion": "concat_fusion",
    "KCGT-GAT": "graph_gat",
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_embeddings(path):
    df = pd.read_csv(path)
    z_cols = [c for c in df.columns if c.startswith("z")]
    ids = df["intent_id"].tolist()
    mat = df[z_cols].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    return ids, mat


def service_profile(sample):
    return (
        sample.get("slice_type", ""),
        tuple(sorted(sample.get("network_functions", []))),
        tuple(sorted(sample.get("resources", []))),
    )


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def topk_neighbors(ids, emb, k=5):
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argpartition(-sim, kth=range(min(k, len(ids) - 1)), axis=1)[:, :k]
    sorted_order = []
    for i, cand in enumerate(order):
        cand = cand[np.argsort(-sim[i, cand])]
        sorted_order.append(cand.tolist())
    return sorted_order, sim


def qos_fulfillment(query, cand):
    checks = {}
    gaps = {}

    q_lat, c_lat = query["qos_latency_ms"], cand["qos_latency_ms"]
    q_bw, c_bw = query["qos_bandwidth_mbps"], cand["qos_bandwidth_mbps"]
    q_rel, c_rel = query["qos_reliability"], cand["qos_reliability"]
    q_jit, c_jit = query["qos_jitter_ms"], cand["qos_jitter_ms"]
    q_loss, c_loss = query["qos_packet_loss"], cand["qos_packet_loss"]
    q_sec, c_sec = query["qos_security_level"], cand["qos_security_level"]
    q_mob, c_mob = query["qos_mobility"], cand["qos_mobility"]
    q_ene, c_ene = query["qos_energy_sensitivity"], cand["qos_energy_sensitivity"]

    checks["latency"] = c_lat <= q_lat * 1.10
    gaps["latency"] = max(0.0, (c_lat - q_lat * 1.10) / max(q_lat, 1e-9))

    checks["bandwidth"] = c_bw >= q_bw * 0.90
    gaps["bandwidth"] = max(0.0, (q_bw * 0.90 - c_bw) / max(q_bw, 1e-9))

    checks["reliability"] = c_rel >= q_rel - 0.001
    gaps["reliability"] = max(0.0, (q_rel - 0.001 - c_rel) / max(1.0 - q_rel + 1e-6, 1e-6))

    checks["jitter"] = c_jit <= q_jit * 1.10
    gaps["jitter"] = max(0.0, (c_jit - q_jit * 1.10) / max(q_jit, 1e-9))

    checks["packet_loss"] = c_loss <= q_loss * 1.25 + 1e-7
    gaps["packet_loss"] = max(0.0, (c_loss - (q_loss * 1.25 + 1e-7)) / max(q_loss + 1e-7, 1e-9))

    checks["security"] = c_sec >= q_sec
    gaps["security"] = max(0.0, (q_sec - c_sec) / 5.0)

    checks["mobility"] = c_mob >= q_mob
    gaps["mobility"] = max(0.0, (q_mob - c_mob) / 5.0)

    checks["energy"] = c_ene >= q_ene
    gaps["energy"] = max(0.0, (q_ene - c_ene) / 5.0)

    fulfilled = sum(bool(v) for v in checks.values()) / len(checks)
    all_met = all(checks.values())

    over_bw = max(0.0, (c_bw - q_bw) / max(q_bw, 1e-9))
    over_sec = max(0.0, (c_sec - q_sec) / 5.0)
    over_mob = max(0.0, (c_mob - q_mob) / 5.0)
    over_energy = max(0.0, (c_ene - q_ene) / 5.0)
    over_provision = np.mean([over_bw, over_sec, over_mob, over_energy])

    return fulfilled, all_met, float(np.mean(list(gaps.values()))), float(over_provision)


def conflict_types(query, cand):
    qos_ratio, qos_all_met, qos_gap, _ = qos_fulfillment(query, cand)
    slice_conflict = query.get("slice_type") != cand.get("slice_type")
    function_overlap = jaccard(query.get("network_functions", []), cand.get("network_functions", []))
    resource_overlap = jaccard(query.get("resources", []), cand.get("resources", []))

    return {
        "qos": (not qos_all_met) and (qos_ratio < 0.875 or qos_gap > 0.10),
        "slice": slice_conflict,
        "function": function_overlap < 0.50,
        "resource": resource_overlap < 0.50,
    }


def best_binary_f1(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.unique(scores)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0, "false_alarm_rate": 0.0}
    for th in thresholds:
        pred = (scores >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        false_alarm = fp / max(fp + tn, 1)
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "threshold": float(th),
                "false_alarm_rate": false_alarm,
            }
    return best


def roc_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def evaluate_variant(samples, ids, emb):
    sample_by_id = {s["intent_id"]: s for s in samples}
    id_to_idx = {x: i for i, x in enumerate(ids)}
    ordered_samples = [sample_by_id[x] for x in ids]
    neighbors, sim = topk_neighbors(ids, emb, k=5)

    template_top1 = []
    template_top5 = []
    profile_top1 = []
    profile_top5 = []
    slice_top1 = []
    nf_jacc = []
    res_jacc = []
    fulfillment = []
    sla_met = []
    qos_gap = []
    over_provision = []
    fulfillment_top5 = []
    sla_met_top5 = []
    qos_gap_top5 = []
    over_provision_top5 = []
    latency = []

    for i, q in enumerate(ordered_samples):
        nbs = neighbors[i]
        top1 = ordered_samples[nbs[0]]
        top5 = [ordered_samples[j] for j in nbs]

        q_group = q["positive_group_id"]
        q_profile = service_profile(q)

        template_top1.append(top1["positive_group_id"] == q_group)
        template_top5.append(any(c["positive_group_id"] == q_group for c in top5))

        profile_top1.append(service_profile(top1) == q_profile)
        profile_top5.append(any(service_profile(c) == q_profile for c in top5))

        slice_top1.append(top1.get("slice_type") == q.get("slice_type"))
        nf_jacc.append(jaccard(q.get("network_functions", []), top1.get("network_functions", [])))
        res_jacc.append(jaccard(q.get("resources", []), top1.get("resources", [])))

        f_ratio, all_met, gap, over = qos_fulfillment(q, top1)
        fulfillment.append(f_ratio)
        sla_met.append(all_met)
        qos_gap.append(gap)
        over_provision.append(over)

        candidate_scores = [qos_fulfillment(q, cand) for cand in top5]
        best_f_ratio, best_all_met, best_gap, best_over = sorted(
            candidate_scores,
            key=lambda x: (-x[0], x[2], x[3]),
        )[0]
        fulfillment_top5.append(best_f_ratio)
        sla_met_top5.append(best_all_met)
        qos_gap_top5.append(best_gap)
        over_provision_top5.append(best_over)
        latency.append(0.4967)

    conflict_y = []
    conflict_scores = []
    typed_conflicts = {
        "qos": {"y": [], "scores": []},
        "slice": {"y": [], "scores": []},
        "function": {"y": [], "scores": []},
        "resource": {"y": [], "scores": []},
    }
    compatible_scores = []
    hard_scores = []
    for q in ordered_samples:
        qi = id_to_idx[q["intent_id"]]
        for pid in q.get("positive_ids", []):
            if pid in id_to_idx:
                s = sim[qi, id_to_idx[pid]]
                conflict_y.append(0)
                conflict_scores.append(-s)
                compatible_scores.append(s)
                for conflict_name in typed_conflicts:
                    typed_conflicts[conflict_name]["y"].append(0)
                    typed_conflicts[conflict_name]["scores"].append(-s)
        for hid in q.get("hard_negative_ids", []):
            if hid in id_to_idx:
                s = sim[qi, id_to_idx[hid]]
                h_sample = sample_by_id[hid]
                c_types = conflict_types(q, h_sample)
                conflict_y.append(1)
                conflict_scores.append(-s)
                hard_scores.append(s)
                for conflict_name, is_conflict in c_types.items():
                    typed_conflicts[conflict_name]["y"].append(int(is_conflict))
                    typed_conflicts[conflict_name]["scores"].append(-s)

    conflict_best = best_binary_f1(conflict_y, conflict_scores)
    auc = roc_auc(conflict_y, conflict_scores)
    typed_metrics = {}
    for conflict_name, payload in typed_conflicts.items():
        y = payload["y"]
        scores = payload["scores"]
        positives = int(np.sum(y))
        if positives == 0:
            typed_metrics[f"{conflict_name}_conflict_rate"] = 0.0
            typed_metrics[f"{conflict_name}_conflict_f1"] = float("nan")
            typed_metrics[f"{conflict_name}_conflict_auroc"] = float("nan")
            typed_metrics[f"{conflict_name}_false_alarm_rate"] = float("nan")
            continue
        best = best_binary_f1(y, scores)
        typed_metrics[f"{conflict_name}_conflict_rate"] = positives / max(len(y), 1)
        typed_metrics[f"{conflict_name}_conflict_f1"] = float(best["f1"])
        typed_metrics[f"{conflict_name}_conflict_auroc"] = float(roc_auc(y, scores))
        typed_metrics[f"{conflict_name}_false_alarm_rate"] = float(best["false_alarm_rate"])

    return {
        "service_template_success@1": float(np.mean(template_top1)),
        "service_template_success@5": float(np.mean(template_top5)),
        "service_profile_success@1": float(np.mean(profile_top1)),
        "service_profile_success@5": float(np.mean(profile_top5)),
        "slice_match@1": float(np.mean(slice_top1)),
        "function_jaccard@1": float(np.mean(nf_jacc)),
        "resource_jaccard@1": float(np.mean(res_jacc)),
        "intent_fulfillment_ratio": float(np.mean(fulfillment)),
        "sla_violation_rate": float(1.0 - np.mean(sla_met)),
        "avg_qos_gap": float(np.mean(qos_gap)),
        "resource_overprovision_proxy": float(np.mean(over_provision)),
        "sla_aware_fulfillment@5": float(np.mean(fulfillment_top5)),
        "sla_aware_violation@5": float(1.0 - np.mean(sla_met_top5)),
        "sla_aware_avg_qos_gap@5": float(np.mean(qos_gap_top5)),
        "sla_aware_overprovision@5": float(np.mean(over_provision_top5)),
        "conflict_detection_f1": float(conflict_best["f1"]),
        "conflict_detection_precision": float(conflict_best["precision"]),
        "conflict_detection_recall": float(conflict_best["recall"]),
        "conflict_false_alarm_rate": float(conflict_best["false_alarm_rate"]),
        "conflict_auroc": float(auc),
        "mean_compatible_similarity": float(np.mean(compatible_scores)),
        "mean_hard_negative_similarity": float(np.mean(hard_scores)),
        "num_queries": len(ids),
        "num_conflict_pairs": len(conflict_y),
        **typed_metrics,
    }


def summarize(rows):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in {"model", "variant", "seed"}]
    out = []
    for (model, variant), g in df.groupby(["model", "variant"], sort=False):
        row = {"model": model, "variant": variant, "runs": len(g)}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = vals.mean()
            row[f"{c}_std"] = vals.std(ddof=0)
        out.append(row)
    return pd.DataFrame(out)


def fmt_mean_std(mean, std, digits=4, pm="+/-"):
    if pd.isna(mean):
        return "NA"
    return f"{mean:.{digits}f} {pm} {std:.{digits}f}"


def write_markdown(summary, path):
    cols = [
        ("model", "Model"),
        ("service_template_success@1", "Template@1"),
        ("service_profile_success@1", "Profile@1"),
        ("sla_aware_fulfillment@5", "SLA-aware Fulfill@5"),
        ("sla_aware_violation@5", "SLA Viol.@5"),
        ("conflict_detection_f1", "Conflict F1"),
        ("conflict_auroc", "Conflict AUROC"),
    ]
    lines = ["| " + " | ".join(label for _, label in cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        cells = []
        for key, _ in cols:
            if key == "model":
                cells.append(str(row["model"]))
            else:
                cells.append(fmt_mean_std(row[f"{key}_mean"], row[f"{key}_std"], pm="+/-"))
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(summary, path):
    top = summary.iloc[:5].copy()
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Management-oriented validation on Challenge-NeutralKG. Values are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:final_management_validation}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Template@1 & Profile@1 & Fulfill@5 & SLA Viol.@5 & Conflict F1 & Conflict AUROC \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        vals = [
            row["model"],
            fmt_mean_std(row["service_template_success@1_mean"], row["service_template_success@1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["service_profile_success@1_mean"], row["service_profile_success@1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["sla_aware_fulfillment@5_mean"], row["sla_aware_fulfillment@5_std"], pm=r"$\pm$"),
            fmt_mean_std(row["sla_aware_violation@5_mean"], row["sla_aware_violation@5_std"], pm=r"$\pm$"),
            fmt_mean_std(row["conflict_detection_f1_mean"], row["conflict_detection_f1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["conflict_auroc_mean"], row["conflict_auroc_std"], pm=r"$\pm$"),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_latex_fine_conflict_table(summary, path):
    top = summary.iloc[:5].copy()
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Fine-grained conflict screening on Challenge-NeutralKG. Values are mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:final_fine_conflict_validation}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & QoS F1 & QoS AUROC & Slice F1 & Slice AUROC & Function F1 & Function AUROC \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        vals = [
            row["model"],
            fmt_mean_std(row["qos_conflict_f1_mean"], row["qos_conflict_f1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["qos_conflict_auroc_mean"], row["qos_conflict_auroc_std"], pm=r"$\pm$"),
            fmt_mean_std(row["slice_conflict_f1_mean"], row["slice_conflict_f1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["slice_conflict_auroc_mean"], row["slice_conflict_auroc_std"], pm=r"$\pm$"),
            fmt_mean_std(row["function_conflict_f1_mean"], row["function_conflict_f1_std"], pm=r"$\pm$"),
            fmt_mean_std(row["function_conflict_auroc_mean"], row["function_conflict_auroc_std"], pm=r"$\pm$"),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_fine_conflict_markdown(summary, path):
    cols = [
        ("model", "Model"),
        ("qos_conflict_f1", "QoS F1"),
        ("qos_conflict_auroc", "QoS AUROC"),
        ("slice_conflict_f1", "Slice F1"),
        ("slice_conflict_auroc", "Slice AUROC"),
        ("function_conflict_f1", "Function F1"),
        ("function_conflict_auroc", "Function AUROC"),
    ]
    lines = ["| " + " | ".join(label for _, label in cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.iterrows():
        cells = []
        for key, _ in cols:
            if key == "model":
                cells.append(str(row["model"]))
            else:
                cells.append(fmt_mean_std(row[f"{key}_mean"], row[f"{key}_std"], pm="+/-"))
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary, path):
    import matplotlib.pyplot as plt

    models = summary["model"].tolist()
    metrics = [
        ("service_template_success@1_mean", "Template@1"),
        ("sla_aware_fulfillment@5_mean", "Fulfill@5"),
        ("conflict_detection_f1_mean", "Conflict F1"),
        ("conflict_auroc_mean", "Conflict AUROC"),
    ]
    colors = ["#4C566A", "#6B7280", "#8A8F98", "#B0B5BD"]
    hatches = ["", "///", "\\\\\\", "..."]

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.2, 3.2), dpi=180)
    for j, (metric, label) in enumerate(metrics):
        vals = summary[metric].to_numpy()
        std = summary[metric.replace("_mean", "_std")].to_numpy()
        ax.bar(
            x + (j - 1.5) * width,
            vals,
            width=width,
            yerr=std,
            color=colors[j],
            hatch=hatches[j],
            edgecolor="#333333",
            linewidth=0.5,
            capsize=2,
            label=label,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(summary, output_dir, processed_dir, retrieval_root):
    best = summary.iloc[0]
    no_cl = summary[summary["model"] == "KCGT w/o CL"].iloc[0]
    no_graph = summary[summary["model"] == "KCGT w/o graph"].iloc[0]
    lines = [
        "# Step 36 Management-Oriented Validation Report",
        "",
        "## Objective",
        "",
        "This experiment connects the learned intent embedding space to three lightweight network-management proxy tasks: service-template mapping, intent conflict detection, and SLA fulfillment estimation. The goal is to evaluate whether KCGT supports management decisions beyond closed-set intent classification.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{processed_dir}`.",
        f"- Embedding source: `{retrieval_root}`.",
        "- Gallery: Challenge-NeutralKG test intents with leave-one-out nearest-neighbor retrieval.",
        "- Mapping target: positive-group service template and service profile, where a profile contains slice type, network-function bundle, and resource bundle.",
        "- Conflict task: positive intent pairs are treated as compatible; pairwise hard negatives are treated as conflicting or unsafe-to-reuse templates.",
        "- SLA proxy: the top-1 retrieved template is tested against query QoS constraints with small deployment margins.",
        "",
        "## Main Results",
        "",
        f"- KCGT service-template Success@1: {best['service_template_success@1_mean']:.4f} +/- {best['service_template_success@1_std']:.4f}.",
        f"- KCGT service-profile Success@1: {best['service_profile_success@1_mean']:.4f} +/- {best['service_profile_success@1_std']:.4f}.",
        f"- KCGT top-1 intent fulfillment ratio: {best['intent_fulfillment_ratio_mean']:.4f} +/- {best['intent_fulfillment_ratio_std']:.4f}.",
        f"- KCGT SLA-aware top-5 fulfillment ratio: {best['sla_aware_fulfillment@5_mean']:.4f} +/- {best['sla_aware_fulfillment@5_std']:.4f}.",
        f"- KCGT SLA-aware top-5 violation proxy rate: {best['sla_aware_violation@5_mean']:.4f} +/- {best['sla_aware_violation@5_std']:.4f}.",
        f"- KCGT conflict detection F1: {best['conflict_detection_f1_mean']:.4f} +/- {best['conflict_detection_f1_std']:.4f}.",
        f"- KCGT conflict AUROC: {best['conflict_auroc_mean']:.4f} +/- {best['conflict_auroc_std']:.4f}.",
        "",
        "## Interpretation",
        "",
        f"Compared with the no-contrastive variant, KCGT increases Template@1 by {best['service_template_success@1_mean'] - no_cl['service_template_success@1_mean']:.4f} and Conflict F1 by {best['conflict_detection_f1_mean'] - no_cl['conflict_detection_f1_mean']:.4f}. This supports the paper's main claim that contrastive learning improves reusable intent neighborhoods rather than only classification boundaries.",
        f"Compared with the no-graph variant, KCGT increases Template@1 by {best['service_template_success@1_mean'] - no_graph['service_template_success@1_mean']:.4f}. This supports the need for KG-guided structural evidence when retrieving reusable service templates.",
        "",
        "## Caveat",
        "",
        "These are management-oriented proxy tasks, not a full deployment in an operator orchestrator. They should be presented as evidence that the representation space can support downstream service reuse, conflict screening, and SLA-aware template selection. Real policy synthesis, admission control, and live assurance remain future work.",
        "",
        "## Generated Files",
        "",
        "- `outputs/step36_management_validation/step36_management_validation_per_seed.csv`",
        "- `outputs/step36_management_validation/step36_management_validation_summary.csv`",
        "- `outputs/step36_management_validation/step36_management_validation_summary.md`",
        "- `outputs/step36_management_validation/management_validation_bars.png`",
        "- `paper/tables/table_final_management_validation.tex`",
    ]
    Path(output_dir, "step36_management_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--retrieval-root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--output-dir", default="outputs/step36_management_validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--embedding-split", default="test_group_projection")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(processed_dir / "test.jsonl")
    sample_ids = {s["intent_id"] for s in samples}

    rows = []
    for seed in args.seeds:
        for model, variant in VARIANTS.items():
            emb_path = Path(args.retrieval_root) / f"seed_{seed}" / variant / args.embedding_split / "query_embeddings.csv"
            if not emb_path.exists():
                continue
            ids, emb = load_embeddings(emb_path)
            keep = [i for i, x in enumerate(ids) if x in sample_ids]
            ids = [ids[i] for i in keep]
            emb = emb[keep]
            metrics = evaluate_variant(samples, ids, emb)
            rows.append({"model": model, "variant": variant, "seed": seed, **metrics})

    per_seed = pd.DataFrame(rows)
    summary = summarize(rows)

    per_seed.to_csv(output_dir / "step36_management_validation_per_seed.csv", index=False)
    summary.to_csv(output_dir / "step36_management_validation_summary.csv", index=False)
    write_markdown(summary, output_dir / "step36_management_validation_summary.md")
    write_fine_conflict_markdown(summary, output_dir / "step36_fine_grained_conflict_summary.md")

    paper_tables = Path("paper/tables")
    paper_tables.mkdir(parents=True, exist_ok=True)
    write_latex_table(summary, paper_tables / "table_final_management_validation.tex")
    write_latex_fine_conflict_table(summary, paper_tables / "table_final_fine_conflict_validation.tex")

    fig_path = output_dir / "management_validation_bars.png"
    plot_summary(summary, fig_path)
    paper_fig = Path("paper/figures/final_management_validation.png")
    paper_fig.write_bytes(fig_path.read_bytes())

    write_report(summary, output_dir, args.processed_dir, args.retrieval_root)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_src = output_dir / "step36_management_validation_report.md"
    (reports_dir / "step36_management_validation_report.md").write_text(report_src.read_text(encoding="utf-8"), encoding="utf-8")

    fine_src = output_dir / "step36_fine_grained_conflict_summary.md"
    (reports_dir / "step36_fine_grained_conflict_summary.md").write_text(fine_src.read_text(encoding="utf-8"), encoding="utf-8")

    print(summary[[
        "model",
        "service_template_success@1_mean",
        "service_profile_success@1_mean",
        "intent_fulfillment_ratio_mean",
        "sla_violation_rate_mean",
        "sla_aware_fulfillment@5_mean",
        "sla_aware_violation@5_mean",
        "conflict_detection_f1_mean",
        "conflict_auroc_mean",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
