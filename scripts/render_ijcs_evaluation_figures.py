from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F2A65A",
    "green": "#72A879",
    "red": "#C76D6D",
    "purple": "#8E7DBE",
    "gray": "#8B8B8B",
    "teal": "#5D9CA0",
    "gold": "#D2AD4F",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "patch.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finish(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def parse_value(value: str) -> tuple[float, float]:
    parts = str(value).split("+/-")
    mean = float(parts[0].strip())
    std = float(parts[1].strip()) if len(parts) == 2 else 0.0
    return mean, std


def render_representation_figure() -> None:
    summary = pd.read_csv(ROOT / "outputs" / "method_novelty_v1" / "method_novelty_summary.csv")
    embedding_path = (
        ROOT
        / "outputs"
        / "method_novelty_v1"
        / "seed_1"
        / "full_structured_relgt"
        / "test_group"
        / "query_embeddings.csv"
    )
    emb = pd.read_csv(embedding_path)
    feature_cols = [c for c in emb.columns if c.startswith("z")]
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1200,
    ).fit_transform(emb[feature_cols].to_numpy())

    with open(
        ROOT / "outputs" / "method_novelty_v1" / "seed_1" / "full_structured_relgt" / "metrics.json",
        encoding="utf-8",
    ) as handle:
        class_to_id = json.load(handle)["class_to_id"]
    abbreviations = {
        "AR/VR Immersive Service": "AR/VR",
        "Autonomous Driving / V2X": "V2X",
        "Cloud Gaming": "Gaming",
        "Industrial URLLC": "Industrial",
        "Massive IoT mMTC": "mMTC",
        "Security-Critical Enterprise": "Security",
        "Telemedicine": "Telemed.",
        "Video Streaming eMBB": "Video",
    }
    palette = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["gray"],
        COLORS["teal"],
        COLORS["gold"],
    ]

    fig = plt.figure(figsize=(7.15, 3.02))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.28, 0.48, 1, 1],
        wspace=0.62,
        hspace=0.82,
    )
    ax0 = fig.add_subplot(grid[:, 0])
    for name, label_id in sorted(class_to_id.items(), key=lambda item: item[1]):
        mask = emb["label"].to_numpy() == label_id
        ax0.scatter(
            tsne[mask, 0],
            tsne[mask, 1],
            s=10,
            alpha=0.72,
            linewidths=0,
            color=palette[label_id],
            label=abbreviations[name],
        )
    ax0.set_xlabel("t-SNE dimension 1")
    ax0.set_ylabel("t-SNE dimension 2")
    ax0.set_xticks([])
    ax0.set_yticks([])
    handles, legend_labels = ax0.get_legend_handles_labels()
    legend_ax = fig.add_subplot(grid[:, 1])
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(-0.94, 0.5),
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.35,
        borderaxespad=0,
        labelspacing=0.36,
    )
    panel_label(ax0, "(a)")

    labels = ["KCGT", "SupCon", "R-GCN", "Shuffled rel."]
    variants = [
        "full_structured_relgt",
        "plain_supcon_relgt",
        "structured_rgcn",
        "structured_relgt_shuffled",
    ]
    ordered = summary.set_index("variant").loc[variants]
    x = np.arange(len(labels))
    width = 0.34

    ax1 = fig.add_subplot(grid[0, 2:])
    ax1.bar(
        x - width / 2,
        ordered["group_recall@1_mean"],
        width,
        yerr=ordered["group_recall@1_std"],
        color=COLORS["blue"],
        edgecolor="black",
        capsize=1.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        label="Group R@1",
    )
    ax1.bar(
        x + width / 2,
        ordered["group_mrr_mean"],
        width,
        yerr=ordered["group_mrr_std"],
        color=COLORS["orange"],
        edgecolor="black",
        capsize=1.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        hatch="///",
        label="Group MRR",
    )
    ax1.set_ylim(0.42, 0.75)
    ax1.set_ylabel("Retrieval score")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.60, 1.02),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    ax1.text(-0.10, 1.11, "(b)", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="bottom")

    ax2 = fig.add_subplot(grid[1, 2:])
    ax2.bar(
        x - width / 2,
        ordered["hard_precision@5_mean"],
        width,
        yerr=ordered["hard_precision@5_std"],
        color=COLORS["green"],
        edgecolor="black",
        capsize=1.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        label="Hard P@5",
    )
    ax2.bar(
        x + width / 2,
        ordered["hard_ndcg@5_mean"],
        width,
        yerr=ordered["hard_ndcg@5_std"],
        color=COLORS["purple"],
        edgecolor="black",
        capsize=1.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        hatch="\\\\\\",
        label="Hard NDCG@5",
    )
    ax2.set_ylim(0.34, 0.64)
    ax2.set_ylabel("Hard-negative score")
    ax2.set_xticks(x, labels)
    ax2.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.60, 1.02),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    ax2.text(-0.10, 1.11, "(c)", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="bottom")

    finish(fig, "ijcs_representation_evidence")


def render_robustness_transfer_figure() -> None:
    robustness = pd.read_csv(
        ROOT / "outputs" / "final_robustness_relgt_oscl" / "step34_robustness_summary.csv"
    )
    transfer = pd.read_csv(
        ROOT / "outputs" / "final_experiment_tables" / "table4_cross_protocol_schema_shift.csv"
    )
    gates = pd.read_csv(
        ROOT / "outputs" / "final_explainability_relgt_oscl" / "gate_by_setting.csv"
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.62})

    selected = [
        "text_paraphrase",
        "text_dropout",
        "qos_noise_15pct",
        "kg_edge_dropout_30pct",
        "kg_node_mask_30pct",
        "relation_mask_has_constraint",
        "relation_mask_uses_resource_family",
    ]
    labels = ["Paraphrase", "Text drop", "QoS noise", "Edge drop", "Node mask", "No constraint", "No resource"]
    robust = robustness.set_index("mode").loc[selected]
    y = np.arange(len(selected))
    height = 0.34
    axes[0].barh(
        y - height / 2,
        robust["delta_recall@1"],
        height,
        color=COLORS["blue"],
        edgecolor="black",
        label="R@1 change",
    )
    axes[0].barh(
        y + height / 2,
        robust["delta_mrr"],
        height,
        color=COLORS["orange"],
        edgecolor="black",
        hatch="///",
        label="MRR change",
    )
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.51, 0.035)
    axes[0].set_xlabel("Change from clean")
    axes[0].grid(axis="x", color="#D9D9D9", linewidth=0.5)
    axes[0].set_axisbelow(True)
    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.58, 1.01),
        columnspacing=0.65,
        handletextpad=0.3,
    )
    axes[0].text(-0.13, 1.16, "(a)", transform=axes[0].transAxes, fontsize=8.5, fontweight="bold", va="top")

    transfer_labels = ["H2C", "C2H", "H2N", "C2N", "N2C", "N2H"]
    recall = np.array([parse_value(v)[0] for v in transfer["Group R@1"]])
    recall_err = np.array([parse_value(v)[1] for v in transfer["Group R@1"]])
    mrr = np.array([parse_value(v)[0] for v in transfer["Group MRR"]])
    mrr_err = np.array([parse_value(v)[1] for v in transfer["Group MRR"]])
    x = np.arange(len(transfer_labels))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        recall,
        width,
        yerr=recall_err,
        color=COLORS["green"],
        edgecolor="black",
        capsize=1.5,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#333333"},
        label="Group R@1",
    )
    axes[1].bar(
        x + width / 2,
        mrr,
        width,
        yerr=mrr_err,
        color=COLORS["purple"],
        edgecolor="black",
        capsize=1.5,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#333333"},
        hatch="\\\\\\",
        label="Group MRR",
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Transfer score")
    axes[1].set_xticks(x, transfer_labels, rotation=35, ha="right")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.56, 1.01),
        columnspacing=0.65,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    axes[1].text(-0.13, 1.16, "(b)", transform=axes[1].transAxes, fontsize=8.5, fontweight="bold", va="top")

    gate_labels = ["In protocol", "C to N", "N to C"]
    x = np.arange(len(gate_labels))
    bottom = np.zeros(len(gates))
    for column, label, color, hatch in [
        ("text_gate", "Text", COLORS["blue"], ""),
        ("graph_gate", "Graph", COLORS["orange"], "///"),
        ("qos_gate", "QoS", COLORS["green"], "\\\\\\"),
    ]:
        axes[2].bar(
            x,
            gates[column],
            bottom=bottom,
            color=color,
            edgecolor="black",
            hatch=hatch,
            label=label,
        )
        bottom += gates[column].to_numpy()
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Mean gate weight")
    axes[2].set_xticks(x, gate_labels, rotation=25, ha="right")
    axes[2].grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[2].set_axisbelow(True)
    axes[2].legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.6,
        handletextpad=0.3,
    )
    accuracy_ax = axes[2].twinx()
    accuracy_ax.plot(x, gates["accuracy"], color=COLORS["red"], marker="o", label="Accuracy")
    accuracy_ax.set_ylim(0, 1.05)
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.tick_params(axis="y", colors=COLORS["red"])
    accuracy_ax.yaxis.label.set_color(COLORS["red"])
    axes[2].text(
        -0.17,
        1.16,
        "(c)",
        transform=axes[2].transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )

    finish(fig, "ijcs_robustness_transfer")


def render_management_figure() -> None:
    scale = pd.read_csv(
        ROOT / "outputs" / "service_template_scale_clean_relgt_oscl" / "clean_scale_summary.csv"
    )
    external = pd.read_csv(
        ROOT / "outputs" / "external_5qi_protocol_relgt_oscl" / "external_5qi_summary.csv"
    )
    end_to_end = pd.read_csv(
        ROOT / "outputs" / "end_to_end_management_simulation" / "end_to_end_summary.csv"
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), gridspec_kw={"wspace": 0.56})

    order = ["KCGT", "GraphTransformer", "Text+GraphTransformer", "Text+GAT", "SBERT-MiniLM"]
    labels = ["KCGT", "GraphTrans.", "Text+GraphT.", "Text+GAT", "SBERT"]
    scale_ordered = scale.set_index("model").loc[order]
    x = np.arange(len(order))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        scale_ordered["template_success@1_mean"],
        width,
        yerr=scale_ordered["template_success@1_sd"],
        color=COLORS["blue"],
        edgecolor="black",
        capsize=2,
        label="Template@1",
    )
    axes[0].bar(
        x + width / 2,
        scale_ordered["template_success@5_mean"],
        width,
        yerr=scale_ordered["template_success@5_sd"],
        color=COLORS["orange"],
        edgecolor="black",
        capsize=2,
        hatch="///",
        label="Template@5",
    )
    axes[0].set_ylim(0, 0.39)
    axes[0].set_ylabel("Template recovery")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].set_axisbelow(True)
    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.58, 1.01),
        columnspacing=0.65,
        handletextpad=0.3,
    )
    axes[0].text(-0.16, 1.16, "(a)", transform=axes[0].transAxes, fontsize=8.5, fontweight="bold", va="top")

    ext_order = ["KCGT", "KCGT + standardized-QoS reranking", "Standardized QoS-only", "SBERT-MiniLM"]
    ext_labels = ["KCGT", "KCGT+QoS", "QoS only", "SBERT"]
    ext = external.set_index("model").loc[ext_order]
    x = np.arange(len(ext_order))
    width = 0.25
    metrics = [
        ("fiveqi_success_at_1", "5QI@1", COLORS["blue"], ""),
        ("service_family_at_1", "Family@1", COLORS["orange"], "///"),
        ("sla_fulfillment_at_1", "SLA fulfill.", COLORS["green"], "\\\\\\"),
    ]
    for index, (column, label, color, hatch) in enumerate(metrics):
        axes[1].bar(
            x + (index - 1) * width,
            ext[column],
            width,
            color=color,
            edgecolor="black",
            hatch=hatch,
            label=label,
        )
    axes[1].set_ylim(0.65, 1.04)
    axes[1].set_ylabel("External-profile score")
    axes[1].set_xticks(x, ext_labels, rotation=30, ha="right")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.55,
        handletextpad=0.3,
    )
    axes[1].text(
        -0.16,
        1.16,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )

    method_labels = {
        "SBERT cosine": "SBERT",
        "QoS-only": "QoS only",
        "KCGT retrieval": "KCGT",
        "KCGT + QoS + admission": "KCGT+verify",
    }
    point_colors = [COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["green"]]
    offsets = {
        "SBERT cosine": (5, 5),
        "QoS-only": (7, -2),
        "KCGT retrieval": (-4, 8),
        "KCGT + QoS + admission": (-5, -15),
    }
    alignments = {
        "KCGT retrieval": "right",
        "KCGT + QoS + admission": "right",
        "QoS-only": "left",
        "SBERT cosine": "left",
    }
    vertical_alignments = {
        "KCGT retrieval": "bottom",
        "KCGT + QoS + admission": "top",
        "QoS-only": "top",
        "SBERT cosine": "bottom",
    }
    for (_, row), color in zip(end_to_end.iterrows(), point_colors):
        compliance = 1.0 - row["sla_violation_mean"]
        size = 70 + 160 * row["resource_utilization_mean"]
        axes[2].scatter(
            row["mapping_success_mean"],
            compliance,
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
        )
        axes[2].annotate(
            f"{method_labels[row['method']]}\n{row['decision_time_ms_mean']:.2f} ms",
            (row["mapping_success_mean"], compliance),
            xytext=offsets[row["method"]],
            textcoords="offset points",
            fontsize=7,
            ha=alignments[row["method"]],
            va=vertical_alignments[row["method"]],
        )
    axes[2].set_xlim(-0.03, 0.74)
    axes[2].set_ylim(0, 0.86)
    axes[2].set_xlabel("Mapping success")
    axes[2].set_ylabel("Post-admission SLA compliance")
    axes[2].grid(color="#D9D9D9", linewidth=0.5)
    axes[2].set_axisbelow(True)
    axes[2].text(
        0.18,
        0.015,
        "Marker area: resource utilization",
        transform=axes[2].transAxes,
        fontsize=7,
        color="#444444",
    )
    axes[2].text(-0.16, 1.16, "(c)", transform=axes[2].transAxes, fontsize=8.5, fontweight="bold", va="top")

    finish(fig, "ijcs_management_evaluation")


def main() -> None:
    configure_style()
    render_representation_figure()
    render_robustness_transfer_figure()
    render_management_figure()


if __name__ == "__main__":
    main()
