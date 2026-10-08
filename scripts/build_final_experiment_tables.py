from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_experiment_tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, str]], fields: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def first_value(v: str) -> str:
    return v.replace(" 卤 ", " +/- ").strip()


def main_classification() -> list[dict[str, str]]:
    baseline = read_csv(ROOT / "outputs" / "reports" / "step30_neutral_kg_baselines_classification_mean_std.csv")
    ablation = read_csv(ROOT / "outputs" / "reports" / "step30_neutral_kg_ablation_classification_mean_std.csv")
    rows = []
    wanted_baselines = {
        "text_gat": "Text+GAT",
        "text_graph_transformer": "Text+GraphTransformer",
        "sbert_minilm": "SBERT-MiniLM",
        "lstm": "LSTM",
        "bert_mini": "BERT-mini",
        "qos_mlp": "QoS-MLP",
        "graph_transformer": "GraphTransformer",
        "rgcn": "R-GCN",
        "hypergraph": "Hypergraph Transformer",
    }
    kcgt = next(r for r in ablation if r["model"] == "full_group")
    rows.append(
        {
            "Model": "KCGT",
            "Runs": kcgt["runs"],
            "Accuracy": first_value(kcgt["accuracy"]),
            "Macro-F1": first_value(kcgt["f1_macro"]),
            "Inference ms/sample": first_value(kcgt["inference_time_ms_per_sample"]),
            "Parameters": kcgt["num_parameters"],
        }
    )
    for r in baseline:
        if r["model"] in wanted_baselines:
            rows.append(
                {
                    "Model": wanted_baselines[r["model"]],
                    "Runs": r["runs"],
                    "Accuracy": first_value(r["accuracy"]),
                    "Macro-F1": first_value(r["f1_macro"]),
                    "Inference ms/sample": first_value(r["inference_time_ms_per_sample"]),
                    "Parameters": r["num_parameters"],
                }
            )
    return rows


def retrieval_table() -> list[dict[str, str]]:
    rows = read_csv(ROOT / "outputs" / "reports" / "step30_neutral_kg_retrieval_ablation_mean_std.csv")
    keep = [
        ("full_group", "test_class_projection", "Class retrieval"),
        ("full_group", "test_group_projection", "Group retrieval"),
        ("full_group", "pairwise_hard_negative_projection", "Pairwise hard-negative"),
        ("full_group", "unseen_composition_projection", "Unseen composition"),
        ("no_contrastive", "test_group_projection", "Group retrieval w/o CL"),
        ("no_graph", "test_group_projection", "Group retrieval w/o graph"),
        ("no_qos", "test_group_projection", "Group retrieval w/o QoS"),
    ]
    out = []
    for ablation, scenario, label in keep:
        r = next(x for x in rows if x["ablation"] == ablation and x["scenario"] == scenario)
        out.append(
            {
                "Setting": label,
                "Embedding": r["embedding"],
                "Recall@1": r["recall@1"],
                "Recall@5": r["recall@5"],
                "MRR": r["mrr"],
                "NDCG@5": r["ndcg@5"],
                "Precision@5": r["precision@5"],
            }
        )
    return out


def ablation_table() -> list[dict[str, str]]:
    cls = read_csv(ROOT / "outputs" / "reports" / "step30_neutral_kg_ablation_classification_mean_std.csv")
    ret = read_csv(ROOT / "outputs" / "reports" / "step30_neutral_kg_retrieval_ablation_mean_std.csv")
    names = [
        ("full_group", "Full KCGT"),
        ("no_contrastive", "w/o contrastive"),
        ("no_graph", "w/o graph encoder"),
        ("no_qos", "w/o QoS encoder"),
        ("concat_fusion", "w/o gated fusion"),
        ("graph_gcn", "GCN graph encoder"),
        ("graph_gat", "GAT graph encoder"),
        ("positive_class", "class positives"),
        ("positive_both", "class+group positives"),
    ]
    out = []
    for key, label in names:
        c = next(r for r in cls if r["model"] == key)
        group = next(r for r in ret if r["ablation"] == key and r["scenario"] == "test_group_projection")
        hard = next(r for r in ret if r["ablation"] == key and r["scenario"] == "pairwise_hard_negative_projection")
        out.append(
            {
                "Variant": label,
                "Macro-F1": first_value(c["f1_macro"]),
                "Group R@1": group["recall@1"],
                "Group MRR": group["mrr"],
                "Hard P@5": hard["precision@5"],
            }
        )
    return out


def transfer_table() -> list[dict[str, str]]:
    cls = read_csv(ROOT / "outputs" / "reports" / "cross_dataset_transfer_classification_mean_std.csv")
    ret = read_csv(ROOT / "outputs" / "reports" / "cross_protocol_transfer_retrieval_mean_std.csv")
    pairs = [
        ("hard_v2", "challenge", "compatible"),
        ("challenge", "hard_v2", "compatible"),
        ("hard_v2", "neutral_kg", "schema shift"),
        ("challenge", "neutral_kg", "schema shift"),
        ("neutral_kg", "challenge", "reverse schema shift"),
        ("neutral_kg", "hard_v2", "reverse schema shift"),
    ]
    out = []
    for source, target, typ in pairs:
        c = next(r for r in cls if r["source"] == source and r["target"] == target and r["model"] == "kcgt_contrastive")
        rr = next(
            r
            for r in ret
            if r["source_protocol"] == source
            and r["target_protocol"] == target
            and r["scenario"] == "target_test_group"
            and r["embedding"] == "projection"
        )
        out.append(
            {
                "Transfer": f"{source} -> {target}",
                "Type": typ,
                "Target Macro-F1": c["target_f1_macro"],
                "Group R@1": rr["recall@1"],
                "Group MRR": rr["mrr"],
                "Group NDCG@5": rr["ndcg@5"],
            }
        )
    return out


def robustness_table() -> list[dict[str, str]]:
    rows = read_csv(ROOT / "outputs" / "step34_robustness" / "step34_robustness_summary.csv")
    pretty = {
        "clean": "Clean",
        "text_paraphrase": "Text paraphrase",
        "text_dropout": "Text dropout",
        "qos_noise_15pct": "QoS noise 15%",
        "kg_edge_dropout_30pct": "KG edge dropout 30%",
        "kg_node_mask_30pct": "KG node mask 30%",
        "relation_mask_has_constraint": "Mask has-constraint",
        "relation_mask_uses_capability": "Mask uses-capability",
        "relation_mask_uses_resource_family": "Mask resource-family",
    }
    wanted = [
        "clean",
        "text_paraphrase",
        "text_dropout",
        "qos_noise_15pct",
        "kg_edge_dropout_30pct",
        "kg_node_mask_30pct",
        "relation_mask_has_constraint",
        "relation_mask_uses_capability",
        "relation_mask_uses_resource_family",
    ]
    out = []
    for mode in wanted:
        r = next(x for x in rows if x["mode"] == mode)
        out.append(
            {
                "Perturbation": pretty.get(mode, mode),
                "Macro-F1": f"{float(r['f1_macro']):.4f}",
                "Delta F1": f"{float(r['delta_f1']):.4f}",
                "Recall@1": f"{float(r['recall@1']):.4f}",
                "Delta R@1": f"{float(r['delta_recall@1']):.4f}",
                "MRR": f"{float(r['mrr']):.4f}",
                "Delta MRR": f"{float(r['delta_mrr']):.4f}",
            }
        )
    return out


def complexity_table() -> list[dict[str, str]]:
    rows = read_csv(ROOT / "outputs" / "step35_complexity_efficiency" / "step35_complexity_efficiency_summary.csv")
    wanted = ["KCGT", "Text+GAT", "Text+GraphTransformer", "SBERT-MiniLM", "BERT-mini", "GraphTransformer", "Hypergraph"]
    out = []
    for name in wanted:
        r = next(x for x in rows if x["model"] == name)
        out.append(
            {
                "Model": r["model"],
                "Runs": r["runs"],
                "Macro-F1": r["macro_f1"],
                "Inference ms/sample": r["inference_ms_per_sample"],
                "Parameters": r["parameters"],
                "Checkpoint MB": r["checkpoint_mb"],
            }
        )
    return out


def write_all() -> None:
    tables = [
        ("table1_main_classification", "Final Table 1: Main NeutralKG Classification Results", main_classification(), ["Model", "Runs", "Accuracy", "Macro-F1", "Inference ms/sample", "Parameters"]),
        ("table2_representation_retrieval", "Final Table 2: Intent Representation and Retrieval Results", retrieval_table(), ["Setting", "Embedding", "Recall@1", "Recall@5", "MRR", "NDCG@5", "Precision@5"]),
        ("table3_ablation", "Final Table 3: KCGT Ablation Results", ablation_table(), ["Variant", "Macro-F1", "Group R@1", "Group MRR", "Hard P@5"]),
        ("table4_cross_protocol_schema_shift", "Final Table 4: Cross-Protocol and Schema-Shift Transfer", transfer_table(), ["Transfer", "Type", "Target Macro-F1", "Group R@1", "Group MRR", "Group NDCG@5"]),
        ("table5_robustness", "Final Table 5: Robustness to Text, QoS, and KG Perturbations", robustness_table(), ["Perturbation", "Macro-F1", "Delta F1", "Recall@1", "Delta R@1", "MRR", "Delta MRR"]),
        ("table6_complexity_efficiency", "Final Table 6: Complexity and Efficiency", complexity_table(), ["Model", "Runs", "Macro-F1", "Inference ms/sample", "Parameters", "Checkpoint MB"]),
    ]
    index_lines = ["# Final Experiment Tables", ""]
    for stem, title, rows, fields in tables:
        write_csv(OUT / f"{stem}.csv", rows, fields)
        write_md(OUT / f"{stem}.md", title, rows, fields)
        index_lines.append(f"- `{stem}.md`: {title}")
    (OUT / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "tables": len(tables)}, indent=2))


if __name__ == "__main__":
    write_all()
