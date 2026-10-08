from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets import KNIIntentDataset, kni_collate_fn
from datasets.kni_intent_dataset import read_jsonl
from evaluate_checkpoint_classification import load_checkpoint
from experiments.train_baseline_classification import move_batch


def short_label(label: str) -> str:
    return (
        label.replace("AR/VR Immersive Service", "AR/VR")
        .replace("Autonomous Driving / V2X", "V2X")
        .replace("Industrial URLLC", "Industrial")
        .replace("Massive IoT mMTC", "mMTC")
        .replace("Security-Critical Enterprise", "Security")
        .replace("Video Streaming eMBB", "Video")
    )


def extract_gates(name: str, checkpoint: Path, jsonl: Path, out_csv: Path) -> list[dict]:
    device = torch.device("cpu")
    model, artifacts, _, _ = load_checkpoint(checkpoint, device)
    dataset = KNIIntentDataset(jsonl, artifacts)
    records = read_jsonl(jsonl)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=kni_collate_fn)
    id_to_class = {v: k for k, v in artifacts.class_to_id.items()}
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            encoded = model.encode(batch)
            logits = model.classifier(encoded["embedding"])
            pred = logits.argmax(dim=-1)
            gates = encoded["gates"].detach().cpu().numpy()
            for i in range(pred.numel()):
                rec = records[offset + i]
                y = int(batch["y"][i].cpu())
                p = int(pred[i].cpu())
                rows.append(
                    {
                        "setting": name,
                        "intent_id": rec["intent_id"],
                        "true_class": id_to_class[y],
                        "pred_class": id_to_class[p],
                        "correct": y == p,
                        "text_gate": float(gates[i, 0]),
                        "graph_gate": float(gates[i, 1]),
                        "qos_gate": float(gates[i, 2]),
                        "slice_type": rec.get("slice_type", ""),
                        "positive_group_id": rec.get("positive_group_id", ""),
                    }
                )
            offset += pred.numel()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def aggregate(rows: list[dict], by: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[by]].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        out.append(
            {
                by: key,
                "n": len(vals),
                "text_gate": float(np.mean([v["text_gate"] for v in vals])),
                "graph_gate": float(np.mean([v["graph_gate"] for v in vals])),
                "qos_gate": float(np.mean([v["qos_gate"] for v in vals])),
                "accuracy": float(np.mean([1.0 if v["correct"] else 0.0 for v in vals])),
            }
        )
    return out


def write_agg_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_gate_by_class(rows: list[dict], out: Path, title: str) -> None:
    agg = aggregate(rows, "true_class")
    labels = [short_label(r["true_class"]) for r in agg]
    text = [r["text_gate"] for r in agg]
    graph = [r["graph_gate"] for r in agg]
    qos = [r["qos_gate"] for r in agg]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, text, label="Text")
    ax.bar(x, graph, bottom=text, label="Graph")
    ax.bar(x, qos, bottom=np.array(text) + np.array(graph), label="QoS")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def relation_counter(jsonl: Path, ids: set[str] | None = None) -> Counter:
    c = Counter()
    for rec in read_jsonl(jsonl):
        if ids is not None and rec["intent_id"] not in ids:
            continue
        for _, rel, _ in rec.get("kg_edges", []):
            c[rel] += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "step33_explainability"))
    parser.add_argument(
        "--in-protocol-checkpoint",
        default=str(ROOT / "outputs" / "step30_neutral_kg_ablation" / "seed_1" / "full_group" / "best.pt"),
    )
    args = parser.parse_args()
    out = Path(args.output_dir)

    settings = [
        (
            "in_protocol_neutral_kg",
            Path(args.in_protocol_checkpoint),
            ROOT / "data" / "processed" / "challenge_neutral_kg" / "test.jsonl",
        ),
        (
            "schema_shift_challenge_to_neutral_kg",
            ROOT / "outputs" / "cross_dataset_transfer" / "seed_1" / "challenge_to_neutral_kg" / "kcgt_contrastive" / "best.pt",
            ROOT / "data" / "processed" / "challenge_neutral_kg" / "test.jsonl",
        ),
        (
            "schema_shift_neutral_kg_to_challenge",
            ROOT / "outputs" / "cross_dataset_transfer" / "seed_1" / "neutral_kg_to_challenge" / "kcgt_contrastive" / "best.pt",
            ROOT / "data" / "processed" / "challenge" / "test.jsonl",
        ),
    ]
    all_rows = []
    for name, ckpt, jsonl in settings:
        rows = extract_gates(name, ckpt, jsonl, out / f"{name}_gates.csv")
        all_rows.extend(rows)
        write_agg_csv(out / f"{name}_gate_by_class.csv", aggregate(rows, "true_class"))
        plot_gate_by_class(rows, out / f"{name}_gate_by_class.png", f"Gate Weights: {name}")

    setting_summary = aggregate(all_rows, "setting")
    write_agg_csv(out / "gate_by_setting.csv", setting_summary)

    # Relation evidence for hard-negative failures from Step 32: compare all test KG relations with
    # relations among the top reported failed queries.
    failure_report = ROOT / "outputs" / "reports" / "step32_error_case_analysis_report.md"
    failure_ids: set[str] = set()
    if failure_report.exists():
        for line in failure_report.read_text(encoding="utf-8").splitlines():
            if line.startswith("| CHAL_"):
                failure_ids.add(line.split("|")[1].strip())
    all_rel = relation_counter(ROOT / "data" / "processed" / "challenge_neutral_kg" / "test.jsonl")
    fail_rel = relation_counter(ROOT / "data" / "processed" / "challenge_neutral_kg" / "test.jsonl", failure_ids)

    lines = [
        "# Step 33 Explainability Analysis",
        "",
        "This report summarizes KCGT gated fusion behavior and KG relation evidence under in-protocol and schema-shift settings.",
        "",
        "## Gate Summary by Setting",
        "",
        "| Setting | n | Text gate | Graph gate | QoS gate | Accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in setting_summary:
        lines.append(f"| {r['setting']} | {r['n']} | {r['text_gate']:.4f} | {r['graph_gate']:.4f} | {r['qos_gate']:.4f} | {r['accuracy']:.4f} |")
    lines += [
        "",
        "## KG Relation Evidence in NeutralKG Hard-Negative Cases",
        "",
        "| Relation | All test count | Failure-query count |",
        "| --- | --- | --- |",
    ]
    for rel, count in all_rel.most_common():
        lines.append(f"| {rel} | {count} | {fail_rel.get(rel, 0)} |")
    lines += [
        "",
        "## Generated Files",
        "",
        f"- Gate summary CSV: `{out / 'gate_by_setting.csv'}`",
        f"- In-protocol gate figure: `{out / 'in_protocol_neutral_kg_gate_by_class.png'}`",
        f"- Challenge to NeutralKG schema-shift gate figure: `{out / 'schema_shift_challenge_to_neutral_kg_gate_by_class.png'}`",
        f"- NeutralKG to Challenge schema-shift gate figure: `{out / 'schema_shift_neutral_kg_to_challenge_gate_by_class.png'}`",
        "",
        "## Paper-Ready Interpretation",
        "",
        "- The gated fusion module provides a direct model-internal signal for how KCGT balances text, KG structure, and QoS features.",
        "- Schema-shift settings should be discussed as changes in modality reliance rather than ordinary accuracy degradation. If graph gates remain high while accuracy drops, the result suggests that unseen ontology/schema semantics can mislead graph-dependent encoders.",
        "- The relation-count analysis is not causal attention, but it identifies which KG relation types dominate the difficult hard-negative cases and motivates later relation-level robustness tests.",
    ]
    report = ROOT / "outputs" / "reports" / "step33_explainability_analysis_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
