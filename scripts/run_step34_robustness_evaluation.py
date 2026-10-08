from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

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

SYNONYMS = {
    "low": ["bounded", "controlled", "reduced"],
    "latency": ["delay", "response time"],
    "reliable": ["dependable", "stable"],
    "reliability": ["continuity", "delivery assurance"],
    "bandwidth": ["throughput", "data rate"],
    "secure": ["protected", "security-aware"],
    "mobility": ["movement support", "handover support"],
    "service": ["application flow", "network service"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clone(x: Any) -> Any:
    return json.loads(json.dumps(x))


def text_paraphrase(records: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for rec in records:
        item = clone(rec)
        text = str(item["intent_text"])
        for src, cands in SYNONYMS.items():
            if src in text.lower() and rng.random() < 0.75:
                text = text.replace(src, rng.choice(cands))
                text = text.replace(src.capitalize(), rng.choice(cands).capitalize())
        item["intent_text"] = text
        item["robustness_mode"] = "text_paraphrase"
        out.append(item)
    return out


def text_dropout(records: list[dict], rng: random.Random, drop_prob: float) -> list[dict]:
    out = []
    for rec in records:
        item = clone(rec)
        words = str(item["intent_text"]).split()
        if len(words) > 5:
            kept = [w for w in words if rng.random() > drop_prob]
            if len(kept) < 5:
                kept = words[:5]
            item["intent_text"] = " ".join(kept)
        item["robustness_mode"] = "text_dropout"
        out.append(item)
    return out


def qos_noise(records: list[dict], rng: random.Random, noise: float) -> list[dict]:
    out = []
    for rec in records:
        item = clone(rec)
        for f in QOS_FIELDS:
            if f in item:
                val = float(item[f])
                item[f] = max(0.0, val * (1.0 + rng.uniform(-noise, noise)))
        item["robustness_mode"] = "qos_noise"
        out.append(item)
    return out


def edge_dropout(records: list[dict], rng: random.Random, drop_prob: float) -> list[dict]:
    out = []
    for rec in records:
        item = clone(rec)
        edges = list(item.get("kg_edges", []))
        if len(edges) > 2:
            kept = [e for e in edges if rng.random() > drop_prob]
            if len(kept) < 1:
                kept = edges[:1]
            item["kg_edges"] = kept
        item["robustness_mode"] = "kg_edge_dropout"
        out.append(item)
    return out


def node_mask(records: list[dict], rng: random.Random, mask_prob: float) -> list[dict]:
    out = []
    for rec in records:
        item = clone(rec)
        nodes = list(item.get("kg_nodes", []))
        protected = [n for n in nodes if str(n).startswith("application:")]
        candidates = [n for n in nodes if n not in protected]
        kept = protected + [n for n in candidates if rng.random() > mask_prob]
        if not kept:
            kept = nodes[:1]
        kept_set = set(kept)
        item["kg_nodes"] = kept
        item["kg_edges"] = [e for e in item.get("kg_edges", []) if e[0] in kept_set and e[2] in kept_set]
        item["robustness_mode"] = "kg_node_mask"
        out.append(item)
    return out


def relation_mask(records: list[dict], relation: str) -> list[dict]:
    out = []
    for rec in records:
        item = clone(rec)
        item["kg_edges"] = [e for e in item.get("kg_edges", []) if e[1] != relation]
        item["robustness_mode"] = f"relation_mask_{relation}"
        out.append(item)
    return out


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_metric(path: Path, metric: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    node = data.get("test", data.get("metrics", data))
    return float(node[metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "challenge_neutral_kg"))
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "outputs" / "step30_neutral_kg_ablation" / "seed_1" / "full_group" / "best.pt"),
        help="Checkpoint used for the no-retraining perturbation suite.",
    )
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "step34_robustness"))
    parser.add_argument("--seed", type=int, default=34)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    processed = Path(args.processed_dir)
    output_root = Path(args.output_root)
    perturb_dir = output_root / "perturbed_jsonl"
    records = read_jsonl(processed / "test.jsonl")
    variants = {
        "clean": records,
        "text_paraphrase": text_paraphrase(records, rng),
        "text_dropout": text_dropout(records, rng, 0.18),
        "qos_noise_15pct": qos_noise(records, rng, 0.15),
        "kg_edge_dropout_30pct": edge_dropout(records, rng, 0.30),
        "kg_node_mask_30pct": node_mask(records, rng, 0.30),
        "relation_mask_has_constraint": relation_mask(records, "has_constraint"),
        "relation_mask_uses_capability": relation_mask(records, "uses_capability"),
        "relation_mask_uses_resource_family": relation_mask(records, "uses_resource_family"),
    }
    for name, rows in variants.items():
        write_jsonl(perturb_dir / f"{name}.jsonl", rows)

    rows = []
    for name in variants:
        cls_out = output_root / "classification" / name
        ret_out = output_root / "retrieval" / name
        run(
            [
                args.python,
                "scripts/evaluate_checkpoint_classification.py",
                "--checkpoint",
                args.checkpoint,
                "--jsonl",
                str(perturb_dir / f"{name}.jsonl"),
                "--output-dir",
                str(cls_out),
            ]
        )
        run(
            [
                args.python,
                "scripts/evaluate_kcgt_retrieval.py",
                "--checkpoint",
                args.checkpoint,
                "--query",
                str(perturb_dir / f"{name}.jsonl"),
                "--gallery",
                str(processed / "test.jsonl"),
                "--output-dir",
                str(ret_out),
                "--relevance",
                "group",
            ]
        )
        cls = json.loads((cls_out / "metrics.json").read_text(encoding="utf-8"))["test"]
        ret = json.loads((ret_out / "retrieval_metrics.json").read_text(encoding="utf-8"))["metrics"]
        rows.append(
            {
                "mode": name,
                "accuracy": cls["accuracy"],
                "f1_macro": cls["f1_macro"],
                "inference_time_ms": cls["inference_time_ms_per_sample"],
                "recall@1": ret["recall@1"],
                "mrr": ret["mrr"],
                "ndcg@5": ret["ndcg@5"],
            }
        )

    clean = next(r for r in rows if r["mode"] == "clean")
    for r in rows:
        r["delta_f1"] = r["f1_macro"] - clean["f1_macro"]
        r["delta_recall@1"] = r["recall@1"] - clean["recall@1"]
        r["delta_mrr"] = r["mrr"] - clean["mrr"]

    csv_path = output_root / "step34_robustness_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Step 34 Robustness Evaluation",
        "",
        "KCGT is evaluated under input perturbations without retraining. Classification uses Macro-F1; retrieval uses group-level Recall@1/MRR against the clean NeutralKG test gallery.",
        "",
        "| Mode | Macro-F1 | Delta F1 | Recall@1 | Delta R@1 | MRR | Delta MRR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['f1_macro']:.4f} | {r['delta_f1']:.4f} | {r['recall@1']:.4f} | {r['delta_recall@1']:.4f} | {r['mrr']:.4f} | {r['delta_mrr']:.4f} |")
    lines += [
        "",
        "## Paper-Ready Interpretation",
        "",
        "- Text and QoS perturbations test whether the model is overfitted to surface templates or exact numeric requirements.",
        "- KG edge/node/relation masking tests whether graph evidence is useful but brittle under missing or coarse ontology information.",
        "- Large drops under relation masking should be framed as evidence that the KG schema is semantically active in KCGT, not just decorative metadata.",
    ]
    report = ROOT / "outputs" / "reports" / "step34_robustness_evaluation_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(csv_path), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
