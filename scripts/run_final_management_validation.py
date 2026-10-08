from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_step36_management_validation import evaluate_variant, load_embeddings, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate management proxy tasks with final KCGT checkpoints.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "challenge_neutral_kg"))
    parser.add_argument("--retrieval-root", default=str(ROOT / "outputs" / "method_novelty_v1"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "final_management_validation_relgt_oscl"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(processed / "test.jsonl")
    sample_ids = {x["intent_id"] for x in samples}
    per_seed = []
    for seed in args.seeds:
        path = Path(args.retrieval_root) / f"seed_{seed}" / "full_structured_relgt" / "test_group" / "query_embeddings.csv"
        ids, emb = load_embeddings(path)
        keep = [i for i, intent_id in enumerate(ids) if intent_id in sample_ids]
        metrics = evaluate_variant(samples, [ids[i] for i in keep], emb[keep])
        per_seed.append({"seed": seed, **metrics})

    metric_map = {
        "template_at_1": "service_template_success@1",
        "profile_at_1": "service_profile_success@1",
        "fulfill_at_5": "sla_aware_fulfillment@5",
        "conflict_f1": "conflict_detection_f1",
        "conflict_auroc": "conflict_auroc",
    }
    summary = []
    for short, full in metric_map.items():
        vals = np.asarray([row[full] for row in per_seed], dtype=float)
        summary.append({"metric": short, "mean": float(vals.mean()), "std": float(vals.std(ddof=0))})

    with (output / "management_validation_per_seed.json").open("w", encoding="utf-8") as f:
        json.dump(per_seed, f, indent=2)
    with (output / "management_validation_final.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Final KCGT Management Validation",
        "",
        "All values are recomputed from the final relation-aware Graph Transformer + OSCL embeddings over five seeds.",
        "",
        "| Metric | Mean | Std |",
        "| --- | ---: | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} |")
    lines += [
        "",
        "These are management-oriented proxy evaluations for service-template reuse, SLA-aware candidate selection, and conflict screening; they do not replace live orchestration or SLA assurance validation.",
    ]
    (output / "management_validation_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "seeds": args.seeds, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
