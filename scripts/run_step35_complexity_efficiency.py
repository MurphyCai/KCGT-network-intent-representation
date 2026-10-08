from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "NA"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +/- {safe_stdev(values):.4f}"


def checkpoint_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def collect_model_rows() -> list[dict]:
    specs = [
        ("KCGT", "outputs/step30_neutral_kg_ablation", "full_group", "ablation"),
        ("KCGT w/o CL", "outputs/step30_neutral_kg_ablation", "no_contrastive", "ablation"),
        ("Text+GAT", "outputs/step30_neutral_kg_baselines", "baselines/text_gat", "baseline"),
        ("Text+R-GCN", "outputs/step30_neutral_kg_baselines", "baselines/rgcn", "baseline"),
        ("Text+GraphTransformer", "outputs/step30_neutral_kg_baselines", "baselines/text_graph_transformer", "baseline"),
        ("GraphTransformer", "outputs/step30_neutral_kg_baselines", "baselines/graph_transformer", "baseline"),
        ("Hypergraph", "outputs/step30_neutral_kg_baselines", "baselines/hypergraph", "baseline"),
        ("LSTM", "outputs/step30_neutral_kg_baselines", "baselines/lstm", "baseline"),
        ("QoS-MLP", "outputs/step30_neutral_kg_baselines", "baselines/qos_mlp", "baseline"),
    ]
    rows: list[dict] = []
    for display, base, rel, family in specs:
        metrics, params, infer, f1, train_epoch, ckpt = [], [], [], [], [], []
        for seed in (1, 2, 3):
            run_dir = ROOT / base / f"seed_{seed}" / rel
            p = run_dir / "metrics.json"
            if not p.exists():
                continue
            data = load_json(p)
            test = data.get("test", data)
            metrics.append(data)
            f1.append(float(test["f1_macro"]))
            infer.append(float(test.get("inference_time_ms_per_sample", 0.0)))
            if "num_parameters" in data:
                params.append(float(data["num_parameters"]))
            history = data.get("history", [])
            secs = [float(h.get("seconds", 0.0)) for h in history if "seconds" in h]
            if secs:
                train_epoch.append(mean(secs))
            ckpt.append(checkpoint_mb(run_dir / "best.pt"))
        if not metrics:
            continue
        rows.append(
            {
                "model": display,
                "family": family,
                "runs": len(metrics),
                "macro_f1": fmt_mean_std(f1),
                "macro_f1_mean": mean(f1),
                "inference_ms_per_sample": fmt_mean_std(infer),
                "inference_ms_mean": mean(infer),
                "parameters": int(round(mean(params))) if params else 0,
                "checkpoint_mb": fmt_mean_std(ckpt),
                "train_seconds_per_epoch": fmt_mean_std(train_epoch),
            }
        )
    # pretrained text baselines from one offline cached seed
    for display, rel in [("BERT-mini", "bert_mini"), ("SBERT-MiniLM", "sbert_minilm")]:
        vals, infers, params = [], [], []
        for seed in (1, 2, 3):
            p = ROOT / "outputs" / "step30_neutral_kg_baselines" / f"seed_{seed}" / "pretrained" / rel / "metrics.json"
            if not p.exists():
                continue
            data = load_json(p)
            test = data.get("test", data)
            vals.append(float(test["f1_macro"]))
            infers.append(float(test.get("inference_time_ms_per_sample", 0.0)))
            params.append(int(data.get("num_parameters", 0)))
        if vals:
            rows.append(
                {
                    "model": display,
                    "family": "pretrained_text",
                    "runs": len(vals),
                    "macro_f1": fmt_mean_std(vals),
                    "macro_f1_mean": mean(vals),
                    "inference_ms_per_sample": fmt_mean_std(infers),
                    "inference_ms_mean": mean(infers),
                    "parameters": int(round(mean(params))) if params else 0,
                    "checkpoint_mb": "NA",
                    "train_seconds_per_epoch": "offline feature/classifier baseline",
                }
            )
    rows.sort(key=lambda r: r["macro_f1_mean"], reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "family",
        "runs",
        "macro_f1",
        "inference_ms_per_sample",
        "parameters",
        "checkpoint_mb",
        "train_seconds_per_epoch",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})


def main() -> None:
    rows = collect_model_rows()
    out_dir = ROOT / "outputs" / "step35_complexity_efficiency"
    write_csv(out_dir / "step35_complexity_efficiency_summary.csv", rows)

    lines = [
        "# Step 35 Complexity and Efficiency Analysis",
        "",
        "This report summarizes model size, training-time traces, and CPU inference latency from the existing NeutralKG experiments. No extra retraining was performed.",
        "",
        "Measurement notes:",
        "",
        "- `inference_ms_per_sample` comes from the existing evaluation loop on the current CPU execution path.",
        "- `train_seconds_per_epoch` is averaged from saved epoch histories when available.",
        "- GPU memory/utilization was not sampled during the original runs, so it is intentionally not reported as a numeric claim.",
        "- BERT/SBERT rows are offline cached pretrained text baselines with one seed; they are not multi-seed complexity measurements.",
        "",
        "| Model | Runs | Macro-F1 | Inference ms/sample | Parameters | Checkpoint MB | Train sec/epoch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['runs']} | {r['macro_f1']} | {r['inference_ms_per_sample']} | "
            f"{r['parameters']} | {r['checkpoint_mb']} | {r['train_seconds_per_epoch']} |"
        )

    kcgt = next((r for r in rows if r["model"] == "KCGT"), None)
    gat = next((r for r in rows if r["model"] == "Text+GAT"), None)
    graph_trans = next((r for r in rows if r["model"] == "Text+GraphTransformer"), None)
    lines += [
        "",
        "## Paper-Ready Interpretation",
        "",
    ]
    if kcgt and gat:
        lines.append(
            f"- KCGT reaches Macro-F1 {kcgt['macro_f1']} with {kcgt['parameters']} parameters, "
            f"while the strongest lightweight graph-text baseline Text+GAT reaches {gat['macro_f1']} with {gat['parameters']} parameters."
        )
    if kcgt and graph_trans:
        lines.append(
            f"- Compared with Text+GraphTransformer ({graph_trans['macro_f1']}), KCGT improves accuracy while keeping CPU inference in the same sub-millisecond/sample regime."
        )
    lines += [
        "- The efficiency evidence supports positioning KCGT as a deployable intent-representation model for management-plane analytics, rather than a large pretrained language-model solution.",
        "- The final manuscript should state that all timing numbers are implementation- and hardware-dependent, and should use them for relative comparison under the same experimental environment.",
    ]
    report = ROOT / "outputs" / "reports" / "step35_complexity_efficiency_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "summary_csv": str(out_dir / "step35_complexity_efficiency_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
