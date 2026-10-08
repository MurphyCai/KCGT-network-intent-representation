from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_from_json(data: dict, metric: str) -> float:
    if "test" in data and metric in data["test"]:
        return float(data["test"][metric])
    if metric in data:
        return float(data[metric])
    raise KeyError(metric)


def read_classification(root: Path) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {}
    kcgt_root = root / "outputs" / "step30_neutral_kg_ablation"
    baseline_root = root / "outputs" / "step30_neutral_kg_baselines"
    for seed in (1, 2, 3):
        p = kcgt_root / f"seed_{seed}" / "full_group" / "metrics.json"
        if p.exists():
            values.setdefault("KCGT", {})[seed] = metric_from_json(load_json(p), "f1_macro")
        for model in ("text_gat", "text_rgcn", "text_graph_transformer", "graph_transformer"):
            # Text+R-GCN is named text_rgcn in older suites and absent in Step 30;
            # fallback to text_gcn/text_gat-like naming is handled by existence checks.
            p = baseline_root / f"seed_{seed}" / "baselines" / model / "metrics.json"
            if p.exists():
                values.setdefault(model, {})[seed] = metric_from_json(load_json(p), "f1_macro")
        for model in ("text_gcn", "rgcn"):
            p = baseline_root / f"seed_{seed}" / "baselines" / model / "metrics.json"
            if p.exists():
                values.setdefault(model, {})[seed] = metric_from_json(load_json(p), "f1_macro")
    for model in ("sbert_minilm", "bert_mini"):
        for seed in (1, 2, 3):
            p = baseline_root / f"seed_{seed}" / "pretrained" / model / "metrics.json"
            if p.exists():
                values.setdefault(model, {})[seed] = metric_from_json(load_json(p), "f1_macro")
    return values


def read_retrieval(root: Path, ablation: str, scenario: str, metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    base = root / "outputs" / "step30_neutral_kg_retrieval"
    for seed in (1, 2, 3):
        p = base / f"seed_{seed}" / ablation / scenario / "retrieval_metrics.json"
        if p.exists():
            data = load_json(p)
            metrics = data.get("metrics", data)
            out[seed] = float(metrics[metric])
    return out


def paired_arrays(a: dict[int, float], b: dict[int, float]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] for s in seeds], dtype=float), np.array([b[s] for s in seeds], dtype=float), seeds


def bootstrap_ci(diff: np.ndarray, n_boot: int = 20000, seed: int = 31) -> tuple[float, float]:
    if len(diff) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_test_rows(task: str, metric: str, primary: dict[int, float], candidates: dict[str, dict[int, float]]) -> list[dict]:
    rows = []
    for name, vals in candidates.items():
        a, b, seeds = paired_arrays(primary, vals)
        if len(seeds) == 0:
            continue
        diff = a - b
        row = {
            "task": task,
            "metric": metric,
            "comparison": f"KCGT - {name}",
            "seeds": " ".join(map(str, seeds)),
            "n": len(seeds),
            "kcgt_mean": float(a.mean()),
            "baseline_mean": float(b.mean()),
            "mean_diff": float(diff.mean()),
            "diff_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "bootstrap_ci95_low": bootstrap_ci(diff)[0],
            "bootstrap_ci95_high": bootstrap_ci(diff)[1],
            "paired_t_p": math.nan,
            "wilcoxon_p": math.nan,
            "note": "",
        }
        if stats is not None and len(diff) >= 2 and float(diff.std(ddof=1)) > 0:
            row["paired_t_p"] = float(stats.ttest_rel(a, b).pvalue)
        if stats is not None and len(diff) >= 2 and np.any(np.abs(diff) > 0):
            try:
                row["wilcoxon_p"] = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
            except ValueError:
                row["wilcoxon_p"] = math.nan
        if len(seeds) < 3:
            row["note"] = "insufficient seeds for formal multi-seed test; descriptive only"
        rows.append(row)
    return rows


def fmt(x: float) -> str:
    if x != x:
        return "NA"
    return f"{x:.6f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "metric",
        "comparison",
        "seeds",
        "n",
        "kcgt_mean",
        "baseline_mean",
        "mean_diff",
        "diff_std",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "paired_t_p",
        "wilcoxon_p",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict]) -> None:
    sig = [r for r in rows if r["n"] >= 3 and (r["paired_t_p"] == r["paired_t_p"]) and r["paired_t_p"] < 0.05]
    lines = [
        "# Step 31 Statistical Significance Tests",
        "",
        "This report evaluates whether KCGT improvements are consistent across seeds. Tests are paired by seed whenever both models have the same seed result.",
        "",
        "Important caveat: with only three random seeds, p-values should be interpreted as supporting evidence rather than standalone proof. Bootstrap confidence intervals over seed-level differences are reported to show effect direction and magnitude. SBERT/BERT currently have only one seed, so they are descriptive comparisons only.",
        "",
        f"- Total comparisons: {len(rows)}",
        f"- Paired comparisons with p < 0.05: {len(sig)}",
        "",
        "| task | metric | comparison | n | KCGT mean | baseline mean | mean diff | 95% bootstrap CI | paired t p | Wilcoxon p | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        ci = f"[{fmt(r['bootstrap_ci95_low'])}, {fmt(r['bootstrap_ci95_high'])}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    r["task"],
                    r["metric"],
                    r["comparison"],
                    str(r["n"]),
                    fmt(r["kcgt_mean"]),
                    fmt(r["baseline_mean"]),
                    fmt(r["mean_diff"]),
                    ci,
                    fmt(r["paired_t_p"]),
                    fmt(r["wilcoxon_p"]),
                    r["note"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Paper-Ready Interpretation",
        "",
        "- On NeutralKG classification, KCGT should be reported with effect sizes and confidence intervals against the strongest graph-text baselines. Because the number of seeds is small, use cautious wording: `KCGT consistently improves Macro-F1 across seeds` rather than overclaiming strict statistical dominance.",
        "- For representation learning, the largest and most stable effect appears in retrieval settings comparing full KCGT with `no_contrastive`, especially group-level Recall@1/MRR and hard-negative Precision@5. This supports the claim that contrastive learning mainly improves embedding-space organization, even when classification scores are already saturated.",
        "- SBERT is useful as a semantic-text reference point, but it needs additional seeds if we want to use formal significance tests in the final manuscript.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "step31_significance")
    args = parser.parse_args()

    class_values = read_classification(args.root)
    primary = class_values.pop("KCGT")
    class_candidates = {
        "Text+GAT": class_values.get("text_gat", {}),
        "Text+GraphTransformer": class_values.get("text_graph_transformer", {}),
        "GraphTransformer": class_values.get("graph_transformer", {}),
        "R-GCN": class_values.get("rgcn", {}),
        "SBERT-MiniLM": class_values.get("sbert_minilm", {}),
    }
    rows = paired_test_rows("NeutralKG classification", "Macro-F1", primary, class_candidates)

    retrieval_specs = [
        ("NeutralKG group retrieval", "test_group_projection", "recall@1"),
        ("NeutralKG group retrieval", "test_group_projection", "mrr"),
        ("NeutralKG hard-negative retrieval", "pairwise_hard_negative_projection", "precision@5"),
        ("NeutralKG hard-negative retrieval", "pairwise_hard_negative_projection", "ndcg@5"),
    ]
    for task, scenario, metric in retrieval_specs:
        primary_r = read_retrieval(args.root, "full_group", scenario, metric)
        cands = {
            "NoContrastive": read_retrieval(args.root, "no_contrastive", scenario, metric),
            "NoGraph": read_retrieval(args.root, "no_graph", scenario, metric),
            "GraphGCN": read_retrieval(args.root, "graph_gcn", scenario, metric),
            "GraphGAT": read_retrieval(args.root, "graph_gat", scenario, metric),
        }
        rows.extend(paired_test_rows(task, metric, primary_r, cands))

    out_dir = args.output_dir
    write_csv(out_dir / "step31_significance_tests.csv", rows)
    write_markdown(args.root / "outputs" / "reports" / "step31_statistical_significance_report.md", rows)
    print(f"Wrote {out_dir / 'step31_significance_tests.csv'}")
    print(f"Wrote {args.root / 'outputs' / 'reports' / 'step31_statistical_significance_report.md'}")


if __name__ == "__main__":
    main()
