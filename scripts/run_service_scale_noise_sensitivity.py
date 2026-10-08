from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import t

from run_service_template_scale_evaluation import evaluate_scale, load_embeddings, read_jsonl


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["template_success@1", "template_success@5", "fulfillment@5", "sla_violation@5"]
    for (noise_std, pool_size), group in frame.groupby(["noise_std", "pool_size"], sort=True):
        row = {"noise_std": noise_std, "pool_size": pool_size, "runs": len(group)}
        critical = float(t.ppf(0.975, len(group) - 1))
        for metric in metrics:
            mean = group[metric].mean()
            sd = group[metric].std(ddof=1)
            half = critical * sd / len(group) ** 0.5
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = max(0.0, mean - half)
            row[f"{metric}_ci95_high"] = min(1.0, mean + half)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    largest = summary[summary["pool_size"] == summary["pool_size"].max()].sort_values("noise_std")
    lines = [
        "# Template Perturbation Sensitivity",
        "",
        "KCGT is evaluated without retraining under multiple Gaussian perturbation standard deviations. Candidate source IDs, query selection, and random seeds remain fixed for a given seed and pool size; only perturbation magnitude changes. Confidence limits for bounded metrics are clipped to [0, 1].",
        "",
        "| Noise std | Template@1 (95% CI) | Template@5 (95% CI) | Fulfill@5 | SLA violation@5 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in largest.iterrows():
        lines.append(
            f"| {row['noise_std']:.3f} | {row['template_success@1_mean']:.4f} "
            f"[{row['template_success@1_ci95_low']:.4f}, {row['template_success@1_ci95_high']:.4f}] | "
            f"{row['template_success@5_mean']:.4f} "
            f"[{row['template_success@5_ci95_low']:.4f}, {row['template_success@5_ci95_high']:.4f}] | "
            f"{row['fulfillment@5_mean']:.4f} | {row['sla_violation@5_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The zero-noise condition contains exact repeated source vectors when the repository exceeds the 1,080 held-out source intents. Nonzero perturbation creates distinct vector variants but does not create independently observed service templates; conclusions therefore concern sensitivity of the controlled variant-repository proxy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/challenge_neutral_kg")
    parser.add_argument("--retrieval-root", default="outputs/step30_neutral_kg_retrieval")
    parser.add_argument("--output-dir", default="outputs/service_template_scale_noise_sensitivity")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.005, 0.015, 0.03, 0.05])
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--embedding-split", default="test_group_projection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(Path(args.processed_dir) / "test.jsonl")
    sample_ids = {sample["intent_id"] for sample in samples}
    rows = []
    for seed in args.seeds:
        path = Path(args.retrieval_root) / f"seed_{seed}" / "full_group" / args.embedding_split / "query_embeddings.csv"
        ids, embeddings = load_embeddings(path)
        keep = [i for i, intent_id in enumerate(ids) if intent_id in sample_ids]
        ids = [ids[i] for i in keep]
        embeddings = embeddings[keep]
        for noise_std in args.noise_stds:
            for pool_size in args.pool_sizes:
                print(f"[noise-sensitivity] seed={seed} noise={noise_std:.3f} pool={pool_size}", flush=True)
                metrics = evaluate_scale(
                    samples=samples,
                    ids=ids,
                    emb=embeddings,
                    pool_size=pool_size,
                    seed=seed,
                    repeats=1,
                    noise_std=noise_std,
                    max_queries=args.max_queries,
                )
                rows.append({"seed": seed, "noise_std": noise_std, "pool_size": pool_size, **metrics})
    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(output_dir / "noise_sensitivity_per_seed.csv", index=False)
    summary.to_csv(output_dir / "noise_sensitivity_summary.csv", index=False)
    write_report(summary, output_dir / "noise_sensitivity_report.md")
    reports = Path("outputs/reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "service_template_noise_sensitivity.md").write_text(
        (output_dir / "noise_sensitivity_report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print((output_dir / "noise_sensitivity_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
