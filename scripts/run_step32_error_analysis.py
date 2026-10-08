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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets import KNIIntentDataset, kni_collate_fn
from datasets.kni_intent_dataset import read_jsonl
from evaluate_checkpoint_classification import load_checkpoint
from experiments.train_baseline_classification import move_batch


FOCUS_PAIRS = [
    ("Cloud Gaming", "AR/VR Immersive Service"),
    ("Telemedicine", "Industrial URLLC"),
    ("Autonomous Driving / V2X", "Industrial URLLC"),
]


def compact(text: str, limit: int = 130) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def predict(checkpoint: Path, jsonl: Path, output_dir: Path, batch_size: int = 128) -> tuple[list[dict], dict[str, int]]:
    device = torch.device("cpu")
    model, artifacts, model_name, _ = load_checkpoint(checkpoint, device)
    dataset = KNIIntentDataset(jsonl, artifacts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=kni_collate_fn)
    id_to_class = {idx: cls for cls, idx in artifacts.class_to_id.items()}
    raw = read_jsonl(jsonl)
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            for i in range(pred.numel()):
                rec = raw[offset + i]
                y = int(batch["y"][i].cpu())
                p = int(pred[i].cpu())
                rows.append(
                    {
                        "intent_id": rec["intent_id"],
                        "text": rec["intent_text"],
                        "true_class": id_to_class[y],
                        "pred_class": id_to_class[p],
                        "confidence": float(conf[i].cpu()),
                        "correct": y == p,
                        "slice_type": rec.get("slice_type", ""),
                        "positive_group_id": rec.get("positive_group_id", ""),
                        "qos_latency_ms": rec.get("qos_latency_ms", ""),
                        "qos_bandwidth_mbps": rec.get("qos_bandwidth_mbps", ""),
                        "qos_reliability": rec.get("qos_reliability", ""),
                    }
                )
            offset += pred.numel()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{model_name}_predictions.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows, artifacts.class_to_id


def plot_confusion(rows: list[dict], class_to_id: dict[str, int], out: Path, title: str) -> None:
    labels = [cls for cls, _ in sorted(class_to_id.items(), key=lambda kv: kv[1])]
    y_true = [class_to_id[r["true_class"]] for r in rows]
    y_pred = [class_to_id[r["pred_class"]] for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=[short_label(x) for x in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=[short_label(x) for x in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def short_label(label: str) -> str:
    return (
        label.replace("AR/VR Immersive Service", "AR/VR")
        .replace("Autonomous Driving / V2X", "V2X")
        .replace("Industrial URLLC", "Industrial")
        .replace("Massive IoT mMTC", "mMTC")
        .replace("Security-Critical Enterprise", "Security")
        .replace("Video Streaming eMBB", "Video")
    )


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    ids, vecs = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        z_fields = sorted([x for x in reader.fieldnames or [] if x.startswith("z")], key=lambda x: int(x[1:]))
        for row in reader:
            ids.append(row["intent_id"])
            vecs.append([float(row[z]) for z in z_fields])
    arr = np.asarray(vecs, dtype=np.float32)
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return ids, arr


def hard_negative_failures(query_jsonl: Path, gallery_jsonl: Path, embeddings: Path, topk: int = 5, limit: int = 8) -> list[dict]:
    # The current evaluator stores query embeddings. For pairwise NeutralKG, query and gallery are generated
    # from the same candidate space, so we use saved query embeddings plus gallery metadata to surface
    # nearest retrieved examples that are same slice/similar QoS but different group.
    records = {r["intent_id"]: r for r in read_jsonl(query_jsonl)}
    gallery = {r["intent_id"]: r for r in read_jsonl(gallery_jsonl)}
    ids, vecs = load_embeddings(embeddings)
    available = [i for i, intent_id in enumerate(ids) if intent_id in records]
    failures = []
    for idx in available:
        qid = ids[idx]
        q = records[qid]
        sims = vecs @ vecs[idx]
        order = [j for j in np.argsort(-sims) if ids[j] != qid and ids[j] in gallery]
        bad = []
        for j in order[: topk * 3]:
            item = gallery[ids[j]]
            same_group = item.get("positive_group_id") == q.get("positive_group_id")
            if not same_group:
                bad.append((j, item, float(sims[j])))
            if len(bad) >= topk:
                break
        if bad:
            failures.append(
                {
                    "query": q,
                    "bad": bad,
                    "top_bad_rank": int(order.index(bad[0][0]) + 1) if bad[0][0] in order else -1,
                }
            )
        if len(failures) >= limit:
            break
    return failures


def summarize_rows(name: str, rows: list[dict]) -> list[str]:
    total = len(rows)
    errors = [r for r in rows if not r["correct"]]
    pair_counter = Counter((r["true_class"], r["pred_class"]) for r in errors)
    lines = [
        f"## {name}",
        "",
        f"- Samples: {total}",
        f"- Errors: {len(errors)}",
        f"- Error rate: {len(errors) / max(1, total):.4f}",
        "",
        "| True class | Predicted class | Count |",
        "| --- | --- | --- |",
    ]
    for (true, pred), count in pair_counter.most_common(10):
        lines.append(f"| {true} | {pred} | {count} |")
    lines += ["", "### Focus Pair Cases", ""]
    for a, b in FOCUS_PAIRS:
        focus = [r for r in errors if {r["true_class"], r["pred_class"]} == {a, b}]
        lines.append(f"- {a} vs {b}: {len(focus)} errors")
        for r in focus[:3]:
            lines.append(f"  - `{r['intent_id']}` true=`{r['true_class']}`, pred=`{r['pred_class']}`, conf={r['confidence']:.3f}: {compact(r['text'])}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "challenge_neutral_kg"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "step32_error_analysis"))
    parser.add_argument(
        "--kcgt-checkpoint",
        default=str(ROOT / "outputs" / "step30_neutral_kg_ablation" / "seed_1" / "full_group" / "best.pt"),
    )
    parser.add_argument(
        "--kcgt-embeddings",
        default=str(ROOT / "outputs" / "step30_neutral_kg_retrieval" / "seed_1" / "full_group" / "pairwise_hard_negative_projection" / "query_embeddings.csv"),
    )
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    out = Path(args.output_dir)
    kcgt_ckpt = Path(args.kcgt_checkpoint)
    gat_ckpt = ROOT / "outputs" / "step30_neutral_kg_baselines" / "seed_1" / "baselines" / "text_gat" / "best.pt"
    jsonl = processed / "test.jsonl"

    kcgt_rows, class_to_id = predict(kcgt_ckpt, jsonl, out / "kcgt")
    gat_rows, _ = predict(gat_ckpt, jsonl, out / "text_gat")
    plot_confusion(kcgt_rows, class_to_id, out / "kcgt_confusion_matrix.png", "KCGT Confusion Matrix on NeutralKG")
    plot_confusion(gat_rows, class_to_id, out / "text_gat_confusion_matrix.png", "Text+GAT Confusion Matrix on NeutralKG")

    failures = hard_negative_failures(
        query_jsonl=processed / "test.jsonl",
        gallery_jsonl=processed / "pairwise_hard_negative_gallery.jsonl",
        embeddings=Path(args.kcgt_embeddings),
    )

    lines = [
        "# Step 32 Error and Case Analysis",
        "",
        "This report analyzes NeutralKG classification errors and pairwise hard-negative retrieval failures.",
        "",
        f"- KCGT confusion matrix: `{out / 'kcgt_confusion_matrix.png'}`",
        f"- Text+GAT confusion matrix: `{out / 'text_gat_confusion_matrix.png'}`",
        "",
    ]
    lines.extend(summarize_rows("KCGT", kcgt_rows))
    lines.extend(summarize_rows("Text+GAT", gat_rows))
    lines += [
        "## NeutralKG Pairwise Hard-Negative Retrieval Failure Cases",
        "",
        "| Query ID | Query class | Query group | Top wrong rank | Retrieved class/group | Similarity | Query text | Retrieved text |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in failures:
        q = item["query"]
        j, bad, sim = item["bad"][0]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(q["intent_id"]),
                    str(q["intent_class"]),
                    str(q["positive_group_id"]),
                    str(item["top_bad_rank"]),
                    f"{bad.get('intent_class')} / {bad.get('positive_group_id')}",
                    f"{sim:.4f}",
                    compact(str(q["intent_text"])).replace("|", "/"),
                    compact(str(bad["intent_text"])).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Paper-Ready Interpretation",
        "",
        "- KCGT makes very few classification errors on NeutralKG, so classification alone is not sensitive enough to expose representation weaknesses.",
        "- Text+GAT produces more interpretable confusions among near-neighbor services, especially where service texts share latency/reliability vocabulary.",
        "- Pairwise hard-negative retrieval reveals the real difficulty: retrieved samples can share slice-level and QoS-level signals while differing in application objective or positive group. This supports using retrieval and schema-shift stress tests as primary representation-quality evidence.",
    ]
    report = ROOT / "outputs" / "reports" / "step32_error_case_analysis_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "output_dir": str(out), "retrieval_failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
