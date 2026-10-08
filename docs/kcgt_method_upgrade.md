# KCGT Method Upgrade: Relation-Aware Graph Transformer and OSCL

## Motivation

The original implementation used an R-GCN as the main graph encoder and a standard supervised contrastive loss. That configuration supported the text--graph--QoS representation study, but it left two avoidable reviewer concerns: the name "Graph Transformer" was not implemented literally, and the optimization objective could be interpreted as a composition of standard modules.

## Final Graph Encoder

The final KCGT graph branch uses two relation-aware Graph Transformer blocks. Each block applies multi-head neighbor attention through `TransformerConv`. A learned relation embedding is supplied as `edge_attr`, so the relation type modifies attention keys and values rather than serving only as a post-hoc label. Each block also contains root/message beta gating, a residual path, layer normalization, and a feed-forward network. GCN, GAT, and R-GCN remain available only as controlled ablations and compatibility backbones.

Implementation: `src/models/kcgt.py`.

## Operationally Structured Contrastive Learning

OSCL augments supervised contrastive learning with network-operational structure:

- KG similarity combines node-set and typed-triple Jaccard similarity.
- QoS similarity uses normalized L1 proximity.
- Slice compatibility contributes a discrete operational signal.
- Positive pairs receive compatibility-dependent weights.
- Declared hard negatives and high-similarity in-batch negatives receive larger denominator weights.
- An adaptive ranking margin increases with operational similarity, forcing plausible but incompatible requests apart.

Implementation: `src/experiments/train_kcgt_contrastive.py`.

## Controlled Evidence

Five identical seeds were evaluated on Challenge-NeutralKG.

| Variant | Macro-F1 | Group R@1 | Group MRR | Hard P@5 | Hard NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Final KCGT | 0.9786 +/- 0.0066 | 0.5685 +/- 0.0237 | 0.6890 +/- 0.0195 | 0.4946 +/- 0.0254 | 0.5867 +/- 0.0209 |
| Relation-aware GT + SupCon | 0.9780 +/- 0.0066 | 0.5472 +/- 0.0321 | 0.6733 +/- 0.0272 | 0.4851 +/- 0.0243 | 0.5784 +/- 0.0203 |
| R-GCN + OSCL | 0.9822 +/- 0.0036 | 0.5219 +/- 0.0777 | 0.6522 +/- 0.0654 | 0.4426 +/- 0.0497 | 0.5465 +/- 0.0410 |
| Final KCGT, shuffled relations | 0.9792 +/- 0.0051 | 0.5509 +/- 0.0304 | 0.6774 +/- 0.0246 | 0.4898 +/- 0.0216 | 0.5819 +/- 0.0185 |

The relation-aware Transformer improves hard-negative P@5 over R-GCN by 0.0520 (`p=0.0273`) and NDCG@5 by 0.0402 (`p=0.0335`). Typed relations improve group Recall@1 over shuffled relations by 0.0176 (`p=0.0113`; paired bootstrap 95% CI `[0.0117, 0.0248]`). OSCL improves mean group Recall@1 over standard SupCon by 0.0213, but its five-seed paired tests are not significant at 0.05; the manuscript therefore presents this as a consistent mean/effect-size improvement rather than a universal significance claim.

## Reproduction

```powershell
python scripts/run_method_novelty_suite.py --python python --processed-dir data/processed/challenge_neutral_kg --output-root outputs/method_novelty_v1 --seeds 1 2 3 4 5 --epochs 6 --batch-size 128 --skip-existing
```

Primary outputs:

- `outputs/method_novelty_v1/method_novelty_summary.csv`
- `outputs/method_novelty_v1/method_novelty_report.md`
- `outputs/method_novelty_v1/method_novelty_significance.csv`
- `outputs/method_novelty_v1/method_novelty_significance.md`
