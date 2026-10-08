# KCGT

Reference implementation for **Knowledge-Guided Contrastive Graph Transformer for Network Intent Representation and QoS-Aware Service Mapping**.

KCGT combines intent text, quantitative QoS attributes, and typed intent knowledge subgraphs. The repository includes the relation-aware graph Transformer, operationally structured contrastive learning (OSCL), controlled baselines, protocol-generation code, retrieval evaluation, transfer and robustness analyses, and management-oriented validation.

## Repository Contents

- `src/`: datasets, model definitions, training, and retrieval utilities.
- `scripts/`: protocol generation, training, evaluation, ablation, transfer, robustness, and management experiments.
- `data/`: compact generation metadata, KG resources, and the external 3GPP 5QI protocol.
- `docs/`: dataset and method specifications.
- `tests/`: integration tests for relation-aware KCGT and OSCL.

Model checkpoints, cached embeddings, generated datasets, temporary files, and historical experiment outputs are intentionally excluded.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For PyTorch and PyTorch Geometric, use the wheel index appropriate for your operating system and CUDA version when the default installation is not suitable.

## Generate the Evaluation Protocols

From the repository root:

```bash
python scripts/generate_kni_intent_dataset.py
python scripts/generate_kni_intent_hard_dataset.py
python scripts/generate_kni_intent_challenge.py
python scripts/make_challenge_neutral_kg.py
python scripts/generate_external_5qi_protocol.py
```

The principal protocol is written to `data/processed/challenge_neutral_kg/`. See `data/README.md` and `docs/` for construction details.

## Train KCGT and Controlled Variants

```bash
python scripts/run_method_novelty_suite.py \
  --python python \
  --processed-dir data/processed/challenge_neutral_kg \
  --output-root outputs/method_novelty_v1 \
  --seeds 1 2 3 4 5 \
  --epochs 6 \
  --batch-size 128
```

The suite evaluates the final relation-aware KCGT, standard supervised contrastive learning, an R-GCN control, and shuffled relation labels under matched seeds.

## External 3GPP 5QI Evaluation

```bash
python scripts/evaluate_external_5qi_protocol.py \
  --checkpoint-root outputs/method_novelty_v1 \
  --checkpoint-pattern "seed_{seed}/full_structured_relgt/best.pt" \
  --output-dir outputs/external_5qi_protocol_relgt_oscl \
  --seeds 1 2 3 4 5
```

## Management-Oriented Evaluation

```bash
python scripts/run_end_to_end_management_simulation.py \
  --processed-dir data/processed/challenge_neutral_kg \
  --kcgt-root outputs/method_novelty_v1 \
  --sbert-root outputs/service_template_scale_baseline_embeddings \
  --output-dir outputs/end_to_end_management_simulation \
  --seeds 1 2 3 4 5 \
  --queries 512 \
  --top-k 10
```

Additional scripts reproduce cross-protocol transfer, schema-shift retrieval, perturbation robustness, statistical tests, service-template scaling, conflict screening, and SLA-aware reranking.

## Tests

After generating `data/processed/challenge_neutral_kg/`:

```bash
python -m unittest discover -s tests -v
```

Without generated data, the integration tests are reported as skipped rather than failed.

## Reproducibility Notes

- All commands use paths relative to the repository root.
- Random seeds are exposed by the experiment runners.
- Large outputs are written under `outputs/` and are excluded from version control.
- Pretrained baselines may download model weights from their original providers.
- The repository does not include trained checkpoints.

## Citation

Citation metadata are provided in `CITATION.cff`. The final journal citation will be added after publication.

## License

The code is released under the MIT License. Dataset and third-party resources may be subject to their respective terms.
