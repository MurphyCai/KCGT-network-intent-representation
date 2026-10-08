# Data

The complete synthetic datasets used in the manuscript are not included in this repository release.

The repository contains:

- generation configuration files under `data/raw/`;
- knowledge-graph schema resources under `data/kg/`;
- the external 3GPP 5QI evaluation protocol under `data/external/5qi/`.

Generate the main protocols from the repository root:

```bash
python scripts/generate_kni_intent_dataset.py
python scripts/generate_kni_intent_hard_dataset.py
python scripts/generate_kni_intent_challenge.py
python scripts/make_challenge_neutral_kg.py
```

Generated files are written under `data/processed/`, which is intentionally ignored by Git because the complete protocols are large. The exact construction rules are documented in `docs/`.

For access to the archived datasets used for the reported experiments, contact the corresponding author as described in the associated manuscript.
