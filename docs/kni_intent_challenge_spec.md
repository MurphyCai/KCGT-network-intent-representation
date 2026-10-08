# KNI-Intent-Challenge Dataset Specification

## Purpose

KNI-Intent-Challenge is a stress-test benchmark derived from KNI-Intent-Hard-v2. It does not replace Hard-v2. Instead, it is designed to test whether a network intent representation model still works when simple cues are weakened.

The benchmark targets the following reviewer concern:

> Does the model learn transferable network intent representations, or does it rely on explicit class keywords, template patterns, slice names, or easy QoS thresholds?

## Design Principles

KNI-Intent-Challenge keeps the same eight intent classes as KNI-Intent-Hard-v2, but modifies the input distribution in six ways.

1. Text leakage reduction

   Strong class and slice indicators are removed or rewritten. Examples include `URLLC`, `eMBB`, `mMTC`, `V2X`, `slice`, `industrial`, `gaming`, `telemedicine`, and similar direct clues. The rewritten text uses service-level descriptions such as "interactive visual session", "mobile coordination flow", or "remote expert operation".

2. Boundary QoS profiles

   QoS values are moved toward class boundaries and overlapping regions. For example, low-latency services are not always assigned extremely small latency; high-throughput services may receive moderate bandwidth values. This reduces reliance on simple threshold rules.

3. Neighbor-class confusion

   Challenge samples emphasize confusable class pairs:

   - AR/VR Immersive Service vs. Cloud Gaming
   - Telemedicine vs. Industrial URLLC
   - Autonomous Driving / V2X vs. Industrial URLLC
   - Video Streaming eMBB vs. Cloud Gaming
   - Security-Critical Enterprise vs. Telemedicine

4. Stronger hard negatives

   Hard negatives are selected from records with similar slice type, overlapping network functions, similar QoS ranges, or predefined confusable classes. This makes retrieval harder than ordinary different-class negative sampling.

5. Unseen composition

   The unseen-composition split holds out specific application, QoS-concept, and function combinations. Individual components may appear in training, but their composition is reserved for stress-test evaluation.

6. Adversarial paraphrases

   The text is rewritten with more natural and less repetitive templates. The generator avoids using the original Hard-v2 template IDs as direct textual patterns.

## Files

The generator produces:

- `data/processed/challenge/train.jsonl`
- `data/processed/challenge/val.jsonl`
- `data/processed/challenge/test.jsonl`
- `data/processed/challenge/unseen_composition_test.jsonl`
- `outputs/reports/kni_intent_challenge_report.md`

## Record Schema

Each JSONL record preserves the fields expected by the existing training pipeline:

- `intent_id`
- `intent_text`
- `intent_class`
- `class_id`
- `slice_type`
- `auxiliary_slice_type`
- QoS fields: `qos_latency_ms`, `qos_bandwidth_mbps`, `qos_reliability`, `qos_jitter_ms`, `qos_packet_loss`, `qos_security_level`, `qos_mobility`, `qos_energy_sensitivity`
- `application`
- `scenario`
- `network_functions`
- `resources`
- `kg_nodes`
- `kg_edges`
- `positive_group_id`
- `positive_ids`
- `hard_negative_ids`
- `easy_negative_ids`

Challenge-specific metadata is also added:

- `challenge_source_id`: original Hard-v2 intent id
- `challenge_type`: stress-test type flags
- `confusable_with`: expected neighbor class for difficult samples
- `composition_key`: application/QoS/function tuple used for unseen-composition splitting

## Leakage Controls

The generator applies three controls:

1. Text keyword control: explicit class and slice words are removed from rewritten text.
2. KG label control: sample-level `intent_class:*` nodes and their edges are removed from the input KG.
3. Group split control: train/validation/test splits are group-disjoint by `positive_group_id`.

## Intended Usage

The standard Hard-v2 benchmark should be used for main reproducible comparison. KNI-Intent-Challenge should be used as a stress-test section in the paper:

- closed-set classification on challenge test
- retrieval on challenge test
- few-shot or low-data training using challenge train
- unseen-composition retrieval
- error analysis and failure cases

The expected result is not necessarily perfect accuracy. A useful Challenge benchmark should expose degradation in text-only and threshold-based baselines, while allowing knowledge-guided contrastive models to preserve stronger representation quality.
