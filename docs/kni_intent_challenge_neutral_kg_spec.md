# KNI-Intent-Challenge-NeutralKG Specification

## Motivation

KNI-Intent-Challenge-NeutralKG is a stricter stress-test split for evaluating network intent representation under reduced knowledge-graph shortcut signals. It extends KNI-Intent-Challenge-KG-Masked by removing not only explicit label-like KG identifiers but also class-specific context terms and fine-grained QoS/capability cues.

The split is intended for robustness analysis in the KCGT manuscript. It should be reported as a stress-test benchmark rather than the primary dataset.

## Source

The dataset is derived from:

`data/processed/challenge`

It preserves the same train/validation/test/unseen-composition sample counts and labels, while rewriting the sample-level KG.

## Neutralization Rules

### Removed Nodes

The following nodes are removed:

- `slice:*`
- `intent_class:*`
- `application:*`
- `scenario:*`
- exact `function:*`
- exact `resource:*`

### Neutral Context

Application and scenario nodes are replaced by class-neutral context nodes:

- `context:service_request`
- `context:performance_management`
- `context:resource_coordination`

These nodes are shared across all classes and therefore cannot directly reveal the intent class.

### Coarse QoS Concepts

Fine-grained QoS concepts are mapped to coarse shared nodes:

- latency-related terms -> `qos_family:latency_constraint`
- bandwidth/throughput terms -> `qos_family:throughput_constraint`
- reliability/loss terms -> `qos_family:reliability_constraint`
- jitter/stability terms -> `qos_family:stability_constraint`
- security/privacy terms -> `qos_family:security_constraint`
- mobility/coverage terms -> `qos_family:mobility_constraint`
- energy terms -> `qos_family:energy_constraint`

The goal is not to remove QoS information completely, but to prevent very specific QoS labels from becoming near-label features.

### Shared Capability Sets

Capabilities are generated at the confusable-pair level rather than the class level. Confusable classes share the same capability set:

- Industrial URLLC and Autonomous Driving / V2X share control/mobility/reliability capabilities.
- Cloud Gaming, AR/VR Immersive Service, and Video Streaming eMBB share media/interactive-delivery capabilities.
- Telemedicine and Security-Critical Enterprise share trusted-critical-service capabilities.
- Massive IoT mMTC uses scalable-device-management capabilities and shares generic resource coordination nodes with all classes.

### Resource Families

Resources are mapped to neutral families:

- `resource_family:shared_compute`
- `resource_family:shared_radio`
- `resource_family:shared_routing`
- `resource_family:shared_queueing`
- `resource_family:shared_storage`
- `resource_family:shared_security`

## Pairwise Hard-Negative Retrieval

The split includes:

`pairwise_hard_negative_gallery.jsonl`

For each test query, the gallery contains:

- positive paraphrases from the same positive group;
- hard negatives from confusable classes with similar QoS ranges;
- no copy of the query itself.

This gallery is designed to evaluate fine-grained representation quality under close semantic and QoS proximity.

## Intended Experiments

Recommended experiments:

1. Classification on `train/val/test`.
2. Unseen-composition classification or retrieval on `unseen_composition_test`.
3. Group-level retrieval on `test`.
4. Pairwise hard-negative group retrieval using `pairwise_hard_negative_gallery.jsonl`.
5. Comparison against Challenge and Challenge-KG-Masked.

## Expected Interpretation

If classification remains saturated, the paper should avoid treating classification as sufficient evidence. Retrieval metrics such as Recall@1, Precision@5, Precision@10, MRR, and NDCG@5 should be emphasized because they better expose representation-space quality under close negative samples.
