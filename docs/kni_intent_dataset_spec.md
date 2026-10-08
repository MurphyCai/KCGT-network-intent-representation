# KNI-Intent Dataset Specification

**Dataset name:** KNI-Intent

**Full name:** Knowledge-Guided Network Intent Dataset

**Project:** Knowledge-Guided Contrastive Graph Transformer for Transferable Network Intent Representation

**Target paper version:** IEEE TNSM-oriented version

**Workspace:** repository root

**Date:** 2026-07-10

---

## 1. Dataset Purpose

KNI-Intent is designed for transferable network intent representation in Intent-Based Networking (IBN) and Zero-Touch Networking (ZTN). It is not designed as a VNF sequence generation dataset. Instead, it supports the following tasks:

1. Network intent classification.
2. Network slice type recognition.
3. Intent similarity retrieval.
4. Intent representation learning.
5. Few-shot intent generalization.
6. Unseen intent generalization.
7. Knowledge-guided graph-text contrastive learning.

The dataset is synthetic but network-domain-grounded. It is generated from transparent rules based on service categories, QoS profiles, network slice types, network functions, resource requirements, and intent paraphrase templates.

---

## 2. Design Principles

The dataset should satisfy five principles:

1. **Network-management relevance**

   Each sample should represent a meaningful network or service management intent.

2. **Heterogeneous intent information**

   Each sample should contain natural-language intent text, QoS values, service category, slice type, network functions, resources, and graph relations.

3. **Contrastive structure**

   Each intent should have positive samples and hard negative samples for contrastive learning.

4. **Transferability**

   The dataset should support few-shot and unseen-intent evaluation.

5. **Reproducibility**

   The generation rules, labels, QoS ranges, templates, and split protocols should be fully documented.

---

## 3. Core Sample Schema

Each sample should be stored as one record with the following fields:

| Field | Type | Description |
|---|---|---|
| `intent_id` | string | Unique sample ID |
| `intent_text` | string | Natural-language network intent |
| `intent_class` | string | High-level intent class |
| `slice_type` | string | Network slice type |
| `application` | string | Application or service name |
| `scenario` | string | Usage scenario |
| `qos_latency_ms` | float | Required end-to-end latency |
| `qos_bandwidth_mbps` | float | Required bandwidth |
| `qos_reliability` | float | Reliability target, from 0 to 1 |
| `qos_jitter_ms` | float | Jitter target |
| `qos_packet_loss` | float | Packet loss target |
| `qos_security_level` | int | Security level from 1 to 5 |
| `qos_mobility` | int | Mobility level from 1 to 5 |
| `qos_energy_sensitivity` | int | Energy sensitivity from 1 to 5 |
| `network_functions` | list[string] | Required network functions |
| `resources` | list[string] | Required resource types |
| `kg_nodes` | list[string] | Nodes in the sample-specific intent subgraph |
| `kg_edges` | list[tuple] | Edges in the sample-specific intent subgraph |
| `positive_group_id` | string | ID for paraphrases/semantically equivalent samples |
| `source_template_id` | string | Template ID used to generate the sample |
| `split_tag` | string | train/validation/test/few-shot/unseen tag |

Recommended raw formats:

```text
data/raw/kni_intent_raw.jsonl
data/raw/kni_intent_raw.csv
```

Recommended processed formats:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/few_shot/
data/processed/unseen/
```

---

## 4. Intent Taxonomy

The first version should use 8 intent classes. These classes are broad enough for network management tasks and distinct enough for classification, retrieval, and contrastive learning.

| Class ID | Intent Class | Typical Slice | Main Objective |
|---|---|---|---|
| C1 | Industrial URLLC | URLLC | Ultra-low latency and high reliability |
| C2 | Video Streaming eMBB | eMBB | High throughput and stable streaming |
| C3 | AR/VR Immersive Service | eMBB/URLLC | High bandwidth, low jitter, low latency |
| C4 | Massive IoT mMTC | mMTC | Massive connectivity and energy efficiency |
| C5 | Autonomous Driving / V2X | URLLC/V2X | Low latency, high mobility, reliability |
| C6 | Telemedicine | URLLC/eMBB | Reliability, security, and low latency |
| C7 | Cloud Gaming | eMBB/URLLC | Low latency, high bandwidth, low jitter |
| C8 | Security-Critical Enterprise | Enterprise/Security | Security, isolation, and policy enforcement |

Optional extension classes:

| Class ID | Intent Class | Typical Slice | Main Objective |
|---|---|---|---|
| C9 | Smart Grid Control | URLLC | Reliability and deterministic control |
| C10 | Emergency Communication | URLLC/Public Safety | Priority, availability, and robustness |
| C11 | Edge AI Inference | Edge/eMBB | Compute-aware low-latency inference |
| C12 | Smart City Monitoring | mMTC | Wide-area connectivity and low energy |

The first implementation should use C1-C8. C9-C12 can be used for unseen-intent generalization experiments.

---

## 5. Application and Service Taxonomy

### C1: Industrial URLLC

Applications:

```text
industrial_robot
motion_control
factory_automation
remote_plc_control
deterministic_video_inspection
```

Scenarios:

```text
smart_factory
industrial_workshop
robotic_assembly_line
automated_quality_inspection
```

### C2: Video Streaming eMBB

Applications:

```text
video_streaming
live_broadcast
4k_video_delivery
online_video_service
surveillance_video_upload
```

Scenarios:

```text
urban_mobile_broadband
stadium_streaming
campus_video_service
home_entertainment
```

### C3: AR/VR Immersive Service

Applications:

```text
ar_assistance
vr_conference
immersive_training
metaverse_interaction
remote_visualization
```

Scenarios:

```text
remote_training
immersive_education
virtual_collaboration
industrial_ar_maintenance
```

### C4: Massive IoT mMTC

Applications:

```text
smart_metering
environmental_sensing
agricultural_monitoring
wearable_sensors
city_sensor_network
```

Scenarios:

```text
smart_city
smart_agriculture
utility_metering
large_scale_monitoring
```

### C5: Autonomous Driving / V2X

Applications:

```text
vehicle_platooning
cooperative_driving
roadside_safety_alert
autonomous_vehicle_control
traffic_coordination
```

Scenarios:

```text
urban_traffic
highway_mobility
intersection_safety
connected_vehicle_network
```

### C6: Telemedicine

Applications:

```text
remote_surgery
medical_video_consultation
patient_monitoring
ambulance_telemetry
medical_image_transfer
```

Scenarios:

```text
remote_hospital
emergency_care
mobile_healthcare
specialist_consultation
```

### C7: Cloud Gaming

Applications:

```text
cloud_game_streaming
interactive_game_service
mobile_cloud_gaming
multiplayer_low_latency_game
```

Scenarios:

```text
home_gaming
mobile_gaming
edge_game_rendering
interactive_entertainment
```

### C8: Security-Critical Enterprise

Applications:

```text
enterprise_vpn
secure_database_access
financial_transaction_service
confidential_file_transfer
zero_trust_access
```

Scenarios:

```text
enterprise_branch
financial_network
private_cloud_access
secure_remote_work
```

---

## 6. Slice Type Taxonomy

Recommended slice labels:

| Slice Label | Description | Typical Classes |
|---|---|---|
| `URLLC` | Ultra-reliable low-latency communication | C1, C5, C6 |
| `eMBB` | Enhanced mobile broadband | C2, C3, C7 |
| `mMTC` | Massive machine-type communication | C4 |
| `V2X` | Vehicle-to-everything service slice | C5 |
| `Enterprise` | Enterprise private/service slice | C8 |
| `Security` | Security-critical service slice | C8 |
| `Edge` | Edge-assisted service slice | C1, C3, C6, C7 |

One sample may have:

- one primary slice type;
- one optional auxiliary slice type.

Recommended fields:

```text
slice_type
auxiliary_slice_type
```

---

## 7. QoS Profile Rules

The following ranges are recommended for synthetic generation. Values should be sampled with small random perturbations inside each range.

| Intent Class | Latency ms | Bandwidth Mbps | Reliability | Jitter ms | Packet Loss | Security | Mobility | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Industrial URLLC | 1-10 | 5-100 | 0.9990-0.999999 | 0.1-5 | 1e-7-1e-4 | 3-5 | 1-3 | 2-4 |
| Video Streaming eMBB | 30-150 | 20-1000 | 0.9500-0.9990 | 5-40 | 1e-4-1e-2 | 1-3 | 1-4 | 1-3 |
| AR/VR Immersive | 5-30 | 100-2000 | 0.9900-0.9999 | 1-10 | 1e-5-1e-3 | 2-4 | 1-4 | 1-3 |
| Massive IoT mMTC | 100-1000 | 0.01-5 | 0.9000-0.9900 | 20-200 | 1e-3-5e-2 | 1-3 | 1-2 | 4-5 |
| Autonomous Driving / V2X | 1-20 | 5-200 | 0.9990-0.999999 | 0.1-10 | 1e-7-1e-4 | 3-5 | 4-5 | 2-4 |
| Telemedicine | 1-50 | 10-1000 | 0.9990-0.999999 | 0.1-20 | 1e-7-1e-4 | 4-5 | 1-4 | 1-3 |
| Cloud Gaming | 5-40 | 50-1000 | 0.9900-0.9999 | 1-15 | 1e-5-1e-3 | 2-4 | 1-4 | 1-3 |
| Security Enterprise | 20-200 | 5-500 | 0.9900-0.9999 | 5-50 | 1e-5-1e-3 | 5 | 1-3 | 1-3 |

QoS normalization for model input:

```text
latency: log-normalize then min-max scale
bandwidth: log-normalize then min-max scale
reliability: transform as -log10(1 - reliability)
packet_loss: transform as -log10(packet_loss)
jitter: log-normalize then min-max scale
security/mobility/energy: divide by 5
```

---

## 8. Network Function Taxonomy

Network functions should be selected according to the intent class and QoS requirements.

| Function Label | Description | Typical Classes |
|---|---|---|
| `traffic_classification` | Identify service/application traffic | All |
| `qos_scheduling` | QoS-aware scheduling | C1, C2, C3, C5, C6, C7 |
| `priority_queueing` | Priority-based forwarding | C1, C5, C6 |
| `network_slicing` | Slice selection and isolation | All |
| `edge_computing` | Edge compute support | C1, C3, C6, C7 |
| `load_balancing` | Traffic load distribution | C2, C3, C7, C8 |
| `bandwidth_reservation` | Reserve bandwidth | C2, C3, C6, C7 |
| `latency_optimization` | Reduce end-to-end delay | C1, C3, C5, C6, C7 |
| `reliability_control` | Improve reliability and redundancy | C1, C5, C6 |
| `mobility_management` | Mobility-aware service continuity | C5, C6 |
| `security_firewall` | Firewall and filtering | C8, C6 |
| `intrusion_detection` | Detect malicious traffic | C8 |
| `encryption` | Protect confidentiality | C6, C8 |
| `access_control` | Enforce access policies | C8 |
| `energy_optimization` | Energy-aware communication | C4 |
| `massive_access_control` | Massive IoT access control | C4 |
| `packet_loss_control` | Reduce packet loss | C1, C3, C5, C6, C7 |
| `jitter_control` | Stabilize packet delay variation | C3, C6, C7 |

Recommended function count:

```text
minimum: 2
typical: 3-5
maximum: 7
```

---

## 9. Resource Taxonomy

| Resource Label | Description | Typical Related Functions |
|---|---|---|
| `radio_resource` | Wireless radio resource | network_slicing, qos_scheduling |
| `bandwidth_resource` | Link bandwidth | bandwidth_reservation |
| `compute_resource` | CPU/GPU/edge compute | edge_computing |
| `storage_resource` | Cache/storage | video_streaming, edge services |
| `queue_resource` | Queue and buffer resource | priority_queueing, qos_scheduling |
| `security_resource` | Security processing capacity | encryption, firewall, IDS |
| `routing_resource` | Routing and forwarding capacity | latency_optimization |
| `spectrum_resource` | Spectrum resource | radio-intensive services |
| `energy_resource` | Device/network energy budget | mMTC, IoT |

Each sample should include 1-4 resource labels.

---

## 10. Knowledge Graph Schema

### Node Types

| Node Type | Examples |
|---|---|
| `Application` | industrial_robot, video_streaming, enterprise_vpn |
| `Scenario` | smart_factory, urban_traffic, secure_remote_work |
| `QoS` | ultra_low_latency, high_bandwidth, high_reliability |
| `Slice` | URLLC, eMBB, mMTC, V2X, Enterprise |
| `Function` | edge_computing, qos_scheduling, encryption |
| `Resource` | bandwidth_resource, compute_resource, security_resource |
| `IntentClass` | Industrial_URLLC, Massive_IoT_mMTC |

### Relation Types

| Relation | Source Type | Target Type | Example |
|---|---|---|---|
| `requires` | Application/IntentClass | QoS | industrial_robot requires ultra_low_latency |
| `belongs_to` | Application/IntentClass | Slice | industrial_robot belongs_to URLLC |
| `supported_by` | Slice/QoS | Function | URLLC supported_by priority_queueing |
| `consumes` | Function/Application | Resource | edge_computing consumes compute_resource |
| `deployed_in` | Application | Scenario | industrial_robot deployed_in smart_factory |
| `has_function` | IntentClass | Function | Telemedicine has_function encryption |
| `has_resource` | IntentClass | Resource | VideoStreaming has_resource bandwidth_resource |
| `conflicts_with` | QoS | QoS | ultra_low_latency conflicts_with energy_saving |
| `similar_to` | Application/IntentClass | Application/IntentClass | cloud_gaming similar_to ar_vr_service |

### Global KG Files

Recommended files:

```text
data/kg/kg_nodes.csv
data/kg/kg_edges.csv
data/kg/relation_types.json
data/kg/node_types.json
```

### Sample Subgraph

For each sample, construct a subgraph containing:

1. Application node.
2. Scenario node.
3. Intent class node.
4. Slice node.
5. QoS nodes.
6. Required network function nodes.
7. Required resource nodes.
8. Edges connecting the above nodes.

---

## 11. Intent Text Generation Rules

Intent text should include diverse forms while keeping network-domain semantics.

### Template Categories

1. Direct requirement:

```text
{application} requires {qos_phrase} communication in {scenario}.
```

2. User-centric request:

```text
The user needs {slice_phrase} service for {application} with {qos_phrase}.
```

3. Service management request:

```text
Provide {network_function_phrase} for {application} under {qos_phrase} constraints.
```

4. Scenario-driven request:

```text
In {scenario}, {application} should maintain {qos_phrase} and {resource_phrase}.
```

5. Operational intent:

```text
Ensure {application} traffic is handled with {slice_type} slicing and {network_function_phrase}.
```

### QoS Phrase Examples

| QoS Type | Phrases |
|---|---|
| Low latency | ultra-low latency, minimal delay, millisecond-level response |
| High bandwidth | high throughput, broadband capacity, large data rate |
| High reliability | ultra-reliable delivery, highly dependable transmission |
| Low jitter | stable delay, low delay variation |
| High security | strong security protection, encrypted and isolated access |
| Energy saving | energy-efficient connectivity, low-power communication |

### Example Intent Texts

```text
Industrial robot requires ultra-reliable low-latency communication in a smart factory.
The user needs high-throughput video streaming with stable bandwidth.
AR maintenance requires high bandwidth, low jitter, and edge computing support.
Massive smart meters need energy-efficient connectivity for large-scale monitoring.
Autonomous vehicles require low-latency and high-mobility V2X communication.
Remote surgery requires secure, reliable, and low-latency network service.
Cloud gaming needs low delay, high bandwidth, and stable packet delivery.
Enterprise VPN traffic requires strong security isolation and encrypted access.
```

---

## 12. Positive Pair Construction

Positive pairs are used for supervised contrastive learning.

Two samples can be considered positive if at least one of the following conditions is true:

1. Same `positive_group_id`.
2. Same intent class.
3. Same primary slice type and similar QoS profile.
4. Same application but different paraphrase template.
5. Same service category and overlapping network functions.

### Positive Group Rule

Each base intent should produce multiple paraphrases:

```text
base_intent -> 3 to 8 paraphrased samples
```

All paraphrases of the same base intent share:

```text
positive_group_id
```

### Similar QoS Rule

Two samples are QoS-similar if:

```text
normalized_qos_distance < delta_qos
```

Recommended first value:

```text
delta_qos = 0.25
```

---

## 13. Negative Pair Construction

Negative pairs should include both easy negatives and hard negatives.

### Easy Negatives

Two samples are easy negatives if:

1. Different intent class.
2. Different slice type.
3. Low overlap in network functions.

### Hard Negatives

Two samples are hard negatives if they look lexically similar but have different network requirements.

Examples:

| Intent A | Intent B | Conflict |
|---|---|---|
| Low-latency industrial robot | Energy-saving IoT sensing | latency vs energy |
| Secure enterprise file transfer | High-throughput video streaming | security vs throughput |
| AR remote maintenance | Cloud gaming | similar bandwidth/latency but different functions |
| Remote surgery | Medical image transfer | same domain, different latency criticality |

### Negative Sampling Rule

For each anchor sample:

```text
positive samples: 2-5
easy negative samples: 5-10
hard negative samples: 2-5
```

---

## 14. Dataset Scale Plan

### Pilot Version

```text
classes: 8
base intents per class: 20
paraphrases per base intent: 3
total samples: about 480
```

Purpose:

- Debug data generation.
- Debug model input/output.
- Validate graph construction.

### CPU-Friendly Version

```text
classes: 8
base intents per class: 100
paraphrases per base intent: 5
total samples: about 4000
```

Purpose:

- Main local experiments.
- Baseline comparison.
- Ablation experiments.

### Full Paper Version

```text
classes: 8
base intents per class: 250
paraphrases per base intent: 5
total samples: about 10000
```

Purpose:

- Main TNSM paper results.

### Extended Version

```text
classes: 12
base intents per class: 300
paraphrases per base intent: 5-8
total samples: 18000-28800
```

Purpose:

- Optional GPU-scale experiment.
- Stronger unseen-class evaluation.

---

## 15. Split Strategy

### Standard Split

```text
train: 70%
validation: 15%
test: 15%
```

Important rule:

> Samples with the same `positive_group_id` should not be split across train, validation, and test for strict generalization evaluation.

Two split modes should be supported:

1. **Paraphrase-mixed split**
   - Allows paraphrases of the same base intent across splits.
   - Easier setting.

2. **Group-disjoint split**
   - Keeps all paraphrases of the same base intent in the same split.
   - Harder and more convincing setting.

The paper should report the group-disjoint split as the main setting.

### Few-shot Split

Training samples per class:

```text
K = 1, 5, 10, 20
```

Validation/test sets remain unchanged.

Report:

```text
Accuracy
Macro-F1
Recall@5
```

### Unseen-Intent Split

Use four unseen settings:

1. **Unseen application**
   - Hold out selected applications from training.

2. **Unseen scenario**
   - Hold out selected scenarios from training.

3. **Unseen template**
   - Hold out selected text templates from training.

4. **Unseen class**
   - Optional: train on 8 classes and test representation transfer on C9-C12.

Recommended first unseen setup:

```text
unseen_application + unseen_template
```

---

## 16. File Layout

Recommended dataset files:

```text
data/
├── raw/
│   ├── kni_intent_raw.jsonl
│   ├── kni_intent_raw.csv
│   └── generation_config.json
├── processed/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── few_shot/
│   └── unseen/
└── kg/
    ├── kg_nodes.csv
    ├── kg_edges.csv
    ├── relation_types.json
    ├── node_types.json
    └── class_kg_rules.json
```

---

## 17. Data Quality Checks

Before using the dataset, run the following checks:

1. No empty intent text.
2. No missing class or slice label.
3. QoS values fall within class-specific ranges.
4. Each sample has at least 2 network functions.
5. Each sample has at least 1 resource label.
6. Each sample has a non-empty KG subgraph.
7. Each positive group has at least 3 samples.
8. Class distribution is balanced.
9. Slice distribution is reported.
10. Train/validation/test leakage is checked by `positive_group_id`.

---

## 18. Dataset Statistics to Report in the Paper

The paper should include a dataset statistics table with:

| Statistic | Value |
|---|---:|
| Number of samples | TBD |
| Number of intent classes | 8 |
| Number of slice types | TBD |
| Number of applications | TBD |
| Number of scenarios | TBD |
| Number of network functions | TBD |
| Number of resource types | TBD |
| Number of KG nodes | TBD |
| Number of KG edges | TBD |
| Average intent length | TBD |
| Average KG nodes per sample | TBD |
| Average KG edges per sample | TBD |
| Average positive samples per anchor | TBD |

---

## 19. Recommended First Implementation Target

The first generated dataset should be:

```text
version: KNI-Intent-Pilot
classes: 8
base intents per class: 20
paraphrases per base intent: 3
estimated samples: 480
split mode: group-disjoint
```

After verifying the pipeline, generate:

```text
version: KNI-Intent-CPU
classes: 8
base intents per class: 100
paraphrases per base intent: 5
estimated samples: 4000
split mode: group-disjoint
```

---

## 20. Immediate Next Step

The next step is:

> **Step 3: Implement the first dataset generator for KNI-Intent-Pilot.**

Recommended output files:

```text
scripts/generate_kni_intent_dataset.py
data/raw/generation_config.json
data/raw/kni_intent_raw.jsonl
data/raw/kni_intent_raw.csv
data/kg/kg_nodes.csv
data/kg/kg_edges.csv
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
```

The first generator should be deterministic and use a fixed random seed.
