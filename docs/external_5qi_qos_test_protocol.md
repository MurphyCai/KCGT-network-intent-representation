# External 3GPP 5QI/QoS Test Protocol

## Objective

This test-only protocol evaluates whether frozen intent representations can map previously unseen service requests to standardized 5G QoS profiles. It is an external standards-profile test, not an additional KNI-Intent training split and not a live 5G-core deployment.

## Standard Source

The profile attributes are transcribed from ETSI TS 123 501 V18.12.0 (3GPP TS 23.501 Release 18), Table 5.7.4-1, *Standardized 5QI to QoS characteristics mapping*.

- Official source: https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/18.12.00_60/ts_123501v181200p.pdf
- Selected profiles: 20
- Standardized attributes: resource type, priority level, packet delay budget (PDB), packet error rate (PER), and MDBV where applicable
- Test queries: 160, comprising eight independently written requests per selected profile

## Separation From KNI-Intent

The requests do not reuse KNI-Intent generation templates. The raw external graph schema contains `standard_profile`, `service_family`, `resource_type`, and `qos_characteristic` nodes linked by `defines`, `supports`, `uses_resource_type`, `has_pdb`, and `has_per`. These names and relations are not added to training.

Because the evaluated checkpoints are frozen, a deterministic ontology adapter maps the external graph into the existing type-level NeutralKG vocabulary. The adapter maps service semantics to capabilities, PDB/PER to latency and reliability constraint families, and bearer treatment to a shared resource family. The raw and adapted records are both retained for audit.

## Leakage Controls

- Query text never contains a 5QI identifier.
- Adapted query graphs contain no `standard_profile:5qi_*` node.
- The protocol is used only at test time.
- QoS-only selection receives only PDB and PER. Priority and resource type are treated as candidate-profile outputs and are not used as query-side answer keys.
- Standardized-QoS reranking applies only PDB/PER feasibility constraints to the KCGT shortlist.
- Auxiliary bandwidth, jitter, security, mobility, and energy fields are required by the frozen model interface but are excluded from core SLA claims.

## Compared Methods

- KCGT projection retrieval, evaluated with three independently trained seeds
- KCGT plus standardized PDB/PER constraint reranking
- Standardized QoS-only nearest-profile selection
- Frozen SBERT-MiniLM text retrieval

## Metrics

- Exact 5QI Success@1 and Success@5
- Mean reciprocal rank
- Service-family match at rank 1
- Resource-type match at rank 1
- Standardized SLA fulfillment at rank 1, based only on PDB and PER
- Decision time per query

## Interpretation Boundary

The protocol tests standards-profile reuse under vocabulary and schema shift. It approximates a management-plane selection stage; it does not validate admission control, packet scheduling, live orchestration, or end-to-end SLA assurance.
