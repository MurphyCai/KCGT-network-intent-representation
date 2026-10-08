"""Build a test-only external 3GPP 5QI/QoS transfer protocol.

The standardized values are transcribed from 3GPP TS 23.501, Table 5.7.4-1.
External graph records retain a separate schema; model records are produced by an
explicit ontology adapter and never expose the target 5QI identifier in the text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE_URL = "https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/18.12.00_60/ts_123501v181200p.pdf"

# 5QI, resource type, priority, PDB ms, PER, MDBV bytes, service family, official example-service summary
PROFILES = [
    (1, "GBR", 20, 100, 1e-2, None, "conversational_voice", "Conversational voice"),
    (2, "GBR", 40, 150, 1e-3, None, "live_video", "Conversational video and live streaming"),
    (3, "GBR", 30, 50, 1e-3, None, "interactive_control", "Real-time gaming, V2X, and process automation"),
    (4, "GBR", 50, 300, 1e-6, None, "buffered_video", "Buffered video streaming"),
    (5, "Non-GBR", 10, 100, 1e-6, None, "ims_signalling", "IMS signalling"),
    (6, "Non-GBR", 60, 300, 1e-6, None, "tcp_data", "Buffered video and TCP-based services"),
    (7, "Non-GBR", 70, 100, 1e-3, None, "interactive_media", "Voice, live video, and interactive gaming"),
    (8, "Non-GBR", 80, 300, 1e-6, None, "tcp_data", "Buffered video and TCP-based services"),
    (9, "Non-GBR", 90, 300, 1e-6, None, "default_data", "Buffered video and TCP-based services"),
    (65, "GBR", 7, 75, 1e-2, None, "mission_critical_voice", "Mission-critical push-to-talk voice"),
    (67, "GBR", 15, 100, 1e-3, None, "mission_critical_video", "Mission-critical video"),
    (69, "Non-GBR", 5, 60, 1e-6, None, "mission_critical_signalling", "Mission-critical signalling"),
    (70, "Non-GBR", 55, 200, 1e-6, None, "mission_critical_data", "Mission-critical data"),
    (75, "GBR", 25, 50, 1e-2, None, "v2x", "V2X messages"),
    (79, "Non-GBR", 65, 50, 1e-2, None, "v2x", "V2X messages"),
    (80, "Non-GBR", 68, 10, 1e-6, None, "low_latency_embb", "Low-latency eMBB, AR, and gaming"),
    (82, "Delay-Critical GBR", 19, 10, 1e-4, 255, "discrete_automation", "Discrete automation"),
    (85, "Delay-Critical GBR", 21, 5, 1e-5, 255, "remote_control", "High-voltage control, remote driving, and split AI"),
    (89, "Delay-Critical GBR", 25, 15, 1e-4, 17000, "cloud_rendering", "Visual cloud and edge rendering"),
    (90, "Delay-Critical GBR", 25, 20, 1e-4, 63000, "cloud_rendering", "Visual cloud and edge rendering"),
]

TEXT_PATTERNS = [
    "The requested service is {service}; it requires a packet delay budget of {pdb} ms and packet error rate no greater than {per}.",
    "Provision {service} traffic with {rtype} treatment, at most {pdb} ms packet delay and a {per} packet error target.",
    "This request carries {service} traffic. The bearer should meet {pdb} ms delay, {per} loss tolerance, and priority level {priority}.",
    "Select a standardized QoS profile for {service}, using {rtype} delivery with a {pdb} ms delay budget and {per} error rate.",
    "The management system must support {service}; latency shall remain within {pdb} ms while packet errors remain below {per}.",
    "Map the flow for {service} to a {rtype} profile characterized by priority {priority}, {pdb} ms PDB, and PER {per}.",
    "A network slice carries {service} traffic and needs bounded delay of {pdb} ms with packet error probability {per}.",
    "Choose the service profile that best fits {service}: {rtype}, priority {priority}, delay budget {pdb} ms, and error target {per}.",
]


def class_and_slice(family: str) -> tuple[str, str]:
    if family in {"v2x", "remote_control"}:
        return "Autonomous Driving / V2X", "V2X"
    if family in {"interactive_control", "discrete_automation"}:
        return "Industrial URLLC", "URLLC"
    if family in {"low_latency_embb", "cloud_rendering"}:
        return "AR/VR Immersive Service", "eMBB"
    if "video" in family or family in {"interactive_media", "buffered_video"}:
        return "Video Streaming eMBB", "eMBB"
    if "mission_critical" in family or family == "ims_signalling":
        return "Security-Critical Enterprise", "Enterprise"
    if family == "conversational_voice":
        return "Telemedicine", "URLLC"
    return "Video Streaming eMBB", "eMBB"


def capabilities(family: str, pdb: int, per: float) -> list[str]:
    caps = ["reliable_delivery"] if per <= 1e-5 else ["interactive_delivery"]
    if pdb <= 20:
        caps.append("closed_loop_control")
    elif pdb <= 100:
        caps.append("edge_assisted_experience")
    else:
        caps.append("adaptive_throughput")
    if family in {"v2x", "remote_control"}:
        caps.append("mobility_aware_assurance")
    if "mission_critical" in family or family == "ims_signalling":
        caps.append("policy_enforcement")
    return sorted(set(caps))


def auxiliary_qos(family: str, pdb: int, per: float) -> dict[str, float | int]:
    bandwidth = 80.0 if family in {"live_video", "cloud_rendering", "low_latency_embb"} else 20.0
    mobility = 5 if family in {"v2x", "remote_control"} else 2
    security = 5 if "mission_critical" in family or family == "ims_signalling" else 3
    return {
        "qos_latency_ms": float(pdb),
        "qos_bandwidth_mbps": bandwidth,
        "qos_reliability": 1.0 - per,
        "qos_jitter_ms": max(1.0, pdb * 0.1),
        "qos_packet_loss": per,
        "qos_security_level": security,
        "qos_mobility": mobility,
        "qos_energy_sensitivity": 2,
    }


def adapted_graph(family: str, pdb: int, per: float, rtype: str) -> tuple[list[str], list[list[str]]]:
    contexts = ["context:service_request", "context:performance_management"]
    constraints = ["qos_family:latency_constraint", "qos_family:reliability_constraint"]
    caps = [f"capability:{x}" for x in capabilities(family, pdb, per)]
    resource = "resource_family:shared_queueing" if rtype == "Non-GBR" else "resource_family:shared_radio"
    if rtype == "Delay-Critical GBR":
        resource = "resource_family:shared_compute"
    nodes = contexts + constraints + caps + [resource]
    edges: list[list[str]] = []
    for context in contexts:
        edges.extend([[context, "has_constraint", q] for q in constraints])
        edges.extend([[context, "uses_capability", cap] for cap in caps])
    edges.extend([[cap, "uses_resource_family", resource] for cap in caps])
    return nodes, edges


def raw_graph(fiveqi: int, family: str, pdb: int, per: float, rtype: str) -> tuple[list[str], list[list[str]]]:
    nodes = [
        f"service_family:{family}", f"standard_profile:5qi_{fiveqi}", f"resource_type:{rtype.lower().replace(' ', '_')}",
        f"qos_characteristic:pdb_{pdb}ms", f"qos_characteristic:per_{per:g}", "standard:3gpp_ts_23_501",
    ]
    edges = [
        ["standard:3gpp_ts_23_501", "defines", f"standard_profile:5qi_{fiveqi}"],
        [f"standard_profile:5qi_{fiveqi}", "supports", f"service_family:{family}"],
        [f"standard_profile:5qi_{fiveqi}", "uses_resource_type", f"resource_type:{rtype.lower().replace(' ', '_')}"],
        [f"standard_profile:5qi_{fiveqi}", "has_pdb", f"qos_characteristic:pdb_{pdb}ms"],
        [f"standard_profile:5qi_{fiveqi}", "has_per", f"qos_characteristic:per_{per:g}"],
    ]
    return nodes, edges


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    root = Path("data/external/5qi")
    root.mkdir(parents=True, exist_ok=True)
    fields = ["fiveqi", "resource_type", "priority_level", "packet_delay_budget_ms", "packet_error_rate", "maximum_data_burst_volume_bytes", "service_family", "example_service", "source_clause", "source_url"]
    with (root / "standardized_5qi_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q, rt, pri, pdb, per, mdbv, fam, service in PROFILES:
            writer.writerow(dict(zip(fields, [q, rt, pri, pdb, per, mdbv or "", fam, service, "Table 5.7.4-1", SOURCE_URL])))

    raw_queries, model_queries, raw_gallery, model_gallery = [], [], [], []
    for q, rt, pri, pdb, per, mdbv, fam, service in PROFILES:
        label, slice_type = class_and_slice(fam)
        raw_nodes, raw_edges = raw_graph(q, fam, pdb, per, rt)
        nodes, edges = adapted_graph(fam, pdb, per, rt)
        common = {
            "intent_class": label, "slice_type": slice_type, "application": fam,
            "scenario": "external_3gpp_5qi", "fiveqi": q, "service_family": fam,
            "resource_type": rt, "priority_level": pri, "packet_delay_budget_ms": pdb,
            "packet_error_rate": per, "maximum_data_burst_volume_bytes": mdbv,
            "network_functions": capabilities(fam, pdb, per), "resources": [rt],
            "positive_group_id": f"5QI_{q}", "source_standard": "3GPP TS 23.501 Table 5.7.4-1",
            **auxiliary_qos(fam, pdb, per),
        }
        gallery_text = f"Standardized profile for {service.lower()} with {rt} treatment, {pdb} ms packet delay budget, packet error rate {per:g}, and priority level {pri}."
        raw_gallery.append({**common, "intent_id": f"EXT_5QI_{q}_G", "intent_text": gallery_text, "external_kg_nodes": raw_nodes, "external_kg_edges": raw_edges})
        model_gallery.append({**common, "intent_id": f"EXT_5QI_{q}_G", "intent_text": gallery_text, "kg_nodes": nodes, "kg_edges": edges})
        for variant, pattern in enumerate(TEXT_PATTERNS, 1):
            text = pattern.format(service=service.lower(), rtype=rt, pdb=pdb, per=f"{per:g}", priority=pri)
            raw_queries.append({**common, "intent_id": f"EXT_5QI_{q}_Q{variant:02d}", "intent_text": text, "external_kg_nodes": [n for n in raw_nodes if not n.startswith("standard_profile:")], "external_kg_edges": [e for e in raw_edges if not any(x.startswith("standard_profile:") for x in (e[0], e[2]))]})
            model_queries.append({**common, "intent_id": f"EXT_5QI_{q}_Q{variant:02d}", "intent_text": text, "kg_nodes": nodes, "kg_edges": edges})
    write_jsonl(root / "raw_external_queries.jsonl", raw_queries)
    write_jsonl(root / "raw_external_gallery.jsonl", raw_gallery)
    write_jsonl(root / "model_queries.jsonl", model_queries)
    write_jsonl(root / "model_gallery.jsonl", model_gallery)
    metadata = {
        "protocol": "external_3gpp_5qi_qos_test_only",
        "source": "ETSI TS 123 501 V18.12.0 (3GPP TS 23.501 Release 18), Table 5.7.4-1",
        "source_url": SOURCE_URL,
        "profiles": len(PROFILES), "queries": len(model_queries), "query_variants_per_profile": len(TEXT_PATTERNS),
        "target_leakage_control": "5QI identifiers are absent from query text and adapted query graphs.",
        "schema_policy": "Raw external schema is retained; an explicit type-level adapter maps it to the frozen NeutralKG vocabulary.",
        "metric_scope": "PDB, PER, priority, and resource type are standardized. Auxiliary KCGT QoS fields are excluded from core SLA claims.",
    }
    (root / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    kni_texts = set()
    for split in ("train", "val", "test"):
        path = Path("data/processed/challenge_neutral_kg") / f"{split}.jsonl"
        if path.exists():
            kni_texts.update(json.loads(line)["intent_text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    audit = {
        "query_count": len(model_queries),
        "gallery_profile_count": len(model_gallery),
        "unique_query_id_count": len({x["intent_id"] for x in model_queries}),
        "unique_gallery_5qi_count": len({x["fiveqi"] for x in model_gallery}),
        "query_text_contains_5qi_token_count": sum("5qi" in x["intent_text"].lower() for x in model_queries),
        "adapted_query_target_node_count": sum(any("5qi_" in n.lower() for n in x["kg_nodes"]) for x in model_queries),
        "exact_kni_text_overlap_count": sum(x["intent_text"] in kni_texts for x in model_queries),
        "all_checks_passed": True,
    }
    audit["all_checks_passed"] = all([
        audit["query_count"] == audit["unique_query_id_count"],
        audit["gallery_profile_count"] == audit["unique_gallery_5qi_count"],
        audit["query_text_contains_5qi_token_count"] == 0,
        audit["adapted_query_target_node_count"] == 0,
        audit["exact_kni_text_overlap_count"] == 0,
    ])
    (root / "protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"External protocol audit failed: {audit}")
    print(json.dumps(metadata, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
