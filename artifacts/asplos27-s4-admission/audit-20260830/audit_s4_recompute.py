#!/usr/bin/env python3
"""External-audit independent recompute for ASPLOS'27 S4-0 (AC3/AC4/AC5).

Stdlib-only, written by the external auditor. Re-derives every headline
number from raw slots/shapes and checks the hash chain independently.
"""
import hashlib
import json
import math
import sys

ART = sys.argv[1] if len(sys.argv) > 1 else \
    "artifacts/asplos27-s4-admission/resnet2b-prop0-v1"

failures = []


def check(cond, label):
    if not cond:
        failures.append(label)
        print("FAIL:", label)


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def chash(value):
    return hashlib.sha256(canon(value).encode()).hexdigest()


def payload_hash(obj, field):
    return chash({k: v for k, v in obj.items() if k != field})


def prod(shape):
    return math.prod(shape)


def main():
    rows = [json.loads(line) for line in
            open(ART + "/raw/workers.jsonl", encoding="utf-8")]
    check(len(rows) == 5, "5 worker rows")
    check([r["admission"]["run_ordinal"] for r in rows] == [0, 1, 2, 3, 4],
          "run ordinals 0..4")

    raw_hashes, admission_hashes = set(), set()
    for idx, row in enumerate(rows):
        check(row["raw_hash"] == payload_hash(row, "raw_hash"),
              f"row{idx} raw_hash")
        check(row["source_hash"] == chash(row["source"]), f"row{idx} source_hash")
        check(row["protocol_hash"] == chash(row["protocol"]),
              f"row{idx} protocol_hash")
        adm = row["admission"]
        check(adm["worker_payload_hash"] == payload_hash(adm, "worker_payload_hash"),
              f"row{idx} worker_payload_hash")
        rec = adm["receipt"]
        check(rec["admission_hash"] == payload_hash(rec, "admission_hash"),
              f"row{idx} admission_hash")
        check(rec["exact_call_identity_hash"] ==
              chash({"exact_call_id": f"asplos27-s4-formal:{idx:03d}"}),
              f"row{idx} exact_call hash")
        raw_hashes.add(row["raw_hash"])
        admission_hashes.add(rec["admission_hash"])

        # AC3 arithmetic from slots
        slots = rec["slots"]
        check(len(slots) == 6, f"row{idx} slot count")
        check([s["slot_ordinal"] for s in slots] == list(range(6)),
              f"row{idx} slot ordinals")
        paths = []
        for s in slots:
            paths += [s["alpha_semantic_path"], s["beta_semantic_path"]]
            check(s["alpha_active_element_count"] == prod(s["alpha_active_shape"]),
                  f"row{idx} slot{s['slot_ordinal']} active arith")
            check(s["alpha_preserved_element_count"]
                  == prod(s["alpha_preserved_shape"]),
                  f"row{idx} slot{s['slot_ordinal']} preserved arith")
            check(s["beta_element_count"] == prod(s["beta_source_shape"]),
                  f"row{idx} slot{s['slot_ordinal']} beta arith")
            check(s["beta_active"] == (s["beta_element_count"] > 0),
                  f"row{idx} slot{s['slot_ordinal']} beta active flag")
            check(s["alpha_live_is_leaf"] is True
                  and s["alpha_live_requires_grad"] is False
                  and s["beta_live_is_leaf"] is True
                  and s["beta_live_requires_grad"] is True,
                  f"row{idx} slot{s['slot_ordinal']} readiness")
        check(len(set(paths)) == 12, f"row{idx} 12 distinct paths")
        check(rec["mutable_path_set_hash"] == chash(sorted(paths)),
              f"row{idx} path set hash")

        stored = sum(prod(s["alpha_source_shape"]) for s in slots)
        active = sum(s["alpha_active_element_count"] for s in slots)
        preserved = sum(s["alpha_preserved_element_count"] for s in slots)
        beta_slots = len(slots)
        active_beta_slots = sum(1 for s in slots if s["beta_active"])
        active_beta_elems = sum(s["beta_element_count"] for s in slots)
        live_elems = stored + active_beta_elems
        mine = {
            "slot_count": 6, "path_count": 12, "alpha_source_count": 6,
            "alpha_stored_element_count": stored,
            "alpha_active_element_count": active,
            "alpha_preserved_element_count": preserved,
            "beta_slot_count": beta_slots,
            "active_beta_slot_count": active_beta_slots,
            "active_beta_element_count": active_beta_elems,
            "live_tensor_count": len(paths),
            "live_element_count_per_pass": live_elems,
            "live_bytes_per_pass": 4 * live_elems,
            "live_content_capture_pass_count":
                rec["live_content_capture_pass_count"],
            "device_to_host_validation_copy_count":
                rec["device_to_host_validation_copy_count"],
            "device_to_host_validation_bytes":
                rec["device_to_host_validation_bytes"],
        }
        expected = {
            "slot_count": 6, "path_count": 12, "alpha_source_count": 6,
            "alpha_stored_element_count": 8496,
            "alpha_active_element_count": 4248,
            "alpha_preserved_element_count": 4248,
            "beta_slot_count": 6, "active_beta_slot_count": 1,
            "active_beta_element_count": 6, "live_tensor_count": 12,
            "live_element_count_per_pass": 8502, "live_bytes_per_pass": 34008,
            "live_content_capture_pass_count": 2,
            "device_to_host_validation_copy_count": 24,
            "device_to_host_validation_bytes": 68016,
        }
        check(mine == expected, f"row{idx} headline arithmetic {mine}")
        for key, value in expected.items():
            if key in ("slot_count", "path_count"):
                continue  # derived-only fields, not stored in the receipt
            check(rec.get(key) == value, f"row{idx} receipt {key}")

        # AC4 lease + claim boundary
        check(adm["lease"] == {
            "state_before_close": "OPEN",
            "retained_tensor_count_before_close": 12,
            "state_after_close": "CLOSED",
            "retained_tensor_count_after_close": 0,
            "single_transfer_observed": False,
            "buffer_prepare_count": 0,
        }, f"row{idx} lease")
        check(adm["counters"] == {
            "provider_core_intercept_count": 1,
            "provider_core_execute_count": 0,
            "provider_compute_bounds_callback_count": 0,
            "provider_update_bounds_callback_count": 0,
            "candidate_kernel_launch_count": 0,
            "candidate_cuda_allocation_count": 0,
            "fallback_count": 0, "retry_count": 0, "mutation_count": 0,
        }, f"row{idx} counters")
        for flag in ("performance_claimed",):
            check(row[flag] is False and adm[flag] is False
                  and rec[flag] is False, f"row{idx} {flag}")
        check(rec["timing_recorded"] is False, f"row{idx} timing flag")
        check(rec["process_global_query_exclusivity_validated"] is False,
              f"row{idx} exclusivity flag")
        check(rec["dense_materialization_observed"] is False,
              f"row{idx} dense flag")

        # AC2 provider structure / projection binding
        prov = adm["provider"]
        check(prov["structure"]["alpha_data"] == "collections.defaultdict",
              f"row{idx} alpha container")
        check(prov["structure"]["beta_data"] == "builtins.dict",
              f"row{idx} beta container")
        check(len(prov["structure"]["slots"]) == 6, f"row{idx} structure slots")
        for srow in prov["structure"]["slots"]:
            check(srow == {
                "alpha_nested": "builtins.dict",
                "alpha_tensor": "torch.Tensor",
                "beta_collection": "builtins.list",
                "beta_entry": "auto_LiRPA.beta_crown.SparseBeta",
                "beta_tensor": "torch.Tensor",
            }, f"row{idx} nested structure")
        proj = prov["live_projection"]
        check(len(proj) == 12, f"row{idx} projection size")
        by_path = {}
        for s in slots:
            by_path[s["alpha_semantic_path"]] = (
                s["alpha_source_shape"], s["alpha_source_dtype"],
                s["alpha_source_device"], s["alpha_live_stride"],
                s["alpha_live_storage_offset"], s["alpha_live_version"],
                s["alpha_live_requires_grad"], s["alpha_live_is_leaf"],
                s["alpha_live_content_hash"])
            by_path[s["beta_semantic_path"]] = (
                s["beta_source_shape"], s["beta_source_dtype"],
                s["beta_source_device"], s["beta_live_stride"],
                s["beta_live_storage_offset"], s["beta_live_version"],
                s["beta_live_requires_grad"], s["beta_live_is_leaf"],
                s["beta_live_content_hash"])
        for prow in proj:
            observed = (prow["shape"], prow["dtype"], prow["device"],
                        prow["stride"], prow["storage_offset"], prow["version"],
                        prow["requires_grad"], prow["is_leaf"],
                        prow["content_hash"])
            check(prow["python_type"] == "torch.Tensor",
                  f"row{idx} projection tensor type")
            check(by_path.get(prow["semantic_path"]) == observed,
                  f"row{idx} projection/receipt binding {prow['semantic_path']}")

    check(len(raw_hashes) == 5 and len(admission_hashes) == 5,
          "5 distinct raw/admission hashes")
    # cross-row identity: deterministic fields identical; per-process capture
    # hashes (plan/snapshot/oracle/plan-binding) legitimately differ per fresh
    # process via snapshot tensor metadata/history content
    first = rows[0]["admission"]["receipt"]
    for row in rows[1:]:
        r = row["admission"]["receipt"]
        for key in ("slots", "topology_hash", "construction_model_hash",
                    "optimizer_policy_hash", "mutable_path_set_hash"):
            check(r[key] == first[key], f"cross-row identity {key}")
        check(row["source"] == rows[0]["source"], "cross-row source")
        check(row["protocol"] == rows[0]["protocol"], "cross-row protocol")
    for key in ("production_plan_hash", "snapshot_hash",
                "oracle_mapping_provenance_hash",
                "plan_binding_projection_hash"):
        distinct = {r["admission"]["receipt"][key] for r in rows}
        check(len(distinct) == 5, f"per-process capture hash {key} distinct=5")

    # AC5 negative registry
    reg = json.load(open(ART + "/negative_registry.json", encoding="utf-8"))
    cases = reg["cases"]
    nodeids = [c["nodeid"] for c in cases]
    check(reg["schema_version"]
          == "boundflow.asplos27-s4-admission-negative-registry/v1",
          "registry schema")
    check(len(cases) == reg["case_count"] == 63, "registry count 63")
    check(len(set(nodeids)) == 63, "registry unique nodeids")
    check(len(cases) >= reg["minimum_required"] == 56, "registry minimum 56")
    check(all(c["fresh_pytest_case"] is True
              and c["exact_detail_and_reason_asserted"] is True for c in cases),
          "registry flags")
    check(reg["targeted_result"] == "pass", "registry targeted result")
    check(all(c["ordinal"] == i for i, c in enumerate(cases)),
          "registry ordinals")
    # registry nodes must exist in the test file
    test_src = open("tests/test_asplos27_s4_mutable_state_admission.py",
                    encoding="utf-8").read()
    missing = [n for n in nodeids
               if n.split("::", 1)[1].split("[", 1)[0] not in test_src]
    check(not missing, f"registry nodeids exist in test file: {missing[:3]}")
    # negative log shows all-pass
    log = open(ART + "/logs/negative-pytest.txt", encoding="utf-8").read()
    check("63 passed" in log, "negative log 63 passed")
    check("failed" not in log.lower(), "negative log no failure")
    print("negative log tail:", log.strip().splitlines()[-1])

    # protocol / summary hash chain
    protocol = json.load(open(ART + "/protocol.json", encoding="utf-8"))
    check(protocol["protocol_hash"] == payload_hash(protocol, "protocol_hash"),
          "protocol self hash")
    check(protocol["workers_jsonl_sha256"] == hashlib.sha256(
        open(ART + "/raw/workers.jsonl", "rb").read()).hexdigest(),
        "protocol raw binding")
    check(protocol["negative_registry_sha256"] == hashlib.sha256(
        open(ART + "/negative_registry.json", "rb").read()).hexdigest(),
        "protocol registry binding")
    check(protocol["source_hash"] == chash(rows[0]["source"]),
          "protocol source hash")
    check(protocol["worker_protocol_hash"] == chash(rows[0]["protocol"]),
          "protocol worker protocol hash")

    summary = json.load(open(ART + "/summary.json", encoding="utf-8"))
    check(summary["summary_hash"] == payload_hash(summary, "summary_hash"),
          "summary self hash")
    check(summary["status"] == "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-0",
          "summary status")
    check(summary["formal_counts"] == expected, "summary formal counts")
    check(summary["fresh_process_count"] == 5
          and summary["run_ordinals"] == [0, 1, 2, 3, 4], "summary inventory")
    check(summary["distinct_raw_hash_count"] == 5
          and summary["distinct_admission_hash_count"] == 5,
          "summary distinct hashes")
    check(summary["negative_case_count"] == 63
          and summary["negative_case_minimum"] == 56, "summary negative counts")
    for key in ("candidate_kernel_launch_count", "candidate_cuda_allocation_count",
                "provider_bound_callback_count", "buffer_prepare_count",
                "mutation_count"):
        check(summary[key] == 0, f"summary {key}=0")
    for flag in ("timing_recorded", "performance_claimed",
                 "process_global_query_exclusivity_validated"):
        check(summary[flag] is False, f"summary {flag} false")

    manifest = json.load(open(ART + "/manifest.json", encoding="utf-8"))
    check(manifest["manifest_hash"] == payload_hash(manifest, "manifest_hash"),
          "manifest self hash")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURES")
        sys.exit(1)
    print("RESULT: S4-0 INDEPENDENT RECOMPUTE PASSED")


if __name__ == "__main__":
    main()
