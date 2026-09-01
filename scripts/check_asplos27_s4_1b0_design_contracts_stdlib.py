#!/usr/bin/env python3
"""Validate the frozen S4-1B0 design contracts without project imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FIXTURE_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IEEE_BIT_FIXTURES_V1_2026_08_30.json"
)
NEGATIVE_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json"
)
ABI_PATH = Path("gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_ABI_CONTRACT_V1_2026_08_30.json")
FORMAL_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_FORMAL_ARTIFACT_CONTRACT_V1_2026_08_30.json"
)
CONSTRUCTION_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md"
)

EXPECTED_SCHEMAS = {
    FIXTURE_PATH: "boundflow.asplos27-s4-1b0-ieee-fixtures/v1",
    NEGATIVE_PATH: "boundflow.asplos27-s4-1b0-negative-contract/v1",
    ABI_PATH: "boundflow.asplos27-s4-1b0-abi-contract/v1",
    FORMAL_PATH: "boundflow.asplos27-s4-1b0-formal-artifact-contract/v1",
}

CODE_CLOSED_PATHS = (
    "boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py",
    "tests/test_asplos27_s4_ternary_endpoint.py",
    "scripts/run_asplos27_s4_1b0_ternary_worker.py",
    "scripts/run_asplos27_s4_1b0_ternary_artifact.py",
    "scripts/replay_asplos27_s4_1b0_ternary_stdlib.py",
    "scripts/probe_asplos27_s4_1b0_ternary_tamper.py",
    "tests/test_asplos27_s4_1b0_ternary_artifact.py",
    "artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1",
)


class ContractFailure(Exception):
    """Raised after collecting one or more deterministic contract failures."""


class Checks:
    """Collect deterministic checks so one run reports every drift at once."""

    def __init__(self) -> None:
        self.count = 0
        self.failures: list[str] = []

    def require(self, condition: bool, label: str) -> None:
        """Record one Boolean invariant and retain its stable failure label."""
        self.count += 1
        if not condition:
            self.failures.append(label)

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        """Record one equality invariant with both values on failure."""
        self.require(
            actual == expected, f"{label}: actual={actual!r} expected={expected!r}"
        )

    def finish(self) -> None:
        """Raise one aggregate failure after all independent checks run."""
        if self.failures:
            raise ContractFailure("\n".join(self.failures))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: top-level JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_construction_model(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    section_start = text.index("## 12. 可重算construction model")
    section = text[section_start:]
    match = re.search(r"```json\n(?P<model>\{.*?\})\n```", section, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"{path}: section 12 canonical JSON block is missing")
    model = json.loads(match.group("model"))
    if not isinstance(model, dict):
        raise TypeError(f"{path}: construction model must be an object")
    hash_block = re.search(r"SHA256：\s*```text\s*([0-9a-f]{64})\s*```", section)
    if hash_block is None:
        raise ValueError(f"{path}: section 12 SHA256 block is missing")
    return model, hash_block.group(1)


def _u32(bits: str) -> int:
    value = int(bits, 16)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"not a uint32 bit pattern: {bits}")
    return value


def _f32_from_bits(bits: str) -> float:
    return struct.unpack(">f", _u32(bits).to_bytes(4, "big"))[0]


def _bits_from_f32(value: float) -> str:
    try:
        raw = struct.pack(">f", value)
    except OverflowError:
        raw = struct.pack(
            ">I", 0xFF800000 if math.copysign(1.0, value) < 0 else 0x7F800000
        )
    return f"0x{int.from_bytes(raw, 'big'):08x}"


def _round_f32(value: float) -> float:
    return _f32_from_bits(_bits_from_f32(value))


def _selector_for_bits(bits: str) -> int:
    raw = _u32(bits)
    if raw & 0x7F800000 == 0x7F800000:
        return -128
    if raw & 0x7FFFFFFF == 0:
        return 0
    return -1 if raw & 0x80000000 else 1


def _selected_bits(selector: int, lower_bits: str, upper_bits: str) -> str:
    if selector == 1:
        return lower_bits.lower()
    if selector == -1:
        return upper_bits.lower()
    if selector == 0:
        lower = _f32_from_bits(lower_bits)
        upper = _f32_from_bits(upper_bits)
        return _bits_from_f32(_round_f32(lower + upper) * 0.5)
    return "0x7fc00000"


def _all_false(values: Mapping[str, Any], names: Iterable[str]) -> bool:
    return all(values.get(name) is False for name in names)


def _asset_map(items: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(item["path"]): str(item["sha256"]) for item in items}


def _validate_dependency_assets(
    checks: Checks,
    root: Path,
    owner: str,
    items: Sequence[Mapping[str, Any]],
    expected_paths: Sequence[Path],
) -> None:
    assets = _asset_map(items)
    checks.equal(
        sorted(assets),
        sorted(str(path) for path in expected_paths),
        f"{owner}.dependency paths",
    )
    for path in expected_paths:
        checks.equal(
            assets.get(str(path)),
            _sha256(root / path),
            f"{owner}.dependency sha256 {path}",
        )


def _validate_fixture(checks: Checks, fixture: Mapping[str, Any]) -> None:
    policies = fixture["policies"]
    pack_cases = fixture["pack_cases"]
    select_cases = fixture["select_cases"]
    observation = fixture["observation"]
    inventory = fixture["future_production_fixture_inventory"]

    checks.equal(len(pack_cases), 16, "fixture pack case count")
    checks.equal(len(select_cases), 16, "fixture select case count")
    checks.equal(
        observation["pack_case_count"], len(pack_cases), "fixture observed pack count"
    )
    checks.equal(
        observation["select_case_count"],
        len(select_cases),
        "fixture observed select count",
    )
    checks.equal(
        len({case["name"] for case in pack_cases}),
        len(pack_cases),
        "fixture unique pack names",
    )
    checks.equal(
        len({case["name"] for case in select_cases}),
        len(select_cases),
        "fixture unique select names",
    )
    for case in pack_cases:
        checks.equal(
            case["expected_selector"],
            _selector_for_bits(case["input_bits"]),
            f"fixture pack semantics {case['name']}",
        )
    for case in select_cases:
        checks.equal(
            case["expected_bits"].lower(),
            _selected_bits(case["selector"], case["lower_bits"], case["upper_bits"]),
            f"fixture select semantics {case['name']}",
        )

    counterexamples = fixture["midpoint_reassociation_counterexamples"]
    checks.equal(len(counterexamples), 2, "fixture midpoint counterexample count")
    for case in counterexamples:
        value = _f32_from_bits(case["input_bits"])
        add_then_mul = _bits_from_f32(_round_f32(value + value) * 0.5)
        mul_then_add = _bits_from_f32(_round_f32(value * 0.5) + _round_f32(value * 0.5))
        checks.equal(
            case["add_then_mul_bits"],
            add_then_mul,
            f"fixture add-then-mul {case['name']}",
        )
        checks.equal(
            case["mul_then_add_bits"],
            mul_then_add,
            f"fixture mul-then-add {case['name']}",
        )
        checks.require(
            add_then_mul != mul_then_add,
            f"fixture reassociation must differ {case['name']}",
        )

    checks.equal(
        inventory["positive"]
        + inventory["negative"]
        + inventory["zero"]
        + inventory["invalid"],
        inventory["numel"],
        "fixture production inventory sum",
    )
    checks.equal(
        inventory["old_binary_zero_misclassified"],
        inventory["zero"],
        "fixture binary zero mismatch count",
    )
    checks.equal(policies["value_dtype"], "float32", "fixture value dtype")
    checks.equal(policies["selector_dtype"], "int8", "fixture selector dtype")
    checks.equal(
        policies["invalid_output_policy"],
        "canonical-qnan-0x7fc00000-v1",
        "fixture qnan policy",
    )
    checks.require(
        _all_false(
            fixture["claims"],
            (
                "implementation_open",
                "production_correctness",
                "formal_validated",
                "timing",
                "performance",
            ),
        ),
        "fixture claims remain closed",
    )


def _validate_negative(checks: Checks, negative: Mapping[str, Any]) -> None:
    reasons = negative["stable_reasons"]
    ordinals = [item["ordinal"] for item in reasons]
    reason_names = [item["reason"] for item in reasons]
    checks.equal(ordinals, list(range(1, 21)), "negative reason ordinals")
    checks.equal(
        len(reason_names), len(set(reason_names)), "negative unique reason strings"
    )
    checks.equal(len(negative["test_layout"]), 16, "negative test layout count")
    checks.equal(len(set(negative["test_layout"])), 16, "negative unique test names")
    topology = negative["formal_topology"]
    checks.equal(
        topology["positive_workers"]
        + topology["cache_workers"]
        + topology["fault_workers"],
        topology["total_fresh_processes"],
        "negative formal topology sum",
    )
    checks.require(topology["raw_first"] is True, "negative raw-first")
    checks.require(
        topology["partial_resume_rejected"] is True, "negative partial resume rejected"
    )
    scope = negative["scope"]
    checks.require(
        scope["isolated_backend"] is True and scope["prepared_probe"] is True,
        "negative isolated scope",
    )
    checks.require(
        _all_false(
            scope,
            (
                "production_evaluator_binding",
                "s4_1a_ticket_consumption",
                "optimizer",
                "timing",
                "performance",
            ),
        ),
        "negative out-of-scope features remain closed",
    )


def _validate_abi(
    checks: Checks,
    fixture: Mapping[str, Any],
    negative: Mapping[str, Any],
    abi: Mapping[str, Any],
) -> None:
    policies = fixture["policies"]
    fixed = abi["build_spec"]["fixed_values"]
    for name in (
        "value_dtype",
        "selector_dtype",
        "endpoint_policy",
        "midpoint_policy",
        "nonfinite_policy",
        "invalid_output_policy",
    ):
        checks.equal(fixed[name], policies[name], f"ABI fixed policy {name}")
    checks.equal(abi["constants"]["qnan_bits"], "0x7fc00000", "ABI qnan bits")
    checks.equal(
        abi["constants"]["default_threads"],
        abi["schedule_spec"]["fixed_values"]["threads_per_block"],
        "ABI threads",
    )
    checks.require(
        abi["tir_builder"]["new_top_level_ir"] is False,
        "ABI introduces no top-level IR",
    )

    negative_cache = negative["cache_contract"]
    abi_cache = abi["cache"]
    checks.equal(
        abi_cache["lookup_key_fields"],
        negative_cache["lookup_key_inputs"],
        "ABI/negative cache key",
    )
    checks.equal(
        abi_cache["first_lookup_excludes"],
        negative_cache["lookup_key_excludes"],
        "ABI/negative cache excludes",
    )
    checks.equal(
        abi_cache["failure_inserts_partial_entry"],
        negative_cache["failed_compile_inserts_entry"],
        "ABI cache failure publish",
    )
    checks.equal(
        abi_cache["failure_fallback"],
        negative_cache["fallback_on_failure"],
        "ABI cache fallback",
    )
    checks.require(
        "compiled_device_source_hash" in abi_cache["hit_revalidates"],
        "ABI cache hit rehashes source",
    )

    prepared = abi["prepared_probe"]
    roles = prepared["caller_owned_tensor_roles"]
    checks.equal(
        len(roles), prepared["unique_tensor_count"], "ABI prepared unique tensor count"
    )
    checks.equal(
        prepared["prepare_dlpack_view_count"],
        prepared["unique_tensor_count"],
        "ABI prepare DLPack count",
    )
    checks.equal(
        [item["role"] for item in roles],
        ["coefficient", "lower", "upper", "selector", "selected"],
        "ABI role order",
    )
    checks.equal(prepared["argument_occurrence_count"], 6, "ABI argument occurrences")
    checks.equal(prepared["center_tensor_count"], 0, "ABI no center tensor")
    checks.equal(prepared["status_tensor_count"], 0, "ABI no status tensor")
    checks.require(
        prepared["caller_provided_device_or_stream_identity"] is False,
        "ABI derives device/stream",
    )
    checks.require(
        prepared["holds_s4_1a_ticket_or_lease"] is False,
        "ABI isolated from S4-1A ticket",
    )

    inventory = fixture["future_production_fixture_inventory"]
    storage = abi["storage"]
    checks.equal(
        storage["production_fixture_numel"], inventory["numel"], "ABI fixture numel"
    )
    checks.equal(
        storage["production_fixture_selector_bytes"],
        inventory["numel"],
        "ABI selector bytes",
    )
    checks.equal(
        storage["production_fixture_selected_bytes"],
        4 * inventory["numel"],
        "ABI selected bytes",
    )
    checks.equal(
        storage["production_fixture_output_bytes"],
        storage["production_fixture_selector_bytes"]
        + storage["production_fixture_selected_bytes"],
        "ABI isolated output bytes",
    )
    checks.equal(storage["global_workspace_bytes"], 0, "ABI global workspace")
    checks.require(
        storage["production_alias_claimed"] is False, "ABI no production alias claim"
    )
    checks.equal(
        abi["warm_launch_receipt"]["fixed_values"]["argument_occurrence_count"],
        prepared["argument_occurrence_count"],
        "ABI warm/prepare argument occurrences",
    )
    checks.require(
        _all_false(
            abi["claims"],
            (
                "implementation_open",
                "production_correctness",
                "formal_validated",
                "timing",
                "performance",
            ),
        ),
        "ABI claims remain closed",
    )


# This routine deliberately keeps the four contracts visible side by side. Splitting
# its locals would make the cross-contract ownership checks harder to audit.
# pylint: disable=too-many-locals
def _validate_formal(
    checks: Checks,
    fixture: Mapping[str, Any],
    negative: Mapping[str, Any],
    abi: Mapping[str, Any],
    formal: Mapping[str, Any],
) -> None:
    negative_topology = negative["formal_topology"]
    topology = formal["worker_topology"]
    for name in (
        "positive_workers",
        "cache_workers",
        "fault_workers",
        "total_fresh_processes",
        "raw_first",
        "partial_resume_rejected",
    ):
        checks.equal(
            topology[name], negative_topology[name], f"formal/negative topology {name}"
        )
    sequence = topology["worker_sequence"]
    checks.equal(
        len(sequence),
        topology["total_fresh_processes"],
        "formal worker sequence length",
    )
    checks.equal(len(set(sequence)), len(sequence), "formal unique worker names")
    checks.equal(
        sum(name.startswith("positive-") for name in sequence),
        topology["positive_workers"],
        "formal positive workers",
    )
    checks.equal(
        sum(name.startswith("cache-") for name in sequence),
        topology["cache_workers"],
        "formal cache workers",
    )
    checks.equal(
        sum(name.startswith("fault-") for name in sequence),
        topology["fault_workers"],
        "formal fault workers",
    )

    reasons = negative["stable_reasons"]
    all_ordinals = {item["ordinal"] for item in reasons}
    partitions = [set(worker["reason_ordinals"]) for worker in formal["fault_workers"]]
    checks.equal(
        set().union(*partitions), all_ordinals, "formal fault ordinal coverage"
    )
    checks.equal(
        sum(len(partition) for partition in partitions),
        len(all_ordinals),
        "formal fault partitions disjoint",
    )
    for worker, partition in zip(formal["fault_workers"], partitions):
        checks.require(
            worker["representative_reason_ordinal"] in partition,
            f"formal representative reason {worker['name']}",
        )
        checks.require(
            f"fault-{worker['name']}" in sequence,
            f"formal fault worker sequence {worker['name']}",
        )
    checks.equal(
        formal["negative_registry_requirements"]["reason_count"],
        len(reasons),
        "formal negative reason count",
    )

    inventory = fixture["future_production_fixture_inventory"]
    sidecar = formal["positive_binary_sidecar"]
    checks.equal(sidecar["numel"], inventory["numel"], "formal sidecar numel")
    dtype_width = {"float32": 4, "int8": 1}
    for record in sidecar["records"]:
        checks.equal(
            record["byte_count"],
            sidecar["numel"] * dtype_width[record["dtype"]],
            f"formal record bytes {record['name']}",
        )
    checks.equal(
        sidecar["sidecar_byte_count"],
        sum(record["byte_count"] for record in sidecar["records"]),
        "formal sidecar total bytes",
    )
    records = {record["name"]: record for record in sidecar["records"]}
    checks.equal(
        records["selector"]["byte_count"],
        abi["storage"]["production_fixture_selector_bytes"],
        "formal/ABI selector bytes",
    )
    checks.equal(
        records["selected"]["byte_count"],
        abi["storage"]["production_fixture_selected_bytes"],
        "formal/ABI selected bytes",
    )
    expected_classes = {
        name: inventory[name] for name in ("positive", "negative", "zero", "invalid")
    }
    requirements = formal["positive_worker_requirements"]
    checks.equal(
        requirements["selector_class_counts"],
        expected_classes,
        "formal selector class counts",
    )
    checks.equal(
        requirements["old_binary_zero_misclassified"],
        inventory["old_binary_zero_misclassified"],
        "formal binary zero count",
    )
    checks.equal(
        requirements["dlpack_pointer_exact"],
        abi["prepared_probe"]["prepare_dlpack_view_count"],
        "formal DLPack pointers",
    )
    for name in (
        "pack_launch_count",
        "select_launch_count",
        "fallback_count",
        "eager_count",
        "native_shadow_count",
        "timing_recorded",
        "performance_claimed",
    ):
        checks.equal(
            requirements[name],
            abi["warm_launch_receipt"]["fixed_values"][name],
            f"formal/ABI warm receipt {name}",
        )

    required = formal["required_files"]
    checks.equal(len(required), len(set(required)), "formal unique required files")
    checks.require(
        all(not Path(path).is_absolute() for path in required),
        "formal required paths are relative",
    )
    binary_files = [
        path
        for path in required
        if path.startswith("raw/binary/positive-") and path.endswith(".bin")
    ]
    checks.equal(
        len(binary_files), topology["positive_workers"], "formal positive sidecar files"
    )
    checks.equal(
        len(formal["outer_resigned_tamper_cases"]),
        formal["tamper_requirements"]["case_count"],
        "formal tamper case count",
    )
    checks.equal(
        formal["tamper_requirements"]["expected_rejected"],
        formal["tamper_requirements"]["case_count"],
        "formal tamper expected rejection",
    )
    checks.require(
        formal["tamper_requirements"]["coherent_full_resign_e0_boundary_disclosed"]
        is True,
        "formal E0 boundary disclosed",
    )
    checks.require(
        formal["stdlib_replay"]["does_not_claim_hardware_authenticity"] is True,
        "formal replay authenticity boundary",
    )
    checks.require(
        _all_false(
            formal["claims"],
            (
                "implementation_open",
                "production_correctness",
                "formal_validated",
                "timing",
                "performance",
                "same_solver",
                "complete_query",
                "tenx",
                "asplos_ready",
            ),
        ),
        "formal claims remain closed",
    )


# The source model and four derived contracts are intentionally explicit here.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def _validate_construction_model(
    checks: Checks,
    model: Mapping[str, Any],
    fixture: Mapping[str, Any],
    negative: Mapping[str, Any],
    abi: Mapping[str, Any],
    formal: Mapping[str, Any],
) -> None:
    """Bind the authoritative markdown model to every machine contract."""
    checks.equal(model["backend_file"], negative["backend_file"], "model backend path")
    checks.equal(model["test_file"], negative["test_file"], "model test path")
    checks.equal(
        model["negative_reason_count"],
        len(negative["stable_reasons"]),
        "model reason count",
    )
    checks.equal(
        model["symbols"],
        [abi["constants"]["pack_symbol"], abi["constants"]["select_symbol"]],
        "model symbols",
    )
    checks.equal(model["threads"], abi["constants"]["default_threads"], "model threads")

    cache = model["cache"]
    lookup_fields = abi["cache"]["lookup_key_fields"]
    excludes = abi["cache"]["first_lookup_excludes"]
    checks.equal(
        "device_source_hash" in lookup_fields,
        cache["device_source_in_lookup_key"],
        "model device source lookup",
    )
    checks.require(
        "device_source_hash" in excludes, "model device source explicitly excluded"
    )
    checks.equal(
        negative["cache_contract"]["hit_rehashes_cached_device_source"],
        cache["hit_rehashes_cached_source"],
        "model cache source rehash",
    )
    checks.equal(
        abi["module_receipt"]["mutable_cache_counts_in_receipt"],
        cache["mutable_counts_in_module_receipt"],
        "model immutable module receipt",
    )
    checks.equal(
        all(
            name in lookup_fields
            for name in ("unscheduled_tir_hash", "scheduled_tir_hash")
        ),
        cache["precompile_tir_hashes_in_lookup_key"],
        "model precompile TIR lookup hashes",
    )

    model_claims = model["claims"]
    checks.equal(
        fixture["claims"]["implementation_open"],
        model_claims["implementation"],
        "model fixture implementation claim",
    )
    checks.equal(
        abi["claims"]["implementation_open"],
        model_claims["implementation"],
        "model ABI implementation claim",
    )
    checks.equal(
        formal["claims"]["implementation_open"],
        model_claims["implementation"],
        "model formal implementation claim",
    )
    checks.equal(
        abi["claims"]["performance"],
        model_claims["performance"],
        "model performance claim",
    )
    checks.equal(
        abi["claims"]["production_correctness"],
        model_claims["production_correctness"],
        "model correctness claim",
    )
    checks.equal(
        abi["storage"]["production_alias_claimed"],
        model_claims["production_alias"],
        "model alias claim",
    )

    model_formal = model["formal"]
    for name in ("positive_workers", "cache_workers", "fault_workers"):
        checks.equal(
            formal["worker_topology"][name], model_formal[name], f"model formal {name}"
        )
    checks.equal(
        negative["formal_topology"]["external_audit_required_for_validated"],
        model_formal["status_requires_external_audit"],
        "model external audit gate",
    )

    model_math = model["math"]
    checks.equal(
        model_math["invalid_output_bits"],
        abi["constants"]["qnan_bits"],
        "model invalid output bits",
    )
    checks.equal(
        model_math["midpoint_policy"],
        fixture["policies"]["midpoint_policy"],
        "model midpoint policy",
    )
    checks.equal(
        model_math["nonfinite_mask"],
        abi["constants"]["nonfinite_mask"],
        "model nonfinite mask",
    )
    selector_values = sorted(
        {case["expected_selector"] for case in fixture["pack_cases"]}
    )
    checks.equal(
        model_math["selector_values"], selector_values, "model selector values"
    )

    production = model["production"]
    checks.require(
        production["selected_output_alias_requires_s4_1b_phase_proof"] is True,
        "model phase alias proof required",
    )
    checks.equal(
        abi["warm_launch_receipt"]["synchronized_content_counts_allowed"],
        production["warm_count_sync"],
        "model warm count sync",
    )
    checks.equal(
        "bit_preserving_content_hashes"
        in abi["formal_observation"]["allowed_after_explicit_synchronize"],
        not production["warm_content_hash"],
        "model content hash deferred to formal",
    )

    scope = model["scope"]
    checks.equal(
        scope["backend_compile"],
        negative["scope"]["isolated_backend"],
        "model backend compile scope",
    )
    checks.equal(
        scope["evaluator_binding"],
        negative["scope"]["production_evaluator_binding"],
        "model evaluator scope",
    )
    checks.equal(
        scope["new_ir"], abi["tir_builder"]["new_top_level_ir"], "model new IR scope"
    )
    checks.equal(
        scope["prepared_probe"],
        negative["scope"]["prepared_probe"],
        "model prepared probe scope",
    )
    checks.equal(scope["timing"], abi["claims"]["timing"], "model timing scope")

    storage = model["storage"]
    checks.equal(
        storage["isolated_dlpack_views"],
        abi["prepared_probe"]["prepare_dlpack_view_count"],
        "model isolated views",
    )
    checks.equal(
        storage["isolated_output_allocated_bytes"],
        abi["storage"]["production_fixture_output_bytes"],
        "model output bytes",
    )
    checks.equal(
        storage["selected_output_bytes"],
        abi["storage"]["production_fixture_selected_bytes"],
        "model selected bytes",
    )
    checks.equal(
        storage["selector_bytes"],
        abi["storage"]["production_fixture_selector_bytes"],
        "model selector bytes",
    )
    checks.equal(storage["s4_1a_base_view_overlap"], 0, "model S4-1A view overlap")


def validate(root: Path, require_code_closed: bool) -> dict[str, Any]:
    """Validate every frozen asset and return a compact deterministic receipt."""
    checks = Checks()
    documents = {path: _load_json(root / path) for path in EXPECTED_SCHEMAS}
    for path, schema in EXPECTED_SCHEMAS.items():
        checks.equal(documents[path].get("schema_version"), schema, f"schema {path}")

    fixture = documents[FIXTURE_PATH]
    negative = documents[NEGATIVE_PATH]
    abi = documents[ABI_PATH]
    formal = documents[FORMAL_PATH]
    construction_model, documented_model_hash = _extract_construction_model(
        root / CONSTRUCTION_PATH
    )
    computed_model_hash = _canonical_hash(construction_model)
    model_hashes = {
        document["source_identity"]["construction_model_hash"]
        for document in documents.values()
    }
    checks.equal(len(model_hashes), 1, "shared construction model hash")
    checks.equal(
        computed_model_hash,
        documented_model_hash,
        "construction model documented SHA256",
    )
    checks.equal(
        model_hashes, {computed_model_hash}, "construction model JSON contract hashes"
    )
    _validate_dependency_assets(
        checks, root, "ABI", abi["dependency_assets"], (FIXTURE_PATH, NEGATIVE_PATH)
    )
    _validate_dependency_assets(
        checks,
        root,
        "formal",
        formal["dependency_assets"],
        (FIXTURE_PATH, NEGATIVE_PATH, ABI_PATH),
    )
    _validate_fixture(checks, fixture)
    _validate_negative(checks, negative)
    _validate_abi(checks, fixture, negative, abi)
    _validate_formal(checks, fixture, negative, abi, formal)
    _validate_construction_model(
        checks, construction_model, fixture, negative, abi, formal
    )
    if require_code_closed:
        for relative in CODE_CLOSED_PATHS:
            checks.require(
                not (root / relative).exists(),
                f"code/formal gate remains closed: {relative}",
            )
    checks.finish()
    return {
        "status": "PASS",
        "schema": "boundflow.asplos27-s4-1b0-design-contract-check/v1",
        "check_count": checks.count,
        "construction_model_hash": next(iter(model_hashes)),
        "construction_package_sha256": _sha256(root / CONSTRUCTION_PATH),
        "asset_sha256": {str(path): _sha256(root / path) for path in EXPECTED_SCHEMAS},
        "reason_count": len(negative["stable_reasons"]),
        "test_layout_count": len(negative["test_layout"]),
        "fresh_process_count": formal["worker_topology"]["total_fresh_processes"],
        "positive_sidecar_bytes": formal["positive_binary_sidecar"][
            "sidecar_byte_count"
        ],
        "code_closed_checked": require_code_closed,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--require-code-closed",
        action="store_true",
        help="also require every future production/formal path to be absent",
    )
    return parser.parse_args()


def main() -> int:
    """Run the checker and emit exactly one canonical JSON result."""
    args = _parse_args()
    try:
        result = validate(args.repo_root.resolve(), args.require_code_closed)
    except (
        ContractFailure,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
