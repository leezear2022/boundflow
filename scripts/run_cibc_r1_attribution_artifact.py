#!/usr/bin/env python3
"""Generate or replay CIBC R1-A additive attribution evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,wrong-import-position

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.cibc_r1_attribution import (
    R1TargetContract,
    canonical_hash,
    canonical_json,
    clock_calibration_from_dict,
    owner_ledger_from_dict,
    target_contract_from_dict,
    timing_ledger_from_dict,
    topology_ledger_from_dict,
)
from boundflow.runtime.cibc_r1_nsys import (
    derive_nsys_attribution,
    nsys_export_receipt_from_dict,
)
from scripts.run_cibc_r1_attribution_worker import (
    CONTROL_GROUP_COUNT,
    CONTROL_REPEATS,
    CONTROL_WARMUP,
    PROFILE_GROUP_COUNT,
    PROFILE_REPEATS,
    PROFILE_WARMUP,
    SEMANTIC_ATOL,
    THREADS_PER_BLOCK,
    WORKER_SCHEMA,
)

PROTOCOL_SCHEMA = "boundflow.cibc-r1-attribution-protocol/v1"
SUMMARY_SCHEMA = "boundflow.cibc-r1-attribution-summary/v1"
MANIFEST_SCHEMA = "boundflow.cibc-r1-attribution-manifest/v1"
PAIR_ORDERS = ("CP", "PC", "CP", "PC", "CP", "PC")
PROFILE_PERTURBATION_MIN = 0.95
PROFILE_PERTURBATION_MAX = 1.05
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PRODUCTION_TOPOLOGY_HASH = (
    "ee3ec3fba1b0deb68e54cb5d0315fb9072def4f8bdf5ff310609e3ea2248d693"
)
EXPECTED_SEMANTIC_MAXIMUM = 0.000244140625
EXPECTED_SEMANTIC_ELEMENT_COUNT = 235_992
ALLOWED_FORMAL_DIRTY_PATHS = (".docops/ev.jsonl",)
CODE_PATHS = (
    "boundflow/runtime/cibc_ibp_graph.py",
    "boundflow/runtime/cibc_r1_attribution.py",
    "boundflow/runtime/cibc_r1_nsys.py",
    "scripts/run_cibc_r1_attribution_worker.py",
    "scripts/run_cibc_r1_attribution_artifact.py",
    "scripts/probe_cibc_r1_attribution_tamper.py",
    "tests/test_cibc_ibp_graph.py",
    "tests/test_cibc_r1_attribution.py",
    "tests/test_cibc_r1_attribution_runner.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R1-A artifact JSON root differs")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _tracked_dirty_paths() -> tuple[str, ...]:
    rows = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    return tuple(sorted(row[3:] for row in rows if len(row) >= 4))


def _historical_code_revision(source: str) -> dict[str, str]:
    result = {}
    for path in CODE_PATHS:
        content = subprocess.run(
            ("git", "show", f"{source}:{path}"),
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        result[path] = hashlib.sha256(content).hexdigest()
    return result


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def protocol(source_capture: Path, model: Path, *, smoke: bool) -> dict[str, object]:
    if file_sha256(source_capture) != SOURCE_CAPTURE_SHA256:
        raise ValueError("R1-A source capture digest differs")
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("R1-A model digest differs")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "source_clean": set(_tracked_dirty_paths()).issubset(
            ALLOWED_FORMAL_DIRTY_PATHS
        ),
        "allowed_formal_dirty_paths": list(ALLOWED_FORMAL_DIRTY_PATHS),
        "code_revision": _code_revision(),
        "source_capture_sha256": file_sha256(source_capture),
        "model_sha256": file_sha256(model),
        "target_contract": R1TargetContract().to_dict(),
        "production_topology_hash": PRODUCTION_TOPOLOGY_HASH,
        "run_kind": "smoke" if smoke else "formal",
        "pair_orders": list(PAIR_ORDERS[:1] if smoke else PAIR_ORDERS),
        "control": {
            "warmup": CONTROL_WARMUP,
            "group_count": CONTROL_GROUP_COUNT,
            "repeats": CONTROL_REPEATS,
        },
        "profile": {
            "warmup": PROFILE_WARMUP,
            "group_count": PROFILE_GROUP_COUNT,
            "repeats": PROFILE_REPEATS,
        },
        "profile_perturbation_range": [
            PROFILE_PERTURBATION_MIN,
            PROFILE_PERTURBATION_MAX,
        ],
        "threads_per_block": THREADS_PER_BLOCK,
        "nsys_required_for_formal": True,
        "torch_profiler_is_smoke_only": True,
        "performance_claimed": False,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def validate_protocol(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "source_git_head",
        "source_clean",
        "allowed_formal_dirty_paths",
        "code_revision",
        "source_capture_sha256",
        "model_sha256",
        "target_contract",
        "production_topology_hash",
        "run_kind",
        "pair_orders",
        "control",
        "profile",
        "profile_perturbation_range",
        "threads_per_block",
        "nsys_required_for_formal",
        "torch_profiler_is_smoke_only",
        "performance_claimed",
        "protocol_hash",
    }
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    run_kind = value.get("run_kind")
    expected_orders = list(PAIR_ORDERS[:1] if run_kind == "smoke" else PAIR_ORDERS)
    source = value.get("source_git_head")
    code_revision = value.get("code_revision")
    if (
        set(value) != expected_keys
        or claimed != canonical_hash(payload)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or not isinstance(source, str)
        or len(source) != 40
        or any(character not in "0123456789abcdef" for character in source)
        or not isinstance(value.get("source_clean"), bool)
        or value.get("allowed_formal_dirty_paths") != list(ALLOWED_FORMAL_DIRTY_PATHS)
        or not isinstance(code_revision, dict)
        or set(code_revision) != set(CODE_PATHS)
        or any(not _valid_digest(code_revision[path]) for path in CODE_PATHS)
        or not _valid_digest(value.get("source_capture_sha256"))
        or not _valid_digest(value.get("model_sha256"))
        or value.get("source_capture_sha256") != SOURCE_CAPTURE_SHA256
        or value.get("model_sha256") != MODEL_SHA256
        or value.get("production_topology_hash") != PRODUCTION_TOPOLOGY_HASH
        or run_kind not in {"smoke", "formal"}
        or value.get("pair_orders") != expected_orders
        or value.get("control")
        != {
            "warmup": CONTROL_WARMUP,
            "group_count": CONTROL_GROUP_COUNT,
            "repeats": CONTROL_REPEATS,
        }
        or value.get("profile")
        != {
            "warmup": PROFILE_WARMUP,
            "group_count": PROFILE_GROUP_COUNT,
            "repeats": PROFILE_REPEATS,
        }
        or value.get("profile_perturbation_range")
        != [PROFILE_PERTURBATION_MIN, PROFILE_PERTURBATION_MAX]
        or value.get("threads_per_block") != THREADS_PER_BLOCK
        or value.get("nsys_required_for_formal") is not True
        or value.get("torch_profiler_is_smoke_only") is not True
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("R1-A protocol differs")
    target_contract_from_dict(value.get("target_contract"))
    if run_kind == "formal" and value.get("source_clean") is not True:
        raise ValueError("R1-A formal source is dirty")
    if run_kind == "smoke" and code_revision != _code_revision():
        raise ValueError("R1-A smoke code revision differs")
    if run_kind == "formal" and code_revision != _historical_code_revision(source):
        raise ValueError("R1-A formal code revision differs")


def _numeric_groups(value: object, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("R1-A timing groups differ")
    groups = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0.0 for item in groups):
        raise ValueError("R1-A timing group value differs")
    return groups


def validate_worker(value: Mapping[str, Any], *, mode: str, pair_ordinal: int) -> None:
    payload = dict(value)
    claimed = payload.pop("worker_hash", None)
    if claimed != canonical_hash(payload):
        raise ValueError("R1-A worker hash differs")
    expected_keys = {
        "schema_version",
        "mode",
        "pair_ordinal",
        "topology",
        "marker_receipt",
        "semantic_receipt",
        "groups_ms",
        "median_ms",
        "warmup_count",
        "group_count",
        "repeats_per_group",
        "input_copy_included",
        "threads_per_block",
        "calibration_receipt",
        "profile_inventory",
        "profiler_epoch_warmup_excluded",
        "environment",
        "cupti_admitted",
        "formal_attribution_admitted",
        "performance_claimed",
        "worker_hash",
    }
    if set(value) != expected_keys:
        raise ValueError("R1-A worker fields differ")
    topology = topology_ledger_from_dict(value["topology"])
    if topology.topology_hash != PRODUCTION_TOPOLOGY_HASH:
        raise ValueError("R1-A production topology differs")
    marker = value.get("marker_receipt")
    if not isinstance(marker, dict):
        raise ValueError("R1-A marker receipt differs")
    expected_markers = [topology.marker_for(node.ordinal) for node in topology.nodes]
    expected_invocations = {str(node.ordinal): 4 for node in topology.nodes}
    if marker != {
        "markers": expected_markers,
        "invocations": expected_invocations,
        "expected_invocations_per_marker": 4,
        "capture_only": True,
    }:
        raise ValueError("R1-A marker derivation differs")
    semantic = value.get("semantic_receipt")
    if (
        not isinstance(semantic, dict)
        or semantic.get("maximum_absolute_difference") != EXPECTED_SEMANTIC_MAXIMUM
        or semantic.get("sign_exact") is not True
        or semantic.get("atol") != SEMANTIC_ATOL
        or semantic.get("rtol") != SEMANTIC_ATOL
        or semantic.get("baseline_launch_count") != 0
        or semantic.get("candidate_launch_count") != 6
        or semantic.get("fallback_count") != 0
        or semantic.get("eager_shadow_count") != 0
        or semantic.get("element_count") != EXPECTED_SEMANTIC_ELEMENT_COUNT
    ):
        raise ValueError("R1-A semantic receipt differs")
    expected_groups = CONTROL_GROUP_COUNT if mode == "control" else PROFILE_GROUP_COUNT
    expected_repeats = CONTROL_REPEATS if mode == "control" else PROFILE_REPEATS
    expected_warmup = CONTROL_WARMUP if mode == "control" else PROFILE_WARMUP
    groups = _numeric_groups(value.get("groups_ms"), expected_groups)
    if (
        value.get("schema_version") != WORKER_SCHEMA
        or value.get("mode") != mode
        or value.get("pair_ordinal") != pair_ordinal
        or value.get("median_ms") != statistics.median(groups)
        or value.get("group_count") != expected_groups
        or value.get("repeats_per_group") != expected_repeats
        or value.get("warmup_count") != expected_warmup
        or value.get("input_copy_included") is not True
        or value.get("threads_per_block") != THREADS_PER_BLOCK
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("R1-A worker identity differs")
    calibration = value.get("calibration_receipt")
    inventory = value.get("profile_inventory")
    if mode == "control":
        if (
            calibration is not None
            or inventory is not None
            or value.get("cupti_admitted") is not False
            or value.get("formal_attribution_admitted") is not False
            or value.get("profiler_epoch_warmup_excluded") is not False
        ):
            raise ValueError("R1-A control profiler boundary differs")
    else:
        rebuilt = clock_calibration_from_dict(calibration)
        if not isinstance(inventory, dict):
            raise ValueError("R1-A profile boundary differs")
        backend = inventory.get("backend")
        if backend == "torch_profiler_smoke":
            if (
                value.get("cupti_admitted") is not rebuilt.cupti_admitted
                or value.get("formal_attribution_admitted") is not False
                or inventory.get("owner_ledger_available") is not False
                or inventory.get("formal_attribution_available") is not False
                or inventory.get("reason") != "nsys_export_unavailable"
                or value.get("profiler_epoch_warmup_excluded") is not True
            ):
                raise ValueError("R1-A torch profile boundary differs")
        elif backend == "nsys_sqlite":
            export_receipt = nsys_export_receipt_from_dict(
                inventory.get("export_receipt")
            )
            timing_ledger_from_dict(inventory.get("timing_ledger"))
            if (
                not rebuilt.formal_admitted
                or value.get("cupti_admitted") is not True
                or value.get("formal_attribution_admitted") is not True
                or inventory.get("formal_attribution_available") is not True
                or inventory.get("reason") is not None
                or inventory.get("owner_ledger_hash") is None
                or inventory.get("owner_ledger_file")
                != f"raw/pair_{pair_ordinal:02d}_owner_ledger.json"
                or inventory.get("sqlite_file")
                != f"raw/pair_{pair_ordinal:02d}_profile.sqlite"
                or inventory.get("nsys_report_file")
                != f"raw/pair_{pair_ordinal:02d}_profile.nsys-rep"
                or value.get("profiler_epoch_warmup_excluded") is not False
                or export_receipt.anchor_errors_ns != rebuilt.nsys_anchor_errors_ns
            ):
                raise ValueError("R1-A Nsight profile boundary differs")
        else:
            raise ValueError("R1-A profile backend differs")


def derive_summary(
    protocol_value: Mapping[str, Any],
    workers: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, object]:
    validate_protocol(protocol_value)
    orders = protocol_value["pair_orders"]
    if not isinstance(orders, list):
        raise ValueError("R1-A pair order inventory differs")
    pairs = []
    for pair_ordinal, order in enumerate(orders):
        control = workers[(pair_ordinal, "control")]
        profile = workers[(pair_ordinal, "profile")]
        validate_worker(control, mode="control", pair_ordinal=pair_ordinal)
        validate_worker(profile, mode="profile", pair_ordinal=pair_ordinal)
        if (
            control["topology"] != profile["topology"]
            or control["semantic_receipt"] != profile["semantic_receipt"]
        ):
            raise ValueError("R1-A control/profile identity differs")
        perturbation = float(profile["median_ms"]) / float(control["median_ms"])
        perturbation_admitted = (
            PROFILE_PERTURBATION_MIN <= perturbation <= PROFILE_PERTURBATION_MAX
        )
        pair_formal = bool(
            perturbation_admitted
            and profile["formal_attribution_admitted"]
            and profile["calibration_receipt"]["formal_admitted"]
        )
        pairs.append(
            {
                "pair_ordinal": pair_ordinal,
                "order": order,
                "control_median_ms": control["median_ms"],
                "profile_median_ms": profile["median_ms"],
                "profile_perturbation": perturbation,
                "perturbation_admitted": perturbation_admitted,
                "cupti_admitted": profile["cupti_admitted"],
                "formal_attribution_admitted": pair_formal,
            }
        )
    formal = bool(
        protocol_value["run_kind"] == "formal"
        and len(pairs) == 6
        and all(pair["formal_attribution_admitted"] for pair in pairs)
    )
    if formal:
        status = "validated-r1a-attribution"
    elif protocol_value["run_kind"] == "formal":
        status = "validated-no-go-r1a-attribution"
    else:
        status = "smoke-only-r1a-attribution-closed"
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "run_kind": protocol_value["run_kind"],
        "pair_count": len(pairs),
        "pairs": pairs,
        "all_cupti_admitted": all(pair["cupti_admitted"] for pair in pairs),
        "all_perturbation_admitted": all(
            pair["perturbation_admitted"] for pair in pairs
        ),
        "formal_attribution_admitted": formal,
        "status": status,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def _run_worker(
    *,
    source_capture: Path,
    model: Path,
    pair_ordinal: int,
    mode: str,
    formal: bool,
    raw_root: Path,
) -> dict[str, Any]:
    worker_command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_cibc_r1_attribution_worker.py"),
        "--source-capture",
        str(source_capture),
        "--model",
        str(model),
        "--pair-ordinal",
        str(pair_ordinal),
        "--mode",
        mode,
    )
    if formal and mode == "profile":
        raw_root.mkdir(parents=True, exist_ok=True)
        report_prefix = raw_root / f"pair_{pair_ordinal:02d}_profile"
        pending_worker = raw_root / f"pair_{pair_ordinal:02d}_profile.pending.json"
        completed = subprocess.run(
            (
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--resolve-symbols=false",
                "--cuda-graph-trace=node",
                "--force-overwrite=true",
                f"--output={report_prefix}",
                *worker_command,
                "--profile-backend",
                "nsys",
                "--output",
                str(pending_worker),
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        sqlite_path = report_prefix.with_suffix(".sqlite")
        report_path = report_prefix.with_suffix(".nsys-rep")
        exported = subprocess.run(
            (
                "nsys",
                "export",
                "--quiet=true",
                "--type=sqlite",
                "--force-overwrite=true",
                "--output",
                str(sqlite_path),
                str(report_path),
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        value = _load_json(pending_worker)
        pending_worker.unlink()
        receipt, owner, timing, calibration = derive_nsys_attribution(
            sqlite_path, value
        )
        owner_payload = owner.to_dict()
        owner_path = raw_root / f"pair_{pair_ordinal:02d}_owner_ledger.json"
        _write_json(owner_path, owner_payload)
        pending_inventory = value["profile_inventory"]
        assert isinstance(pending_inventory, dict)
        value["calibration_receipt"] = calibration
        value["cupti_admitted"] = True
        value["formal_attribution_admitted"] = True
        value["profile_inventory"] = {
            "backend": "nsys_sqlite",
            "anchors": pending_inventory["anchors"],
            "export_receipt": receipt.to_dict(),
            "timing_ledger": timing.to_dict(),
            "owner_ledger_hash": owner_payload["owner_ledger_hash"],
            "owner_ledger_file": f"raw/pair_{pair_ordinal:02d}_owner_ledger.json",
            "sqlite_file": f"raw/pair_{pair_ordinal:02d}_profile.sqlite",
            "nsys_report_file": f"raw/pair_{pair_ordinal:02d}_profile.nsys-rep",
            "formal_attribution_available": True,
            "reason": None,
        }
        value["worker_hash"] = canonical_hash(
            {key: item for key, item in value.items() if key != "worker_hash"}
        )
        (raw_root / f"pair_{pair_ordinal:02d}_nsys.log").write_text(
            completed.stdout + completed.stderr + exported.stdout + exported.stderr,
            encoding="utf-8",
        )
        return value
    completed = subprocess.run(
        worker_command,
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    if not isinstance(value, dict):
        raise TypeError("R1-A worker output differs")
    return value


def _manifest(
    root: Path, protocol_value: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, object]:
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "files": files,
        "protocol_hash": protocol_value["protocol_hash"],
        "summary_hash": summary["summary_hash"],
    }
    value["manifest_hash"] = canonical_hash(value)
    return value


def generate(
    root: Path, *, source_capture: Path, model: Path, smoke: bool
) -> dict[str, object]:
    if root.exists():
        raise FileExistsError("R1-A artifact target already exists")
    if not smoke and shutil.which("nsys") is None:
        raise RuntimeError("R1-A formal requires Nsight Systems before worker launch")
    protocol_value = protocol(source_capture, model, smoke=smoke)
    if not smoke and protocol_value["source_clean"] is not True:
        raise RuntimeError("R1-A formal requires a clean source commit")
    temporary = Path(tempfile.mkdtemp(prefix="cibc-r1a-", dir=root.parent))
    try:
        _write_json(temporary / "protocol.json", protocol_value)
        workers: dict[tuple[int, str], Mapping[str, Any]] = {}
        orders = protocol_value["pair_orders"]
        assert isinstance(orders, list)
        for pair_ordinal, order in enumerate(orders):
            for mode_code in order:
                mode = "control" if mode_code == "C" else "profile"
                worker = _run_worker(
                    source_capture=source_capture,
                    model=model,
                    pair_ordinal=pair_ordinal,
                    mode=mode,
                    formal=not smoke,
                    raw_root=temporary / "raw",
                )
                validate_worker(worker, mode=mode, pair_ordinal=pair_ordinal)
                workers[(pair_ordinal, mode)] = worker
                _write_json(
                    temporary / "raw" / f"pair_{pair_ordinal:02d}_{mode}.json", worker
                )
        summary = derive_summary(protocol_value, workers)
        _write_json(temporary / "summary.json", summary)
        _write_json(
            temporary / "manifest.json", _manifest(temporary, protocol_value, summary)
        )
        os.replace(temporary, root)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(root: Path) -> dict[str, object]:
    protocol_value = _load_json(root / "protocol.json")
    summary = _load_json(root / "summary.json")
    manifest = _load_json(root / "manifest.json")
    validate_protocol(protocol_value)
    manifest_payload = dict(manifest)
    claimed_manifest = manifest_payload.pop("manifest_hash", None)
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        claimed_manifest != canonical_hash(manifest_payload)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("files") != files
        or manifest.get("protocol_hash") != protocol_value.get("protocol_hash")
        or manifest.get("summary_hash") != summary.get("summary_hash")
    ):
        raise ValueError("R1-A manifest differs")
    orders = protocol_value["pair_orders"]
    assert isinstance(orders, list)
    workers = {
        (pair_ordinal, mode): _load_json(
            root / "raw" / f"pair_{pair_ordinal:02d}_{mode}.json"
        )
        for pair_ordinal in range(len(orders))
        for mode in ("control", "profile")
    }
    if protocol_value["run_kind"] == "formal":
        for pair_ordinal in range(len(orders)):
            profile = workers[(pair_ordinal, "profile")]
            validate_worker(profile, mode="profile", pair_ordinal=pair_ordinal)
            receipt, owner, timing, calibration = derive_nsys_attribution(
                root / "raw" / f"pair_{pair_ordinal:02d}_profile.sqlite",
                profile,
            )
            inventory = profile["profile_inventory"]
            assert isinstance(inventory, dict)
            frozen_owner = owner_ledger_from_dict(
                _load_json(root / "raw" / f"pair_{pair_ordinal:02d}_owner_ledger.json")
            )
            if (
                inventory["export_receipt"] != receipt.to_dict()
                or inventory["timing_ledger"] != timing.to_dict()
                or inventory["owner_ledger_hash"]
                != frozen_owner.to_dict()["owner_ledger_hash"]
                or owner.to_dict() != frozen_owner.to_dict()
                or profile["calibration_receipt"] != calibration
            ):
                raise ValueError("R1-A Nsight semantic replay differs")
    rebuilt = derive_summary(protocol_value, workers)
    if rebuilt != summary:
        raise ValueError("R1-A summary derivation differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        result = replay(args.artifact)
    else:
        if args.source_capture is None or args.model is None:
            parser.error("generation requires source capture and model")
        result = generate(
            args.artifact,
            source_capture=args.source_capture,
            model=args.model,
            smoke=args.smoke,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
