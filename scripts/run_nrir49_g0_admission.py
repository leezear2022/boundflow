#!/usr/bin/env python3
"""Generate or replay the fail-closed NRIR49 G0 admission artifact."""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=import-outside-toplevel,missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Mapping, Sequence

from packaging.version import Version

ARTIFACT_SCHEMA_VERSION = "boundflow.nrir49-g0-admission-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.nrir49-g0-admission/v1"
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "admission.json"
SELECTED_CROWN_SUPPORTED_OPS = frozenset(
    {"linear", "relu", "conv2d", "flatten", "reshape", "add", "concat"}
)
NON_UNKNOWN = frozenset({"verified", "unsafe"})
QUEUE_TARGET = 1.20
COMPLETE_QUERY_TARGET = 1.15
MAX_REQUIRED_REGION_SPEEDUP = 10.0


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Return deterministic JSON without accepting NaN/Infinity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    """Hash a semantic JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_region_speedup(share: float, target: float) -> float | None:
    """Invert Amdahl's law; return None when the target is unreachable."""

    if not 0.0 <= share <= 1.0 or target < 1.0:
        raise ValueError("invalid Amdahl share/target")
    denominator = share + 1.0 / target - 1.0
    if denominator <= 0.0:
        return None
    return share / denominator


def projected_scope_speedup(share: float, region_speedup: float) -> float:
    """Project end-to-end speedup from a measured region share."""

    if not 0.0 <= share <= 1.0 or region_speedup < 1.0:
        raise ValueError("invalid Amdahl share/speedup")
    return 1.0 / ((1.0 - share) + share / region_speedup)


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=dict(env) if env is not None else None,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[-4096:],
            "stderr": completed.stderr.strip()[-4096:],
        }
    except subprocess.TimeoutExpired as error:
        return {
            "returncode": 124,
            "stdout": str(error.stdout or "")[-4096:],
            "stderr": "command timed out after 15 seconds",
        }
    except OSError as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}


def _git_value(root: Path, *args: str) -> str | None:
    result = _run(("git", *args), cwd=root)
    if result["returncode"] != 0:
        return None
    return str(result["stdout"])


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def absolute_without_resolving_symlink(path: Path) -> Path:
    """Make a CLI path absolute while preserving virtualenv python symlinks."""

    return Path(os.path.abspath(path))


def _relative_identity(path: Path | None, anchor: Path | None) -> str | None:
    if path is None:
        return None
    if anchor is not None:
        try:
            return str(path.absolute().relative_to(anchor.absolute()))
        except ValueError:
            pass
    return path.name


def _repo_source(root: Path) -> dict[str, object]:
    status = _git_value(root, "status", "--porcelain=v1") or ""
    dirty_paths = sorted(
        line[3:] for line in status.splitlines() if len(line) >= 4 and line[3:]
    )
    return {
        "branch": _git_value(root, "branch", "--show-current"),
        "commit": _git_value(root, "rev-parse", "HEAD"),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "probe_sha256": file_sha256(Path(__file__).resolve()),
    }


def probe_gpu_environment() -> dict[str, object]:
    """Collect current GPU facts and distinguish firmware disable from stack failure."""

    import torch  # pylint: disable=import-outside-toplevel

    try:
        import tvm  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel,import-error

        tvm_version: str | None = str(tvm.__version__)
        tvm_cuda_enabled: bool | None = bool(tvm.runtime.enabled("cuda"))
    except (ImportError, OSError) as error:
        tvm_version = None
        tvm_cuda_enabled = None
        tvm_import_error: str | None = str(error)
    else:
        tvm_import_error = None

    try:
        import tvm_ffi  # pylint: disable=import-outside-toplevel,import-error,unused-import

        tvm_ffi_importable = True
        tvm_ffi_error: str | None = None
    except (ImportError, OSError) as error:
        tvm_ffi_importable = False
        tvm_ffi_error = str(error)

    dgpu_disable_path = Path("/sys/devices/platform/asus-nb-wmi/dgpu_disable")
    mux_path = Path("/sys/devices/platform/asus-nb-wmi/gpu_mux_mode")
    dgpu_disable = _read_optional(dgpu_disable_path)
    gpu_mux_mode = _read_optional(mux_path)
    smi = _run(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        )
    )
    lspci = _run(("lspci", "-nn"))
    journal = _run(("journalctl", "-u", "asusd", "-b", "--no-pager", "-n", "250"))
    queued_enable = "Queueing GPU attribute dgpu_disable = 0 for delayed apply" in str(
        journal["stdout"]
    )
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    devices: list[dict[str, object]] = []
    if cuda_available:
        for index in range(device_count):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    nvidia_nodes = sorted(glob.glob("/dev/nvidia*"))
    nvidia_module_files = sorted(
        glob.glob(f"/lib/modules/{platform.release()}/**/nvidia.ko*", recursive=True)
    )
    loaded_modules = _run(("lsmod",))
    nvidia_module_loaded = any(
        line.startswith("nvidia ")
        for line in str(loaded_modules["stdout"]).splitlines()
    )
    nvidia_pci_visible = "NVIDIA" in str(lspci["stdout"])
    if cuda_available and device_count > 0 and smi["returncode"] == 0:
        status = "ready"
    elif dgpu_disable == "1" and queued_enable:
        status = "blocked_reboot_required"
    elif dgpu_disable == "1":
        status = "blocked_firmware_disabled"
    else:
        status = "blocked_cuda_runtime_unavailable"
    return {
        "status": status,
        "performance_claimed": False,
        "host": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": devices,
        },
        "tvm": {
            "version": tvm_version,
            "cuda_enabled": tvm_cuda_enabled,
            "import_error": tvm_import_error,
            "tvm_ffi_importable": tvm_ffi_importable,
            "tvm_ffi_error": tvm_ffi_error,
        },
        "driver": {
            "nvidia_smi": smi,
            "nvidia_pci_visible": nvidia_pci_visible,
            "nvidia_device_nodes": nvidia_nodes,
            "nvidia_module_files": nvidia_module_files,
            "nvidia_module_loaded": nvidia_module_loaded,
        },
        "asus_firmware": {
            "dgpu_disable": dgpu_disable,
            "gpu_mux_mode": gpu_mux_mode,
            "enable_queued_for_delayed_apply": queued_enable,
        },
    }


def _parse_auto_lirpa_torch_range(setup_text: str) -> tuple[str | None, str | None]:
    lower_match = re.search(r"pytorch_version_l\s*=\s*['\"]([^'\"]+)", setup_text)
    upper_match = re.search(r"pytorch_version_u\s*=\s*['\"]([^'\"]+)", setup_text)
    return (
        lower_match.group(1) if lower_match else None,
        upper_match.group(1) if upper_match else None,
    )


def probe_competitor_environment(
    root: Path, *, abcrown_root: Path | None, abcrown_python: Path | None
) -> dict[str, object]:
    """Audit source and interpreter availability without installing anything."""

    import torch  # pylint: disable=import-outside-toplevel

    auto_root = root / "boundflow" / "3rdparty" / "auto_LiRPA"
    setup_path = auto_root / "setup.py"
    conv_path = auto_root / "auto_LiRPA" / "operators" / "convolution.py"
    setup_text = _read_optional(setup_path) or ""
    conv_text = _read_optional(conv_path) or ""
    lower, upper = _parse_auto_lirpa_torch_range(setup_text)
    torch_version = Version(str(torch.__version__).split("+", maxsplit=1)[0])
    torch_supported = bool(
        lower and upper and Version(lower) <= torch_version < Version(upper)
    )
    source_oracle = {
        "path": str(conv_path.relative_to(root)),
        "sha256": file_sha256(conv_path) if conv_path.is_file() else None,
        "onehot_to_dense": "onehotc_to_dense" in conv_text,
        "dense_conv_transpose2d": "F.conv_transpose2d" in conv_text,
        "patches_path": "type(last_A) == Patches" in conv_text,
    }
    abcrown_source = bool(abcrown_root and (abcrown_root / ".git").exists())
    abcrown_commit = (
        _git_value(abcrown_root, "rev-parse", "HEAD")
        if abcrown_root is not None and abcrown_source
        else None
    )
    abcrown_dirty = bool(
        _git_value(abcrown_root, "status", "--porcelain=v1")
        if abcrown_root is not None and abcrown_source
        else False
    )
    submodule_status = (
        _git_value(abcrown_root, "submodule", "status")
        if abcrown_root is not None and abcrown_source
        else None
    )
    uv_lock = abcrown_root / "uv.lock" if abcrown_root is not None else None
    pyproject = abcrown_root / "pyproject.toml" if abcrown_root is not None else None
    interpreter_probe: dict[str, object] | None = None
    if abcrown_python is not None:
        competitor_env = dict(os.environ)
        for name in (
            "BOUNDFLOW_ROOT",
            "PYTHONPATH",
            "TVM_HOME",
            "TVM_LIBRARY_PATH",
        ):
            competitor_env.pop(name, None)
        competitor_env["PYTHONNOUSERSITE"] = "1"
        interpreter_probe = _run(
            (
                str(abcrown_python),
                "-c",
                "import abcrown,auto_LiRPA,json,sys,torch;print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'auto_lirpa':auto_LiRPA.__version__,'abcrown':abcrown.__version__},sort_keys=True))",
            ),
            cwd=root,
            env=competitor_env,
        )
        if abcrown_root is not None:
            for stream in ("stdout", "stderr"):
                interpreter_probe[stream] = str(interpreter_probe[stream]).replace(
                    str(abcrown_root), "<ABCROWN_ROOT>"
                )
    if not abcrown_source:
        status = "blocked_missing_alpha_beta_crown_source"
    elif interpreter_probe is None or interpreter_probe["returncode"] != 0:
        status = "blocked_missing_compatible_interpreter"
    else:
        status = "ready"
    return {
        "status": status,
        "performance_claimed": False,
        "vendored_auto_lirpa": {
            "commit": _git_value(auto_root, "rev-parse", "HEAD"),
            "dirty": bool(_git_value(auto_root, "status", "--porcelain=v1")),
            "declared_torch_lower_inclusive": lower,
            "declared_torch_upper_exclusive": upper,
            "current_torch_version": str(torch.__version__),
            "current_torch_declared_supported": torch_supported,
            "boundconv_source_oracle": source_oracle,
        },
        "alpha_beta_crown": {
            "root_identity": abcrown_root.name if abcrown_root else None,
            "source_available": abcrown_source,
            "commit": abcrown_commit,
            "dirty": abcrown_dirty,
            "submodule_status": submodule_status,
            "uv_lock_sha256": (
                file_sha256(uv_lock)
                if uv_lock is not None and uv_lock.is_file()
                else None
            ),
            "pyproject_sha256": (
                file_sha256(pyproject)
                if pyproject is not None and pyproject.is_file()
                else None
            ),
            "python_identity": _relative_identity(abcrown_python, abcrown_root),
            "interpreter_probe": interpreter_probe,
        },
    }


def probe_user_boundconv(source: Path | None) -> dict[str, object]:
    """Keep the reported 40x result unclaimed until its source is supplied."""

    if source is None or not source.is_file():
        return {
            "status": "not_auditable_source_missing",
            "performance_claimed": False,
            "source_identity": source.name if source else None,
            "sha256": None,
        }
    return {
        "status": "source_available_benchmark_pending",
        "performance_claimed": False,
        "source_identity": source.name,
        "sha256": file_sha256(source),
    }


def load_qualification_candidate(
    *,
    native_result_path: Path,
    abcrown_result_path: Path,
    model: Path,
    property_path: Path,
    vnncomp_root: Path,
    abcrown_root: Path,
) -> dict[str, object]:
    """Freeze one same-verdict public workload used only for solveability admission."""

    native = json.loads(native_result_path.read_text(encoding="utf-8"))
    abcrown = json.loads(abcrown_result_path.read_text(encoding="utf-8"))
    if not isinstance(native, Mapping) or not isinstance(abcrown, Mapping):
        raise TypeError("G0 qualification results must be JSON objects")
    workload_id = native.get("workload_id")
    if (
        not isinstance(workload_id, str)
        or abcrown.get("workload_id") != workload_id
        or native.get("backend") != "boundflow_native"
        or abcrown.get("backend") != "external_abcrown"
        or native.get("performance_claimed") is not False
        or abcrown.get("performance_claimed") is not False
        or native.get("solver_status") not in NON_UNKNOWN
        or abcrown.get("solver_status") != native.get("solver_status")
    ):
        raise ValueError(
            "G0 qualification results do not share one non-unknown verdict"
        )
    vnncomp_commit = _git_value(vnncomp_root, "rev-parse", "HEAD")
    abcrown_commit = _git_value(abcrown_root, "rev-parse", "HEAD")
    if abcrown.get("abcrown_commit") != abcrown_commit:
        raise ValueError("G0 qualification alpha-beta-CROWN commit differs")
    return {
        "status": "qualified_same_non_unknown",
        "performance_claimed": False,
        "workload_id": workload_id,
        "verdict": native["solver_status"],
        "selection_role": "solveability_qualification_only_not_performance_tuning",
        "protocol": {
            "device": "cpu",
            "timeout_seconds": 30,
            "torch_threads": 8,
            "native_alpha_steps": 5,
            "abcrown_alpha_steps": 5,
            "abcrown_beta_steps": 10,
            "native_search_steps": 4,
            "native_max_nodes": 1,
        },
        "source": {
            "vnncomp_commit": vnncomp_commit,
            "abcrown_commit": abcrown_commit,
            "model_relative_path": str(
                model.resolve().relative_to(vnncomp_root.resolve())
            ),
            "property_relative_path": str(
                property_path.resolve().relative_to(vnncomp_root.resolve())
            ),
            "model_sha256": file_sha256(model),
            "property_sha256": file_sha256(property_path),
            "native_result_sha256": file_sha256(native_result_path),
            "abcrown_result_sha256": file_sha256(abcrown_result_path),
        },
        "native_result": dict(native),
        "abcrown_result": dict(abcrown),
    }


def _load_and_verify_prior_evidence(artifact_dir: Path) -> dict[str, Any]:
    from scripts.run_multiworkload_competitor_e2e_artifact import (
        validate_evidence_structure,
    )

    manifest_path = artifact_dir / MANIFEST_FILE
    evidence_path = artifact_dir / "evidence.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, Mapping) or files.get("evidence.json") != file_sha256(
        evidence_path
    ):
        raise ValueError("prior multiworkload evidence digest differs")
    if manifest.get("evidence_hash") != canonical_hash(evidence):
        raise ValueError("prior multiworkload semantic hash differs")
    validate_evidence_structure(evidence)
    return evidence


def audit_frontend_and_solveability(
    artifact_dir: Path,
    *,
    qualification_candidate: Mapping[str, Any] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Derive coverage and shared-verdict admission from replayed historical records."""

    evidence = _load_and_verify_prior_evidence(artifact_dir)
    root = Path(__file__).resolve().parents[1]
    try:
        source_artifact = str(artifact_dir.resolve().relative_to(root))
    except ValueError:
        source_artifact = artifact_dir.name
    by_workload: dict[str, dict[str, str]] = {}
    op_inventory: dict[str, list[str]] = {}
    for record in evidence["records"]:
        workload = str(record["workload_id"])
        result = record["result"]
        by_workload.setdefault(workload, {})[str(record["backend"])] = str(
            result["solver_status"]
        )
        if record["backend"] == "boundflow_native":
            op_inventory[workload] = sorted(set(result.get("primal_ops", [])))
    if qualification_candidate is not None:
        workload = str(qualification_candidate["workload_id"])
        native = _mapping(
            qualification_candidate["native_result"], "qualification native result"
        )
        abcrown = _mapping(
            qualification_candidate["abcrown_result"], "qualification abcrown result"
        )
        by_workload[workload] = {
            "boundflow_native": str(native["solver_status"]),
            "external_abcrown": str(abcrown["solver_status"]),
        }
        op_inventory[workload] = sorted(set(native.get("primal_ops", [])))
    unsupported = {
        workload: sorted(set(ops) - SELECTED_CROWN_SUPPORTED_OPS)
        for workload, ops in op_inventory.items()
    }
    observed_importable = all(not values for values in unsupported.values())
    shared_non_unknown = sorted(
        workload
        for workload, statuses in by_workload.items()
        if statuses.get("boundflow_native") in NON_UNKNOWN
        and statuses.get("external_abcrown") in NON_UNKNOWN
    )
    frontend: dict[str, object] = {
        "status": "validated_reduced" if observed_importable else "no_go",
        "performance_claimed": False,
        "selected_crown_supported_ops": sorted(SELECTED_CROWN_SUPPORTED_OPS),
        "observed_workload_ops": op_inventory,
        "unsupported_observed_ops": unsupported,
        "average_pool_supported": False,
        "claim_boundary": "only the three replayed NRIR-18 workloads; broader held-out model-family coverage remains pending",
    }
    solveability: dict[str, object] = {
        "status": "admitted" if shared_non_unknown else "no_go",
        "performance_claimed": False,
        "workload_solver_status": by_workload,
        "shared_non_unknown_workloads": shared_non_unknown,
        "required_shared_non_unknown_count": 1,
        "source_artifact": source_artifact,
        "source_evidence_sha256": file_sha256(artifact_dir / "evidence.json"),
        "qualification_candidate": (
            dict(qualification_candidate)
            if qualification_candidate is not None
            else None
        ),
    }
    return frontend, solveability


def build_evidence(
    root: Path,
    *,
    prior_artifact_dir: Path,
    abcrown_root: Path | None = None,
    abcrown_python: Path | None = None,
    user_boundconv_source: Path | None = None,
    qualification_candidate: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Build one current-host G0 snapshot without making a performance claim."""

    gpu = probe_gpu_environment()
    competitor = probe_competitor_environment(
        root, abcrown_root=abcrown_root, abcrown_python=abcrown_python
    )
    user_boundconv = probe_user_boundconv(user_boundconv_source)
    frontend, solveability = audit_frontend_and_solveability(
        prior_artifact_dir, qualification_candidate=qualification_candidate
    )
    gpu_ready = gpu["status"] == "ready"
    gate_results = {
        "gpu_infrastructure_ready": gpu_ready,
        "competitor_environment_ready": competitor["status"] == "ready",
        "frontend_observed_matrix_importable": frontend["status"]
        == "validated_reduced",
        "shared_non_unknown_workload_present": solveability["status"] == "admitted",
    }
    blockers = [name for name, passed in gate_results.items() if not passed]
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "ready_for_g1" if not blockers else "blocked",
        "performance_claimed": False,
        "source": _repo_source(root),
        "gpu": gpu,
        "competitor": competitor,
        "user_boundconv_40x": user_boundconv,
        "frontend": frontend,
        "solveability": solveability,
        "memory_reachability": {
            "status": (
                "pending_g1_measurement"
                if gpu_ready
                else "not_auditable_gpu_unavailable"
            ),
            "performance_claimed": False,
            "required_peak_budget_fraction": 0.80,
            "b80_allocated": None,
            "b80_reserved": None,
            "b_oom": None,
        },
        "amdahl_preregistration": {
            "status": (
                "pending_g1_measurement"
                if gpu_ready
                else "not_auditable_gpu_unavailable"
            ),
            "performance_claimed": False,
            "queue_target": QUEUE_TARGET,
            "complete_query_target": COMPLETE_QUERY_TARGET,
            "maximum_required_region_speedup": MAX_REQUIRED_REGION_SPEEDUP,
            "formula_projected": "1 / ((1 - s) + s / r)",
            "formula_required": "s / (s + 1 / T - 1)",
            "gpu_queue_share": None,
            "gpu_complete_query_share": None,
            "required_region_speedup": None,
        },
        "admission": {
            "g1_ready": not blockers,
            "gate_results": gate_results,
            "blockers": blockers,
        },
        "limitations": [
            "This G0 artifact makes no GPU performance, memory, or 40x claim.",
            "The frontend result covers only workloads already present in the replayed NRIR-18 artifact.",
            "A shared non-unknown workload must be frozen before G8 TTV/solved metrics have a denominator.",
            "G1 Amdahl and physical-memory values remain unset until a real CUDA device is visible.",
        ],
    }
    validate_evidence(evidence)
    return evidence


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject claim upgrades or contradictions in a G0 record."""

    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("performance_claimed") is not False
        or evidence.get("status") not in {"blocked", "ready_for_g1"}
    ):
        raise ValueError("G0 evidence header differs")
    gpu = _mapping(evidence.get("gpu"), "gpu")
    competitor = _mapping(evidence.get("competitor"), "competitor")
    frontend = _mapping(evidence.get("frontend"), "frontend")
    solveability = _mapping(evidence.get("solveability"), "solveability")
    memory = _mapping(evidence.get("memory_reachability"), "memory")
    amdahl = _mapping(evidence.get("amdahl_preregistration"), "amdahl")
    admission = _mapping(evidence.get("admission"), "admission")
    if any(
        section.get("performance_claimed") is not False
        for section in (gpu, competitor, frontend, solveability, memory, amdahl)
    ):
        raise ValueError("G0 subsection attempted a performance claim")
    statuses = _mapping(solveability.get("workload_solver_status"), "solver statuses")
    expected_shared = sorted(
        workload
        for workload, raw in statuses.items()
        if _mapping(raw, "workload status").get("boundflow_native") in NON_UNKNOWN
        and _mapping(raw, "workload status").get("external_abcrown") in NON_UNKNOWN
    )
    if solveability.get("shared_non_unknown_workloads") != expected_shared:
        raise ValueError("G0 shared solveability derivation differs")
    expected_solve_status = "admitted" if expected_shared else "no_go"
    if solveability.get("status") != expected_solve_status:
        raise ValueError("G0 solveability status contradicts records")
    candidate = solveability.get("qualification_candidate")
    if candidate is not None:
        candidate_map = _mapping(candidate, "qualification candidate")
        native = _mapping(candidate_map.get("native_result"), "candidate native")
        abcrown = _mapping(candidate_map.get("abcrown_result"), "candidate abcrown")
        if (
            candidate_map.get("performance_claimed") is not False
            or candidate_map.get("selection_role")
            != "solveability_qualification_only_not_performance_tuning"
            or native.get("performance_claimed") is not False
            or abcrown.get("performance_claimed") is not False
            or native.get("solver_status") not in NON_UNKNOWN
            or abcrown.get("solver_status") != native.get("solver_status")
            or native.get("workload_id") != candidate_map.get("workload_id")
            or abcrown.get("workload_id") != candidate_map.get("workload_id")
        ):
            raise ValueError("G0 qualification candidate contract differs")
    expected_gates = {
        "gpu_infrastructure_ready": gpu.get("status") == "ready",
        "competitor_environment_ready": competitor.get("status") == "ready",
        "frontend_observed_matrix_importable": frontend.get("status")
        == "validated_reduced",
        "shared_non_unknown_workload_present": expected_solve_status == "admitted",
    }
    if admission.get("gate_results") != expected_gates:
        raise ValueError("G0 gate derivation differs")
    blockers = [name for name, passed in expected_gates.items() if not passed]
    if admission.get("blockers") != blockers or admission.get("g1_ready") is not (
        not blockers
    ):
        raise ValueError("G0 blocker derivation differs")
    expected_header = "ready_for_g1" if not blockers else "blocked"
    if evidence.get("status") != expected_header:
        raise ValueError("G0 header contradicts admission gates")
    if gpu.get("status") != "ready":
        if (
            memory.get("status") != "not_auditable_gpu_unavailable"
            or amdahl.get("status") != "not_auditable_gpu_unavailable"
        ):
            raise ValueError("G0 unavailable GPU must not produce measured opportunity")
    if amdahl.get("formula_required") != "s / (s + 1 / T - 1)":
        raise ValueError("G0 Amdahl preregistration differs")
    if amdahl.get("maximum_required_region_speedup") != 10.0:
        raise ValueError("G0 Amdahl kill threshold differs")


def generate_artifact(artifact_dir: Path, evidence: Mapping[str, Any]) -> None:
    """Write an immutable admission snapshot and digest manifest."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = artifact_dir / EVIDENCE_FILE
    evidence_path.write_text(
        canonical_json(evidence, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": evidence["status"],
        "performance_claimed": False,
        "evidence_hash": canonical_hash(evidence),
        "files": {EVIDENCE_FILE: file_sha256(evidence_path)},
    }
    (artifact_dir / MANIFEST_FILE).write_text(
        canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )


def replay_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Verify file digest, semantic hash, and all derived admission fields."""

    manifest = json.loads((artifact_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    evidence_path = artifact_dir / EVIDENCE_FILE
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {EVIDENCE_FILE: file_sha256(evidence_path)}
        or manifest.get("evidence_hash") != canonical_hash(evidence)
    ):
        raise ValueError("G0 artifact manifest differs")
    validate_evidence(evidence)
    if manifest.get("status") != evidence.get("status"):
        raise ValueError("G0 manifest/evidence status differs")
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--prior-artifact-dir", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path)
    generate.add_argument("--abcrown-python", type=Path)
    generate.add_argument("--user-boundconv-source", type=Path)
    generate.add_argument("--candidate-native-result", type=Path)
    generate.add_argument("--candidate-abcrown-result", type=Path)
    generate.add_argument("--candidate-model", type=Path)
    generate.add_argument("--candidate-property", type=Path)
    generate.add_argument("--vnncomp-root", type=Path)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "generate":
        candidate_values = (
            args.candidate_native_result,
            args.candidate_abcrown_result,
            args.candidate_model,
            args.candidate_property,
            args.vnncomp_root,
        )
        if any(value is not None for value in candidate_values) and not all(
            value is not None for value in candidate_values
        ):
            raise ValueError("G0 qualification candidate arguments must be complete")
        if args.candidate_native_result is not None and args.abcrown_root is None:
            raise ValueError(
                "G0 qualification candidate requires alpha-beta-CROWN root"
            )
        qualification_candidate = (
            load_qualification_candidate(
                native_result_path=args.candidate_native_result.resolve(),
                abcrown_result_path=args.candidate_abcrown_result.resolve(),
                model=args.candidate_model.resolve(),
                property_path=args.candidate_property.resolve(),
                vnncomp_root=args.vnncomp_root.resolve(),
                abcrown_root=args.abcrown_root.resolve(),
            )
            if args.candidate_native_result is not None
            else None
        )
        evidence = build_evidence(
            root,
            prior_artifact_dir=args.prior_artifact_dir.resolve(),
            abcrown_root=args.abcrown_root.resolve() if args.abcrown_root else None,
            abcrown_python=(
                absolute_without_resolving_symlink(args.abcrown_python)
                if args.abcrown_python
                else None
            ),
            user_boundconv_source=(
                args.user_boundconv_source.resolve()
                if args.user_boundconv_source
                else None
            ),
            qualification_candidate=qualification_candidate,
        )
        generate_artifact(args.artifact_dir.resolve(), evidence)
    else:
        evidence = replay_artifact(args.artifact_dir.resolve())
    admission = _mapping(evidence["admission"], "admission")
    print(
        canonical_json(
            {
                "status": evidence["status"],
                "g1_ready": admission["g1_ready"],
                "blockers": admission["blockers"],
            }
        )
    )


if __name__ == "__main__":
    main()
