#!/usr/bin/env python3
"""Run or replay the post-reboot NRIR49 G0 CUDA admission smoke matrix."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-boolean-expressions
# pylint: disable=import-outside-toplevel,line-too-long

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

ARTIFACT_SCHEMA_VERSION = "boundflow.nrir49-g0-cuda-smoke-artifact/v1"
EVIDENCE_SCHEMA_VERSION = "boundflow.nrir49-g0-cuda-smoke/v1"
MANIFEST_FILE = "manifest.json"
EVIDENCE_FILE = "cuda_smoke.json"
JSON_MARKER = "NRIR49_CUDA_SMOKE_JSON="
GATE_ORDER = (
    "nvidia_driver_device",
    "boundflow_torch_cuda",
    "tvm_cuda_build_run",
    "tvm_ffi_custom_stream",
    "competitor_torch_cuda",
    "cross_environment_identity_digest",
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Encode deterministic JSON while rejecting non-finite values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    """Hash one semantic JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file incrementally."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash a contiguous array's exact bytes."""

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def python_type_identity(value: object) -> str:
    """Return stable Python type metadata without probing dynamic module functions."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[-8192:],
            "stderr": completed.stderr.strip()[-8192:],
        }
    except subprocess.TimeoutExpired as error:
        return {
            "returncode": 124,
            "stdout": str(error.stdout or "")[-8192:],
            "stderr": f"command timed out after {timeout} seconds",
        }
    except OSError as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _git_value(root: Path, *args: str) -> str | None:
    result = _run(("git", *args), cwd=root)
    return str(result["stdout"]) if result["returncode"] == 0 else None


def _record(
    status: str,
    *,
    facts: Mapping[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    if status not in {"pass", "blocked", "fail"}:
        raise ValueError("invalid CUDA smoke status")
    return {
        "status": status,
        "performance_claimed": False,
        "facts": dict(facts or {}),
        "error": error,
    }


def parse_marked_json(output: str) -> dict[str, Any]:
    """Parse the last marker-prefixed JSON record from noisy subprocess output."""

    rows = [
        line[len(JSON_MARKER) :]
        for line in output.splitlines()
        if line.startswith(JSON_MARKER)
    ]
    if not rows:
        raise ValueError("CUDA smoke subprocess JSON marker is missing")
    value = json.loads(rows[-1])
    if not isinstance(value, dict):
        raise TypeError("CUDA smoke subprocess JSON must be an object")
    return value


def expected_vector_contract() -> dict[str, object]:
    """Return the exact cross-environment input/output oracle."""

    source = np.arange(256, dtype="float32")
    expected = source * np.float32(2.0) + np.float32(1.0)
    return {
        "dtype": "float32",
        "shape": [256],
        "input_sha256": array_sha256(source),
        "output_sha256": array_sha256(expected),
    }


def run_nvidia_gate() -> dict[str, object]:
    """Require the firmware-enabled device, driver, PCI visibility and nvidia-smi."""

    dgpu_disable = _read_optional(
        Path("/sys/devices/platform/asus-nb-wmi/dgpu_disable")
    )
    gpu_mux_mode = _read_optional(
        Path("/sys/devices/platform/asus-nb-wmi/gpu_mux_mode")
    )
    smi = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    lspci = _run(("lspci", "-nn"))
    device_nodes = sorted(Path("/dev").glob("nvidia*"))
    rows = [line.strip() for line in str(smi["stdout"]).splitlines() if line.strip()]
    facts: dict[str, object] = {
        "dgpu_disable": dgpu_disable,
        "gpu_mux_mode": gpu_mux_mode,
        "nvidia_smi": smi,
        "nvidia_pci_visible": "NVIDIA" in str(lspci["stdout"]),
        "nvidia_device_nodes": [path.name for path in device_nodes],
        "gpu_rows": rows,
    }
    if smi["returncode"] != 0 or not rows:
        return _record("blocked", facts=facts, error="nvidia-smi/device unavailable")
    if dgpu_disable == "1" or not facts["nvidia_pci_visible"] or not device_nodes:
        return _record(
            "fail",
            facts=facts,
            error="firmware/PCI/device-node facts contradict nvidia-smi",
        )
    return _record("pass", facts=facts)


def run_boundflow_torch_gate(driver_ready: bool) -> dict[str, object]:
    """Execute an exact CUDA vector contract in the BoundFlow environment."""

    import torch

    basic = {
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if (
        not driver_ready
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
    ):
        return _record("blocked", facts=basic, error="BoundFlow torch CUDA unavailable")
    try:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        source = torch.arange(256, dtype=torch.float32, device=device)
        custom = torch.cuda.Stream(device=device)
        with torch.cuda.stream(custom):
            output = source * 2.0 + 1.0
        custom.synchronize()
        source_cpu = source.detach().cpu().numpy()
        output_cpu = output.detach().cpu().numpy()
        oracle = expected_vector_contract()
        facts = {
            **basic,
            "device_name": props.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(props.total_memory),
            "custom_stream_nonzero": int(custom.cuda_stream) != 0,
            "input_sha256": array_sha256(source_cpu),
            "output_sha256": array_sha256(output_cpu),
        }
        if (
            facts["input_sha256"] != oracle["input_sha256"]
            or facts["output_sha256"] != oracle["output_sha256"]
            or not facts["custom_stream_nonzero"]
        ):
            return _record(
                "fail",
                facts=facts,
                error="BoundFlow torch CUDA vector/stream contract differs",
            )
        return _record("pass", facts=facts)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return _record("fail", facts=basic, error=f"{type(error).__name__}: {error}")


def run_tvm_cuda_gate(torch_ready: bool) -> dict[str, object]:
    """Build and execute a real CUDA TIR kernel, not only query TVM capability."""

    try:
        import tvm  # type: ignore[import-untyped]
        from scripts.smoke_tvm_cuda import add_one
    except (ImportError, OSError) as error:
        return _record("fail", error=f"TVM import failed: {error}")
    basic = {
        "tvm_version": str(tvm.__version__),
        "tvm_cuda_enabled": bool(tvm.runtime.enabled("cuda")),
    }
    if not torch_ready:
        return _record(
            "blocked", facts=basic, error="driver/torch prerequisite unavailable"
        )
    if not basic["tvm_cuda_enabled"]:
        return _record("fail", facts=basic, error="TVM was not built with CUDA")
    try:
        count = 256
        module = tvm.build(add_one, target="cuda")
        device = tvm.cuda(0)
        source = np.arange(count, dtype="float32")
        output = np.zeros_like(source)
        tvm_source = tvm.runtime.tensor(source, device)
        tvm_output = tvm.runtime.tensor(output, device)
        module(tvm_source, tvm_output)
        device.sync()
        actual = tvm_output.numpy()
        expected = source + np.float32(1.0)
        facts = {
            **basic,
            "device_exist": bool(device.exist),
            "input_sha256": array_sha256(source),
            "output_sha256": array_sha256(actual),
            "expected_output_sha256": array_sha256(expected),
            "kernel_contract": "float32[256]: output = input + 1",
            "module_python_type": python_type_identity(module),
        }
        if facts["output_sha256"] != facts["expected_output_sha256"]:
            return _record("fail", facts=facts, error="TVM CUDA output digest differs")
        return _record("pass", facts=facts)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return _record("fail", facts=basic, error=f"{type(error).__name__}: {error}")


def run_tvm_ffi_stream_gate(torch_ready: bool) -> dict[str, object]:
    """Prove the current Torch custom stream is visible through TVM-FFI."""

    try:
        import torch
        import tvm_ffi  # type: ignore[import-untyped]
    except (ImportError, OSError) as error:
        return _record("fail", error=f"TVM-FFI import failed: {error}")
    if not torch_ready:
        return _record("blocked", error="driver/torch prerequisite unavailable")
    try:
        stream = torch.cuda.Stream(device=torch.device("cuda:0"))
        ffi_device = tvm_ffi.device("cuda:0")
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
            torch_raw = int(torch.cuda.current_stream().cuda_stream)
            ffi_raw = int(tvm_ffi.get_raw_stream(ffi_device))
            output = torch.arange(256, device="cuda", dtype=torch.float32) * 2.0 + 1.0
        stream.synchronize()
        facts = {
            "torch_raw_stream": torch_raw,
            "ffi_raw_stream": ffi_raw,
            "stream_match": torch_raw == ffi_raw == int(stream.cuda_stream),
            "output_sha256": array_sha256(output.detach().cpu().numpy()),
        }
        oracle = expected_vector_contract()
        if (
            not facts["stream_match"]
            or facts["output_sha256"] != oracle["output_sha256"]
        ):
            return _record(
                "fail", facts=facts, error="TVM-FFI stream/output contract differs"
            )
        return _record("pass", facts=facts)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return _record("fail", error=f"{type(error).__name__}: {error}")


def _competitor_program(model: Path, property_path: Path) -> str:
    return "\n".join(
        (
            "import abcrown, auto_LiRPA, hashlib, json, pathlib, sys, torch",
            f"model = pathlib.Path({str(model)!r})",
            f"property_path = pathlib.Path({str(property_path)!r})",
            "def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()",
            "if not torch.cuda.is_available() or torch.cuda.device_count() < 1: raise RuntimeError('competitor CUDA unavailable')",
            "device = torch.device('cuda:0')",
            "props = torch.cuda.get_device_properties(device)",
            "source = torch.arange(256, dtype=torch.float32, device=device)",
            "stream = torch.cuda.Stream(device=device)",
            "with torch.cuda.stream(stream): output = source * 2.0 + 1.0",
            "stream.synchronize()",
            "def tensor_digest(value): return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()",
            "record = {'python':sys.version.split()[0], 'torch_version':torch.__version__, 'torch_cuda_build':torch.version.cuda, 'cudnn_version':torch.backends.cudnn.version(), 'auto_lirpa_version':auto_LiRPA.__version__, 'abcrown_version':abcrown.__version__, 'cuda_available':True, 'device_count':torch.cuda.device_count(), 'device_name':props.name, 'capability':list(torch.cuda.get_device_capability(device)), 'total_memory_bytes':int(props.total_memory), 'custom_stream_nonzero':int(stream.cuda_stream) != 0, 'input_sha256':tensor_digest(source), 'output_sha256':tensor_digest(output), 'model_sha256':digest(model), 'property_sha256':digest(property_path)}",
            f"print({JSON_MARKER!r} + json.dumps(record, sort_keys=True))",
        )
    )


def run_competitor_gate(
    *,
    driver_ready: bool,
    root: Path,
    python: Path,
    model: Path,
    property_path: Path,
) -> dict[str, object]:
    """Run the same CUDA vector and source digest contract in the isolated env."""

    basic = {
        "root_identity": root.name,
        "python_identity": str(python.absolute().relative_to(root.absolute())),
        "repo_commit": _git_value(root, "rev-parse", "HEAD"),
        "repo_dirty": bool(_git_value(root, "status", "--porcelain=v1")),
        "submodule_status": _git_value(root, "submodule", "status"),
    }
    if not driver_ready:
        return _record(
            "blocked", facts=basic, error="NVIDIA driver prerequisite unavailable"
        )
    if not python.is_file() or not model.is_file() or not property_path.is_file():
        return _record(
            "fail", facts=basic, error="competitor interpreter/source file missing"
        )
    competitor_env = dict(os.environ)
    for name in ("BOUNDFLOW_ROOT", "PYTHONPATH", "TVM_HOME", "TVM_LIBRARY_PATH"):
        competitor_env.pop(name, None)
    competitor_env["PYTHONNOUSERSITE"] = "1"
    completed = _run(
        (str(python), "-c", _competitor_program(model, property_path)),
        cwd=Path(__file__).resolve().parents[1],
        env=competitor_env,
        timeout=90,
    )
    for stream_name in ("stdout", "stderr"):
        completed[stream_name] = str(completed[stream_name]).replace(
            str(root), "<ABCROWN_ROOT>"
        )
    if completed["returncode"] != 0:
        return _record(
            "fail",
            facts={**basic, "process": completed},
            error="competitor CUDA smoke failed",
        )
    try:
        payload = parse_marked_json(str(completed["stdout"]))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return _record("fail", facts={**basic, "process": completed}, error=str(error))
    return _record(
        "pass", facts={**basic, **payload, "process_stderr": completed["stderr"]}
    )


def derive_cross_environment_gate(
    *,
    boundflow: Mapping[str, Any],
    tvm: Mapping[str, Any],
    competitor: Mapping[str, Any],
    model: Path,
    property_path: Path,
) -> dict[str, object]:
    """Require both environments and TVM to consume the same frozen bytes."""

    if any(record.get("status") != "pass" for record in (boundflow, tvm, competitor)):
        return _record("blocked", error="one or more execution prerequisites failed")
    bf = _mapping(boundflow.get("facts"), "BoundFlow torch facts")
    tvm_facts = _mapping(tvm.get("facts"), "TVM facts")
    comp = _mapping(competitor.get("facts"), "competitor facts")
    oracle = expected_vector_contract()
    expected_model = file_sha256(model)
    expected_property = file_sha256(property_path)
    checks = {
        "input_digest_match": bf.get("input_sha256")
        == comp.get("input_sha256")
        == oracle["input_sha256"],
        "output_digest_match": bf.get("output_sha256")
        == comp.get("output_sha256")
        == oracle["output_sha256"],
        "tvm_input_digest_match": tvm_facts.get("input_sha256")
        == oracle["input_sha256"],
        "tvm_output_digest_match": tvm_facts.get("output_sha256")
        == tvm_facts.get("expected_output_sha256"),
        "device_name_match": bf.get("device_name") == comp.get("device_name"),
        "capability_match": bf.get("capability") == comp.get("capability"),
        "total_memory_match": bf.get("total_memory_bytes")
        == comp.get("total_memory_bytes"),
        "model_digest_match": comp.get("model_sha256") == expected_model,
        "property_digest_match": comp.get("property_sha256") == expected_property,
    }
    facts = {
        "checks": checks,
        "model_identity": model.name,
        "property_identity": property_path.name,
        "model_sha256": expected_model,
        "property_sha256": expected_property,
        "vector_contract": oracle,
    }
    if not all(checks.values()):
        return _record(
            "fail", facts=facts, error="cross-environment identity/digest differs"
        )
    return _record("pass", facts=facts)


def aggregate_gates(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    """Derive G0 readiness from the exact six-gate matrix."""

    if set(gates) != set(GATE_ORDER):
        raise ValueError("CUDA smoke gate set differs")
    for name, gate in gates.items():
        if gate.get("status") not in {"pass", "blocked", "fail"}:
            raise ValueError(f"CUDA smoke gate status differs: {name}")
        if gate.get("performance_claimed") is not False:
            raise ValueError(f"CUDA smoke gate attempted a performance claim: {name}")
    blockers = [name for name in GATE_ORDER if gates[name]["status"] != "pass"]
    return {
        "g0_cuda_ready": not blockers,
        "blockers": blockers,
        "status": "ready_for_g1" if not blockers else "blocked",
    }


def build_evidence(
    *,
    root: Path,
    abcrown_root: Path,
    abcrown_python: Path,
    model: Path,
    property_path: Path,
) -> dict[str, object]:
    """Execute all gates even after failures so the artifact is diagnostic."""

    nvidia = run_nvidia_gate()
    driver_ready = nvidia["status"] == "pass"
    boundflow = run_boundflow_torch_gate(driver_ready)
    torch_ready = boundflow["status"] == "pass"
    tvm = run_tvm_cuda_gate(torch_ready)
    ffi = run_tvm_ffi_stream_gate(torch_ready)
    competitor = run_competitor_gate(
        driver_ready=driver_ready,
        root=abcrown_root,
        python=abcrown_python,
        model=model,
        property_path=property_path,
    )
    cross = derive_cross_environment_gate(
        boundflow=boundflow,
        tvm=tvm,
        competitor=competitor,
        model=model,
        property_path=property_path,
    )
    gates: dict[str, Mapping[str, Any]] = {
        "nvidia_driver_device": nvidia,
        "boundflow_torch_cuda": boundflow,
        "tvm_cuda_build_run": tvm,
        "tvm_ffi_custom_stream": ffi,
        "competitor_torch_cuda": competitor,
        "cross_environment_identity_digest": cross,
    }
    admission = aggregate_gates(gates)
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": admission["status"],
        "performance_claimed": False,
        "source": {
            "branch": _git_value(root, "branch", "--show-current"),
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "abcrown_commit": _git_value(abcrown_root, "rev-parse", "HEAD"),
        },
        "gates": gates,
        "admission": admission,
        "limitations": [
            "This artifact is a functional environment admission, not a latency or memory benchmark.",
            "No G1 profiling or G2/G3/TIR optimization is authorized by a blocked artifact.",
            "The synthetic vector checks execution/identity only; model semantics remain covered by the separate solveability artifact.",
        ],
    }
    validate_evidence(evidence)
    return evidence


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    """Recompute the admission header and reject claim/status tampering."""

    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("performance_claimed") is not False
    ):
        raise ValueError("CUDA smoke evidence header differs")
    gates = _mapping(evidence.get("gates"), "CUDA smoke gates")
    typed_gates = {name: _mapping(value, name) for name, value in gates.items()}
    actual = aggregate_gates(typed_gates)
    if (
        evidence.get("admission") != actual
        or evidence.get("status") != actual["status"]
    ):
        raise ValueError("CUDA smoke admission derivation differs")
    if (
        not isinstance(evidence.get("limitations"), list)
        or len(evidence["limitations"]) != 3
    ):
        raise ValueError("CUDA smoke limitation ledger differs")


def generate_artifact(artifact_dir: Path, evidence: Mapping[str, Any]) -> None:
    """Write a digest-bound CUDA smoke artifact without overwriting evidence."""

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
    """Replay file and semantic digests plus every derived gate."""

    manifest = json.loads((artifact_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    evidence_path = artifact_dir / EVIDENCE_FILE
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("files") != {EVIDENCE_FILE: file_sha256(evidence_path)}
        or manifest.get("evidence_hash") != canonical_hash(evidence)
        or manifest.get("status") != evidence.get("status")
    ):
        raise ValueError("CUDA smoke artifact manifest differs")
    validate_evidence(evidence)
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--property", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Execute the requested generate/replay command and fail closed on blockers."""

    args = _parse_args()
    if args.command == "generate":
        evidence = build_evidence(
            root=Path(__file__).resolve().parents[1],
            abcrown_root=args.abcrown_root.resolve(),
            abcrown_python=Path(os.path.abspath(args.abcrown_python)),
            model=args.model.resolve(),
            property_path=args.property.resolve(),
        )
        generate_artifact(args.artifact_dir.resolve(), evidence)
    else:
        evidence = replay_artifact(args.artifact_dir.resolve())
    admission = _mapping(evidence["admission"], "CUDA smoke admission")
    print(canonical_json(admission))
    if admission["g0_cuda_ready"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
