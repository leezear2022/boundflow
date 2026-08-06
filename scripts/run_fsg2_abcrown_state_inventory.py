#!/usr/bin/env python3
"""Generate or replay the real αβ-CROWN state-ownership inventory for FSG2."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,import-outside-toplevel,protected-access
# pylint: disable=missing-function-docstring,line-too-long
# pylint: disable=import-error

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import FrameType
from typing import Any, Iterator, Mapping, Sequence

SCHEMA_VERSION = "boundflow.fsg2-abcrown-state-inventory/v1"
WORKER_SCHEMA_VERSION = "boundflow.fsg2-abcrown-state-worker/v1"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_RELATIVE_PATH = "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
PROPERTY_RELATIVE_PATH = (
    "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
    "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
ARTIFACT_FILES = (
    "inventory.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = ("scripts/run_fsg2_abcrown_state_inventory.py",)
STATE_ATTRIBUTES = ("alpha", "sparse_beta", "beta", "split_beta")
KWARG_STATE_NAMES = (
    "intermediate_constr",
    "interm_bounds",
    "reference_bounds",
    "aux_reference_bounds",
)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    return {path: file_sha256(root / path) for path in CODE_PATHS}


def _code_paths_clean() -> bool:
    root = _repo_root()
    return not _git_value(root, "status", "--porcelain=v1", "--", *CODE_PATHS)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    source_head = manifest.get("source_git_head")
    revisions = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revisions, Mapping):
        raise ValueError("FSG2 inventory source provenance differs")
    if _git_value(root, "rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {}
        for path in CODE_PATHS:
            blob = subprocess.run(
                ("git", "show", f"{source_head}:{path}"),
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            observed[path] = _sha256_bytes(blob)
    if dict(revisions) != observed:
        raise ValueError("FSG2 inventory source code revision differs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--benchmark-root", type=Path, required=True)
    worker.add_argument("--abcrown-root", type=Path, required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _phase_from_stack(method: str) -> tuple[str, str]:
    external = "unclassified_compute_bounds"
    frame: FrameType = sys._getframe(1)
    try:
        for _ in range(20):
            parent = frame.f_back
            if parent is None:
                break
            frame = parent
            filename = frame.f_code.co_filename.replace("\\", "/")
            function = frame.f_code.co_name
            if function == "update_bounds_core":
                return "beta_split", "activation_bab_bound"
            if "/input_split/" in filename:
                return "beta_split", "input_bab_bound"
            if "incomplete_verifier" in filename:
                return "initial_crown", "incomplete_verification"
            if "beta_CROWN_solver" in filename:
                normalized = method.lower().replace("_", "-")
                return (
                    "alpha_optimize" if "optimized" in normalized else "initial_crown",
                    "alpha_crown_initialization",
                )
    finally:
        del frame
    if "optimized" in method.lower().replace("_", "-"):
        return "alpha_optimize", external
    return "unclassified", external


def _tensor_digest(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _walk_tensors(
    value: Any,
    path: str,
    torch_module: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> list[dict[str, object]]:
    if seen is None:
        seen = set()
    if torch_module.is_tensor(value):
        return [
            {
                "path": path,
                "shape": [int(dimension) for dimension in value.shape],
                "rank": int(value.dim()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": int(value.numel()),
                "requires_grad": bool(value.requires_grad),
                "content_sha256": _tensor_digest(value),
            }
        ]
    if depth >= 8 or id(value) in seen:
        return []
    if isinstance(value, Mapping):
        seen.add(id(value))
        rows: list[dict[str, object]] = []
        for key in sorted(value, key=str):
            rows.extend(
                _walk_tensors(
                    value[key],
                    f"{path}.{key}",
                    torch_module,
                    depth=depth + 1,
                    seen=seen,
                )
            )
        return rows
    if isinstance(value, (tuple, list)):
        seen.add(id(value))
        rows = []
        for index, item in enumerate(value):
            rows.extend(
                _walk_tensors(
                    item, f"{path}[{index}]", torch_module, depth=depth + 1, seen=seen
                )
            )
        return rows
    return []


def _module_state(
    instance: Any, torch_module: Any
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {name: [] for name in STATE_ATTRIBUTES}
    nodes_method = getattr(instance, "nodes", None)
    if not callable(nodes_method):
        return result
    for node_index, node in enumerate(nodes_method()):
        node_name = str(getattr(node, "name", type(node).__name__))
        for attribute in STATE_ATTRIBUTES:
            value = getattr(node, attribute, None)
            result[attribute].extend(
                _walk_tensors(
                    value,
                    f"node[{node_index}:{node_name}].{attribute}",
                    torch_module,
                )
            )
    return result


def _kwarg_state(
    kwargs: Mapping[str, Any], torch_module: Any
) -> dict[str, list[dict[str, object]]]:
    return {
        name: _walk_tensors(kwargs.get(name), f"kwargs.{name}", torch_module)
        for name in KWARG_STATE_NAMES
    }


@contextmanager
def _inventory_calls(
    bounded_module: Any, torch_module: Any, calls: list[dict[str, Any]]
) -> Iterator[None]:
    original = bounded_module.compute_bounds
    stack: list[int] = []

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = len(calls)
        method = str(kwargs.get("method", "backward"))
        phase, external_phase = _phase_from_stack(method)
        row: dict[str, Any] = {
            "call_id": call_id,
            "parent_call_id": stack[-1] if stack else None,
            "depth": len(stack),
            "method": method,
            "phase": phase,
            "external_phase": external_phase,
            "bound_lower": bool(kwargs.get("bound_lower", True)),
            "bound_upper": bool(kwargs.get("bound_upper", True)),
            "kwargs_keys": sorted(kwargs),
            "pre_module_state": _module_state(instance, torch_module),
            "kwarg_state": _kwarg_state(kwargs, torch_module),
        }
        calls.append(row)
        stack.append(call_id)
        try:
            result = original(instance, *args, **kwargs)
            row["result_tensors"] = _walk_tensors(result, "result", torch_module)
            return result
        finally:
            row["post_module_state"] = _module_state(instance, torch_module)
            if stack.pop() != call_id:
                raise RuntimeError("FSG2 state inventory call stack differs")

    bounded_module.compute_bounds = wrapped
    try:
        yield
    finally:
        bounded_module.compute_bounds = original


def _visited_domains(result: Any) -> list[int]:
    stats = getattr(result, "stats", None)
    if not isinstance(stats, dict) or not isinstance(stats.get("bab"), list):
        return []
    return [
        int(row[2])
        for row in stats["bab"]
        if isinstance(row, (tuple, list)) and len(row) >= 3
    ]


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import ABCrownSolver, ConfigBuilder, IOConstraints  # type: ignore[import-not-found]
    from auto_LiRPA import BoundedModule  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("FSG2 state inventory requires CUDA")
    if _git_value(args.abcrown_root, "rev-parse", "HEAD") != ABCROWN_COMMIT:
        raise ValueError("FSG2 alpha-beta-CROWN commit differs")
    if (
        _git_value(args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
        != AUTO_LIRPA_COMMIT
    ):
        raise ValueError("FSG2 auto_LiRPA commit differs")
    if _git_value(args.benchmark_root, "rev-parse", "HEAD") != VNNCOMP_COMMIT:
        raise ValueError("FSG2 VNN-COMP commit differs")
    calls: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg2-property-") as workspace:
        isolated_property = Path(workspace) / args.property.name
        shutil.copy2(args.property, isolated_property)
        config = (
            ConfigBuilder.from_defaults()
            .set("general/device", "cuda")
            .set("general/seed", 100)
            .set("general/reset_seed_after_precompile", True)
            .set("general/complete_verifier", "bab")
            .set("attack/pgd_order", "skip")
            .set("bab/timeout", 60)
            .set("bab/max_iterations", 1)
            .set("solver/batch_size", 64)
            .set("solver/auto_enlarge_batch_size", False)
            .set("solver/alpha-crown/iteration", 5)
            .set("solver/beta-crown/iteration", 10)
        )
        with _inventory_calls(BoundedModule, torch, calls):
            solver = ABCrownSolver(str(args.model), config=config)
            result = solver.verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    record: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "source": {
            "abcrown_commit": ABCROWN_COMMIT,
            "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "model_relative_path": MODEL_RELATIVE_PATH,
            "property_relative_path": PROPERTY_RELATIVE_PATH,
            "model_sha256": file_sha256(args.model),
            "property_sha256": file_sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "complete_verifier": "bab",
            "timeout_seconds": 60,
            "max_iterations": 1,
            "batch_size": 64,
            "auto_enlarge_batch_size": False,
            "alpha_steps": 5,
            "beta_steps": 10,
            "property_cache": "cold_isolated_copy",
        },
        "result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "calls": calls,
        "performance_claimed": False,
    }
    _write_json(args.result_json, record)
    print(
        canonical_json({"status": record["result"]["status"], "call_count": len(calls)})
    )


def _state_count(call: Mapping[str, Any], timing: str, name: str) -> int:
    state = call.get(f"{timing}_module_state")
    if not isinstance(state, Mapping) or not isinstance(state.get(name), list):
        return 0
    return len(state[name])


def _kwarg_count(call: Mapping[str, Any], name: str) -> int:
    state = call.get("kwarg_state")
    if not isinstance(state, Mapping) or not isinstance(state.get(name), list):
        return 0
    return len(state[name])


def derive_summary(inventory: Mapping[str, Any]) -> dict[str, object]:
    calls_raw = inventory.get("calls")
    if not isinstance(calls_raw, list) or not all(
        isinstance(row, Mapping) for row in calls_raw
    ):
        raise ValueError("FSG2 inventory calls differ")
    calls: Sequence[Mapping[str, Any]] = calls_raw
    phase_counts = {
        phase: sum(call.get("phase") == phase for call in calls)
        for phase in ("initial_crown", "alpha_optimize", "beta_split", "unclassified")
    }
    alpha_calls = [call for call in calls if call.get("phase") == "alpha_optimize"]
    beta_calls = [call for call in calls if call.get("phase") == "beta_split"]
    alpha_pre_counts = [_state_count(call, "pre", "alpha") for call in alpha_calls]
    beta_pre_counts = [
        sum(
            _state_count(call, "pre", name)
            for name in ("sparse_beta", "beta", "split_beta")
        )
        for call in beta_calls
    ]
    beta_post_counts = [
        sum(
            _state_count(call, "post", name)
            for name in ("sparse_beta", "beta", "split_beta")
        )
        for call in beta_calls
    ]
    beta_constraint_counts = [
        _kwarg_count(call, "intermediate_constr") for call in beta_calls
    ]
    beta_intermediate_bound_counts = [
        _kwarg_count(call, "interm_bounds") for call in beta_calls
    ]
    beta_aux_reference_counts = [
        _kwarg_count(call, "aux_reference_bounds") for call in beta_calls
    ]
    intermediate_constraint_key_observed = any(
        "intermediate_constr" in call.get("kwargs_keys", []) for call in beta_calls
    )
    observed_alpha = bool(alpha_calls) and max(alpha_pre_counts, default=0) > 0
    observed_beta_phase = bool(beta_calls)
    beta_state_explicit = observed_beta_phase and min(beta_pre_counts, default=0) > 0
    nested_split_tensor_context = max(beta_constraint_counts, default=0) > 0
    reasons = ["native_rvir_v3_backend_supports_initial_crown_only"]
    if observed_alpha:
        reasons.append("production_alpha_is_nested_start_node_keyed_state")
    if observed_beta_phase and not beta_state_explicit:
        reasons.append("beta_state_not_explicit_on_bounded_module_before_call")
    if intermediate_constraint_key_observed and not nested_split_tensor_context:
        reasons.append("intermediate_constr_key_has_no_owned_tensor_leaf")
    if max(beta_intermediate_bound_counts, default=0) > 0:
        reasons.append("provider_intermediate_bounds_are_not_explicit_beta_split_state")
    summary: dict[str, object] = {
        "status": "validated_reduced_state_boundary",
        "workload_id": "cifar10_resnet:000",
        "phase_call_counts": phase_counts,
        "alpha_pre_state_counts": alpha_pre_counts,
        "beta_pre_state_counts": beta_pre_counts,
        "beta_post_state_counts": beta_post_counts,
        "beta_intermediate_constraint_tensor_counts": beta_constraint_counts,
        "beta_intermediate_bound_tensor_counts": beta_intermediate_bound_counts,
        "beta_aux_reference_bound_tensor_counts": beta_aux_reference_counts,
        "production_alpha_state_observed": observed_alpha,
        "production_beta_phase_observed": observed_beta_phase,
        "production_beta_state_explicit_before_call": beta_state_explicit,
        "intermediate_constraint_key_observed": intermediate_constraint_key_observed,
        "provider_nested_split_tensor_context_observed": nested_split_tensor_context,
        "initial_crown_replacement_admitted": True,
        "alpha_beta_split_replacement_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "rejection_reasons": reasons,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "alpha_beta_split_replacement_admitted": summary[
            "alpha_beta_split_replacement_admitted"
        ],
        "b2_same_solver_timing_admitted": summary["b2_same_solver_timing_admitted"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# FSG2 αβ-CROWN Production State Inventory\n\n"
        "This artifact inventories tensor ownership at real ResNet2B compute_bounds calls. "
        "It records why initial-CROWN replacement cannot be generalized to alpha/beta/split "
        "without a new lossless provider-state mapping. It makes no performance claim.\n"
    )


def _external_env() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("BOUNDFLOW_ROOT", "PYTHONPATH", "TVM_HOME", "TVM_LIBRARY_PATH"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _validate_inputs(
    benchmark_root: Path, abcrown_root: Path, abcrown_python: Path
) -> None:
    checks = (
        (_git_value(benchmark_root, "rev-parse", "HEAD") == VNNCOMP_COMMIT),
        (_git_value(abcrown_root, "rev-parse", "HEAD") == ABCROWN_COMMIT),
        (
            _git_value(abcrown_root / "auto_LiRPA", "rev-parse", "HEAD")
            == AUTO_LIRPA_COMMIT
        ),
        abcrown_python.is_file(),
        file_sha256(benchmark_root / MODEL_RELATIVE_PATH) == MODEL_SHA256,
        file_sha256(benchmark_root / PROPERTY_RELATIVE_PATH) == PROPERTY_SHA256,
    )
    if not all(checks):
        raise ValueError("FSG2 state inventory source inputs differ")


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError(
            "FSG2 inventory code path must be clean before formal generation"
        )
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = args.benchmark_root.resolve()
    abcrown_root = args.abcrown_root.resolve()
    abcrown_python = Path(os.path.abspath(args.abcrown_python))
    _validate_inputs(benchmark_root, abcrown_root, abcrown_python)
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg2-inventory-") as temporary:
        result_path = Path(temporary) / "inventory.json"
        completed = subprocess.run(
            (
                str(abcrown_python),
                str(Path(__file__).resolve()),
                "worker",
                "--benchmark-root",
                str(benchmark_root),
                "--abcrown-root",
                str(abcrown_root),
                "--model",
                str(benchmark_root / MODEL_RELATIVE_PATH),
                "--property",
                str(benchmark_root / PROPERTY_RELATIVE_PATH),
                "--result-json",
                str(result_path),
            ),
            cwd=_repo_root(),
            env=_external_env(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=240,
        )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"FSG2 inventory worker failed: {completed.stdout[-8000:]}"
            )
        shutil.copy2(result_path, artifact_dir / "inventory.json")
        print(completed.stdout.strip()[-2000:], flush=True)
    inventory = _load_json(artifact_dir / "inventory.json")
    summary = derive_summary(inventory)
    _write_json(artifact_dir / "summary.json", summary)
    replay_result = _replay_result(summary)
    (artifact_dir / "replay_stdout.txt").write_text(
        canonical_json(replay_result) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(_readme(), encoding="utf-8")
    root = _repo_root()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_git_head": _git_value(root, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)
    return replay_result


def _replay(artifact_dir: Path) -> dict[str, object]:
    manifest = _load_json(artifact_dir / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG2 inventory manifest envelope differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("FSG2 inventory artifact file inventory differs")
    for name, digest in files.items():
        if file_sha256(artifact_dir / name) != digest:
            raise ValueError("FSG2 inventory artifact file digest differs")
    summary = derive_summary(_load_json(artifact_dir / "inventory.json"))
    if _load_json(artifact_dir / "summary.json") != summary:
        raise ValueError("FSG2 inventory semantic replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("FSG2 inventory summary projection differs")
    result = _replay_result(summary)
    if (artifact_dir / "replay_stdout.txt").read_text(
        encoding="utf-8"
    ) != canonical_json(result) + "\n":
        raise ValueError("FSG2 inventory replay stdout differs")
    if (artifact_dir / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG2 inventory README differs")
    return result


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
        return
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve())
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
