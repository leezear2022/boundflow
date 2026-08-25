#!/usr/bin/env python3
"""Generate or replay the MR3-0 real-provider P-anchor hook preflight."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,import-outside-toplevel,protected-access
# pylint: disable=missing-function-docstring,line-too-long,import-error
# pylint: disable=wrong-import-position,too-many-instance-attributes
# pylint: disable=too-few-public-methods,too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import FrameType
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    EXPECTED_RUNS,
    MR3_HOOK_SCHEMA,
    MR3_HOOK_WORKER_SCHEMA,
    VNNCOMP_COMMIT,
    canonical_hash,
    derive_summary,
)

MODEL_RELATIVE_PATH = "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
PROPERTY_RELATIVE_PATH = (
    "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
    "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
ARTIFACT_FILES = ("raw.json", "summary.json", "replay_stdout.txt", "README.md")
CODE_PATHS = (
    "boundflow/runtime/mr3_provider_hook_feasibility.py",
    "scripts/run_mr3_provider_hook_feasibility.py",
    "scripts/probe_mr3_provider_hook_feasibility_tamper.py",
    "tests/test_mr3_provider_hook_feasibility.py",
)
TARGET_RELU = "/input-24"
TARGET_CONV = "/input-20"
TARGET_START = "/49"


def _json_text(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(_json_text(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in CODE_PATHS}


def _code_clean() -> bool:
    return not _git(ROOT, "status", "--porcelain=v1", "--", *CODE_PATHS)


def _tensor_digest(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_receipt(value: Any, torch_module: Any) -> dict[str, object]:
    if not torch_module.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value)!r}")
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "stride": [int(dimension) for dimension in value.stride()],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "numel": int(value.numel()),
        "data_ptr": int(value.data_ptr()),
        "version": int(value._version),
        "content_sha256": _tensor_digest(value),
    }


def _semantic_tensor_receipt(value: Any, torch_module: Any) -> dict[str, object]:
    receipt = _tensor_receipt(value, torch_module)
    receipt.pop("data_ptr")
    receipt.pop("version")
    return receipt


def _tensor_state(value: Any, torch_module: Any) -> dict[str, object]:
    receipt = _semantic_tensor_receipt(value, torch_module)
    receipt["values"] = [
        float(item) for item in value.detach().cpu().contiguous().reshape(-1).tolist()
    ]
    return receipt


def _walk_tensor_values(value: Any, torch_module: Any, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if torch_module.is_tensor(value):
        yield value
        return
    if id(value) in seen:
        return
    if isinstance(value, Mapping):
        seen.add(id(value))
        for key in sorted(value, key=str):
            yield from _walk_tensor_values(value[key], torch_module, seen)
    elif isinstance(value, (tuple, list)):
        seen.add(id(value))
        for item in value:
            yield from _walk_tensor_values(item, torch_module, seen)
    elif hasattr(value, "val"):
        seen.add(id(value))
        yield from _walk_tensor_values(value.val, torch_module, seen)


def _result_state(value: Any, torch_module: Any) -> list[dict[str, object]]:
    return [
        _tensor_state(tensor, torch_module)
        for tensor in _walk_tensor_values(value, torch_module)
    ]


def _module_state(instance: Any, torch_module: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in instance.nodes():
        for attribute in ("alpha", "sparse_betas", "beta", "split_beta"):
            value = getattr(node, attribute, None)
            for ordinal, tensor in enumerate(_walk_tensor_values(value, torch_module)):
                rows.append(
                    {
                        "node": str(getattr(node, "name", "")),
                        "attribute": attribute,
                        "ordinal": ordinal,
                        "tensor": _tensor_state(tensor, torch_module),
                    }
                )
    return rows


def _target_alpha(instance: Any, torch_module: Any) -> Any:
    node = next(
        node for node in instance.nodes() if getattr(node, "name", None) == TARGET_RELU
    )
    alpha = getattr(node, "alpha", None)
    if not isinstance(alpha, Mapping) or TARGET_START not in alpha:
        raise ValueError("MR3-0 target alpha is absent")
    value = alpha[TARGET_START]
    if not torch_module.is_tensor(value):
        raise TypeError("MR3-0 target alpha is not a tensor")
    return value


def _phase_from_stack(method: str) -> str:
    frame: FrameType = sys._getframe(1)
    try:
        for _ in range(24):
            parent = frame.f_back
            if parent is None:
                break
            frame = parent
            if frame.f_code.co_name == "update_bounds_core":
                return "beta_split"
    finally:
        del frame
    return "other" if "optimized" not in method.lower() else "optimized_other"


def _extract_lower_a(result: Any, torch_module: Any) -> Any:
    try:
        value = result[0][0][0]
    except (IndexError, TypeError) as error:
        raise ValueError("MR3-0 bound_backward result structure differs") from error
    if not torch_module.is_tensor(value):
        raise TypeError("MR3-0 lower A is not a tensor")
    return value


def _extract_lower_bias(result: Any, torch_module: Any) -> Any:
    try:
        value = result[1]
    except (IndexError, TypeError) as error:
        raise ValueError("MR3-0 bound_backward bias structure differs") from error
    if not torch_module.is_tensor(value):
        raise TypeError("MR3-0 lower bias is not a tensor")
    return value


class _ProbeRecorder:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.current_evaluation: int | None = None
        self.evaluations: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.relu_calls = 0
        self.conv_calls = 0
        self.replacement_count = 0
        self.fallback_count = 0
        self.eager_count = 0
        self.native_shadow_count = 0
        self.device_before = int(torch_module.cuda.current_device())
        self.stream_before = int(torch_module.cuda.current_stream().cuda_stream)
        self.device_after: int | None = None
        self.stream_after: int | None = None
        self.topology: dict[str, object] = {}

    @contextmanager
    def install(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes.get(TARGET_RELU)
        conv = nodes.get(TARGET_CONV)
        if relu is None or conv is None:
            raise ValueError("MR3-0 target nodes are absent")
        relu_inputs = getattr(relu, "inputs", ())
        self.topology = {
            "provider_start_node": TARGET_START,
            "relu_name": TARGET_RELU,
            "relu_class": type(relu).__name__,
            "conv_name": TARGET_CONV,
            "conv_class": type(conv).__name__,
            "relu_input_is_conv": bool(relu_inputs and relu_inputs[0] is conv),
        }
        original_relu = relu.bound_backward
        original_conv = conv.bound_backward

        def relu_wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original_relu(*args, **kwargs)
            start_node = kwargs.get("start_node")
            start_name = str(getattr(start_node, "name", ""))
            if self.current_evaluation is None or start_name != TARGET_START:
                return result
            if self.pending is not None:
                raise ValueError("MR3-0 P-anchor ReLU repeated before Conv")
            if len(args) < 3:
                raise ValueError("MR3-0 ReLU arguments differ")
            incoming = args[0]
            preactivation = args[2]
            alpha = getattr(relu, "alpha", {}).get(TARGET_START)
            indices = getattr(relu, "alpha_indices", None)
            if not self.torch.is_tensor(alpha) or not isinstance(indices, list):
                raise ValueError("MR3-0 compressed alpha layout differs")
            index_tensors = list(indices)
            if not all(self.torch.is_tensor(item) for item in index_tensors):
                raise TypeError("MR3-0 alpha feature index differs")
            feature_tuples = set(
                zip(*(item.detach().cpu().tolist() for item in index_tensors))
            )
            beta_tensors = []
            for owner in (relu, conv):
                for attribute in ("sparse_betas", "beta", "split_beta"):
                    beta_tensors.extend(
                        _walk_tensor_values(getattr(owner, attribute, None), self.torch)
                    )
            output_a = _extract_lower_a(result, self.torch)
            row: dict[str, Any] = {
                "evaluation_ordinal": self.current_evaluation,
                "start_node": start_name,
                "relu_name": TARGET_RELU,
                "conv_name": TARGET_CONV,
                "relu_incoming_lower_a": _tensor_receipt(incoming, self.torch),
                "preactivation_lower": _tensor_receipt(preactivation.lower, self.torch),
                "preactivation_upper": _tensor_receipt(preactivation.upper, self.torch),
                "compressed_alpha": _tensor_receipt(alpha, self.torch),
                "alpha_feature_index_shapes": [
                    [int(dimension) for dimension in item.shape]
                    for item in index_tensors
                ],
                "alpha_feature_index_unique_count": len(feature_tuples),
                "target_beta_tensor_count": len(beta_tensors),
                "target_beta_numel": sum(int(item.numel()) for item in beta_tensors),
                "relu_output_lower_a": _tensor_receipt(output_a, self.torch),
                "relu_lower_bias": _tensor_receipt(
                    _extract_lower_bias(result, self.torch), self.torch
                ),
            }
            self.pending = row
            self.relu_calls += 1
            return result

        def conv_wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.current_evaluation is None:
                return original_conv(*args, **kwargs)
            if self.pending is None:
                raise ValueError("MR3-0 Conv arrived without target ReLU")
            if len(args) < 5:
                raise ValueError("MR3-0 Conv arguments differ")
            incoming = args[0]
            weight = args[3].lower
            bias = args[4].lower
            result = original_conv(*args, **kwargs)
            self.pending.update(
                {
                    "conv_input_lower_a": _tensor_receipt(incoming, self.torch),
                    "conv_weight": _tensor_receipt(weight, self.torch),
                    "conv_bias": _tensor_receipt(bias, self.torch),
                    "conv_output_lower_a": _tensor_receipt(
                        _extract_lower_a(result, self.torch), self.torch
                    ),
                    "conv_lower_bias": _tensor_receipt(
                        _extract_lower_bias(result, self.torch), self.torch
                    ),
                }
            )
            self.evaluations.append(self.pending)
            self.pending = None
            self.conv_calls += 1
            return result

        relu.bound_backward = relu_wrapped
        conv.bound_backward = conv_wrapped
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            conv.bound_backward = original_conv
            self.device_after = int(self.torch.cuda.current_device())
            self.stream_after = int(self.torch.cuda.current_stream().cuda_stream)
            if self.pending is not None:
                raise ValueError("MR3-0 ReLU-to-Conv observation remained partial")

    def receipt(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "evaluations": self.evaluations,
            "counters": {
                "outer_exact_call_count": 1,
                "inner_evaluation_count": len(self.evaluations),
                "relu_original_call_count": self.relu_calls,
                "conv_original_call_count": self.conv_calls,
                "replacement_count": self.replacement_count,
                "fallback_count": self.fallback_count,
                "eager_count": self.eager_count,
                "native_shadow_count": self.native_shadow_count,
            },
            "device_before": self.device_before,
            "device_after": self.device_after,
            "stream_before": self.stream_before,
            "stream_after": self.stream_after,
        }


class _ExactCallTracker:
    def __init__(self, torch_module: Any, *, mode: str) -> None:
        self.torch = torch_module
        self.mode = mode
        self.stack: list[int] = []
        self.active_outer = False
        self.outer_count = 0
        self.inner_states: list[list[dict[str, object]]] = []
        self.outer_result_state: list[dict[str, object]] | None = None
        self.final_target_alpha_state: dict[str, object] | None = None
        self.final_module_state: list[dict[str, object]] | None = None
        self.inner_hashes: list[str] = []
        self.outer_result_hash: str | None = None
        self.final_target_alpha_hash: str | None = None
        self.final_module_state_hash: str | None = None
        self.probe: _ProbeRecorder | None = None

    @contextmanager
    def install(self, bounded_module: Any) -> Iterator[None]:
        original = bounded_module.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            method = str(kwargs.get("method", "backward"))
            phase = _phase_from_stack(method)
            is_outer = (
                not self.stack
                and phase == "beta_split"
                and "optimized" in method.lower()
            )
            is_inner = (
                self.active_outer
                and len(self.stack) == 1
                and phase == "beta_split"
                and method.lower() == "backward"
            )
            if is_outer:
                if self.outer_count:
                    raise ValueError("MR3-0 beta optimized exact call repeated")
                self.outer_count += 1
                self.active_outer = True
                self.probe = (
                    _ProbeRecorder(self.torch) if self.mode == "probe" else None
                )
            call_id = len(self.stack)
            self.stack.append(call_id)
            if is_inner and self.probe is not None:
                self.probe.current_evaluation = len(self.inner_states)
            hook = (
                self.probe.install(instance)
                if is_outer and self.probe is not None
                else nullcontext()
            )
            try:
                with hook:
                    result = original(instance, *args, **kwargs)
                if is_inner:
                    state = _result_state(result, self.torch)
                    self.inner_states.append(state)
                    self.inner_hashes.append(canonical_hash(state))
                if is_outer:
                    self.outer_result_state = _result_state(result, self.torch)
                    self.outer_result_hash = canonical_hash(self.outer_result_state)
                    self.final_target_alpha_state = _tensor_state(
                        _target_alpha(instance, self.torch), self.torch
                    )
                    self.final_target_alpha_hash = canonical_hash(
                        self.final_target_alpha_state
                    )
                    self.final_module_state = _module_state(instance, self.torch)
                    self.final_module_state_hash = canonical_hash(
                        self.final_module_state
                    )
                return result
            finally:
                if is_inner and self.probe is not None:
                    self.probe.current_evaluation = None
                if self.stack.pop() != call_id:
                    raise RuntimeError("MR3-0 compute_bounds stack differs")
                if is_outer:
                    self.active_outer = False

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
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped]

    if not torch.cuda.is_available():
        raise RuntimeError("MR3-0 requires CUDA")
    tracker = _ExactCallTracker(torch, mode=args.mode)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr3-hook-property-"
    ) as workspace:
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
        with tracker.install(BoundedModule):
            solver = ABCrownSolver(str(args.model), config=config)
            result = solver.verify(
                constraints=IOConstraints(vnnlib_path=str(isolated_property))
            )
    if (
        tracker.outer_count != 1
        or len(tracker.inner_states) != 10
        or tracker.outer_result_hash is None
        or tracker.outer_result_state is None
        or tracker.final_target_alpha_hash is None
        or tracker.final_target_alpha_state is None
        or tracker.final_module_state_hash is None
        or tracker.final_module_state is None
    ):
        raise ValueError("MR3-0 exact-call tracker did not close")
    record: dict[str, object] = {
        "schema_version": MR3_HOOK_WORKER_SCHEMA,
        "pair_index": args.pair_index,
        "position": args.position,
        "mode": args.mode,
        "source": {
            "abcrown_commit": _git(args.abcrown_root, "rev-parse", "HEAD"),
            "auto_lirpa_commit": _git(
                args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD"
            ),
            "vnncomp_commit": _git(args.benchmark_root, "rev-parse", "HEAD"),
            "model_sha256": _sha256(args.model),
            "property_sha256": _sha256(args.property),
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "property_cache": "cold_isolated_copy",
        },
        "solver_result": {
            "status": str(result.status),
            "success": bool(result.success),
            "visited_domains": _visited_domains(result),
        },
        "outer_beta_exact_call_count": tracker.outer_count,
        "inner_beta_evaluation_count": len(tracker.inner_states),
        "outer_result_hash": tracker.outer_result_hash,
        "outer_result_state": tracker.outer_result_state,
        "inner_result_hashes": tracker.inner_hashes,
        "inner_result_states": tracker.inner_states,
        "final_target_alpha_hash": tracker.final_target_alpha_hash,
        "final_target_alpha_state": tracker.final_target_alpha_state,
        "final_module_state_hash": tracker.final_module_state_hash,
        "final_module_state": tracker.final_module_state,
        "hook": tracker.probe.receipt() if tracker.probe is not None else None,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    record["worker_hash"] = canonical_hash(record)
    _write_json(args.result_json, record)
    print(_json_text({"status": result.status, "mode": args.mode}))


def _validate_inputs(benchmark_root: Path, abcrown_root: Path, python: Path) -> None:
    checks = (
        _git(benchmark_root, "rev-parse", "HEAD") == VNNCOMP_COMMIT,
        _git(abcrown_root, "rev-parse", "HEAD") == ABCROWN_COMMIT,
        _git(abcrown_root / "auto_LiRPA", "rev-parse", "HEAD") == AUTO_LIRPA_COMMIT,
        python.is_file(),
        _sha256(benchmark_root / MODEL_RELATIVE_PATH) == MODEL_SHA256,
        _sha256(benchmark_root / PROPERTY_RELATIVE_PATH) == PROPERTY_SHA256,
    )
    if not all(checks):
        raise ValueError("MR3-0 frozen inputs differ")


def _external_env() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("BOUNDFLOW_ROOT", "PYTHONPATH", "TVM_HOME", "TVM_LIBRARY_PATH"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _readme() -> str:
    return (
        "# MR3-0 Provider Hook Feasibility\n\n"
        "This artifact proves a pass-through node hook can bind the real beta-split "
        "optimized exact call at `/49`, `/input-24 -> /input-20` without changing "
        "provider semantics. It records no timing and makes no performance claim.\n"
    )


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "candidate_bridge_implementation_open": summary[
            "candidate_bridge_implementation_open"
        ],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_clean():
        raise ValueError("MR3-0 code paths must be clean before formal generation")
    artifact = args.artifact_dir.resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    benchmark_root = args.benchmark_root.resolve()
    abcrown_root = args.abcrown_root.resolve()
    python = Path(os.path.abspath(args.abcrown_python))
    _validate_inputs(benchmark_root, abcrown_root, python)
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr3-hook-runs-") as workspace:
        temporary = Path(workspace)
        for ordinal, (pair_index, position, mode) in enumerate(EXPECTED_RUNS):
            result_json = temporary / f"run_{ordinal:02d}.json"
            completed = subprocess.run(
                (
                    str(python),
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
                    "--pair-index",
                    str(pair_index),
                    "--position",
                    str(position),
                    "--mode",
                    mode,
                    "--result-json",
                    str(result_json),
                ),
                cwd=ROOT,
                env=_external_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
            print(completed.stdout[-2000:], flush=True)
            if completed.returncode or not result_json.is_file():
                raise RuntimeError(
                    f"MR3-0 worker {ordinal} failed: {completed.stdout[-8000:]}"
                )
            runs.append(_load_json(result_json))
    raw = {"schema_version": MR3_HOOK_SCHEMA, "runs": runs}
    summary = derive_summary(raw)
    _write_json(artifact / "raw.json", raw)
    _write_json(artifact / "summary.json", summary)
    replay_result = _replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        _json_text(replay_result) + "\n", encoding="utf-8"
    )
    (artifact / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": MR3_HOOK_SCHEMA,
        "source_git_head": _git(ROOT, "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: _sha256(artifact / name) for name in ARTIFACT_FILES},
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return replay_result


def _verify_code_revision(manifest: Mapping[str, Any]) -> None:
    source_head = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revision, Mapping):
        raise ValueError("MR3-0 code provenance differs")
    if _git(ROOT, "rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {}
        for path in CODE_PATHS:
            blob = subprocess.run(
                ("git", "show", f"{source_head}:{path}"),
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            observed[path] = hashlib.sha256(blob).hexdigest()
    if dict(revision) != observed:
        raise ValueError("MR3-0 code revision differs")


def replay(artifact: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != MR3_HOOK_SCHEMA
        or manifest.get("manifest_hash") != canonical_hash(unsigned)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("MR3-0 manifest envelope differs")
    _verify_code_revision(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("MR3-0 artifact inventory differs")
    for name, digest in files.items():
        if _sha256(artifact / name) != digest:
            raise ValueError("MR3-0 artifact digest differs")
    summary = derive_summary(_load_json(artifact / "raw.json"))
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("MR3-0 semantic replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("MR3-0 summary projection differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != _json_text(
        result
    ) + "\n":
        raise ValueError("MR3-0 replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("MR3-0 README differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--benchmark-root", type=Path, required=True)
    worker.add_argument("--abcrown-root", type=Path, required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--pair-index", type=int, choices=(0, 1), required=True)
    worker.add_argument("--position", type=int, choices=(0, 1), required=True)
    worker.add_argument("--mode", choices=("control", "probe"), required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _worker(args)
        return
    result = (
        _generate(args)
        if args.command == "generate"
        else replay(args.artifact_dir.resolve())
    )
    print(_json_text(result))


if __name__ == "__main__":
    main()
