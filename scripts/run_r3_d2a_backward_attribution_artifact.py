#!/usr/bin/env python3
"""Generate or replay the formal five-fresh D2-A backward attribution artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code,too-many-boolean-expressions
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d2a-backward-attribution-v1"
WORKER = ROOT / "scripts/run_r3_d2a_backward_attribution_worker.py"
D1C_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1c-wrapper-formal-v1"
D1B_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1b-schedule-formal-v1"
RUN_COUNT = 5
COOLDOWN_SECONDS = 30
PHASES = (
    "backward",
    "coefficient_sign",
    "effective_value",
    "recompute_a26",
    "terminal_backward_residual",
)
CODE_PATHS = (
    "boundflow/runtime/r3_compiled_p_alpha_vjp.py",
    "boundflow/runtime/r3_optimizer_trajectory_timing.py",
    "boundflow/runtime/r3_d1c_cumulative_wrapper.py",
    "scripts/run_r3_d2a_backward_attribution_worker.py",
    "scripts/run_r3_d2a_backward_attribution_artifact.py",
    "scripts/probe_r3_d2a_backward_attribution_tamper.py",
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _clean() -> None:
    dirty = [
        line
        for line in _git("status", "--porcelain").splitlines()
        if not line.endswith("docs/CIBC_for_DAC.pdf") and ".docops/ev.jsonl" not in line
    ]
    if dirty:
        raise RuntimeError(f"R3-D2A formal source is dirty: {dirty}")


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D2A raw root differs")
    return value


def _d1c_reference(run_index: int) -> dict[str, Any]:
    matches = sorted((D1C_ARTIFACT / "raw").glob(f"run-{run_index:02d}-*-d1c.pt"))
    if len(matches) != 1:
        raise ValueError("R3-D2A D1-C reference inventory differs")
    return _load(matches[0])


def _positive_timing_list(value: object, count: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            not isinstance(item, float) or not math.isfinite(item) or item <= 0.0
            for item in value
        )
    ):
        raise ValueError("R3-D2A timing list differs")
    return value


def _validate_worker(raw: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_index",
        "source_capture_sha256",
        "model_sha256",
        "d1c_manifest_sha256",
        "plan_hash",
        "trace_hash",
        "warmup_count",
        "host_wrapper_ns",
        "formal_reference_native_ns",
        "formal_reference_d1c_ns",
        "phase_ms",
        "phase_totals_ms",
        "terminal_backward_residual_ms",
        "symbol_profile_host_ns",
        "symbol_ms",
        "terminal_lower",
        "terminal_alpha",
        "terminal_lower_sha256",
        "terminal_alpha_sha256",
        "reference_lower_diff",
        "reference_alpha_diff",
        "reference_sign_exact",
        "symbol_phase_lower_diff",
        "symbol_phase_alpha_diff",
        "execution",
        "environment",
        "single_stream_no_overlap",
        "symbol_profile_headline_forbidden",
        "diagnostic_only",
        "performance_claimed",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != "boundflow.r3-d2a-backward-attribution-worker/v1"
        or raw["run_index"] not in range(RUN_COUNT)
        or raw["warmup_count"] != 3
        or raw["single_stream_no_overlap"] is not True
        or raw["symbol_profile_headline_forbidden"] is not True
        or raw["diagnostic_only"] is not True
        or raw["performance_claimed"] is not False
    ):
        raise ValueError("R3-D2A worker envelope differs")
    host_ns = raw["host_wrapper_ns"]
    symbol_host_ns = raw["symbol_profile_host_ns"]
    reference_ns = raw["formal_reference_d1c_ns"]
    if (
        not isinstance(host_ns, int)
        or not isinstance(symbol_host_ns, int)
        or not isinstance(reference_ns, (int, float))
        or host_ns <= 0
        or symbol_host_ns <= 0
        or not 0.85 <= host_ns / float(reference_ns) <= 1.15
    ):
        raise ValueError("R3-D2A host timing sanity differs")
    phase_ms = raw["phase_ms"]
    totals = raw["phase_totals_ms"]
    if not isinstance(phase_ms, Mapping) or not isinstance(totals, Mapping):
        raise TypeError("R3-D2A phase payload differs")
    if set(phase_ms) != {
        "forward",
        "backward",
        "coefficient_sign",
        "effective_value",
        "recompute_a26",
    }:
        raise ValueError("R3-D2A phase inventory differs")
    rebuilt = {
        name: sum(_positive_timing_list(values, 10))
        for name, values in phase_ms.items()
    }
    if rebuilt != totals:
        raise ValueError("R3-D2A phase total differs")
    residual = (
        rebuilt["backward"]
        - rebuilt["coefficient_sign"]
        - rebuilt["effective_value"]
        - rebuilt["recompute_a26"]
    )
    if (
        residual <= 0.0
        or abs(residual - float(raw["terminal_backward_residual_ms"])) > 1e-9
    ):
        raise ValueError("R3-D2A backward conservation differs")
    if rebuilt["forward"] + rebuilt["backward"] > host_ns / 1_000_000.0:
        raise ValueError("R3-D2A host/phase containment differs")
    symbols = raw["symbol_ms"]
    if not isinstance(symbols, Mapping) or not symbols:
        raise TypeError("R3-D2A symbol ledger differs")
    for name, values in symbols.items():
        if not isinstance(name, str) or not (
            name.startswith("b1:") or name.startswith("b2:")
        ):
            raise ValueError("R3-D2A symbol name differs")
        _positive_timing_list(values, len(values))
    lower, alpha = raw["terminal_lower"], raw["terminal_alpha"]
    if (
        not torch.is_tensor(lower)
        or not torch.is_tensor(alpha)
        or tuple(lower.shape) != (6, 1)
        or tuple(alpha.shape) != (2, 1, 6, 86)
        or _tensor_hash(lower) != raw["terminal_lower_sha256"]
        or _tensor_hash(alpha) != raw["terminal_alpha_sha256"]
        or float(raw["reference_lower_diff"]) > 2e-4
        or float(raw["reference_alpha_diff"]) > 2e-5
        or raw["reference_sign_exact"] is not True
        or float(raw["symbol_phase_lower_diff"]) > 2e-4
        or float(raw["symbol_phase_alpha_diff"]) > 2e-5
    ):
        raise ValueError("R3-D2A terminal semantics differ")
    reference = _d1c_reference(int(raw["run_index"]))
    reference_lower = reference["terminal_lower"]
    reference_alpha = reference["terminal_alpha"]
    if not torch.is_tensor(reference_lower) or not torch.is_tensor(reference_alpha):
        raise TypeError("R3-D2A frozen reference differs")
    rebuilt_lower_diff = float((lower - reference_lower).abs().max().item())
    rebuilt_alpha_diff = float((alpha - reference_alpha).abs().max().item())
    rebuilt_sign = torch.equal(torch.sign(lower), torch.sign(reference_lower))
    if (
        rebuilt_lower_diff != float(raw["reference_lower_diff"])
        or rebuilt_alpha_diff != float(raw["reference_alpha_diff"])
        or rebuilt_sign is not raw["reference_sign_exact"]
    ):
        raise ValueError("R3-D2A frozen terminal replay differs")
    if raw["execution"] != {
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "custom_forward_count": 10,
        "custom_backward_count": 10,
        "fallback_count": 0,
        "eager_candidate_count": 0,
        "native_shadow_count": 0,
    }:
        raise ValueError("R3-D2A execution counters differ")


def _required(
    region_ms: float, total_ms: float, target_ms: float
) -> tuple[bool, float | None]:
    other_ms = total_ms - region_ms
    if target_ms <= other_ms:
        return False, None
    return True, region_ms / (target_ms - other_ms)


def _row(raw: Mapping[str, Any]) -> dict[str, Any]:
    _validate_worker(raw)
    total_ms = float(raw["formal_reference_d1c_ns"]) / 1_000_000.0
    native_ms = float(raw["formal_reference_native_ns"]) / 1_000_000.0
    totals = raw["phase_totals_ms"]
    assert isinstance(totals, Mapping)
    phase_values = {
        "backward": float(totals["backward"]),
        "coefficient_sign": float(totals["coefficient_sign"]),
        "effective_value": float(totals["effective_value"]),
        "recompute_a26": float(totals["recompute_a26"]),
        "terminal_backward_residual": float(raw["terminal_backward_residual_ms"]),
    }
    phases = {}
    for name, region_ms in phase_values.items():
        parity_physical, parity_required = _required(region_ms, total_ms, native_ms)
        research_physical, research_required = _required(
            region_ms, total_ms, native_ms / 1.20
        )
        phases[name] = {
            "duration_ms": region_ms,
            "formal_wrapper_share": region_ms / total_ms,
            "parity_physical": parity_physical,
            "parity_required_speedup": parity_required,
            "research_physical": research_physical,
            "research_required_speedup": research_required,
        }
    symbol_totals = {name: sum(values) for name, values in raw["symbol_ms"].items()}
    return {
        "run_index": raw["run_index"],
        "profile_host_ms": float(raw["host_wrapper_ns"]) / 1_000_000.0,
        "formal_d1c_ms": total_ms,
        "formal_native_ms": native_ms,
        "profile_sanity_ratio": float(raw["host_wrapper_ns"])
        / float(raw["formal_reference_d1c_ns"]),
        "phases": phases,
        "symbol_profile_host_ms": float(raw["symbol_profile_host_ns"]) / 1_000_000.0,
        "symbol_totals_ms": symbol_totals,
    }


def _summary(raws: list[dict[str, Any]]) -> dict[str, Any]:
    if len(raws) != RUN_COUNT or [raw["run_index"] for raw in raws] != list(
        range(RUN_COUNT)
    ):
        raise ValueError("R3-D2A fresh inventory differs")
    for raw in raws:
        _validate_worker(raw)
    environments = {_canonical(raw["environment"]) for raw in raws}
    identities = {
        (
            raw["plan_hash"],
            raw["trace_hash"],
            raw["source_capture_sha256"],
            raw["model_sha256"],
            raw["d1c_manifest_sha256"],
        )
        for raw in raws
    }
    symbol_counts = {
        tuple(sorted((name, len(values)) for name, values in raw["symbol_ms"].items()))
        for raw in raws
    }
    if len(environments) != 1 or len(identities) != 1 or len(symbol_counts) != 1:
        raise ValueError("R3-D2A fresh identity/count receipt differs")
    rows = [_row(raw) for raw in raws]
    route_table = {}
    for phase in PHASES:
        values = [row["phases"][phase] for row in rows]
        required = [value["research_required_speedup"] for value in values]
        physical = all(value["research_physical"] for value in values)
        cap = 15.50 if phase == "coefficient_sign" else 10.0
        maximum_required = (
            max(float(value) for value in required if value is not None)
            if physical
            else None
        )
        admitted = (
            physical
            and min(float(value["formal_wrapper_share"]) for value in values) >= 0.20
            and maximum_required is not None
            and maximum_required <= cap
        )
        route_table[phase] = {
            "minimum_share": min(
                float(value["formal_wrapper_share"]) for value in values
            ),
            "maximum_parity_required": (
                max(
                    float(value["parity_required_speedup"])
                    for value in values
                    if value["parity_required_speedup"] is not None
                )
                if all(value["parity_physical"] for value in values)
                else None
            ),
            "maximum_research_required": maximum_required,
            "admission_cap": cap,
            "admitted": admitted,
        }
    dominant_symbols = []
    for row in rows:
        ordered = sorted(row["symbol_totals_ms"].items(), key=lambda item: -item[1])
        dominant_symbols.append([name for name, _value in ordered[:3]])
    expected_prefix = [
        "b1:boundflow_r31b1_residual6",
        "b1:boundflow_r31b1_residual11",
    ]
    signature_mapping = all(
        symbols[:2] == expected_prefix for symbols in dominant_symbols
    )
    d2b_open = route_table["coefficient_sign"]["admitted"] and signature_mapping
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-d2a-backward-attribution-summary/v1",
        "run_count": RUN_COUNT,
        "rows": rows,
        "route_table": route_table,
        "dominant_symbols": dominant_symbols,
        "d1b_residual_signature_mapping": signature_mapping,
        "selected_route": (
            "coefficient-sign-staged-residual-reuse" if d2b_open else None
        ),
        "d2b_open": d2b_open,
        "r3_3_open": False,
        "same_solver_open": False,
        "diagnostic_only": True,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def _protocol(revision: str, capture: Path, model: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "boundflow.r3-d2a-backward-attribution-protocol/v1",
        "source_revision": revision,
        "run_count": RUN_COUNT,
        "warmup_count": 3,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "source_capture_sha256": _file_hash(capture),
        "model_sha256": _file_hash(model),
        "d1c_manifest_sha256": _file_hash(D1C_ARTIFACT / "manifest.json"),
        "d1b_manifest_sha256": _file_hash(D1B_ARTIFACT / "manifest.json"),
        "generic_required_cap": 10.0,
        "verified_residual_required_cap": 15.50,
        "minimum_share": 0.20,
        "phase_profile_sanity": [0.85, 1.15],
        "single_stream_no_overlap": True,
        "symbol_profile_headline_forbidden": True,
        "code_revision": {name: _file_hash(ROOT / name) for name in CODE_PATHS},
    }
    result["protocol_hash"] = _hash(result)
    return result


def generate(output: Path, capture: Path, model: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3-D2A artifact exists: {output}")
    _clean()
    protocol = _protocol(_git("rev-parse", "HEAD"), capture, model)
    temporary = Path(tempfile.mkdtemp(prefix="r3-d2a-formal-", dir=output.parent))
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir(parents=True)
        raws = []
        for run_index in range(RUN_COUNT):
            if run_index:
                time.sleep(COOLDOWN_SECONDS)
            target = raw_dir / f"run-{run_index:02d}.pt"
            subprocess.run(
                (
                    sys.executable,
                    str(WORKER),
                    "--source-capture",
                    str(capture),
                    "--model",
                    str(model),
                    "--run-index",
                    str(run_index),
                    "--result",
                    str(target),
                ),
                cwd=ROOT,
                check=True,
                env=os.environ.copy(),
            )
            raws.append(_load(target))
        summary = _summary(raws)
        (temporary / "protocol.json").write_text(
            _canonical(protocol) + "\n", encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            _canonical(summary) + "\n", encoding="utf-8"
        )
        files = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "schema_version": "boundflow.r3-d2a-backward-attribution-manifest/v1",
            "source_revision": protocol["source_revision"],
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = _hash(manifest)
        (temporary / "manifest.json").write_text(
            _canonical(manifest) + "\n", encoding="utf-8"
        )
        replay(temporary)
        temporary.rename(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(artifact: Path) -> dict[str, Any]:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if (
        manifest_hash != _hash(unsigned_manifest)
        or manifest.get("schema_version")
        != "boundflow.r3-d2a-backward-attribution-manifest/v1"
    ):
        raise ValueError("R3-D2A manifest differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or any(
        _file_hash(artifact / name) != digest for name, digest in files.items()
    ):
        raise ValueError("R3-D2A file digest differs")
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol_hash != _hash(unsigned_protocol)
        or protocol_hash != manifest["protocol_hash"]
    ):
        raise ValueError("R3-D2A protocol hash differs")
    frozen = {
        "run_count": RUN_COUNT,
        "warmup_count": 3,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "generic_required_cap": 10.0,
        "verified_residual_required_cap": 15.50,
        "minimum_share": 0.20,
        "phase_profile_sanity": [0.85, 1.15],
        "single_stream_no_overlap": True,
        "symbol_profile_headline_forbidden": True,
    }
    if any(protocol.get(name) != value for name, value in frozen.items()):
        raise ValueError("R3-D2A frozen protocol differs")
    raws = [_load(path) for path in sorted((artifact / "raw").glob("*.pt"))]
    summary = _summary(raws)
    if (
        summary != json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-D2A semantic replay differs")
    route = summary["route_table"]["coefficient_sign"]
    print(
        f"R3-D2A replay PASS: share={route['minimum_share']:.4f} "
        f"required={route['maximum_research_required']:.4f}x d2b_open={summary['d2b_open']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output.absolute())
    else:
        generate(
            args.output.absolute(),
            args.source_capture.absolute(),
            args.model.absolute(),
        )


if __name__ == "__main__":
    main()
