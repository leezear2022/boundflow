#!/usr/bin/env python3
"""Generate or replay the formal five-triplet D2-B timing artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code,too-many-boolean-expressions
# pylint: disable=line-too-long,multiple-imports,import-outside-toplevel

from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.runtime.r3_d1c_cumulative_wrapper import R3D1CCumulativeReceiptV1
from boundflow.runtime.r3_d2b_staged_backward import R3D2BStagedBackwardReceiptV1

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
OUTPUT = ROOT / "artifacts/r3-structured-owner/r3-d2b-wrapper-timing-v1"
WORKER = ROOT / "scripts/run_r3_d2b_timing_worker.py"
ORDER = (
    ("native", "d1c", "d2b"),
    ("d2b", "native", "d1c"),
    ("d1c", "d2b", "native"),
    ("native", "d2b", "d1c"),
    ("d1c", "native", "d2b"),
)
CODE = (
    "boundflow/runtime/r3_d2b_staged_backward.py",
    "scripts/run_r3_d2b_timing_worker.py",
    "scripts/run_r3_d2b_timing_artifact.py",
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


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D2B timing raw differs")
    return value


def _validate(raw: Mapping[str, Any]) -> None:
    if (
        raw["schema_version"] != "boundflow.r3-d2b-wrapper-timing-worker/v1"
        or raw["mode"] not in {"native", "d1c", "d2b"}
        or raw["warmup_count"] != 3
        or raw["sample_count"] != 30
        or raw["performance_claimed"] is not False
        or raw["region_headline_forbidden"] is not True
    ):
        raise ValueError("R3-D2B timing envelope differs")
    samples = raw["latency_ns"]
    if (
        not isinstance(samples, list)
        or len(samples) != 30
        or any(not isinstance(x, int) or x <= 0 for x in samples)
        or statistics.median(samples) != raw["median_latency_ns"]
    ):
        raise ValueError("R3-D2B timing samples differ")
    lower, alpha = raw["terminal_lower"], raw["terminal_alpha"]
    if (
        not torch.is_tensor(lower)
        or not torch.is_tensor(alpha)
        or _tensor_hash(lower) != raw["terminal_lower_sha256"]
        or _tensor_hash(alpha) != raw["terminal_alpha_sha256"]
    ):
        raise ValueError("R3-D2B timing terminal differs")
    mode = raw["mode"]
    if raw["execution"] != {
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "custom_forward_count": 0 if mode == "native" else 10,
        "custom_backward_count": 0 if mode == "native" else 10,
        "fallback_count": 0,
        "eager_candidate_count": 0,
        "native_shadow_count": 0,
        "timing_capture_count": 0,
    }:
        raise ValueError("R3-D2B timing execution differs")
    if mode == "native":
        if (
            raw["d1c_receipt"] is not None
            or raw["d2b_receipt"] is not None
            or raw["coefficient_sign_region_ms"] is not None
            or raw["region_event_count"] != 0
        ):
            raise ValueError("R3-D2B native receipt differs")
    else:
        R3D1CCumulativeReceiptV1(**raw["d1c_receipt"]).validate()
        if (
            not isinstance(raw["coefficient_sign_region_ms"], float)
            or raw["coefficient_sign_region_ms"] <= 0
            or raw["region_event_count"] != 10
        ):
            raise ValueError("R3-D2B region receipt differs")
        if mode == "d2b":
            R3D2BStagedBackwardReceiptV1(**raw["d2b_receipt"]).validate()
        elif raw["d2b_receipt"] is not None:
            raise ValueError("R3-D2B control candidate receipt differs")


def _triplet(by: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    if set(by) != {"native", "d1c", "d2b"}:
        raise ValueError("R3-D2B timing triplet differs")
    for raw in by.values():
        _validate(raw)
    native, control, candidate = by["native"], by["d1c"], by["d2b"]
    for name in (
        "run_index",
        "source_capture_sha256",
        "model_sha256",
        "plan_hash",
        "trace_hash",
        "environment",
    ):
        if len({str(x[name]) for x in by.values()}) != 1:
            raise ValueError(f"R3-D2B timing identity differs: {name}")
    for reference in (native, control):
        if (
            not torch.allclose(
                reference["terminal_lower"],
                candidate["terminal_lower"],
                atol=2e-4,
                rtol=2e-4,
            )
            or not torch.equal(
                torch.sign(reference["terminal_lower"]),
                torch.sign(candidate["terminal_lower"]),
            )
            or not torch.allclose(
                reference["terminal_alpha"],
                candidate["terminal_alpha"],
                atol=2e-5,
                rtol=2e-5,
            )
        ):
            raise ValueError("R3-D2B timing semantics differ")
    n, c, d = (float(x["median_latency_ns"]) for x in (native, control, candidate))
    region = float(control["coefficient_sign_region_ms"]) / float(
        candidate["coefficient_sign_region_ms"]
    )
    cm, dm = control["memory"], candidate["memory"]
    if (
        dm["peak_allocated"] > cm["peak_allocated"]
        or dm["peak_reserved"] > cm["peak_reserved"]
    ):
        raise ValueError("R3-D2B timing memory differs")
    return {
        "native_ms": n / 1e6,
        "d1c_ms": c / 1e6,
        "d2b_ms": d / 1e6,
        "candidate_native_speedup": n / d,
        "d1c_recovery": c / d,
        "region_speedup": region,
    }


def _summary(raws: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        _triplet({x["mode"]: x for x in raws if x["run_index"] == i}) for i in range(5)
    ]

    def geo(name: str) -> float:
        return math.exp(sum(math.log(x[name]) for x in rows) / 5)

    region_worst = min(x["region_speedup"] for x in rows)
    speed_worst = min(x["candidate_native_speedup"] for x in rows)
    speed_geo = geo("candidate_native_speedup")
    research = region_worst >= 11.8762 and speed_geo >= 1.20 and speed_worst >= 1.20
    parity = region_worst >= 11.8762 and speed_geo >= 1.0 and speed_worst >= 1.0
    result = {
        "schema_version": "boundflow.r3-d2b-wrapper-timing-summary/v1",
        "rows": rows,
        "candidate_native_geomean": speed_geo,
        "candidate_native_worst": speed_worst,
        "d1c_recovery_geomean": geo("d1c_recovery"),
        "d1c_recovery_worst": min(x["d1c_recovery"] for x in rows),
        "region_speedup_worst": region_worst,
        "region_gate": 11.8762,
        "parity_gate": parity,
        "research_gate": research,
        "provisional_verdict": (
            "RESEARCH-GATE-PASSED-PENDING-TAMPER"
            if research
            else (
                "PARITY-GATE-PASSED-PENDING-TAMPER"
                if parity
                else "NO-GO-GATE-PENDING-TAMPER"
            )
        ),
        "timing_closure_pending_tamper": True,
        "r3_3_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def replay(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_hash", None)
    if claimed != _hash(unsigned) or any(
        _file_hash(root / name) != digest for name, digest in manifest["files"].items()
    ):
        raise ValueError("R3-D2B timing manifest differs")
    protocol = json.loads((root / "protocol.json").read_text())
    pu = dict(protocol)
    ph = pu.pop("protocol_hash", None)
    if (
        ph != _hash(pu)
        or ph != manifest["protocol_hash"]
        or protocol["order"] != [list(x) for x in ORDER]
        or protocol["region_gate"] != 11.8762
    ):
        raise ValueError("R3-D2B timing protocol differs")
    summary = _summary([_load(p) for p in sorted((root / "raw").glob("*.pt"))])
    if (
        summary != json.loads((root / "summary.json").read_text())
        or summary["summary_hash"] != manifest["summary_hash"]
    ):
        raise ValueError("R3-D2B timing replay differs")
    print(
        f"R3-D2B timing replay PASS: geomean={summary['candidate_native_geomean']:.4f} worst={summary['candidate_native_worst']:.4f} verdict={summary['provisional_verdict']}"
    )
    return summary


def generate(root: Path) -> None:
    if root.exists():
        raise FileExistsError(root)
    dirty = [
        x
        for x in subprocess.check_output(
            ("git", "status", "--porcelain"), cwd=ROOT, text=True
        ).splitlines()
        if not x.endswith("docs/CIBC_for_DAC.pdf") and ".docops/ev.jsonl" not in x
    ]
    if dirty:
        raise RuntimeError(f"R3-D2B timing source dirty: {dirty}")
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    temp = Path(tempfile.mkdtemp(prefix="r3-d2b-timing-", dir=root.parent))
    try:
        (temp / "raw").mkdir()
        raws = []
        for i, modes in enumerate(ORDER):
            for j, mode in enumerate(modes):
                target = temp / "raw" / f"run-{i:02d}-{j}-{mode}.pt"
                subprocess.run(
                    (
                        sys.executable,
                        str(WORKER),
                        "--source-capture",
                        str(CAPTURE),
                        "--model",
                        str(MODEL),
                        "--mode",
                        mode,
                        "--run-index",
                        str(i),
                        "--result",
                        str(target),
                    ),
                    cwd=ROOT,
                    check=True,
                    env=os.environ.copy(),
                )
                raws.append(_load(target))
        summary = _summary(raws)
        protocol = {
            "schema_version": "boundflow.r3-d2b-wrapper-timing-protocol/v1",
            "source_revision": revision,
            "order": [list(x) for x in ORDER],
            "warmup_count": 3,
            "sample_count": 30,
            "region_gate": 11.8762,
            "parity_gate": 1.0,
            "research_gate": 1.2,
            "code_revision": {x: _file_hash(ROOT / x) for x in CODE},
        }
        protocol["protocol_hash"] = _hash(protocol)
        (temp / "protocol.json").write_text(_canonical(protocol) + "\n")
        (temp / "summary.json").write_text(_canonical(summary) + "\n")
        files = {
            str(p.relative_to(temp)): _file_hash(p)
            for p in sorted(temp.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "schema_version": "boundflow.r3-d2b-wrapper-timing-manifest/v1",
            "source_revision": revision,
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": files,
        }
        manifest["manifest_hash"] = _hash(manifest)
        (temp / "manifest.json").write_text(_canonical(manifest) + "\n")
        replay(temp)
        temp.rename(root)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output)
    else:
        generate(args.output)


if __name__ == "__main__":
    main()
