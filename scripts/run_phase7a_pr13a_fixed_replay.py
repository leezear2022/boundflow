"""Generate the PR-13A real-BaB fixed-query replay artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.bab_query import FixedBabQueryRecorder, replay_fixed_query_trace
from boundflow.runtime.task_executor import InputSpec

ARTIFACT_SCHEMA_VERSION = "boundflow.pr13a-fixed-replay-artifact/v1"
REPLAY_SCHEMA_VERSION = "boundflow.pr13a-fixed-replay-result/v1"


def make_workload() -> tuple[BFTaskModule, InputSpec, BabConfig]:
    """Return the deterministic real-BaB smoke workload shared by PR-13A/B."""

    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="linear",
                name="linear1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="linear",
                name="linear2",
                inputs=["r1", "W2", "b2"],
                outputs=["h2"],
            ),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(
                op_type="linear",
                name="linear3",
                inputs=["r2", "W3", "b3"],
                outputs=["out"],
            ),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    params = {
        "W1": torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        "b1": torch.tensor([0.1, -0.1], dtype=torch.float32),
        "W2": torch.tensor([[1.0, -0.5], [-0.25, 1.0]], dtype=torch.float32),
        "b2": torch.tensor([-0.2, 0.15], dtype=torch.float32),
        "W3": torch.tensor([[1.0, -0.75]], dtype=torch.float32),
        "b3": torch.tensor([-0.1], dtype=torch.float32),
    }
    module = BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})
    spec = InputSpec.linf(
        value_name="input",
        center=torch.tensor([[0.0]], dtype=torch.float32),
        eps=1.0,
    )
    config = BabConfig(
        max_nodes=9,
        oracle="alpha_beta",
        node_batch_size=1,
        enable_node_eval_cache=False,
        alpha_steps=0,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.35,
        tol=1e-8,
    )
    return module, spec, config


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(out_dir: Path) -> None:
    """Record one real search, replay it, and write immutable evidence files."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    module, spec, config = make_workload()
    recorder = FixedBabQueryRecorder()
    result = solve_bab_mlp(
        module,
        spec,
        config=config,
        query_recorder=recorder,
    )
    recorder.validate_complete()
    comparisons = replay_fixed_query_trace(module, recorder.entries)

    trace_path = out_dir / "query_trace.jsonl"
    recorder.write_jsonl(trace_path)
    replay_path = out_dir / "replay.jsonl"
    replay_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": REPLAY_SCHEMA_VERSION,
                    "query_id": comparison.query_id,
                    "status_match": comparison.status_match,
                    "branch_match": comparison.branch_match,
                    "finite": comparison.finite,
                    "ordered": comparison.ordered,
                    "allclose": comparison.allclose,
                    "max_abs_diff": comparison.max_abs_diff,
                    "passed": comparison.passed,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for comparison in comparisons
        ),
        encoding="utf-8",
    )
    all_passed = bool(comparisons) and all(item.passed for item in comparisons)
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok" if all_passed else "fail",
        "scope": "PR-13A contract/replay smoke; not a performance claim",
        "workload": "deterministic-two-relu-mlp-real-bab-driver",
        "solver_result": {
            "status": result.status,
            "nodes_visited": result.nodes_visited,
            "nodes_evaluated": result.nodes_evaluated,
            "nodes_expanded": result.nodes_expanded,
        },
        "query_count": len(recorder.entries),
        "replay_passed": sum(item.passed for item in comparisons),
        "replay_failed": sum(not item.passed for item in comparisons),
        "max_abs_diff": max(item.max_abs_diff for item in comparisons),
        "query_loss": result.nodes_visited - len(recorder.entries),
        "duplicate_query_ids": len(recorder.entries)
        - len({entry.query.query_id for entry in recorder.entries}),
    }
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "torch_version": torch.__version__,
        "device": str(spec.center.device),
        "seed": 0,
        "command": (
            "python scripts/run_phase7a_pr13a_fixed_replay.py " f"--out-dir {out_dir}"
        ),
        "files": {
            path.name: _sha256(path) for path in (trace_path, replay_path, summary_path)
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    if not all_passed:
        raise RuntimeError("fixed-query replay mismatch; inspect replay.jsonl")


def main() -> None:
    """Parse the immutable artifact destination and run the smoke workflow."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
