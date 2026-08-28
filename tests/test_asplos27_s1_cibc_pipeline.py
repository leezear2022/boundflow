"""Contracts for the ASPLOS'27 S1 canonical mixed Relax/TIR path."""

# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=import-outside-toplevel,protected-access

from dataclasses import replace
from pathlib import Path
import argparse

import pytest
import torch

from boundflow.backends.tvm.relax_interval_task_ops import (
    IntervalTaskLoweringConfig,
)
from boundflow.ir.task import BoundTask, BufferSpec, StoragePlan, TaskKind, TaskOp
from boundflow.runtime.asplos27_s1_cibc_pipeline import (
    PreparedS1CIBCCUDAGraphV1,
    S1CIBCCompileReceiptV1,
    prepare_s1_cibc_program_v1,
    specialize_storage_plan_batch_v1,
)


def _small_task() -> BoundTask:
    return BoundTask(
        task_id="small",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                "conv2d",
                "conv",
                ["input", "weight", "bias"],
                ["output"],
                attrs={
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                },
            )
        ],
        input_values=["input"],
        output_values=["output"],
        params=["weight", "bias"],
    )


def test_s1_lowering_config_fails_closed_on_schedule_inventory() -> None:
    task = _small_task()
    with pytest.raises(ValueError, match="does not cover every Conv"):
        IntervalTaskLoweringConfig(conv_backend="cibc_tir").validate(task)
    with pytest.raises(ValueError, match="unknown ops"):
        IntervalTaskLoweringConfig(
            conv_backend="cibc_tir",
            cibc_threads_by_op=(("other", 128),),
        ).validate(task)
    with pytest.raises(ValueError, match="schedule differs"):
        IntervalTaskLoweringConfig(
            conv_backend="cibc_tir",
            cibc_threads_by_op=(("conv", 32),),
        ).validate(task)
    config = IntervalTaskLoweringConfig(
        conv_backend="cibc_tir",
        cibc_threads_by_op=(("conv", 128),),
    )
    config.validate(task)
    assert config.threads_for("conv") == 128


def test_s1_batch_specialization_changes_only_activation_buffers() -> None:
    storage = StoragePlan(
        buffers={
            "input": BufferSpec("input", "float32", [1, 3, 4, 4]),
            "output": BufferSpec("output", "float32", [1, 2, 4, 4]),
            "weight": BufferSpec("weight", "float32", [2, 3, 3, 3], scope="param"),
        },
        value_to_buffer={
            "input": "input",
            "output": "output",
            "weight": "weight",
        },
    )
    specialized = specialize_storage_plan_batch_v1(storage, batch_size=6)
    assert specialized.buffers["input"].shape == [6, 3, 4, 4]
    assert specialized.buffers["output"].shape == [6, 2, 4, 4]
    assert specialized.buffers["weight"].shape == [2, 3, 3, 3]
    assert storage.buffers["input"].shape == [1, 3, 4, 4]


def test_s1_compile_receipt_rejects_claim_and_missing_cublas() -> None:
    digest = "a" * 64
    receipt = S1CIBCCompileReceiptV1(
        source_task_hash=digest,
        specialized_storage_hash=digest,
        plan_hash=digest,
        source_relax_ir_hash=digest,
        lowered_relax_ir_hash=digest,
        device_source_hashes=(digest,),
        target="cuda",
        op_count=1,
        cibc_conv_ops=("conv",),
        cibc_threads_by_op=(("conv", 128),),
        cublas_partition_count=1,
        compile_ms=1.0,
    )
    receipt.validate()
    with pytest.raises(ValueError, match="compile receipt differs"):
        replace(receipt, performance_claimed=True).validate()
    with pytest.raises(ValueError, match="compile receipt differs"):
        replace(receipt, cublas_partition_count=0).validate()


def test_s1_formal_artifact_replays_and_passes_all_qualification_gates() -> None:
    from scripts import run_asplos27_s1_cibc_artifact as artifact

    root = Path("artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2")
    if not root.is_dir():
        pytest.skip("S1 formal artifact unavailable")
    result = artifact.replay(root)
    summary = artifact.load_json(root / "summary.json")
    assert result["status"] == "replay-passed"
    assert summary["status"] == "validated-s1-cibc-pipeline"
    assert summary["run_count"] == 6
    assert summary["op_count"] == 17
    assert summary["cibc_conv_coverage"] == 6
    assert summary["cublas_partition_count"] == 2
    assert summary["pipeline_speedup_geomean"] >= 2.20
    assert summary["pipeline_speedup_worst"] >= 2.00
    assert summary["pipeline_direct_propagation_geomean"] >= 0.90
    assert summary["s1_performance_admitted"] is True
    assert summary["same_solver_claimed"] is False
    assert summary["performance_claimed"] is False


def test_s1_formal_artifact_rejects_every_outer_resigned_tamper() -> None:
    root = Path("artifacts/asplos27-s1-cibc-pipeline/resnet2b-prop0-v2")
    if not root.is_dir():
        pytest.skip("S1 formal artifact unavailable")
    report = __import__("json").loads(
        (root / "tamper_report.json").read_text(encoding="utf-8")
    )
    assert report["case_count"] == report["rejected_count"] == 8
    assert all(row["outer_resigned"] is True for row in report["rows"])
    assert all(row["rejected"] is True for row in report["rows"])
    assert report["performance_claimed"] is False


def test_s1_historical_code_revision_ignores_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_asplos27_s1_cibc_artifact as artifact

    source = artifact.git("rev-parse", "HEAD")
    expected = artifact.code_revision(source)
    monkeypatch.setattr(
        artifact,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("read working tree")),
    )
    assert artifact.code_revision(source) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_s1_resnet2b_pipeline_matches_reference_and_rejects_mutation() -> None:
    from scripts import run_cibc_ibp_horizontal_worker as worker
    from boundflow.runtime.cibc_ibp_graph import run_cibc_ibp_graph_once_v1

    source = Path(
        "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
    )
    model = Path("../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx")
    if not source.is_file() or not model.is_file():
        pytest.skip("S1 production fixture unavailable")
    args = argparse.Namespace(source_capture=source, model=model, run_ordinal=0)
    module, spec, lower, upper = worker._prepare(
        args
    )  # pylint: disable=protected-access
    task = module.get_entry_task()
    schedules = tuple((op.name, 128) for op in task.ops if op.op_type == "conv2d")
    reference, _launches = run_cibc_ibp_graph_once_v1(
        module,
        input_value=spec.value_name,
        input_lower=lower,
        input_upper=upper,
        threads_per_block=None,
    )
    prepared = prepare_s1_cibc_program_v1(
        module,
        input_lower=lower,
        input_upper=upper,
        cibc_threads_by_op=schedules,
    )
    graph = PreparedS1CIBCCUDAGraphV1(prepared)
    observed, receipt = graph.run(input_lower=lower, input_upper=upper)
    torch.cuda.synchronize()
    final = reference[task.output_values[0]]
    assert torch.allclose(observed.lower, final.lower, atol=3e-4, rtol=3e-4)
    assert torch.allclose(observed.upper, final.upper, atol=3e-4, rtol=3e-4)
    assert torch.equal(torch.sign(observed.lower), torch.sign(final.lower))
    assert torch.equal(torch.sign(observed.upper), torch.sign(final.upper))
    assert receipt.cibc_conv_call_tir_count == 6
    assert receipt.cuda_graph_replay_count == 1
    assert receipt.warm_dlpack_view_count == 0
    assert receipt.fallback_count == receipt.eager_shadow_count == 0
    assert receipt.output_materialization_copy_included is False
    mutated = lower.clone()
    prepared.admit_dynamic_input(mutated, upper)
    mutated.add_(1.0)
    before = prepared.invocation_count
    with pytest.raises(ValueError, match="not admitted"):
        graph.run(input_lower=mutated, input_upper=upper)
    assert prepared.invocation_count == before
