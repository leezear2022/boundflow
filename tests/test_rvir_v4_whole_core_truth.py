"""Contract and semantic-tamper tests for RVIR-v4 V4-3A truth."""

# pylint: disable=missing-function-docstring,protected-access

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from boundflow.runtime.rvir_v4_whole_core_truth import (
    compare_rvir_v4_whole_core_truth,
    validate_rvir_v4_whole_core_truth,
    whole_core_truth_metadata,
)
from scripts import run_rvir_v4_production_state_capture as capture_runner
from scripts import run_rvir_v4_whole_core_truth_artifact as artifact_runner

ARTIFACT = Path("artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1")
TRUTH = ARTIFACT / "truth.pt"


def _payload() -> dict[str, object]:
    value = torch.load(TRUTH, map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    return value


def test_formal_truth_closes_branch_input_before_return_consumption() -> None:
    payload = _payload()
    summary = validate_rvir_v4_whole_core_truth(
        payload["whole_core_truths"][0], payload["whole_post_truths"][0]  # type: ignore[index]
    )

    assert summary["lA_count"] == 6
    assert summary["kfsb_candidate_count"] == 3
    assert summary["provider_update_bounds_call_count"] == 3
    assert summary["whole_core_replacement_admitted"] is False
    assert summary["b2_same_solver_timing_admitted"] is False
    assert summary["performance_claimed"] is False


def test_tensor_digest_tamper_is_rejected() -> None:
    payload = _payload()
    core = payload["whole_core_truths"][0]  # type: ignore[index]
    post = payload["whole_post_truths"][0]  # type: ignore[index]
    core["fields"]["lb"]["value"] = core["fields"]["lb"]["value"].clone()
    core["fields"]["lb"]["value"][0, 0] += 1.0

    with pytest.raises(ValueError, match="tensor identity differs"):
        validate_rvir_v4_whole_core_truth(core, post)


def test_fully_resigned_l_a_tamper_fails_semantic_parity() -> None:
    payload = _payload()
    clean_core = payload["whole_core_truths"][0]  # type: ignore[index]
    clean_post = payload["whole_post_truths"][0]  # type: ignore[index]
    changed_core = deepcopy(clean_core)
    record = changed_core["branch_trace"]["input"]["lAs"]["_data"]["/48"]
    record["value"] = record["value"].clone()
    record["value"].flatten()[0] += 1.0
    record["content_sha256"] = production_tensor_sha256(record["value"])
    semantic = {
        key: value for key, value in changed_core.items() if key != "truth_hash"
    }
    changed_core["truth_hash"] = artifact_runner._canonical_hash(
        whole_core_truth_metadata(semantic)
    )
    validate_rvir_v4_whole_core_truth(changed_core, clean_post)

    with pytest.raises(ValueError, match="numeric semantic parity differs"):
        compare_rvir_v4_whole_core_truth(
            changed_core, clean_post, clean_core, clean_post
        )


def test_worker_modes_are_named_independently() -> None:
    assert (
        capture_runner.WHOLE_CORE_WORKER_SCHEMA_VERSION
        != capture_runner.OPTIMIZER_WORKER_SCHEMA_VERSION
    )
