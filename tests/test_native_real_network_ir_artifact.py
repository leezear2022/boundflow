"""Semantic identity gates for the native real-network artifact."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.runtime.abcrown_adapter import (
    deserialize_intermediate_bounds,
    serialize_intermediate_bounds,
)
from scripts.run_native_real_network_ir_artifact import (
    ABCROWN_COMMIT,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    SOURCE_PAYLOAD_SCHEMA_VERSION,
    VNNLIB_SHA256,
    _validate_source_capture,
)


def _frozen_payload() -> dict:
    root = Path(__file__).resolve().parents[1]
    payload = torch.load(
        root / "artifacts/native-real-network-ir/"
        "vnncomp21-resnet2b-prop0-cpu-v1/payload.pt",
        map_location="cpu",
        weights_only=True,
    )
    payload["schema_version"] = SOURCE_PAYLOAD_SCHEMA_VERSION
    return payload


def _source_manifest() -> dict:
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "payload_schema_version": SOURCE_PAYLOAD_SCHEMA_VERSION,
        "status": "ok",
        "abcrown_commit": ABCROWN_COMMIT,
        "model_sha256": MODEL_SHA256,
        "vnnlib_sha256": VNNLIB_SHA256,
        "capture": {
            "method": "crown",
            "solver_phase": "alpha_crown_initialization",
            "intermediate_bound_count": 6,
            "intermediate_bounds_hash": INTERMEDIATE_BOUNDS_SHA256,
            "intermediate_bound_source": "external_verifier",
            "relu_lower_slope_policy": "adaptive",
        },
        "boundflow": {
            "pytorch_eager": {
                "lower_vs_external": {
                    "allclose": True,
                    "sign_agreement": 9,
                }
            }
        },
    }


def test_source_capture_rejects_rehashed_external_bound_tamper() -> None:
    payload = _frozen_payload()
    _validate_source_capture(_source_manifest(), payload)
    bounds = deserialize_intermediate_bounds(payload["external_intermediate_bounds"])
    tampered = (
        replace(bounds[0], lower=bounds[0].lower - 1.0),
        *bounds[1:],
    )
    payload["external_intermediate_bounds"] = serialize_intermediate_bounds(tampered)

    with pytest.raises(ValueError, match="external intervals differ"):
        _validate_source_capture(_source_manifest(), payload)
