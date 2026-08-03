"""Immutable artifact writer and verifier for Plan IR v1 selections."""

# Artifact verification intentionally keeps all fail-closed checks together.
# pylint: disable=too-many-branches,too-many-locals

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..ir.bound import BFBoundModule
from ..ir.plan import PlanInstance, PlanTemplate

PLAN_SELECTION_ARTIFACT_SCHEMA = "boundflow.plan-selection-artifact/v1"


def write_plan_selection_artifact(
    output_dir: Path,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
) -> Path:
    """Write a new immutable Bound/Template/Instance evidence directory."""

    bound_module.validate()
    template.validate(bound_module=bound_module)
    instance.validate(template=template, bound_module=bound_module)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Plan IR artifact: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "bound_module.json": bound_module.canonical_json(),
        "plan_template.json": template.canonical_json(bound_module=bound_module),
        "plan_instance.json": instance.canonical_json(
            template=template, bound_module=bound_module
        ),
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(payload + "\n", encoding="utf-8")

    manifest = {
        "schema_version": PLAN_SELECTION_ARTIFACT_SCHEMA,
        "bound_module_hash": bound_module.stable_hash(),
        "plan_template_hash": template.stable_hash(bound_module=bound_module),
        "plan_instance_hash": instance.stable_hash(
            template=template, bound_module=bound_module
        ),
        "files": {
            filename: _sha256_text(payload + "\n")
            for filename, payload in sorted(payloads.items())
        },
        "replay_contract": (
            "exact canonical Bound IR and PlanTemplate inputs are required"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def verify_plan_selection_artifact(
    output_dir: Path,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
) -> PlanInstance:
    """Verify hashes and replay an artifact against exact typed inputs."""

    bound_module.validate()
    template.validate(bound_module=bound_module)
    required = (
        "bound_module.json",
        "plan_template.json",
        "plan_instance.json",
        "manifest.json",
    )
    missing = tuple(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise ValueError(f"Plan IR artifact is missing files: {missing}")

    texts = {
        name: (output_dir / name).read_text(encoding="utf-8") for name in required[:-1]
    }
    if texts["bound_module.json"] != bound_module.canonical_json() + "\n":
        raise ValueError("artifact Bound IR does not match the supplied module")
    if (
        texts["plan_template.json"]
        != template.canonical_json(bound_module=bound_module) + "\n"
    ):
        raise ValueError("artifact PlanTemplate does not match the supplied template")

    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise ValueError("artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    if manifest.get("schema_version") != PLAN_SELECTION_ARTIFACT_SCHEMA:
        raise ValueError("unsupported Plan IR artifact schema")
    expected_hashes = {
        "bound_module_hash": bound_module.stable_hash(),
        "plan_template_hash": template.stable_hash(bound_module=bound_module),
    }
    for key, expected in expected_hashes.items():
        if manifest.get(key) != expected:
            raise ValueError(f"artifact {key} mismatch")
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, dict):
        raise ValueError("artifact manifest files must be a JSON object")
    for filename, text in texts.items():
        if file_hashes.get(filename) != _sha256_text(text):
            raise ValueError(f"artifact file hash mismatch: {filename}")

    encoded_instance = texts["plan_instance.json"]
    if not encoded_instance.endswith("\n"):
        raise ValueError("artifact PlanInstance must end with one newline")
    instance = PlanInstance.from_canonical_json(
        encoded_instance[:-1],
        template=template,
        bound_module=bound_module,
    )
    instance_hash = instance.stable_hash(template=template, bound_module=bound_module)
    if manifest.get("plan_instance_hash") != instance_hash:
        raise ValueError("artifact plan_instance_hash mismatch")
    return instance


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
