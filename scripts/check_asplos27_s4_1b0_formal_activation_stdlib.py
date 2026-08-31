#!/usr/bin/env python3
"""Fail-closed formal-artifact activation for the frozen S4-1B0 implementation."""

# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from check_asplos27_s4_1b0_design_contracts_stdlib import (
    ContractFailure,
    validate as validate_design_contracts,
)

EXPECTED_BRANCH = "feat/rvir-v4-production-state-ownership-v1"
IMPLEMENTATION_COMMIT = "f61e917"
CONSTRUCTION_HASH = "5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a"
SOURCE_PATH = Path("boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py")
TEST_PATH = Path("tests/test_asplos27_s4_ternary_endpoint.py")
CHANGELOG_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_TERNARY_ENDPOINT_IMPLEMENTATION_CHANGELOG_2026_08_31.md"
)
EXPECTED_SHA256 = {
    SOURCE_PATH: "2a9ecfd4f0183a1febb596d52f4aad8d938b65b0057dbe595f6e51fad82c7997",
    TEST_PATH: "776343dd22b03c8f9210d8f6bc29f9b229cabe01c33a9eca212da9a96497a9f4",
    CHANGELOG_PATH: "f68bfd79f7150b490ca5c0a446551cc3282daf67b935aae87a8ac58083004b28",
}
CRITICAL_PATHS = (
    SOURCE_PATH,
    TEST_PATH,
    CHANGELOG_PATH,
    Path(
        "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md"
    ),
    Path("gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IEEE_BIT_FIXTURES_V1_2026_08_30.json"),
    Path("gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json"),
    Path("gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_ABI_CONTRACT_V1_2026_08_30.json"),
    Path(
        "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_FORMAL_ARTIFACT_CONTRACT_V1_2026_08_30.json"
    ),
    Path(".docops/s.md"),
    Path("scripts/check_asplos27_s4_1b0_formal_activation_stdlib.py"),
)
EXPECTED_TOP_LEVEL_IMPORT_ROOTS = {
    "__future__",
    "dataclasses",
    "hashlib",
    "json",
    "math",
    "struct",
    "typing",
}
FORBIDDEN_SOURCE_TOKENS = (
    "time.perf_counter",
    "time.time",
    "torch.cuda.Event",
    "benchmark",
    "S4MutableStateAdmission",
    "PreparedS4MutableStateAdmission",
    "asplos27_s4_ordered_buffer_abi",
)


class GateError(Exception):
    """One deterministic activation failure."""


@dataclass
class Checks:
    """Collect independent checks before emitting one canonical receipt."""

    count: int = 0
    failures: list[str] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    def require(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            assert self.failures is not None
            self.failures.append(label)

    def equal(self, actual: Any, expected: Any, label: str) -> None:
        self.require(
            actual == expected, f"{label}: actual={actual!r} expected={expected!r}"
        )

    def finish(self) -> None:
        if self.failures:
            raise GateError("\n".join(self.failures))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GateError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _git_blob(root: Path, revision: str, path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GateError(f"git show {revision}:{path} failed: {detail}")
    return result.stdout


def _is_ancestor(root: Path, revision: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _top_level_import_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".", 1)[0])
    return roots


def _class_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _parse_state(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def validate(root: Path, *, require_published: bool = True) -> dict[str, Any]:
    """Validate immutable implementation identity and return formal authority."""

    checks = Checks()
    design = validate_design_contracts(root, require_code_closed=False)
    checks.equal(design["construction_model_hash"], CONSTRUCTION_HASH, "design hash")
    checks.equal(_git(root, ("branch", "--show-current")), EXPECTED_BRANCH, "branch")
    head = _git(root, ("rev-parse", "HEAD"))
    upstream = _git(root, ("rev-parse", "@{upstream}"))
    checks.require(_is_ancestor(root, IMPLEMENTATION_COMMIT), "implementation ancestor")
    if require_published:
        checks.equal(head, upstream, "HEAD/upstream")
    if require_published:
        dirty = _git(
            root,
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *(str(path) for path in CRITICAL_PATHS),
            ),
        )
        checks.equal(dirty, "", "critical paths clean")

    for path, expected in EXPECTED_SHA256.items():
        checks.require((root / path).is_file(), f"file exists {path}")
        checks.equal(_sha256(root / path), expected, f"sha256 {path}")
        checks.equal(
            hashlib.sha256(_git_blob(root, IMPLEMENTATION_COMMIT, path)).hexdigest(),
            expected,
            f"implementation blob {path}",
        )

    source = (root / SOURCE_PATH).read_text(encoding="utf-8")
    test = (root / TEST_PATH).read_text(encoding="utf-8")
    source_tree = ast.parse(source, filename=str(SOURCE_PATH))
    test_tree = ast.parse(test, filename=str(TEST_PATH))
    checks.equal(
        _top_level_import_roots(source_tree),
        EXPECTED_TOP_LEVEL_IMPORT_ROOTS,
        "source top-level imports",
    )
    checks.require(
        not any(token in source for token in FORBIDDEN_SOURCE_TOKENS),
        "no timing or production-owner imports",
    )
    checks.require(CONSTRUCTION_HASH in source, "source binds construction hash")
    required_classes = {
        "TernaryEndpointBuildSpecV1",
        "TernaryEndpointScheduleSpecV1",
        "CompiledTernaryEndpointV1",
        "TernaryEndpointModuleReceiptV1",
        "TernaryEndpointCacheObservationV1",
        "TernaryEndpointModuleCacheV1",
        "TernaryEndpointTensorDescriptorV1",
        "TernaryEndpointWarmLaunchReceiptV1",
        "PreparedTernaryEndpointProbeV1",
    }
    checks.require(
        required_classes.issubset(_class_names(source_tree)), "required classes"
    )
    required_functions = {
        "_build_pack_primfunc",
        "_build_select_primfunc",
        "_schedule_elementwise",
        "build_ternary_endpoint_modules_v1",
        "compile_ternary_endpoint_v1",
        "ternary_pack_bit_oracle_v1",
        "ternary_select_bit_oracle_v1",
        "validate_selected_output_after_sync_v1",
        "validate_ternary_endpoint_construction_model_v1",
    }
    checks.require(
        required_functions.issubset(_function_names(source_tree)), "required functions"
    )

    negative = json.loads(
        (
            root
            / "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json"
        ).read_text(encoding="utf-8")
    )
    reasons = [row["reason"] for row in negative["stable_reasons"]]
    checks.equal(len(reasons), 20, "negative reason count")
    for reason in reasons:
        checks.require(reason in source, f"source reason {reason}")
        checks.require(reason in test, f"test reason {reason}")
    test_names = _function_names(test_tree)
    for name in negative["test_layout"]:
        checks.require(name in test_names, f"frozen test {name}")

    state = _parse_state(root / ".docops/s.md")
    checks.equal(state.get("st"), "s04", "DocOps stage")
    checks.equal(state.get("blk"), "none", "DocOps blocker")
    checks.require(
        state.get("next")
        in {
            "freeze-s4-1b0-implementation-and-open-formal-artifact",
            "generate-s4-1b0-eleven-process-correctness-artifact",
        },
        "DocOps next action",
    )
    checks.finish()
    return {
        "schema": "boundflow.asplos27-s4-1b0-formal-activation/v1",
        "status": "PROCEED",
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "head": head,
        "upstream": upstream,
        "check_count": checks.count,
        "design_check_count": design["check_count"],
        "construction_model_hash": CONSTRUCTION_HASH,
        "source_sha256": EXPECTED_SHA256[SOURCE_PATH],
        "test_sha256": EXPECTED_SHA256[TEST_PATH],
        "implementation_authority": True,
        "formal_authority": True,
        "timing_authority": False,
        "performance_claimed": False,
    }


def main() -> int:
    """Emit one canonical activation receipt."""

    root = Path(__file__).resolve().parents[1]
    try:
        receipt = validate(root)
    except (
        ContractFailure,
        GateError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema": "boundflow.asplos27-s4-1b0-formal-activation/v1",
                    "status": "ERROR",
                    "reason": str(error),
                    "formal_authority": False,
                    "timing_authority": False,
                    "performance_claimed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
