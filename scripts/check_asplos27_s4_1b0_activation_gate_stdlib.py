#!/usr/bin/env python3
"""Fail-closed activation gate for S4-1B0 implementation/correctness."""

# The sibling design checker intentionally exposes the same small CLI shape.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from check_asplos27_s4_1b0_design_contracts_stdlib import (
    ContractFailure,
    validate as validate_design_contracts,
)

TASK_ID = "asplos27-s4-1a-ordered-buffer-20260830"
EXPECTED_BRANCH = "feat/rvir-v4-production-state-ownership-v1"
EXPECTED_S4_1A_BASELINE_COMMIT = "f773370"
REQUIRED_ANCESTORS = (
    "f773370",  # S4-1A formal validation
    "a66e383",  # S4-1B0 formal artifact contract freeze
    "44b01ab",  # construction-root contract checker
)
CRITICAL_PATHS = (
    ".docops/s.md",
    f".docops/exchange/{TASK_ID}",
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md",
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IEEE_BIT_FIXTURES_V1_2026_08_30.json",
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json",
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_ABI_CONTRACT_V1_2026_08_30.json",
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_FORMAL_ARTIFACT_CONTRACT_V1_2026_08_30.json",
    "scripts/check_asplos27_s4_1b0_design_contracts_stdlib.py",
    "scripts/check_asplos27_s4_1b0_activation_gate_stdlib.py",
)
WAIT_STATUSES = {
    "in_progress": "s4-1a-exchange-in-progress",
    "ready_for_audit": "external-audit-s4-1a-pending",
    "auditing": "external-audit-s4-1a-running",
    "changes_requested": "external-audit-s4-1a-changes-requested",
    "disputed": "external-audit-s4-1a-disputed",
}


@dataclass(frozen=True)
class GateDecision:
    """One deterministic state-machine decision."""

    status: str
    reason: str


class GateError(Exception):
    """Raised when authoritative state is internally inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: top-level JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_state_markdown(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line or raw_line.lstrip().startswith("#"):
            continue
        key, value = raw_line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _classify_exchange(state: Mapping[str, Any]) -> GateDecision:
    status = state.get("status")
    if status in WAIT_STATUSES:
        return GateDecision("WAIT", WAIT_STATUSES[str(status)])
    if status == "approved":
        return GateDecision("WAIT", "executor-exchange-close-required")
    if status != "closed":
        return GateDecision("ERROR", f"unsupported-exchange-status:{status}")
    approved_round = state.get("approved_round")
    if not isinstance(approved_round, int) or approved_round <= 0:
        return GateDecision("ERROR", "closed-exchange-without-approved-round")
    if state.get("round") != approved_round:
        return GateDecision("ERROR", "closed-exchange-round-mismatch")
    return GateDecision("PROCEED", "closed-approved-exchange")


def _run_git(root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _enforce_publication_state(ahead: int, behind: int, dirty: Sequence[str]) -> None:
    if ahead or behind:
        raise GateError(
            f"activation-head-upstream-diverged:ahead={ahead}:behind={behind}"
        )
    if dirty:
        raise GateError(f"activation-critical-paths-dirty:{'|'.join(dirty)}")


def _validate_git_identity(root: Path, require_published: bool) -> dict[str, Any]:
    branch = _run_git(root, ("branch", "--show-current"))
    if branch != EXPECTED_BRANCH:
        raise GateError(f"wrong-branch:{branch or 'detached'}")
    head = _run_git(root, ("rev-parse", "HEAD"))
    for ancestor in REQUIRED_ANCESTORS:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            raise GateError(f"required-ancestor-missing:{ancestor}")
    upstream = _run_git(root, ("rev-parse", "@{upstream}"))
    divergence = _run_git(
        root, ("rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    )
    ahead_text, behind_text = divergence.split()
    ahead, behind = int(ahead_text), int(behind_text)
    dirty_output = _run_git(
        root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *CRITICAL_PATHS,
        ),
    )
    dirty = [line for line in dirty_output.splitlines() if line]
    if require_published:
        _enforce_publication_state(ahead, behind, dirty)
    return {
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "critical_paths_clean": not dirty,
        "critical_path_changes": dirty,
        "publication_required": require_published,
        "required_ancestors": list(REQUIRED_ANCESTORS),
    }


def _require_closed_exchange_files(
    exchange_root: Path, round_name: str
) -> dict[str, Path]:
    paths = {
        "request_json": exchange_root / "request.json",
        "request_md": exchange_root / "request.md",
        "delivery_json": exchange_root / round_name / "delivery.json",
        "delivery_md": exchange_root / round_name / "delivery.md",
        "audit_json": exchange_root / round_name / "audit.json",
        "audit_md": exchange_root / round_name / "audit.md",
        "closure_json": exchange_root / "closure.json",
        "closure_md": exchange_root / "closure.md",
    }
    for path in paths.values():
        if not path.is_file():
            raise GateError(
                f"closed-exchange-file-missing:{path.relative_to(exchange_root)}"
            )
    return paths


def _validate_request_document(
    paths: Mapping[str, Path], state: Mapping[str, Any], expected_doc: str
) -> dict[str, Any]:
    request = _load_json(paths["request_json"])
    if request.get("task") != TASK_ID or request.get("doc") != expected_doc:
        raise GateError("request-task-or-doc-mismatch")
    if request.get("round") != 0 or request.get("type") != "request":
        raise GateError("request-round-or-type-mismatch")
    if request.get("executor") != state.get("executor") or request.get(
        "auditor"
    ) != state.get("auditor"):
        raise GateError("request-participant-mismatch")
    if request.get("md_sha256") != _sha256(paths["request_md"]):
        raise GateError("request-md-sha256-mismatch")
    return request


def _validate_delivery_document(
    paths: Mapping[str, Path],
    state: Mapping[str, Any],
    approved_round: int,
    expected_doc: str,
) -> dict[str, Any]:
    delivery = _load_json(paths["delivery_json"])
    if delivery.get("task") != TASK_ID or delivery.get("doc") != expected_doc:
        raise GateError("delivery-task-or-doc-mismatch")
    if delivery.get("round") != approved_round:
        raise GateError("delivery-round-mismatch")
    if delivery.get("from") != state.get("executor") or delivery.get("to") != state.get(
        "auditor"
    ):
        raise GateError("delivery-participant-mismatch")
    if delivery.get("md_sha256") != _sha256(paths["delivery_md"]):
        raise GateError("delivery-md-sha256-mismatch")
    return delivery


def _validate_audit_document(
    paths: Mapping[str, Path],
    state: Mapping[str, Any],
    *,
    approved_round: int,
    expected_doc: str,
    expected_delivery_doc: str,
) -> dict[str, Any]:
    audit = _load_json(paths["audit_json"])
    if audit.get("task") != TASK_ID or audit.get("doc") != expected_doc:
        raise GateError("audit-task-or-doc-mismatch")
    if audit.get("round") != approved_round or audit.get("verdict") != "approve":
        raise GateError("audit-round-or-verdict-mismatch")
    if audit.get("delivery") != expected_delivery_doc:
        raise GateError("audit-delivery-link-mismatch")
    if audit.get("from") != state.get("auditor") or audit.get("to") != state.get(
        "executor"
    ):
        raise GateError("audit-participant-mismatch")
    if audit.get("md_sha256") != _sha256(paths["audit_md"]):
        raise GateError("audit-md-sha256-mismatch")
    findings = audit.get("findings")
    if not isinstance(findings, list):
        raise GateError("audit-findings-not-list")
    blocking = [
        item.get("id", "unknown")
        for item in findings
        if isinstance(item, dict) and item.get("severity") in {"blocker", "major"}
    ]
    if blocking:
        raise GateError(f"approved-audit-has-blocking-findings:{','.join(blocking)}")
    return audit


def _validate_closure_document(
    paths: Mapping[str, Path],
    state: Mapping[str, Any],
    approved_round: int,
    expected_doc: str,
) -> dict[str, Any]:
    closure = _load_json(paths["closure_json"])
    if closure.get("task") != TASK_ID or closure.get("doc") != expected_doc:
        raise GateError("closure-task-or-doc-mismatch")
    if (
        closure.get("round") != approved_round
        or closure.get("approved_round") != approved_round
    ):
        raise GateError("closure-round-mismatch")
    if closure.get("resolution") != "approved":
        raise GateError("closure-resolution-not-approved")
    if closure.get("from") != state.get("executor"):
        raise GateError("closure-executor-mismatch")
    if closure.get("md_sha256") != _sha256(paths["closure_md"]):
        raise GateError("closure-md-sha256-mismatch")
    return closure


def _validate_registered_docs(
    state: Mapping[str, Any], expected: Sequence[str]
) -> None:
    docs = state.get("docs")
    if not isinstance(docs, list):
        raise GateError("exchange-docs-not-list")
    if len(docs) != len(set(docs)):
        raise GateError("exchange-docs-contain-duplicates")
    for document in expected:
        if document not in docs:
            raise GateError(f"exchange-doc-not-registered:{document}")


def _validate_closed_exchange(
    exchange_root: Path, state: Mapping[str, Any]
) -> dict[str, Any]:
    approved_round = int(state["approved_round"])
    round_name = f"r{approved_round:03d}"
    paths = _require_closed_exchange_files(exchange_root, round_name)
    request_doc = f"{TASK_ID}/request"
    delivery_doc = f"{TASK_ID}/{round_name}/delivery"
    audit_doc = f"{TASK_ID}/{round_name}/audit"
    closure_doc = f"{TASK_ID}/closure"
    request = _validate_request_document(paths, state, request_doc)
    delivery = _validate_delivery_document(paths, state, approved_round, delivery_doc)
    audit = _validate_audit_document(
        paths,
        state,
        approved_round=approved_round,
        expected_doc=audit_doc,
        expected_delivery_doc=delivery_doc,
    )
    closure = _validate_closure_document(paths, state, approved_round, closure_doc)
    _validate_registered_docs(
        state, (request_doc, delivery_doc, audit_doc, closure_doc)
    )
    return {
        "approved_round": approved_round,
        "audit_verdict": audit["verdict"],
        "blocking_finding_count": 0,
        "closure_resolution": closure["resolution"],
        "request_md_sha256": request["md_sha256"],
        "delivery_md_sha256": delivery["md_sha256"],
        "audit_md_sha256": audit["md_sha256"],
        "closure_md_sha256": closure["md_sha256"],
    }


def _validate_delivery_result_commit(
    root: Path, exchange_root: Path, approved_round: int
) -> dict[str, str]:
    delivery = _load_json(exchange_root / f"r{approved_round:03d}" / "delivery.json")
    result_commit = delivery.get("result_commit")
    if not isinstance(result_commit, str) or not result_commit:
        raise GateError("delivery-result-commit-missing")
    resolved = _run_git(root, ("rev-parse", f"{result_commit}^{{commit}}"))
    baseline = _run_git(
        root, ("rev-parse", f"{EXPECTED_S4_1A_BASELINE_COMMIT}^{{commit}}")
    )
    baseline_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, resolved],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if baseline_check.returncode:
        raise GateError("delivery-result-commit-does-not-contain-s4-1a-baseline")
    head_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved, "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if head_check.returncode:
        raise GateError("delivery-result-commit-not-head-ancestor")
    return {"declared": result_commit, "resolved": resolved, "baseline": baseline}


def _validate_docops_state(values: Mapping[str, str], exchange_closed: bool) -> None:
    if values.get("st") != "s04" or values.get("stat") != "active":
        raise GateError("docops-stage-not-active-s04")
    if values.get("health") != "green":
        raise GateError("docops-health-not-green")
    blocker = values.get("blk", "")
    next_action = values.get("next", "")
    if exchange_closed:
        if blocker == "external-audit-s4-1a-pending":
            raise GateError("closed-exchange-but-s4-1a-blocker-remains")
        if "s4-1b0" not in next_action:
            raise GateError("closed-exchange-next-is-not-s4-1b0")
    else:
        if blocker != "external-audit-s4-1a-pending":
            raise GateError("open-exchange-without-s4-1a-audit-blocker")
        if next_action != "wait-for-external-audit-s4-1a-round1":
            raise GateError("open-exchange-next-action-mismatch")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


# The synthetic closed exchange keeps each linked document visible in one place.
# pylint: disable=too-many-locals
def _closed_exchange_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="boundflow-s4-1b0-gate-") as raw_root:
        root = Path(raw_root)
        round_root = root / "r001"
        round_root.mkdir(parents=True)
        request_md = root / "request.md"
        delivery_md = round_root / "delivery.md"
        audit_md = round_root / "audit.md"
        closure_md = root / "closure.md"
        request_md.write_text("# Request\n", encoding="utf-8")
        delivery_md.write_text("# Delivery\n", encoding="utf-8")
        audit_md.write_text("# Audit\n", encoding="utf-8")
        closure_md.write_text("# Closure\n", encoding="utf-8")
        _write_json(
            root / "request.json",
            {
                "doc": f"{TASK_ID}/request",
                "type": "request",
                "task": TASK_ID,
                "round": 0,
                "executor": "codex",
                "auditor": "external-model",
                "md_sha256": _sha256(request_md),
            },
        )
        delivery_doc = f"{TASK_ID}/r001/delivery"
        audit_doc = f"{TASK_ID}/r001/audit"
        closure_doc = f"{TASK_ID}/closure"
        delivery = {
            "doc": delivery_doc,
            "task": TASK_ID,
            "round": 1,
            "from": "codex",
            "to": "external-model",
            "md_sha256": _sha256(delivery_md),
        }
        audit = {
            "doc": audit_doc,
            "task": TASK_ID,
            "round": 1,
            "from": "external-model",
            "to": "codex",
            "verdict": "approve",
            "delivery": delivery_doc,
            "findings": [{"id": "I1", "severity": "info"}],
            "md_sha256": _sha256(audit_md),
        }
        closure = {
            "doc": closure_doc,
            "task": TASK_ID,
            "round": 1,
            "approved_round": 1,
            "from": "codex",
            "resolution": "approved",
            "md_sha256": _sha256(closure_md),
        }
        _write_json(round_root / "delivery.json", delivery)
        _write_json(round_root / "audit.json", audit)
        _write_json(root / "closure.json", closure)
        state = {
            "task": TASK_ID,
            "status": "closed",
            "round": 1,
            "approved_round": 1,
            "executor": "codex",
            "auditor": "external-model",
            "docs": [f"{TASK_ID}/request", delivery_doc, audit_doc, closure_doc],
        }
        _validate_closed_exchange(root, state)
        audit_md.write_text("# Tampered Audit\n", encoding="utf-8")
        try:
            _validate_closed_exchange(root, state)
        except GateError as exc:
            if str(exc) != "audit-md-sha256-mismatch":
                raise GateError(f"self-test-wrong-tamper-reason:{exc}") from exc
        else:
            raise GateError("self-test-audit-markdown-tamper-accepted")
    return 2


def _publication_state_self_test() -> int:
    _enforce_publication_state(0, 0, ())
    for ahead, behind, dirty, expected in (
        (1, 0, (), "activation-head-upstream-diverged:ahead=1:behind=0"),
        (0, 0, (" M critical.py",), "activation-critical-paths-dirty: M critical.py"),
    ):
        try:
            _enforce_publication_state(ahead, behind, dirty)
        except GateError as exc:
            if str(exc) != expected:
                raise GateError(f"self-test-wrong-publication-reason:{exc}") from exc
        else:
            raise GateError("self-test-invalid-publication-state-accepted")
    return 3


def _state_machine_self_test() -> dict[str, Any]:
    cases = [
        (
            {"status": "ready_for_audit", "round": 1, "approved_round": None},
            GateDecision("WAIT", "external-audit-s4-1a-pending"),
        ),
        (
            {"status": "approved", "round": 1, "approved_round": 1},
            GateDecision("WAIT", "executor-exchange-close-required"),
        ),
        (
            {"status": "closed", "round": 1, "approved_round": 1},
            GateDecision("PROCEED", "closed-approved-exchange"),
        ),
        (
            {"status": "closed", "round": 1, "approved_round": None},
            GateDecision("ERROR", "closed-exchange-without-approved-round"),
        ),
        (
            {"status": "closed", "round": 2, "approved_round": 1},
            GateDecision("ERROR", "closed-exchange-round-mismatch"),
        ),
    ]
    for state, expected in cases:
        actual = _classify_exchange(state)
        if actual != expected:
            raise GateError(f"self-test-mismatch:{state}:{actual}:{expected}")
    closed_cases = _closed_exchange_self_test()
    publication_cases = _publication_state_self_test()
    return {
        "status": "PASS",
        "classifier_case_count": len(cases),
        "closed_exchange_case_count": closed_cases,
        "publication_case_count": publication_cases,
        "case_count": len(cases) + closed_cases + publication_cases,
    }


def evaluate(root: Path) -> tuple[GateDecision, dict[str, Any]]:
    """Evaluate the full pre-activation gate without changing repository state."""
    design = validate_design_contracts(root, require_code_closed=True)
    exchange_root = root / ".docops" / "exchange" / TASK_ID
    exchange_state = _load_json(exchange_root / "state.json")
    if exchange_state.get("task") != TASK_ID:
        raise GateError("exchange-task-mismatch")
    decision = _classify_exchange(exchange_state)
    git_identity = _validate_git_identity(
        root, require_published=decision.status == "PROCEED"
    )
    docops = _parse_state_markdown(root / ".docops" / "s.md")
    if decision.status == "ERROR":
        raise GateError(decision.reason)
    _validate_docops_state(docops, exchange_closed=decision.status == "PROCEED")
    evidence: dict[str, Any] = {
        "design_check_count": design["check_count"],
        "construction_model_hash": design["construction_model_hash"],
        "code_closed_checked": design["code_closed_checked"],
        "git": git_identity,
        "exchange_status": exchange_state["status"],
        "exchange_round": exchange_state["round"],
        "docops_blocker": docops.get("blk"),
        "docops_next": docops.get("next"),
        "delivery_result_commit": _validate_delivery_result_commit(
            root, exchange_root, int(exchange_state["round"])
        ),
    }
    if decision.status == "PROCEED":
        evidence["closure"] = _validate_closed_exchange(exchange_root, exchange_state)
    return decision, evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the in-memory exchange state-machine tests",
    )
    return parser.parse_args()


def main() -> int:
    """Print one canonical receipt and use 0=GO, 3=WAIT, 1=ERROR."""
    args = _parse_args()
    try:
        if args.self_test:
            result = _state_machine_self_test()
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        decision, evidence = evaluate(args.repo_root.resolve())
        receipt = {
            "schema": "boundflow.asplos27-s4-1b0-activation-gate/v1",
            "status": decision.status,
            "reason": decision.reason,
            "implementation_authority": decision.status == "PROCEED",
            "formal_authority": False,
            "timing_authority": False,
            "performance_claimed": False,
            "evidence": evidence,
        }
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0 if decision.status == "PROCEED" else 3
    except (
        ContractFailure,
        GateError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": "boundflow.asplos27-s4-1b0-activation-gate/v1",
                    "status": "ERROR",
                    "reason": str(exc),
                    "implementation_authority": False,
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


if __name__ == "__main__":
    raise SystemExit(main())
