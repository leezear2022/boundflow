#!/usr/bin/env python3
"""Fail-closed activation gate for S4-1B0 implementation/correctness."""

# The sibling design checker intentionally exposes the same small CLI shape.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from check_asplos27_s4_1b0_design_contracts_stdlib import (
    ContractFailure,
    validate as validate_design_contracts,
)

TASK_ID = "asplos27-s4-1a-ordered-buffer-20260830"
EXPECTED_BRANCH = "feat/rvir-v4-production-state-ownership-v1"
REQUIRED_ANCESTORS = (
    "f773370",  # S4-1A formal validation
    "a66e383",  # S4-1B0 formal artifact contract freeze
    "44b01ab",  # construction-root contract checker
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


def _validate_git_identity(root: Path) -> dict[str, Any]:
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
    return {
        "branch": branch,
        "head": head,
        "required_ancestors": list(REQUIRED_ANCESTORS),
    }


# Audit, closure, and state identities stay visible together for review.
# pylint: disable=too-many-locals
def _validate_closed_exchange(
    exchange_root: Path, state: Mapping[str, Any]
) -> dict[str, Any]:
    approved_round = int(state["approved_round"])
    round_name = f"r{approved_round:03d}"
    audit_path = exchange_root / round_name / "audit.json"
    audit_md_path = exchange_root / round_name / "audit.md"
    closure_path = exchange_root / "closure.json"
    closure_md_path = exchange_root / "closure.md"
    for path in (audit_path, audit_md_path, closure_path, closure_md_path):
        if not path.is_file():
            raise GateError(
                f"closed-exchange-file-missing:{path.relative_to(exchange_root)}"
            )

    audit = _load_json(audit_path)
    closure = _load_json(closure_path)
    expected_audit_doc = f"{TASK_ID}/{round_name}/audit"
    expected_closure_doc = f"{TASK_ID}/closure"
    if audit.get("task") != TASK_ID or audit.get("doc") != expected_audit_doc:
        raise GateError("audit-task-or-doc-mismatch")
    if audit.get("round") != approved_round or audit.get("verdict") != "approve":
        raise GateError("audit-round-or-verdict-mismatch")
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

    if closure.get("task") != TASK_ID or closure.get("doc") != expected_closure_doc:
        raise GateError("closure-task-or-doc-mismatch")
    if (
        closure.get("round") != approved_round
        or closure.get("approved_round") != approved_round
    ):
        raise GateError("closure-round-mismatch")
    if closure.get("resolution") != "approved":
        raise GateError("closure-resolution-not-approved")
    docs = state.get("docs")
    if not isinstance(docs, list):
        raise GateError("exchange-docs-not-list")
    for expected in (expected_audit_doc, expected_closure_doc):
        if expected not in docs:
            raise GateError(f"exchange-doc-not-registered:{expected}")
    return {
        "approved_round": approved_round,
        "audit_verdict": audit["verdict"],
        "blocking_finding_count": 0,
        "closure_resolution": closure["resolution"],
    }


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
    return {"status": "PASS", "case_count": len(cases)}


def evaluate(root: Path) -> tuple[GateDecision, dict[str, Any]]:
    """Evaluate the full pre-activation gate without changing repository state."""
    design = validate_design_contracts(root, require_code_closed=True)
    git_identity = _validate_git_identity(root)
    exchange_root = root / ".docops" / "exchange" / TASK_ID
    exchange_state = _load_json(exchange_root / "state.json")
    if exchange_state.get("task") != TASK_ID:
        raise GateError("exchange-task-mismatch")
    decision = _classify_exchange(exchange_state)
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
