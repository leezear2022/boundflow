from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from boundflow.runtime.bound_planner import phase7b_cost_model_rules_jsonable

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _load_cost_model(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("meta", {}).get("schema_version") != "phase7b_cost_model_v1":
        raise ValueError(f"expected phase7b_cost_model_v1, got {payload.get('meta', {}).get('schema_version')}")
    return payload


def _rule_key(rule: Dict[str, Any], *, device: str) -> tuple[str, str, str, str]:
    return (
        str(device),
        str(rule["workload"]),
        str(rule["scale_id"]),
        str(rule["recommended_final_concretization_policy"]),
    )


def build_planner_v2_report(
    cost_model: Dict[str, Any],
    *,
    device: str = "cpu",
    min_confidence: str = "high",
) -> Dict[str, Any]:
    if min_confidence not in _CONFIDENCE_RANK:
        raise ValueError(f"unknown min confidence: {min_confidence}")
    threshold = _CONFIDENCE_RANK[min_confidence]
    embedded_rules = phase7b_cost_model_rules_jsonable()
    embedded_keys = {
        (
            str(rule["device"]),
            str(rule["workload"]),
            str(rule["scale_id"]),
            str(rule["final_concretization_policy"]),
        )
        for rule in embedded_rules["rules"]
    }

    promoted: List[Dict[str, Any]] = []
    missing_promotions: List[Dict[str, Any]] = []
    held_back: List[Dict[str, Any]] = []

    for rule in cost_model.get("rules", []):
        confidence = str(rule["confidence"])
        guardrails = rule.get("guardrails", {})
        is_safe = (
            int(guardrails.get("unknown_materialization_calls", 0)) == 0
            and int(guardrails.get("split_pos_neg_dense_total", 0)) == 0
        )
        eligible = _CONFIDENCE_RANK[confidence] >= threshold and is_safe
        key = _rule_key(rule, device=device)
        item = {
            "workload": rule["workload"],
            "scale_id": rule["scale_id"],
            "recommended_final_concretization_policy": rule["recommended_final_concretization_policy"],
            "confidence": confidence,
            "relative_gap_to_second_best": rule["relative_gap_to_second_best"],
            "guardrails": guardrails,
        }
        if eligible and key in embedded_keys:
            promoted.append(item)
        elif eligible:
            missing_promotions.append(item)
        else:
            held_back.append(item)

    return {
        "meta": {
            "schema_version": "phase7b_planner_v2_candidates.v1",
            "source_schema_version": cost_model.get("meta", {}).get("schema_version"),
            "source_git_sha": cost_model.get("meta", {}).get("source_git_sha"),
            "source_device": cost_model.get("meta", {}).get("source_device"),
            "device": device,
            "min_confidence": min_confidence,
            "embedded_cost_model_rules_schema_version": embedded_rules["schema_version"],
        },
        "promoted_rules": promoted,
        "missing_promotions": missing_promotions,
        "held_back_rules": held_back,
        "summary": {
            "promoted_count": len(promoted),
            "missing_promotion_count": len(missing_promotions),
            "held_back_count": len(held_back),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7B PR-22: audit planner-v2 cost-model promotions.")
    parser.add_argument("cost_model_json", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min-confidence", type=str, default="high", choices=["low", "medium", "high"])
    args = parser.parse_args(argv)

    payload = _load_cost_model(args.cost_model_json)
    out = build_planner_v2_report(payload, device=str(args.device), min_confidence=str(args.min_confidence))
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
