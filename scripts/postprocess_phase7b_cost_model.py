from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("meta", {}).get("schema_version") != "phase7b_crossover_matrix.v1":
        raise ValueError(f"expected phase7b_crossover_matrix.v1, got {payload.get('meta', {}).get('schema_version')}")
    return payload


def _safe_ratio(num: float, den: float) -> float | None:
    if den == 0.0:
        return None
    return float(num) / float(den)


def _relative_gap(best_ms: float, second_ms: float) -> float:
    if second_ms <= 0.0:
        return 0.0
    return max(0.0, float(second_ms - best_ms) / float(second_ms))


def _confidence_from_gap(gap: float, *, min_relative_margin: float) -> str:
    if gap < float(min_relative_margin):
        return "low"
    if gap < max(float(min_relative_margin) * 2.0, 0.10):
        return "medium"
    return "high"


def _cap_confidence(confidence: str, *, max_confidence: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    inverse = {v: k for k, v in rank.items()}
    return inverse[min(rank[confidence], rank[max_confidence])]


def _group_rows(rows: Iterable[Dict[str, Any]]) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["workload"]), str(row["scale_id"])), []).append(row)
    return grouped


def _policy_ms(row: Dict[str, Any]) -> float:
    return float(row["metrics"]["structured_ms_p50"])


def _policy_final(row: Dict[str, Any]) -> str:
    return str(row["metrics"]["planner_final_concretization_policy"])


def _rule_for_group(
    *,
    workload: str,
    scale_id: str,
    rows: List[Dict[str, Any]],
    min_relative_margin: float,
    max_confidence: str,
) -> Dict[str, Any]:
    ordered = sorted(rows, key=_policy_ms)
    best = ordered[0]
    best_by_final_policy: Dict[str, Dict[str, Any]] = {}
    for row in ordered:
        best_by_final_policy.setdefault(_policy_final(row), row)
    final_policy_ordered = sorted(best_by_final_policy.values(), key=_policy_ms)
    second_final = final_policy_ordered[1] if len(final_policy_ordered) > 1 else final_policy_ordered[0]
    gap = _relative_gap(_policy_ms(best), _policy_ms(second_final))
    by_policy = {str(row["policy_request"]): row for row in rows}
    structured = by_policy.get("structured")
    dense = by_policy.get("dense_barrier")
    dense_ratio = None
    if structured is not None and dense is not None:
        dense_ratio = _safe_ratio(_policy_ms(dense), _policy_ms(structured))

    return {
        "schema_version": 1,
        "workload": workload,
        "scale_id": scale_id,
        "compare_target": str(best["compare_target"]),
        "recommended_policy_request": str(best["policy_request"]),
        "recommended_final_concretization_policy": _policy_final(best),
        "confidence": _cap_confidence(
            _confidence_from_gap(gap, min_relative_margin=min_relative_margin),
            max_confidence=max_confidence,
        ),
        "relative_gap_to_second_best": float(gap),
        "dense_barrier_vs_structured_ms_ratio": dense_ratio,
        "evidence": {
            "policy_ms_p50": {
                str(row["policy_request"]): _policy_ms(row)
                for row in sorted(rows, key=lambda item: str(item["policy_request"]))
            },
            "policy_final_concretization": {
                str(row["policy_request"]): _policy_final(row)
                for row in sorted(rows, key=lambda item: str(item["policy_request"]))
            },
            "materialized_bytes": {
                str(row["policy_request"]): int(row["metrics"]["materialized_bytes"])
                for row in sorted(rows, key=lambda item: str(item["policy_request"]))
            },
            "right_matmul_exact_bytes": {
                str(row["policy_request"]): int(row["metrics"]["right_matmul_exact_bytes"])
                for row in sorted(rows, key=lambda item: str(item["policy_request"]))
            },
            "cache_hits": {
                str(row["policy_request"]): int(row["metrics"]["cache_hits"])
                for row in sorted(rows, key=lambda item: str(item["policy_request"]))
            },
            "cache_misses": {
                str(row["policy_request"]): int(row["metrics"]["cache_misses"])
                for row in sorted(rows, key=lambda item: str(item["policy_request"]))
            },
        },
        "guardrails": {
            "unknown_materialization_calls": max(
                int(row["metrics"]["unknown_materialization_calls"]) for row in rows
            ),
            "split_pos_neg_dense_total": max(
                int(row["metrics"]["split_pos_neg_dense_total"]) for row in rows
            ),
        },
    }


def _summarize_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        grouped.setdefault(str(rule["workload"]), []).append(rule)

    out: List[Dict[str, Any]] = []
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    for workload, items in sorted(grouped.items()):
        counts: Dict[str, int] = {}
        for item in items:
            final_policy = str(item["recommended_final_concretization_policy"])
            counts[final_policy] = counts.get(final_policy, 0) + 1
        recommended = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        min_confidence = min(items, key=lambda item: confidence_rank[str(item["confidence"])])["confidence"]
        out.append(
            {
                "workload": workload,
                "scales_observed": [str(item["scale_id"]) for item in sorted(items, key=lambda item: str(item["scale_id"]))],
                "recommended_default_final_concretization_policy": recommended,
                "final_policy_vote_counts": dict(sorted(counts.items())),
                "min_confidence": str(min_confidence),
            }
        )
    return out


def build_cost_model(
    matrix_payload: Dict[str, Any],
    *,
    min_relative_margin: float = 0.05,
    min_iters_for_confidence: int = 3,
) -> Dict[str, Any]:
    grouped = _group_rows(matrix_payload.get("rows", []))
    source_iters = int(matrix_payload.get("meta", {}).get("iters", 0) or 0)
    max_confidence = "high" if source_iters >= int(min_iters_for_confidence) else "low"
    rules = [
        _rule_for_group(
            workload=workload,
            scale_id=scale_id,
            rows=rows,
            min_relative_margin=float(min_relative_margin),
            max_confidence=max_confidence,
        )
        for (workload, scale_id), rows in sorted(grouped.items())
    ]
    return {
        "meta": {
            "schema_version": "phase7b_cost_model_v1",
            "source_schema_version": matrix_payload.get("meta", {}).get("schema_version"),
            "source_git_sha": matrix_payload.get("meta", {}).get("git_sha"),
            "source_device": matrix_payload.get("meta", {}).get("device"),
            "source_dtype": matrix_payload.get("meta", {}).get("dtype"),
            "source_timer": matrix_payload.get("meta", {}).get("timer"),
            "source_warmup": matrix_payload.get("meta", {}).get("warmup"),
            "source_iters": matrix_payload.get("meta", {}).get("iters"),
            "min_relative_margin": float(min_relative_margin),
            "min_iters_for_confidence": int(min_iters_for_confidence),
            "max_confidence_from_measurement_reliability": max_confidence,
            "capability_table": matrix_payload.get("meta", {}).get("capability_table"),
        },
        "rules": rules,
        "summary": _summarize_rules(rules),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7B PR-20: derive cost model v1 from crossover matrix JSON.")
    parser.add_argument("matrix_json", type=str)
    parser.add_argument("--min-relative-margin", type=float, default=0.05)
    parser.add_argument("--min-iters-for-confidence", type=int, default=3)
    args = parser.parse_args(argv)

    payload = _load_json(args.matrix_json)
    out = build_cost_model(
        payload,
        min_relative_margin=float(args.min_relative_margin),
        min_iters_for_confidence=int(args.min_iters_for_confidence),
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
