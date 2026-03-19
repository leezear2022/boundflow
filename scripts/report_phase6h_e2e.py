from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _flatten(prefix: str, obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(f"{prefix}{k}_", v))
        return out
    out[prefix[:-1]] = obj
    return out


@dataclass(frozen=True)
class FlatRow:
    data: Dict[str, Any]


def _iter_flat_rows(payload: Dict[str, Any]) -> Iterable[FlatRow]:
    meta = payload.get("meta", {})
    rows = payload.get("rows", [])
    for row in rows:
        base: Dict[str, Any] = {}
        base.update(_flatten("meta_", meta))
        # switch + comparability
        for k in (
            "workload",
            "enable_node_eval_cache",
            "use_branch_hint",
            "enable_batch_infeasible_prune",
            "comparable",
            "note_code",
            "note",
            "speedup",
            "speedup_p90",
            "speedup_p99",
        ):
            if k in row:
                base[k] = row[k]
        # per-path records: batch + serial
        for path in ("batch", "serial"):
            rec = dict(base)
            rec["path"] = path
            rec["verdict"] = row.get(f"{path}_verdict")
            rec["time_ms_p50"] = row.get(f"{path}_ms_p50")
            rec["time_ms_p90"] = row.get(f"{path}_ms_p90")
            rec["time_ms_p99"] = row.get(f"{path}_ms_p99")
            rec["runs_count"] = row.get(f"{path}_runs_count")
            rec["valid_runs_count"] = row.get(f"{path}_valid_runs_count")
            rec["timeouts_count"] = row.get(f"{path}_timeouts_count")
            stats = row.get(f"{path}_stats", {})
            counts = row.get(f"counts_{path}", {})
            rec.update(_flatten("stats_", stats))
            rec.update(_flatten("counts_", counts))
            yield FlatRow(data=rec)


def _write_csv(path: Path, flat_rows: List[FlatRow]) -> None:
    cols: List[str] = []
    col_set = set()
    for r in flat_rows:
        for k in r.data.keys():
            if k not in col_set:
                col_set.add(k)
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r.data)


def _write_summary_md(path: Path, *, flat_rows: List[FlatRow], payloads: List[Dict[str, Any]]) -> None:
    # One section per meta signature.
    def sig_key(d: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            d.get("meta_device"),
            d.get("meta_dtype"),
            d.get("meta_workload"),
            d.get("meta_p"),
            d.get("meta_specs"),
            d.get("meta_max_nodes"),
            d.get("meta_node_batch_size"),
            d.get("meta_oracle"),
            d.get("meta_steps"),
            d.get("meta_lr"),
            d.get("meta_timer"),
        )

    by_sig: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in flat_rows:
        d = r.data
        if d.get("path") != "batch":
            continue
        by_sig.setdefault(sig_key(d), []).append(d)

    # Collect failed runs from raw payloads (error records have rows=[]).
    failed_runs: List[Dict[str, Any]] = []
    for p in payloads:
        meta = p.get("meta", {})
        if meta.get("run_status") == "error":
            failed_runs.append(meta)

    lines: List[str] = []
    lines.append("# Phase 6H E2E 报告（自动生成）")
    lines.append("")
    if failed_runs:
        lines.append("## 失败运行（run_status=error）")
        lines.append("")
        lines.append("| device | dtype | workload | timer | error |")
        lines.append("|---|---|---|---|---|")
        for d in failed_runs:
            lines.append(
                f"| {d.get('device')} | {d.get('dtype')} | {d.get('workload')} | {d.get('timer')} | {str(d.get('error',''))[:80]} |"
            )
        lines.append("")

    for sig, ds in by_sig.items():
        lines.append("## 配置")
        lines.append("")
        lines.append("```json")
        meta = {k: v for k, v in ds[0].items() if str(k).startswith("meta_")}
        lines.append(json.dumps(meta, ensure_ascii=False, sort_keys=True))
        lines.append("```")
        lines.append("")

        comparable = [d for d in ds if int(d.get("comparable") or 0) == 1]
        incomparable = [d for d in ds if int(d.get("comparable") or 0) == 0]

        lines.append("### Comparable 行（主表）")
        lines.append("")
        if not comparable:
            lines.append("- （无）")
            lines.append("")
        else:
            # sort by speedup desc
            comparable_sorted = sorted(comparable, key=lambda x: float(x.get("speedup") or 0.0), reverse=True)
            lines.append(
                "| cache | hint | prune | verdict | time_ms_p50 | time_ms_p90 | speedup_p50 | speedup_p90 | oracle_calls | forward_trace_calls | cache_hit_rate | pruned_infeasible | avg_batch_fill |"
            )
            lines.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for d in comparable_sorted:
                lines.append(
                    f"| {d.get('enable_node_eval_cache')} | {d.get('use_branch_hint')} | {d.get('enable_batch_infeasible_prune')} | {d.get('verdict')} | "
                    f"{float(d.get('time_ms_p50') or 0.0):.3f} | {float(d.get('time_ms_p90') or 0.0):.3f} | "
                    f"{float(d.get('speedup') or 0.0):.3f} | {float(d.get('speedup_p90') or 0.0):.3f} | "
                    f"{int(d.get('counts_oracle_calls') or 0)} | {int(d.get('counts_forward_trace_calls') or 0)} | {float(d.get('counts_cache_hit_rate') or 0.0):.3f} | "
                    f"{int(d.get('counts_pruned_infeasible_count') or 0)} | {float(d.get('stats_avg_batch_fill_rate') or 0.0):.3f} |"
                )
            lines.append("")

        lines.append("### Non-comparable 行（仅供参考）")
        lines.append("")
        if not incomparable:
            lines.append("- （无）")
            lines.append("")
        else:
            lines.append("| cache | hint | prune | batch_verdict | note |")
            lines.append("|---:|---:|---:|---|---|")
            for d in incomparable:
                lines.append(
                    f"| {d.get('enable_node_eval_cache')} | {d.get('use_branch_hint')} | {d.get('enable_batch_infeasible_prune')} | {d.get('verdict')} | {d.get('note','')} |"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6H PR-2: report JSONL -> CSV + summary.md.")
    parser.add_argument("--in-jsonl", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--out-summary-md", type=str, required=True)
    args = parser.parse_args(argv)

    payloads = list(_read_jsonl(Path(args.in_jsonl)))
    flat_rows: List[FlatRow] = []
    for p in payloads:
        flat_rows.extend(list(_iter_flat_rows(p)))

    _write_csv(Path(args.out_csv), flat_rows)
    _write_summary_md(Path(args.out_summary_md), flat_rows=flat_rows, payloads=payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
