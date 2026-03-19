from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _sig(meta: Dict[str, Any]) -> str:
    parts = [
        str(meta.get("workload")),
        str(meta.get("device")),
        str(meta.get("dtype")),
        f"p={meta.get('p')}",
        f"S={meta.get('specs')}",
        f"K={meta.get('node_batch_size')}",
        f"maxN={meta.get('max_nodes')}",
        f"steps={meta.get('steps')}",
        str(meta.get("timer")),
    ]
    return "_".join(parts).replace("/", "_")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6H PR-2: plot E2E ablations from JSONL.")
    parser.add_argument("--in-jsonl", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = _read_jsonl(Path(args.in_jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, p in enumerate(payloads):
        meta = p.get("meta", {})
        rows = p.get("rows", [])
        comparable = [r for r in rows if int(r.get("comparable", 0)) == 1]
        if not comparable:
            continue

        labels: List[str] = []
        speedups: List[float] = []
        speedups_p90: List[float] = []
        oracle_calls: List[int] = []
        forward_calls: List[int] = []
        cache_hit_rates: List[float] = []
        pruned: List[int] = []
        fill_rates: List[float] = []

        for r in comparable:
            labels.append(f"c{r['enable_node_eval_cache']}_h{r['use_branch_hint']}_p{r['enable_batch_infeasible_prune']}")
            speedups.append(float(r.get("speedup", 0.0)))
            speedups_p90.append(float(r.get("speedup_p90", 0.0)))
            cb = r.get("counts_batch", {})
            oracle_calls.append(int(cb.get("oracle_calls", 0)))
            forward_calls.append(int(cb.get("forward_trace_calls", 0)))
            cache_hit_rates.append(float(cb.get("cache_hit_rate", 0.0)))
            pruned.append(int(cb.get("pruned_infeasible_count", 0)))
            bs = r.get("batch_stats", {})
            fill_rates.append(float(bs.get("avg_batch_fill_rate", 0.0)))

        sig = _sig(meta)

        # 1) speedup bar
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 3.5))
        ax.bar(labels, speedups)
        ax.set_title(f"E2E speedup (comparable)\\n{sig}")
        ax.set_ylabel("serial_ms_p50 / batch_ms_p50")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:03d}_{sig}_speedup.png", dpi=160)
        plt.close(fig)

        # 1b) speedup (p90) bar
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 3.5))
        ax.bar(labels, speedups_p90)
        ax.set_title(f"E2E speedup p90 (comparable)\\n{sig}")
        ax.set_ylabel("serial_ms_p90 / batch_ms_p90")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:03d}_{sig}_speedup_p90.png", dpi=160)
        plt.close(fig)

        # 2) counters bar (oracle/forward/pruned)
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 3.5))
        x = range(len(labels))
        ax.plot(list(x), oracle_calls, label="oracle_calls", marker="o")
        ax.plot(list(x), forward_calls, label="forward_trace_calls", marker="o")
        ax.plot(list(x), pruned, label="pruned_infeasible", marker="o")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Counters (batch)\\n{sig}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:03d}_{sig}_counters.png", dpi=160)
        plt.close(fig)

        # 3) fill_rate vs speedup scatter
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(fill_rates, speedups)
        for i, lab in enumerate(labels):
            ax.annotate(lab, (fill_rates[i], speedups[i]), fontsize=7)
        ax.set_xlabel("avg_batch_fill_rate")
        ax.set_ylabel("speedup")
        ax.set_title(f"Fill rate vs speedup\\n{sig}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:03d}_{sig}_fill_vs_speedup.png", dpi=160)
        plt.close(fig)

        # optional: cache hit rate vs speedup (not required by DoD, but useful)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(cache_hit_rates, speedups)
        for i, lab in enumerate(labels):
            ax.annotate(lab, (cache_hit_rates[i], speedups[i]), fontsize=7)
        ax.set_xlabel("cache_hit_rate")
        ax.set_ylabel("speedup")
        ax.set_title(f"Cache hit rate vs speedup\\n{sig}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:03d}_{sig}_cache_vs_speedup.png", dpi=160)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
