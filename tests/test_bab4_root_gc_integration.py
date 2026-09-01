"""Cumulative root-CROWN plus BAB4 integration contracts."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

from pathlib import Path

from scripts import run_bab4_root_gc_three_fresh as diagnostic
from scripts import run_bab4_root_gc_worker as worker


def test_cumulative_root_protocol_is_symmetric_and_counterbalanced() -> None:
    assert worker.CONTROL == "B4-A-GC"
    assert worker.CANDIDATE == "BAB4-GC-ROOT"
    assert diagnostic.CONTROL == worker.CONTROL
    assert diagnostic.CANDIDATE == worker.CANDIDATE
    assert len(diagnostic.PAIR_ORDERS) == 3
    assert all(
        set(order) == {worker.CONTROL, worker.CANDIDATE}
        for order in diagnostic.PAIR_ORDERS
    )


def test_cumulative_root_worker_reuses_prepared_owner_after_exact_warmup() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    warm_suffix_install = source.index("warm_stack.enter_context(warm_suffix.install")
    warmup_call = source.index("receipt = original_warmup")
    reuse_reset = source.index("warm_suffix.reset_after_exact_warmup_v1()")
    suffix_install = source.index("stack.enter_context(suffix.install")
    query_call = source.index("s4_worker._worker(base_args)")
    assert warm_suffix_install < warmup_call < reuse_reset < suffix_install < query_call
    assert (
        '"root_prepared_runtime_reused_after_exact_warmup": bool(candidate)' in source
    )
    assert 'configuration="BAB4-GC" if candidate else "B4-A-GC"' in source


def test_root_segment_attribution_is_opt_in_and_non_claiming() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert (
        'parser.add_argument("--attribute-root-segments", action="store_true")'
        in source
    )
    assert '"diagnostic_only": True' in source
    assert '"included_in_performance_claim": False' in source
    assert source.index("warm_executor.reset_after_exact_warmup_v1()") < source.index(
        "root_segment_observer.install("
    )


def test_root_compute_transaction_capture_is_opt_in_and_non_claiming() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert (
        'parser.add_argument("--capture-root-compute-transaction", action="store_true")'
        in source
    )
    assert '"schema_version": "boundflow.root-compute-transaction-capture/v1"' in source
    assert '"included_in_performance_claim": False' in source
    assert source.index("warm_executor.reset_after_exact_warmup_v1()") < source.index(
        "root_compute_capture.install(BoundedModule)"
    )


def test_root_direct_backward_is_opt_in_and_keeps_prior_bounds_native() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    bridge = (
        Path(__file__).parents[1]
        / "boundflow/runtime/root_crown_backward_general_live.py"
    ).read_text(encoding="utf-8")
    assert (
        'parser.add_argument("--direct-root-backward", action="store_true")' in source
    )
    assert '"replacement_seam": "BoundedModule.backward_general:/49"' in bridge
    assert '"native_deque_traversal_count": 0' in bridge
    assert "check_prior_bounds" not in bridge


def test_root_prior_bound_attribution_is_opt_in_and_non_claiming() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert (
        'parser.add_argument("--attribute-root-prior-bounds", action="store_true")'
        in source
    )
    assert '"schema_version": "boundflow.root-prior-bounds-attribution/v1"' in source
    assert '"diagnostic_only": True' in source
    assert '"included_in_performance_claim": False' in source


def test_root_sparse_patches_replacement_is_opt_in_and_non_claiming() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    bridge = (
        Path(__file__).parents[1]
        / "boundflow/runtime/root_crown_sparse_patches_live.py"
    ).read_text(encoding="utf-8")
    backend = (
        Path(__file__).parents[1]
        / "boundflow/backends/tvm/root_crown_sparse_patches_seed.py"
    ).read_text(encoding="utf-8")
    for mode in ("shadow", "replace", "direct"):
        assert (
            f'parser.add_argument("--{mode}-root-sparse-patches", '
            'action="store_true")'
        ) in source
    assert '"replacement_seam": "BoundedModule.backward_general:/44"' in bridge
    assert '"performance_claimed": False' in bridge
    assert '"dense_seed_external_allocation": False' in backend
    assert source.index("warm_executor.reset_after_exact_warmup_v1()") < source.index(
        "root_sparse_patches_bridge.install(BoundedModule)"
    )
