"""Prepared-runtime GC isolation contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import gc
from types import SimpleNamespace

from boundflow.runtime.prepared_gc_isolation import prepared_gc_isolation_v1


def test_prepared_gc_isolation_collects_young_and_restores_module_owner() -> None:
    owner = SimpleNamespace(gc=gc)
    enabled_before = gc.isenabled()
    with prepared_gc_isolation_v1(
        gc_module=gc, complete_verifier_module=owner
    ) as receipt:
        assert owner.gc is not gc
        assert receipt.restored is False
        cycle = []
        cycle.append(cycle)
        del cycle
        assert owner.gc.collect() >= 1
    assert owner.gc is gc
    assert gc.isenabled() is enabled_before
    payload = receipt.to_dict()
    assert payload["query_collect_generation"] == 1
    assert payload["query_collect_call_count"] == 1
    assert payload["prepared_old_generation_scan_excluded"] is True
    assert payload["query_collection_preserved"] is True
    assert payload["restored"] is True
