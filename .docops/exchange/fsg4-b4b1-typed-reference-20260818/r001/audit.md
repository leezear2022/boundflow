# Audit fsg4-b4b1-typed-reference-20260818/r001/audit

- round: 1
- delivery: fsg4-b4b1-typed-reference-20260818/r001/delivery
- verdict: request_changes
- from: external-model -> to: codex
- ts: 2026-08-22T17:33:11Z

## Findings

### F1 [major] boundflow/runtime/fsg4_b4b1_pytorch_reference.py:516

- evidence: DifferentiableLowerReferenceReceiptV1.validate accepts metrics=() with semantic_passed=true and accepts a flipped incoming_lower_a_gradient_present for both S and P; receipt therefore does not exactly bind the required metric/result/gradient-target inventory.
- advice: Derive the exact metric inventory and gradient ownership from IR/contracts, enforce nonempty exact names/count/targets and consistent beta/incoming presence flags, and add direct negative tests.

### F2 [major] scripts/run_fsg4_b4b1_pytorch_reference_artifact.py:263

- evidence: With caller state deterministic=true warn_only=true debug_mode=1, _reference_execution_policy exits as deterministic=true warn_only=false debug_mode=2; the caller deterministic warn/debug state is not restored exactly.
- advice: Save and restore deterministic debug/warn mode, and test modes 0/1/2 on normal and exceptional exits together with threads, precision, and MKLDNN state.

## Summary

Request changes: AC1 and AC3 fail on two independently reproduced major interface-consistency defects; AC2, AC4, AC5 and AC6 pass. Independent raw replay: 5 runs, 10 captures, 60 metrics, 196380 elements, max diff 6.109476089477539e-07, allclose/sign exact true. Tests: targeted 23 passed; related 119 passed, 12 skipped; full 1357 passed, 51 skipped, 7 warnings. Full evidence: r001/audit_report_full.md.
