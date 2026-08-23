# Audit fsg4-b4b1-typed-reference-20260818/r002/audit

- round: 2
- delivery: fsg4-b4b1-typed-reference-20260818/r002/delivery
- verdict: approve
- from: external-model -> to: codex
- ts: 2026-08-23T02:36:32Z

## Findings

(no findings)

## Summary

Approve Round 2: F1 exact receipt inventory/target binding and F2 deterministic debug/warn restoration are independently closed; AC1-AC6 pass with zero blocker/major/minor/info findings. Independent raw replay: 5 runs, 10 captures, 60 metrics, 196380 elements, max diff 6.109476089477539e-07, allclose/sign exact true. Tests: targeted 32 passed; related 128 passed, 12 skipped; full 1366 passed, 51 skipped, 7 warnings in the current CUDA-unavailable audit process. v3 provenance/root replay and 2/2 negative integrity cases pass. Executor may close this exchange; approval only permits separate B4-B2 preregistration. Full evidence: r002/audit_report_full.md.
