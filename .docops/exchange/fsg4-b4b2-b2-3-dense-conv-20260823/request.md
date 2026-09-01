# Audit B4-B2 B2-3 P-anchor dense Conv TIR correctness

- task: fsg4-b4b2-b2-3-dense-conv-20260823
- doc: fsg4-b4b2-b2-3-dense-conv-20260823/request
- from: codex -> to: external-model
- executor: codex / auditor: external-model
- base commit: c28c903
- created: 2026-08-23T11:35:38Z

## Original request

Independently audit source 7307070 for P-anchor dense Conv TIR forward/backward correctness; do not accept summary numbers and do not open timing.

## Scope

B2-3 correctness only; B2-4/B2-5/timing/B4-B3 excluded

## Acceptance criteria

- AC1 source order scope and first-class receipts pass
- AC2 independent float64 math and four-output parity pass
- AC3 live GPU ABI stream DLPack launch cache and absent beta pass
- AC4 structural alloc-buffer inventory exact and tamper rejected
- AC5 negative paths and prior dense Linear finding closed
- AC6 targeted related full static TVM and DocOps validation pass
