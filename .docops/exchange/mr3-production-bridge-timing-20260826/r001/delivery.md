# Delivery mr3-production-bridge-timing-20260826/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: d75f60f
- result commit: 4cbdb2e2bc6933bd7577a9183801bc2500b3531d
- ts: 2026-08-25T08:21:37Z

## Changed files

- boundflow/runtime/mr3_production_bridge_timing.py
- scripts/run_mr3_production_bridge_timing_worker.py
- scripts/run_mr3_production_bridge_timing_formal.py
- scripts/probe_mr3_production_bridge_timing_tamper.py
- tests/test_mr3_production_bridge_timing.py
- tests/test_mr3_production_bridge_timing_artifact.py
- artifacts/measurement-recovery/mr3-p-production-bridge-timing-v1
- gemini_doc/BOUNDFLOW_MR3_P_PRODUCTION_BRIDGE_TIMING_FORMAL_NO_GO_CLOSURE_2026_08_26.md

## Claims

- MR3 P-anchor production bridge correctness remains valid, but frozen complete outer exact-call timing closes NO-GO: geomean/bootstrap-lower/worst=0.979727/0.939360/0.916094x; no performance claim; complete-query timing closed.

## Validation

- `pytest tests` -> pass

## Known limitations

- Single ResNet2B property-0, warm candidate module, one P anchor; does not invalidate CIBC full-graph IBP 2.45631x or isolated TIR results.

## Risks

- Recompute timing/bootstrap/memory from raw; inspect timing boundary and module warm exclusion; run replay/tamper; verify claim scope.

## Open questions

- Approve NO-GO closure only if AC1-AC7 independently reproduce with no claim drift.
