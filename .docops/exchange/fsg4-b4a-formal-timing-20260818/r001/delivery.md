# Delivery fsg4-b4a-formal-timing-20260818/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: adc175b
- result commit: d387a7c
- ts: 2026-08-18T03:37:19Z

## Changed files

- scripts/run_fsg3_same_solver_timing.py
- scripts/run_fsg4_b4a_same_solver_worker.py
- scripts/run_fsg4_b4a_formal_timing.py
- scripts/probe_fsg4_b4a_formal_timing_tamper.py
- tests/test_fsg3_same_solver_worker.py
- tests/test_fsg4_b4a_formal_timing.py
- tests/test_fsg4_b4a_formal_timing_tamper.py
- tests/test_fsg4_b4a_formal_timing_artifact.py
- artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5
- artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5-tamper-report.json
- gemini_doc/change_2026-08-18_fsg4_b4a_formal_timing_internal_closure.md

## Claims

- B4-A correctness/mechanism validated; preregistered performance gate is NO-GO; no performance claim and no cumulative admission

## Validation

- `related73-full1356-replay-tamper-static-docops` -> pass

## Known limitations

- kernel launch delta deferred; profile is attribution only; external audit required

## Risks

- mobile GPU thermal telemetry required power-policy and interval-delta hardening; rejected v1-v4 are excluded

## Open questions

- Approve VALIDATED-NO-GO-B4-A-PERFORMANCE and permit only a separately preregistered B4-B decision?
