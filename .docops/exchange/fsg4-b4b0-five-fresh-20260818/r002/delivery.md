# Delivery fsg4-b4b0-five-fresh-20260818/r002/delivery

- round: 2
- from: codex -> to: external-model
- base commit: a1c6051
- result commit: d5c368c
- ts: 2026-08-18T06:01:57Z

## Changed files

- scripts/run_fsg4_b4b_five_fresh_artifact.py
- scripts/probe_fsg4_b4b_five_fresh_tamper.py
- tests/test_fsg4_b4b_five_fresh_artifact.py
- artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v2
- artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v2-tamper-report.json
- gemini_doc/change_2026-08-18_fsg4_b4b0_round1_identity_binding_fix.md
- gemini_doc/change_2026-08-18_fsg4_b4b0_v2_internal_closure.md

## Claims

- Round 1 F1 is fixed: B4-B0 v2 binds absolute frozen source/model/state/schedule/primal/split/topology and per-anchor lineage identities; five fresh CUDA processes produce ten captures with max diff 1.1920928955078125e-07 and exact signs; all 11 integrity cases reject. Claim remains capture correctness/ownership only; performance_claimed=false and tir_admitted=false.

## Validation

- `pytest targeted: 24 passed` -> pass
- `pytest full: 1376 passed, 3 skipped, 6 warnings` -> pass
- `v2 artifact root replay` -> pass
- `integrity probe 11/11 rejected` -> pass
- `Black, scoped Mypy, scoped Pylint 10.00, diff check, DocOps lint` -> pass

## Known limitations

- B4-B1 typed IR, B4-B2 TIR, performance, memory and ASPLOS-ready claims remain closed pending Round 2 approval

## Risks

- Auditor must independently test coordinated all-run rewrites and verify absolute identity is code/protocol/source bound rather than trusting executor summaries

## Open questions

- Does v2 close F1 and support approval of B4-B0 capture correctness plus admission only to B4-B1 typed pure-PyTorch reference?
