# Delivery fsg4-b4b0-five-fresh-20260818/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: 1dbb2de
- result commit: a1c6051
- ts: 2026-08-18T05:22:07Z

## Changed files

- boundflow/runtime/fsg4_b4b_production_region_capture.py
- boundflow/runtime/crown_ibp.py
- boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py
- scripts/run_fsg4_b4b_capture_worker.py
- scripts/run_fsg4_b4b_five_fresh_artifact.py
- scripts/probe_fsg4_b4b_five_fresh_tamper.py
- tests/test_fsg4_b4b_five_fresh_artifact.py
- artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v1
- artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v1-tamper-report.json
- gemini_doc/change_2026-08-18_fsg4_b4b0_five_fresh_internal_closure.md

## Claims

- B4-B0 production evaluation-zero capture correctness/ownership only: five fresh CUDA processes, ten typed captures, raw replay and nine outer-resigned tamper categories. No TIR or performance claim.

## Validation

- `pytest targeted 20 passed` -> pass
- `pytest full 1372 passed 3 skipped 6 warnings` -> pass
- `artifact root replay` -> pass
- `tamper 9 of 9 rejected` -> pass
- `Black Mypy Pylint diff DocOps lint` -> pass

## Known limitations

- B4-B1 and B4-B2 remain closed pending audit; performance_claimed=false; tir_admitted=false; no region or whole-query speedup

## Risks

- Auditor should independently recompute raw tensor comparisons and not trust summary.json; CUDA stream numeric id is process-local and excluded from cross-run equality while default-stream property remains mandatory

## Open questions

- Does evidence support external approval of B4-B0 capture correctness and opening only B4-B1 typed pure-PyTorch reference?
