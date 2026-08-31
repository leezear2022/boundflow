# Delivery asplos27-s4-1b0-ternary-20260831/r001/delivery

- round: 1
- from: codex-executor -> to: external-model
- base commit: 20f57bb
- result commit: 50c5ff642f0eb99150cc0b1bc01f414beda28ab2
- ts: 2026-08-31T02:43:10Z

## Changed files

- boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py
- tests/test_asplos27_s4_ternary_endpoint.py
- scripts/run_asplos27_s4_1b0_ternary_worker.py
- scripts/run_asplos27_s4_1b0_ternary_artifact.py
- scripts/replay_asplos27_s4_1b0_ternary_stdlib.py
- scripts/probe_asplos27_s4_1b0_ternary_tamper.py
- tests/test_asplos27_s4_1b0_ternary_artifact.py
- artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1
- gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_FORMAL_EXTERNAL_AUDIT_HANDOFF_2026_08_31.md

## Claims

- FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0 only; isolated ternary endpoint correctness, cache/receipt/prepared probe and fail-closed evidence; no timing/performance claim

## Validation

- `python scripts/replay_asplos27_s4_1b0_ternary_stdlib.py artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1` -> pass
- `python scripts/probe_asplos27_s4_1b0_ternary_tamper.py artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1` -> pass
- `conda run -n boundflow pytest -q tests/test_asplos27_s4_ternary_endpoint.py tests/test_asplos27_s4_1b0_ternary_artifact.py` -> pass
- `conda run -n boundflow pytest -q` -> pass
- `black mypy pylint git-diff-check dol-lint` -> pass

## Known limitations

- External audit pending; S4-1B production, evaluator, optimizer, timing, performance, same-solver, complete-query and 10x remain closed

## Risks

- First dry-run exposed 9/10 replay gap; fixed at 4e2a261 and formally regenerated; coherent full-resign E0 boundary remains

## Open questions

- Do AC1-AC7 independently support VALIDATED-S4-1B0-TERNARY-ENDPOINT and only opening S4-1B implementation/correctness?
