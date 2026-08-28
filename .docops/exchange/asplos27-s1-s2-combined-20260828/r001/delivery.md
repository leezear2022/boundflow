# Delivery asplos27-s1-s2-combined-20260828/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: d7adec6
- result commit: cd1670c
- ts: 2026-08-28T04:49:36Z

## Changed files

- boundflow/backends/tvm/asplos27_s2_selected_value.py
- boundflow/runtime/asplos27_s2_crown_pipeline.py
- scripts/run_asplos27_s2_crown_worker.py
- scripts/run_asplos27_s2_crown_artifact.py
- scripts/probe_asplos27_s2_crown_tamper.py
- tests/test_asplos27_s2_crown_pipeline.py
- artifacts/asplos27-s2-crown-pipeline/resnet2b-p-anchor-v2
- gemini_doc/BOUNDFLOW_ASPLOS27_S2_FORMAL_CLOSURE_2026_08_28.md

## Claims

- S1 closes canonical IBP plumbing; S2 closes only P-anchor single-evaluation coarse CROWN at 4.2453819646x geomean and 3.5407988567x worst with correctness and fail-closed receipts; only S3 preregistration opens

## Validation

- `pytest -q tests/test_asplos27_s2_crown_pipeline.py` -> pass
- `pytest -q tests` -> pass
- `S2 artifact raw replay` -> pass
- `S2 ten resigned tamper probes` -> pass
- `black mypy pylint` -> pass
- `dol lint --soft` -> pass

## Known limitations

- No optimizer 10/9, same-solver, complete-query, cross-model, overall 10x, or ASPLOS-ready claim

## Risks

- PDN worst pair is 3.5407988567x and therefore close to the frozen 3.50x gate; audit it explicitly

## Open questions

- Do AC1-AC7 support both closures and opening only S3 preregistration?
