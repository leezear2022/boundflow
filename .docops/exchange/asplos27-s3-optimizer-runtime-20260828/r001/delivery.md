# Delivery asplos27-s3-optimizer-runtime-20260828/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: ad58afb
- result commit: 6ef12b55dfc1c9f1207adcc55694460a72e14821
- ts: 2026-08-28T11:28:37Z

## Changed files

- artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2
- boundflow/runtime/asplos27_s3_optimizer_pipeline.py
- boundflow/runtime/asplos27_s2_crown_pipeline.py
- boundflow/backends/tvm/asplos27_s2_selected_value.py
- scripts/run_asplos27_s3_optimizer_artifact_v2.py
- scripts/probe_asplos27_s3_optimizer_v2_tamper.py
- gemini_doc/BOUNDFLOW_ASPLOS27_S3_FORMAL_CLOSURE_2026_08_28.md

## Claims

- Preserve v1 NO-GO; validate v2 P-anchor local 10/9 wrapper at 3.243894x geomean; close unsafe selected-value CUDA Graph ownership; open only S4 implementation/correctness

## Validation

- `pytest -q targeted` -> pass
- `pytest -q tests` -> pass
- `artifact v2 replay` -> pass
- `outer-resigned tamper 10/10` -> pass
- `black mypy pylint` -> pass

## Known limitations

- No same-solver, complete-query, cross-model, 10x or ASPLOS-ready performance claim

## Risks

- Historical S2 selected-value CUDA Graph mechanism is unsafe and superseded; whole-wrapper dynamic allocated is 13824 B

## Open questions

- Do AC1-AC7 support closing S3 v2 and opening only S4 implementation/correctness?
