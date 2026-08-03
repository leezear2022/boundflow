# Delivery rvir-post-hardening-20260803/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: b01225b3ffa7a4bfc6fcdfe42ab9fe0973d631ae
- result commit: 1a6eb65
- ts: 2026-08-03T13:54:43Z

## Changed files

- tests/test_real_verifier_ir_integration.py
- scripts/run_phase6h_artifact.sh
- tests/test_phase6h_artifact_runner_smoke.py
- scripts/run_real_verifier_ir_artifact.py
- tests/test_real_verifier_ir_artifact.py
- artifacts/rvir/rvir-cpu-correctness-v2-20260803/manifest.json
- artifacts/rvir/rvir-cpu-correctness-v2-20260803/online_queries.jsonl
- artifacts/rvir/rvir-cpu-correctness-v2-20260803/online_typed_ir.jsonl
- gemini_doc/change_2026-08-03_rvir_resnet_raw_rerun.md

## Claims

- PR5-PR8 close F1,F2,F4,F5 with dedicated fail-closed tests, PATH-independent Python selection, raw online semantic replay, and fixed-source ResNet rerun evidence; F3 is preserved through a new immutable exchange task; RVIR remains VALIDATED-REDUCED with no performance or CUDA claim.

## Validation

- `python -m pytest -q targeted` -> pass
- `RVIR v2 fresh semantic replay` -> pass
- `python -m pytest -q tests (460 passed, 37 skipped)` -> pass
- `Black, mypy, Pylint` -> pass
- `fresh fixed-source ResNet rerun and tensor digest comparison` -> pass
- `DocOps exchange validate and lint --soft` -> pass

## Known limitations

- CPU correctness/integration only; fused coverage remains 0/394; no fresh CUDA, performance, complete verifier E2E, or ASPLOS-ready claim.

## Risks

- Auditor may mistake digest validation for semantic replay, or typed external admission for BoundFlow fused execution; F5 requires a separately cloned external alpha-beta-CROWN environment.

## Open questions

- Do AC1-AC6 independently prove that F1-F5 are closed or correctly preserved without claim drift?
