# Delivery fsg4-b4b2-b2-3-dense-conv-20260823/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: c28c903
- result commit: 73070706935f2e6610d4e12903e1d9b4f67b0f83
- ts: 2026-08-23T11:35:38Z

## Changed files

- boundflow/ir/differentiable_lower_dense_conv_tir.py
- boundflow/backends/tvm/differentiable_lower_dense_conv.py
- boundflow/runtime/fsg4_b4b2_dense_conv_tir.py
- scripts/run_fsg4_b4b2_dense_conv_tir_correctness.py
- tests/test_fsg4_b4b2_dense_conv_tir.py
- tests/test_fsg4_b4b2_dense_linear_tir.py

## Claims

- P-anchor dense Conv TIR correctness only: 5 raw/20 metrics/92190 elements pass, beta gradient absent, structural workspace exact, no performance claim

## Validation

- `B2-0-through-B2-3-targeted` -> pass
- `B4-B-related` -> pass
- `full-pytest` -> pass
- `black-mypy-pylint-diff` -> pass
- `tvm-rebuild-docops-lint` -> pass

## Known limitations

- No timing, B2-4/B2-5/B4-B3, performance, memory, B0 parity, whole-core/query or ASPLOS-ready claim

## Risks

- Fresh labels denote five independently captured production raws; B2-5 formal independent-process artifact remains closed

## Open questions

- Approve B2-3 correctness and at most open B2-4, or report evidence-backed findings
