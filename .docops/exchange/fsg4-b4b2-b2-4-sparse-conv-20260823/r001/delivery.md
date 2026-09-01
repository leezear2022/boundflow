# Delivery fsg4-b4b2-b2-4-sparse-conv-20260823/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: b18fad483fcfa9bbef61337628f368a7ca2fd7c2
- result commit: 1f8d47a8acd55f9b315e207a549f515e29a6f35e
- ts: 2026-08-23T13:59:31Z

## Changed files

- boundflow/ir/differentiable_lower_sparse_conv_tir.py
- boundflow/backends/tvm/differentiable_lower_sparse_conv.py
- boundflow/runtime/fsg4_b4b2_sparse_conv_tir.py
- scripts/run_fsg4_b4b2_sparse_conv_tir_correctness.py
- tests/test_fsg4_b4b2_sparse_conv_tir.py
- tests/test_fsg4_b4b2_dense_conv_tir.py

## Claims

- P-anchor sparse-source Conv P0 five-raw correctness plus 12 unique pre-timing schedule candidates compile/correct; no timing winner or performance claim

## Validation

- `B2-0-through-B2-4-targeted` -> pass
- `B4-B-related` -> pass
- `full-pytest` -> pass
- `runner-p0-and-12-candidates` -> pass
- `black-mypy-pylint-diff` -> pass
- `tvm-rebuild-docops-lint` -> pass

## Known limitations

- No timing raw, winner, B2-5, B4-B3, speedup, memory, whole-core/query, B0 parity or ASPLOS-ready claim

## Risks

- Module TIR/device hash independent recompilation remains explicitly deferred to B2-5 replay; candidate ledger is immutable at 12

## Open questions

- Approve B2-4 and at most open B2-5 formal artifact/timing implementation, or report evidence-backed findings
