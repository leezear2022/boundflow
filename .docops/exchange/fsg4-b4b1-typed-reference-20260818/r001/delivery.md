# Delivery fsg4-b4b1-typed-reference-20260818/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: 88e0e7a
- result commit: 56fcc2b
- ts: 2026-08-18T07:52:51Z

## Changed files

- boundflow/ir/differentiable_lower_region.py
- boundflow/runtime/fsg4_b4b1_pytorch_reference.py
- scripts/run_fsg4_b4b1_pytorch_reference_artifact.py
- scripts/probe_fsg4_b4b1_pytorch_reference_integrity.py
- tests/test_fsg4_b4b1_pytorch_reference.py
- artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2

## Claims

- VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT; no TIR or performance claim

## Validation

- `targeted` -> pass
- `related` -> pass
- `full` -> pass
- `static` -> pass
- `integrity` -> pass

## Known limitations

- Frozen S/P production anchors only; B4-B2/TIR/performance/memory/whole-query/ASPLOS-ready closed

## Risks

- v1 was thread-policy-sensitive and is explicitly superseded; auditor must verify v2 policy isolation and add an independent coordinated attack

## Open questions

- Approve B4-B1 correctness closure and only allow separate B4-B2 preregistration?
