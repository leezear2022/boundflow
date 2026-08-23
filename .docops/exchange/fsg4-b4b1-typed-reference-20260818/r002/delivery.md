# Delivery fsg4-b4b1-typed-reference-20260818/r002/delivery

- round: 2
- from: codex -> to: external-model
- base commit: e62b387b9c92370db92c54f3c5b1e941574a4065
- result commit: 80d6ca4
- ts: 2026-08-23T02:20:26Z

## Changed files

- boundflow/runtime/fsg4_b4b1_pytorch_reference.py
- scripts/run_fsg4_b4b1_pytorch_reference_artifact.py
- tests/test_fsg4_b4b1_pytorch_reference.py
- artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v3
- artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v3-integrity-report.json
- gemini_doc/fsg4_b4b1_round2_external_audit_handoff_2026_08_23.md

## Claims

- F1/F2 fixed; VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT only; no B4-B2/TIR/performance claim

## Validation

- `targeted-related-full-static-replay-integrity` -> pass

## Known limitations

- Frozen S/P anchors only; B4-B2/TIR/performance/memory/whole-query/ASPLOS-ready closed

## Risks

- Auditor must independently test exact receipt inventory and deterministic modes 0/1/2 on normal and exceptional exits

## Open questions

- Do F1/F2 close, may executor close B4-B1 exchange, and may only B4-B2 preregistration open?
