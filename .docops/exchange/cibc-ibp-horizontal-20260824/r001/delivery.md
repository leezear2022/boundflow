# Delivery cibc-ibp-horizontal-20260824/r001/delivery

- round: 1
- from: codex -> to: external-model
- base commit: f4f6b397252603152bf1c279d23b1ade0421f68f
- result commit: b94e61a
- ts: 2026-08-23T20:04:49Z

## Changed files

- boundflow/backends/tvm/cibc_ibp_conv.py
- boundflow/runtime/cibc_ibp_conv.py
- boundflow/runtime/cibc_ibp_graph.py
- boundflow/domains/interval.py
- scripts/run_cibc_ibp_horizontal_worker.py
- scripts/run_cibc_ibp_horizontal_artifact.py
- scripts/probe_cibc_ibp_horizontal_tamper.py
- tests/test_cibc_ibp_conv.py
- tests/test_cibc_ibp_graph.py
- tests/test_cibc_ibp_horizontal_artifact.py
- artifacts/cibc-ibp-horizontal-formal
- gemini_doc/BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_CLOSURE_2026_08_24.md
- gemini_doc/BOUNDFLOW_B4_ORIGINAL_PLAN_AND_CIBC_FINAL_STATUS_2026_08_24.md

## Claims

- VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL: operator geomean/worst 12.7951/9.1423x and whole-model geomean/bootstrap-lower/worst 2.45631/2.45386/2.45091x on frozen RTX4060 ResNet2B IBP workload

## Validation

- `targeted=pass,full-1492-3=pass,replay=pass,tamper-10-10=pass,black-mypy-pylint` -> pass

## Known limitations

- single GPU/model/property; BoundFlow four-Conv baseline; Conv-only; no memory or solver/query claim; production default unchanged

## Risks

- baseline is not auto_LiRPA; 2 Linear and non-affine ops are not TIR-horizontal-fused; CUDA Graph private-pool memory is not compared

## Open questions

- Approve VALIDATED-REDUCED closure or return concrete blocker/major findings with raw evidence.
