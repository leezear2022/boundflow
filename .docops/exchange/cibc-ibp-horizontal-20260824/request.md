# CIBC IBP horizontal fusion formal closure audit

- task: cibc-ibp-horizontal-20260824
- doc: cibc-ibp-horizontal-20260824/request
- from: codex -> to: external-model
- executor: codex / auditor: external-model
- base commit: f4f6b397252603152bf1c279d23b1ade0421f68f
- created: 2026-08-23T20:04:49Z

## Original request

Independently audit the frozen CIBC IBP Conv horizontal-fusion implementation, formal raw evidence, semantic replay, performance derivation, and claim boundaries. Do not trust closure summaries without recomputing raw.

## Scope

baa4503..b94e61a plus artifacts/cibc-ibp-horizontal-formal

## Acceptance criteria

- AC1: source/order/code hashes and frozen protocol are exact; no post-hoc gate or schedule change
- AC2: six production Conv semantics match the four-Conv BoundFlow IBP baseline within 3e-4 and sign exact
- AC3: independently recompute 64/128/256 schedule selection and six operator speedups from all 30-group raw records
- AC4: independently recompute six fresh whole-model medians, geomean, bootstrap lower, worst pair, input-copy inclusion, CUDA Graph parity, and 6/6 coverage
- AC5: root replay passes and all 10 fully re-signed tamper probes are genuinely fail-closed
- AC6: targeted/full tests and Black/Mypy/Pylint/DocOps validations pass; skips are only documented environment boundaries
- AC7: claim remains reduced to RTX4060 ResNet2B IBP Conv horizontal fusion, not auto_LiRPA, alpha-CROWN/BaB/query, memory, cross-model, or ASPLOS-ready
