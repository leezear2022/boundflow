# FSG4 B4-B0 five-fresh production capture external audit

- task: fsg4-b4b0-five-fresh-20260818
- doc: fsg4-b4b0-five-fresh-20260818/request
- from: codex-executor -> to: external-model-auditor
- executor: codex / auditor: external-model
- base commit: 1dbb2de4bc29eb92457e2d24c3e627d638b6607a
- created: 2026-08-18T05:21:50Z

## Original request

Independently audit B4-B0 production evaluation-zero dual-anchor capture. Do not trust executor summary values; recompute from raw PT/JSON and source.

## Scope

B4-B0 typed lineage, live capture, five fresh artifact, root replay, tamper resistance, regression and claim boundary

## Acceptance criteria

- AC1 provenance: independently verify source 1dbb2de, code_revision, manifest/protocol/file hashes, source capture/model hashes and absence of host-local paths
- AC2 fresh capture: verify five isolated CUDA workers, ten captures, exact S/P anchor structure, evaluation 0, 10/9 schedule, default stream, no alias, provider/fallback zero
- AC3 semantics: rebuild typed captures from raw PT, independently compare 108 tensors and 664744 elements at atol/rtol 2e-4 with exact signs; verify active-beta S and empty-beta P ownership
- AC4 tamper: independently inspect and if possible extend the nine outer-resigned state/start-node/topology/shape/alpha-index/beta-location/gradient/alias/stream probes; all must fail semantically
- AC5 validation: reproduce targeted 20 passed, full 1372 passed 3 skipped 6 warnings, Black, scoped Mypy, Pylint 10.00/10, diff check and DocOps validate/lint
- AC6 claims: confirm performance_claimed=false and tir_admitted=false throughout; approve only B4-B0 capture correctness and admission to B4-B1 typed reference, never B4-B2 TIR or performance
