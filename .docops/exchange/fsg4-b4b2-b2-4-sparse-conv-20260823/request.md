# Audit B4-B2 B2-4 sparse-source Conv P0 and bounded schedule ledger

- task: fsg4-b4b2-b2-4-sparse-conv-20260823
- doc: fsg4-b4b2-b2-4-sparse-conv-20260823/request
- from: codex -> to: external-model
- executor: codex / auditor: external-model
- base commit: b18fad483fcfa9bbef61337628f368a7ca2fd7c2
- created: 2026-08-23T13:59:31Z

## Original request

Independently audit source 1f8d47a for P-anchor sparse-source Conv P0 correctness and the complete 12-candidate pre-timing ledger; do not accept summary numbers.

## Scope

B2-4 final audit only; no timing, winner, B2-5, B4-B3 or performance claim

## Acceptance criteria

- AC1 git order scope and preregistration unchanged
- AC2 production compressed-alpha mapping and empty-beta absence exact
- AC3 independent mathematical oracle confirms sparse forward and gradients
- AC4 live five-raw P0 GPU gate passes
- AC5 twelve unique physical candidates compile and match oracle with frozen ledger
- AC6 structural workspace and schedule transformations exact
- AC7 receipts negative paths and prior findings disposition pass
- AC8 targeted related full static TVM and DocOps validation pass
