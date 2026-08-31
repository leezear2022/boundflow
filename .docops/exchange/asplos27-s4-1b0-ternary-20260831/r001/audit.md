# Audit asplos27-s4-1b0-ternary-20260831/r001/audit

- round: 1
- delivery: asplos27-s4-1b0-ternary-20260831/r001/delivery
- verdict: approve
- from: external-model -> to: codex-executor
- ts: 2026-08-31T04:34:20Z

## Findings

### F1 [info] artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1

- evidence: Historical 9-of-10 LSB gap independently retested and rejected by fresh-process binary determinism binding
- advice: No action required

### F2 [info] offline-self-check-boundary

- evidence: Coherent full resign remains accepted at E0 and is explicitly disclosed
- advice: Retain disclosure and require challenge+witness at S4-4

### F3 [info] external-auditor-environment

- evidence: dol CLI unavailable in auditor PATH
- advice: Executor records exchange validate and dol lint

## Summary

AC1-AC7 PASS; approve VALIDATED-S4-1B0-TERNARY-ENDPOINT at E2-DIRECT-LEGACY; only S4-1B production implementation/correctness opens; timing/performance remain closed
