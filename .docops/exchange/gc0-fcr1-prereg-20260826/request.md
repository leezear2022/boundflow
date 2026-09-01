# GC-0/FCR-1 Verification Graph ABI Correctness Preregistration Audit

- task: gc0-fcr1-prereg-20260826
- doc: gc0-fcr1-prereg-20260826/request
- from: codex-executor -> to: external-model-auditor
- executor: codex-executor / auditor: external-model-auditor
- base commit: 9c5f3867c657078cb6ba980a613b686c5a08f2d2
- created: 2026-08-25T17:34:09Z

## Original request

Independently audit the documentation-only GC-0/FCR-1 preregistration at exact result commit 68dd54c332f6d9a46640a70cdb83bbd4fb81c3f8. Do not assume implementation, correctness, timing, speedup, query/queue benefit, or ASPLOS readiness. Verify consistency against MR7-R closure, the MR7 graph-compiler research plan, and current code contracts.

## Scope

Exact diff 9c5f3867c657078cb6ba980a613b686c5a08f2d2..68dd54c332f6d9a46640a70cdb83bbd4fb81c3f8; preregistration plan, changelog, and authority-document synchronization only. User-owned docs/CIBC_for_DAC.pdf is outside scope.

## Acceptance criteria

- AC1: verify source identity, documentation-only ordering, no implementation/raw/timing, and authority-doc status consistency.
- AC2: verify GC-0/GC-1/GC-2 staged gates match the existing research plan and cannot be skipped or collapsed.
- AC3: verify graph/value/op/effect/VJP schema is model/site independent and covers P empty-beta, S active-beta, and multi-site 10/9 instances without schema hardcoding.
- AC4: verify analysis-only legality, stable rejection reasons, effect/alias/external-use/postdominator/dense-escape witnesses, and fail-closed boundaries are sufficient and falsifiable.
- AC5: verify Relax/TIR lowering receipts, symbolic then physical arena ABI, prepared runtime, minimal saved state, and replay-by-relowering requirements are precise and do not prematurely claim implementation.
- AC6: verify five-fresh dual-oracle trajectory protocol, frozen tolerances, rollback, structural counters, artifact manifest, and 22 fully re-signed tamper classes are independently auditable.
- AC7: verify no claim drift: implementation_open=false, timing_open=false, performance_claimed=false, ASPLOS-ready=false; report blocker/major/minor/info findings and approve only if all blocking/major issues are absent.
