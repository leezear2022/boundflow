# GC0-1 Verification Graph Capture + Analysis 预注册变更记录

date: 2026-08-26
stage: GC0-1-preregistration
parent: `94166b6`
status: documentation-only-pending-external-audit
implementation-open: false
timing-open: false
performance-claimed: false

## 1. 变更范围

本轮只新增 GC0-1 capture/analysis 预注册与权威索引，不修改 `boundflow/`、`tests/`、`scripts/`或
artifact。独立外审批准本计划前，GC0-1 实现门禁保持关闭。

## 2. 冻结内容

- provider-neutral snapshot、capture request/receipt 与三类 source adapter；
- deterministic ID、mapping/omission ledger、canonical module construction；
- A0—A8 schema/use-def/boundary/postdominator/effect/residual/alias/dense-VJP analyses；
- typed witness ledger；admitted结果也必须以coverage-bound positive proof witness闭合，禁止`ok`占位；
- shallow policy rejection 与 full analysis witness 的强制分层；
- 七类 analysis reason 的 causal witness 与至少14个schema-valid negative graph；
- 15类 direct-through-capture、四类 multi-reason、三类 positive signature；
- five-fresh metadata artifact、semantic replay 与16类fully re-signed tamper；
- Plan-AC1—Plan-AC7、GO/NO-GO/INVALID与唯一后继GC0-2预注册。

## 3. GC0-0 findings 处置

- F1 minor：`DENSE_A_ESCAPE`的 shallow VJP policy拒绝不等于dense lineage analysis；GC0-1必须
  输出source→sink path才算full coverage；
- F2 info：`EFFECT_ORDER_CONFLICT`、`REGION_EXTERNAL_USE`、`UNSAFE_ALIAS_OR_LIFETIME`
  不以constructor测试关闭，必须由schema-valid graph在A2/A4/A6产生完整witness；
- 同名reason新增`evidence_kind/pass_id/subject/path/detail_code/witness_hash`分层合同。

## 4. Claim 边界

本轮只可claim GC0-1 protocol已冻结。不得claim capture/analysis实现、任一region admitted、rule
rewrite、lowering、runtime、production correctness、speedup或ASPLOS-ready。
