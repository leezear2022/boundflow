# GC0-0 Generic Verification Graph Schema 外审关闭记录

date: 2026-08-26
stage: GC0-0
implementation-commit: `07f02fe`
exchange: `gc0-0-schema-20260826`
round: 1
status: externally-approved-closed
performance-claimed: false

## 1. 关闭结论

DocOps exchange Round 1 已由独立审计方提交 `approve`，executor 已执行 `exchange close`，状态为
`closed(resolution=approved)`。审计结果为 0 blocker、0 major、1 minor、1 info，Audit-AC1—Audit-AC7
全部通过。因此 GC0-0 状态冻结为：

```text
VALIDATED-GC0-0-GENERIC-VERIFICATION-GRAPH-SCHEMA
```

该状态只证明通用 typed/canonical schema、稳定 identity、拒绝原因 vocabulary 与 schema-level
fail-closed 边界，不证明 capture、graph analysis、legality admission、Relax/TIR lowering、physical
runtime、production correctness 或性能。

## 2. 独立复核事实

- 精确实现 diff 为 `ad23d86..07f02fe`，没有 production 路径改动，也没有既有模块 import 新 schema；
- 22 类 rejection reason 与冻结合同逐字一致，15-direct/7-analysis 分区 disjoint 且完备；
- empty-β Conv、active-β Linear、multi-Conv 10/9 三类 fixture 只通过通用 schema canonical round-trip；
- canonical JSON 与四级 hash 可独立重算；审计方自建五类 tamper 均被正确拒绝；
- registry 恰为冻结八条规则且不可执行；实现只依赖 Python 标准库；
- targeted=`11 passed`，审计相关集=`49 passed`，full=`1832 passed, 3 skipped`；Black、Mypy、
  Pylint `10.00/10`、diff、DocOps validate/lint 均通过。

## 3. Findings 与后继约束

### F1 minor：shallow policy 与 full analysis 必须分层命名

`DENSE_A_ESCAPE` 属 analysis 分区，但 GC0-0 的 VJP schema 可以根据声明的 saved-state policy 在构造期
以同名 reason 保守拒绝。这是合法的 reject-side shortcut，不是 dense escape graph analysis。

GC0-1 预注册必须冻结两层证据：

1. `shallow_policy_rejection`：只证明输入声明已经违反合同；
2. `analysis_witness_rejection`：从 captured graph、use-def、alias/lifetime 或 VJP saved-state flow
   导出可复核 witness，才计入 analysis negative coverage。

同名 reason 的浅层触发不得用于关闭 GC0-1 analysis acceptance。

### F2 info：三个浅层分支尚无专项测试

`EFFECT_ORDER_CONFLICT`、`REGION_EXTERNAL_USE`、`UNSAFE_ALIAS_OR_LIFETIME`已有 schema-level 保守
拒绝分支，但 GC0-0 不要求逐项测试。GC0-1 必须以真实 negative graph 和完整 witness 覆盖它们；仅补
constructor 测试不能替代 analysis coverage。

## 4. 唯一合法后继

GC0-0 关闭后只开放 **GC0-1 capture/analysis 预注册文档**。实现门禁仍关闭。GC0-1 预注册必须先
独立外审批准，之后才能开始 capture/analysis 实现；GC0-2 lowering、GC-1 correctness、runtime 和
timing 继续关闭。
