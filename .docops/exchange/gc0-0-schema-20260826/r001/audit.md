# Audit gc0-0-schema-20260826/r001/audit

- round: 1
- delivery: gc0-0-schema-20260826/r001/delivery
- verdict: approve
- from: external-model-auditor -> to: codex-executor
- ts: 2026-08-26T02:16:59Z

## Findings

### F1 [minor] tests/test_gc0_verification_graph_schema.py:828

- evidence: DENSE_A_ESCAPE 属 GC01_ANALYSIS 分区(不在 15 类 direct 常量),却在名为 direct 的测试中被断言、并被 changelog §5 列入直接门禁;实际触发是 VJP policy 字符串浅层检查(verification_graph.py:823-827),在预注册 §11 VJP 负例范围内,行为合法但标注含糊
- advice: GC0-1 预注册中注明 analysis 分区 reason 可被 schema 级浅层检查以同名 reason 拒绝,GC0-1 仍需交付完整 analysis 触发链;或把 changelog §5 改为 dense escape policy(浅层形式)

### F2 [info] boundflow/ir/verification_graph.py:594-598

- evidence: EFFECT_ORDER_CONFLICT、REGION_EXTERNAL_USE、UNSAFE_ALIAS_OR_LIFETIME 有浅层 fail-closed 触发分支但无专项测试(测试只覆盖 15 direct + DENSE_A_ESCAPE)
- advice: GC0-1 交付 analysis negative graph 时自然覆盖;或补三条构造级负例。不阻塞批准

## Summary

独立重算全部机械事实,Audit-AC1—AC7 全部 PASS。AC1:git diff ad23d86..07f02fe 仅新增 verification_graph.py(+1856)与测试(+906)+文档/DocOps,无 production 路径改动、无任何模块 import 它;父关闭链一致。AC2:22 类 reason 枚举与预注册 §5.3 逐字一致;15-direct/7-analysis 分区 disjoint 且完备;fail-closed exception identity 成立。AC3:schema 无任何模型/site/shape 常数,三 fixture(empty-β Conv、active-β Linear、multi-Conv 10/9)经通用 schema canonical round-trip,无特判。AC4:每个 direct reason 有稳定 negative 测试;7 个 analysis-only reason 未虚假声称已执行——审计确认 5 个在 validate 中有预注册 §11 允许的浅层 fail-closed 形式(保守侧),2 个从不触发;LegalityResult rejected 强制带 reason、admitted 强制完整 witness;无 launch/execute 入口。AC5:canonical JSON(sort_keys+紧凑分隔符+allow_nan=False)确定性;四级 hash 经审计方 hashlib 独立重算一致;strict round-trip 与 identity binding 成立;审计方自建 5 个 tamper 变体(非 canonical、全重签 performance_claimed=true、删规则、过期 hash、删 postdominator witness)全部以正确 reason 拒绝。AC6:registry 恰为冻结 8 条且 execution/timing/performance 标志 fail closed;实现仅 import 标准库。AC7:targeted 11、related 49、全量 1832 passed/3 skipped(skip 为既有环境边界);black/mypy/pylint 10.00/diff/dol validate+lint 全过;上轮 3 minor+1 info 逐条确认已关闭;claim 边界无漂移,未越权开放 GC0-1 或性能。结论 approve,同意关闭 GC0-0;唯一后继是 GC0-1 预注册(不是实现)。详见 r001/audit_report_full.md。
