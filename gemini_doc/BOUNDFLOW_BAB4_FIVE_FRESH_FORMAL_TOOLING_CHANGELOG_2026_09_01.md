# BAB4 five-fresh formal 工具链修改记录

status: tooling-ready-before-formal-generation
date: 2026-09-01
performance-claimed: false
external-audit-requested: false

## 1. 目标

把 `B4-A vs BAB4` 的 3-pair 本地开发信号升级成 source-bound、raw-first、可独立 replay
的 5-pair formal candidate。工具链先提交，再从该 commit 启动 worker，避免 artifact 的
`source_git_head` 与实际生成代码不一致。

## 2. 实现

### 2.1 复用并参数化既有协议

`scripts/run_asplos27_s4_same_solver_five_fresh.py` 保留历史默认候选 `S4`，新增内部
`CANDIDATE_CONFIGURATION`。旧 S4 artifact replay 已重新执行，manifest 与 summary hash
保持原值并 PASS。

新增 `scripts/run_bab4_same_solver_five_fresh.py` 作为薄入口，只冻结：

- schema：`boundflow.bab4-same-solver-five-fresh/v1`；
- candidate：`BAB4`；
- 五组交替顺序：B4-A/BAB4、BAB4/B4-A、B4-A/BAB4、BAB4/B4-A、B4-A/BAB4；
- 19 个 code/model-plan 证据入口。

### 2.2 BAB4 replay 的额外语义约束

每个 candidate receipt 必须满足：

- schema 为 `boundflow.bab-four-segment-exact-call/v1`；
- 去除 worker 后加字段后，receipt hash 可从 exact-call core 字段重算；
- evaluation/mutation=`10/9`；
- compiled segments=`4`；
- compiled forward/backward launches=`76/36`；
- fallback/provider/compile-inside-exact-call=`0/0/0`；
- static warmup=`10/9`、fallback 0、无 source capture runtime dependency；
- 五个 candidate 的 combined compiled-assets hash 唯一且稳定。

原有门禁继续保留：环境必须准入、discrete semantics exact、lower 误差不超过 `2e-4`、
sign exact、query parity `1.00x`、query research `1.15x`、core research `1.20x`。

### 2.3 篡改探针

新增 `scripts/probe_bab4_same_solver_five_fresh_tamper.py`，覆盖 10 类 raw-worker mutation：

1. core latency；
2. query latency；
3. lower value；
4. discrete semantics；
5. environment admission；
6. compiled assets identity；
7. forward launch count；
8. fallback count；
9. warmup source dependency；
10. performance claim flag。

每类攻击都会重写最外层 manifest digest，因此测试的是 outer-resigned tamper，而不是只改文件
却保留旧 SHA256。工具不声称能拒绝攻击者同时伪造全部 raw、派生 summary 与所有身份来源的
coherent full resign；该边界显式记录为
`coherent-full-resign_claimed=false`。

## 3. 生成前门禁

formal 生成只能从工具提交后的 clean tracked source 开始。历史用户未提交文件允许留在工作树，
但不得进入本次 commit 或 `CODE_PATHS`。生成完成后必须执行：

- BAB4 stdlib replay；
- 10/10 outer-resigned tamper probe；
- targeted pytest；
- mypy / pylint / black / diff check；
- DocOps lint。

本轮按用户要求不发外审；artifact 与完整数字完成后再统一交给下一轮外审。

## 4. 生成演练修正

第一次生成已完成 10 个环境准入 worker，但旧汇总逻辑读取 S4 receipt 的
`region_template_hash`；BAB4 receipt 的同一物理身份字段名为 `production_plan_hash`，导致
汇总 fail closed。现已按 candidate schema 显式选择字段。已有 10 份 raw 经修正逻辑完整重算
通过，证明没有第二个汇总问题；这批 raw 只用于验证修复，不作为最终 artifact。正式数据必须从
包含该修复的新 source commit 重新运行。
