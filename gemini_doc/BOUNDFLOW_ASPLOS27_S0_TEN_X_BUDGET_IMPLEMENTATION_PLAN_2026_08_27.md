# BoundFlow ASPLOS’27 S0 全栈事务归因与10×预算执行计划

date: 2026-08-27
status: first-batch-executed-next-marker-slice-open
parent-plan: `BOUNDFLOW_README_PIPELINE_END_TO_END_ACCELERATION_DRAFT_PLAN_2026_08_26.md`
external-audit: deferred-by-user
performance-claimed: false

## 1. 本阶段回答什么

S0不优化kernel。它先回答三个会决定ASPLOS’27路线生死的问题：

1. `10×`的分母究竟是哪一个official B0 scope；
2. 求解器、CROWN/IBP、optimizer、branching、runtime、memory等事务是否被互斥且完整地统计；
3. 现有CIBC/B4/R3/RVIR机制即使全部接入，在Amdahl预算上是否可能达到10×。

第一批只复用冻结raw，不重跑GPU，也不升级性能claim。第二批才增加低扰动solver transaction marker并重跑
同一control/profile protocol。

## 2. 第一批实现

新增`boundflow/runtime/asplos27_tenx_budget.py`：

- `ClaimMode`严格区分fixed-trajectory systems与solved-query TTV；
- `BudgetBucket/TenXBudget`要求B0 critical-path bucket互斥并闭合到1；
- semantic coverage与时间轴closure分离，默认`coverage>=97%`且`unclassified<=3%`；
- `sum(s_i/r_i)+h`直接计算projected speedup和10×可达性；
- `DirectCumulativeObservation`保存真实direct ratio，但不同scope禁止聚合；
- FSG1 transaction inventory把bound call、setup/termination、同阶段host control与阶段转换分开；相邻阶段
  只能提供topology context，不能冒充机制已解析。

新增`run_asplos27_s0_tenx_budget_artifact.py`：

- 绑定FSG1 closure/worker raw、FSG4-B3、MR5、MR6、CIBC summary的SHA256；
- 生成protocol、budget report、transaction inventory、direct ledger、summary和manifest；
- replay重新读取冻结raw、重算所有语义派生值，不只验外层digest；
- 即使攻击者改report并重签file/manifest digest，semantic replay仍拒绝。

## 3. 第一批结果与解释

artifact为：

`artifacts/asplos27-s0-tenx-budget/fsg1-diagnostic-and-history-v1`

关键结果：

| 项 | 结果 | 含义 |
|---|---:|---|
| B0 profile | 10 | ResNet2B与MNISTFC各5 |
| transaction topology context | 10/10 closed | 每段有bound call或邻接阶段上下文 |
| mechanism admission | 5/10 | 仅MNISTFC通过；ResNet未通过 |
| ResNet unresolved mechanism | 30.62%—31.51% | 不能用“solver control”一个词吞掉 |
| existing-operator-only max projection | 2.3188568× | 假设全部operator都达CIBC local 12.795× |
| operator-infinite max ceiling | 2.6107780× | host/runtime不动时的绝对上限 |
| 10× feasible runs | 0/10 | 只证伪operator-only，不证伪coarse full-stack |
| historical direct observations | 4 | scope隔离，禁止相乘 |

因此S0当前是`NOT ADMITTED`。这和历史CIBC `2.45631×`、B4-B2 `4.89834×`并不冲突；它说明这些
local/graph机制必须和O4 optimizer transition、O5 prepared runtime、O6 batching/branching一起形成coarse
cumulative candidate，才能有机会接近10×。

## 4. 下一批唯一代码范围

实现一个low-perturbation transaction observer，目标函数只来自固定αβ-CROWN commit，并至少标记：

1. model/property/front-end setup；
2. incomplete verification；
3. incomplete→complete/BaB handoff；
4. domain pick/prepare；
5. branching/KFSB score；
6. split/history/state assembly；
7. bound pre/core/post；
8. domain commit/prune/queue；
9. termination/report。

门禁：control/profile交替fresh，median perturbation`<=1.05`；机制覆盖`>=97%`；unclassified`<=3%`；所有
marker必须是observer-only，结果、visited domains、调用序列和trajectory不变。通过后才为每个bucket写
`mechanism → candidate optimization → attainable r_i → residual contribution`，再决定是否开放S1/S2。

## 5. STOP条件

- 把outer `verify`或`general_bab`一个大span写成“机制已解析”；
- 用函数邻接推断替代真实marker；
- 把CIBC/B4/MR数字跨scope相乘；
- 在S0未通过时宣称10×可达、ASPLOS-ready或开始S1正式计时。
