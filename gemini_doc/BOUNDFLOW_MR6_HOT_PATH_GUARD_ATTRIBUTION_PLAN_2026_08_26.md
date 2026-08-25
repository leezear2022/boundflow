---
status: preregistered
updated: 2026-08-26T22:20:00+08:00
type: plan
topic: boundflow
slug: mr6-hot-path-guard-attribution
stage: s01
---

# MR6 Hot-Path Guard Attribution 预注册计划

## 1. 问题与边界

MR5 formal显示candidate host geomean=`0.83440665x`，约慢`19.84%`，同时compile排除、三cache全hit、
memory gate通过。代码账本发现outer内至少360次device→host同步value guard。MR6只回答：这些同步
guard是否足以解释当前损失，以及去掉它们后的物理上限是否值得实现安全版本。

MR6不是优化claim。diagnostic mode故意不具备production fail-closed value checking，永不默认启用、
不得进入claims map的speedup栏；它只用于决定MR6-B是否开工。

## 2. 冻结变量

- workload/model/property、solver config、seed、10/9 optimizer、C2→C1→C0顺序全部继承MR5；
- compile和dummy warm仍在outer外；module、schedule、thread extent和TIR数学不变；
- 三方=`provider`、`bridge-full-guards`、`bridge-structural-only`；
- structural-only保留shape/dtype/device/contiguous/requires-grad、state/order/cache/module/stream/DLPack
  pointer与launch receipts；只屏蔽输入finite/range和handoff content共10类同步guard，保留两项
  output-finite guard；
- outer结束后仍保存完整semantic state并与full-guard独立fresh process配对比较；
- 不改provider、optimizer、termination、alpha/beta、TIR schedule或阈值。

## 3. 实验设计

先实现显式`GuardPolicy`，默认且production唯一合法值仍为`FULL`；`STRUCTURAL_ONLY_DIAGNOSTIC`
必须由独立diagnostic worker构造，receipt强制`performance_claimed=false`和
`production_admitted=false`。

正式诊断为3 triplet/9 fresh，Latin顺序=`PFD/FDP/DPF`：

- `P`：原生provider；
- `F`：当前full guard bridge；
- `D`：同一bridge、同一module，只屏蔽输入/区间/α/handoff共10类同步value guard，输出finite仍检查；
- headline仍是完整outer host wall；CUDA event只校验方向；
- 报告full/diagnostic host ratio、bootstrap lower、worst、语义max diff、30/27、cache和guard counters。

## 4. 冻结路由门禁

MR6-B安全实现只在以下条件全部满足时开放：

1. 3/3 triplet内provider/full/diagnostic allclose/sign exact，optimizer/final state到既有`2e-4`容差；
2. full动态同步guard账本=`360`，diagnostic=`60`，其他launch/cache/module receipts exact；
3. `full_time / diagnostic_time` geomean `>=1.10x`；
4. `provider_time / diagnostic_time` geomean `>=0.98x`；
5. 三triplet中各三方语义exact，三个pair的host/event方向一致，diagnostic/provider worst
   `>=0.95x`。

若(3)失败：guard不是dominant，关闭MR6-B，转kernel/launch/materialization profiler；
若(3)通过但(4)失败：guard重要但不充分，先做剩余launch/materialization attribution，不直接传播；
只有(3)(4)都过，才允许MR6-B实现安全的device-resident status aggregation。

## 5. MR6-B安全设计（未开放）

若门禁开放，安全版本必须同时满足：

- static shape/dtype/device/layout/module identity移到compile/admission并由typed receipt绑定；
- finite、lower≤upper、α range和output finite在TIR内归约到device status buffer；
- 三site/10 evaluation共享一个Plan-owned status ledger，outer commit前只同步检查一次；
- status非零必须在任何provider-owned state mutation前fail closed；
- content equality改为typed handoff lineage + version/pointer identity；若物理pointer不能复用则仍需显式
  lineage hash，不允许悄悄删除语义合同；
- formal correctness、fault injection、fully re-signed tamper和6-pair timing重新从零执行。

## 6. 明确排除

- 不复活B4-C2 dense-retention；
- 不跨层保存dense A或autograd history；
- 不在本阶段改TIR schedule、做cross-site kernel fusion、CUDA Graph或allocator；
- 不宣称auto_LiRPA、complete-query、queue、memory或ASPLOS-ready收益。
