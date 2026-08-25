---
status: validated-invalid-attribution
updated: 2026-08-26T00:25:00+08:00
type: closure
topic: boundflow
slug: mr7-launch-materialization-formal-invalid-closure
stage: s01
---

# MR7 Launch / Materialization 正式归因 INVALID Closure

> **后续状态**：本 closure 的MR7-R后继已于2026-08-26正式通过；当前下一步已变为GC-0/FCR-1
> verification graph ABI+correctness预注册。见MR7-R formal closure。

## 1. 结论

MR7 已按冻结协议完整执行，但状态只能是：

`INVALID_MR7_ATTRIBUTION`

原因不是 correctness、launch、device envelope 或 replay 失败，而是 3 组中第 1 组
`profile/control CUDA-event ratio=1.239399 > 1.10`。另外两组分别为`1.039553`与`1.096733`。
冻结协议要求3/3通过，因此不得删除第1组、取后两组或将诊断数字升级为正式路线准入。

## 2. 已通过门禁

- 6 fresh / 3 counterbalanced control-profile pair；
- semantics 3/3 allclose、sign exact，最大差`2.4065375e-6`；
- 30 forward / 27 backward marker，module/cache/stream/fallback lifecycle exact；
- host五类互斥账本 closure error=`0`；
- 三组 device kernel/device marker aggregate envelope error分别为
  `1.1855% / 0.8652% / 0.9773%`，均`<=2%`；
- 11/11 fully re-signed host/device/semantic/count/module/order/source tamper fail-closed；
- replay逐次导出同一`summary_hash=c889a4e5…d09d`；
- 定向`7 passed`，全量`1815 passed, 3 skipped, 6 warnings`；
- `performance_claimed=false`、`production_admitted=false`。

## 3. 只可作为路线设计输入的诊断数字

以下数字来自完整 raw 重算，但因 profiler perturbation gate 失败，不能写成正式 opportunity admission：

- unprofiled host boundary 中位：`25,891,032.5 ns`，outer share=`19.8183%`；
- host share：FFI/DLPack/stream=`8.3612%`，layout/materialization=`1.3019%`，
  post-output=`10.1553%`，admission/handoff=`2.7129%`；
- profiled device kernel share=`8.6915%`，其中forward=`8.5460%`、backward=`0.1454%`；
- C0/C1/C2 kernel总量三次几乎不漂移，且C0在3/3均最慢；
- 若仅把host boundary share作为乐观输入，到当前candidate parity所需region speedup约`1.95853x`。

这些事实反对“继续只调单个TIR kernel”作为默认下一步：device kernel只占该outer约8.7%，而host
boundary诊断接近20%。但严格来说，MR7-A/B/C均未被本轮正式开放。

## 4. 对全编译路线的影响

本轮不关闭`BOUNDFLOW_FULLY_COMPILED_VERIFIER_RUNTIME_V1_ARCHITECTURE_2026_08_25.md`。该路线还包括
optimizer、branch/queue、execution graph、parallel scheduling与memory/arena planning，不等同于当前
逐站点Conv bridge。MR7 INVALID只说明“带CUPTI的pair不能作为host share准入协议”，不是这些系统假设
已被证伪。

## 5. 历史唯一后继（MR7-R已完成）

下一步只开放`MR7-R`无profiler host recovery：用独立进程成对比较MR6原始diagnostic与MR7 ledger
control，验证ledger本身不扰动outer，并用5个unprofiled fresh决定host boundary门禁。CUPTI数字只作
non-admission附录，不再控制host路线资格。MR7-R通过后最多开放FCR-1 compiled-region/arena/FFI的
ABI与correctness预注册；不得直接形成speedup claim。

## 6. Artifact

`artifacts/measurement-recovery/mr7-launch-materialization-attribution-v1`
