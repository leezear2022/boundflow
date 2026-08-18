---
status: internally-validated-five-fresh-correctness
updated: 2026-08-16T23:10:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_FSG4_B4A_TERMINAL_LOWER_ADJOINT_HANDOFF
stage: s01
---

# FSG4/B4-A terminal lower/lA typed handoff 计划

## 0. 判定与边界

B4-0 已由 Round 1 外审批准并由 executor 关闭为
`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`。本计划只开放 B4-A：复用 optimizer 第 10 次、
不发生 update 的 CROWN evaluation 所产生的 terminal lower 与六层 lower adjoint（lA），让 terminal
export 只做 typed assembly，不再执行第 11 次 parent CROWN backward。

本文是实现前预注册，不表示 B4-A 已实现、正确或加速。B4-B differentiable CUDA/TIR、B4-C/D、
B5—B7 均关闭；artifact 在外审批准前持续写 `performance_claimed=false`。

## 1. 冻结事实与直接假设

- preregistration source：`c894f4d`；
- direct baseline：externally approved B3；cumulative control：B0；
- B3 optimizer 固定 `10 evaluation / 9 update`，第 10 次 evaluation 后无 Adam/clamp/scheduler update；
- B3 optimizer 只保存第 10 次 `terminal_lower`，未保存该次 backward 中已存在的六层 lA；
- `export_rvir_v4_native_backward` 随后复用 forward trace，但仍完整调用
  `run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace` 一次；
- B4-0 formal raw 证明 `terminal_export.crown.00` 是完整独立 CROWN call，其 36 个 terminal kernel
  names 与 `optimizer.crown.09` 完全相同；该结构证据不等于数值复用正确性；
- B4-0 外审确认 production CUDA kernel 行没有 input shape；shape 必须从 correlation-parent CPU
  operator 恢复并绑定 lineage，不得从 kernel name 或时间邻近关系猜测。

唯一单变量假设：第 10 次 evaluation 在同一次 backward 中额外导出 detached typed lA，不改变前九次
autograd/update 或 terminal state，并可无重算地组装与 B3 terminal export 数值等价的 export。

## 2. Typed handoff 所有权

新增 B4-A 专属 handoff，不修改 B3 artifact/schema 的默认含义。handoff 至少绑定：

1. `source_state_hash`、`mutation_policy_hash`、`schedule_hash`；
2. terminal `scope_hash`、`terminal_state_hash`、`primal_graph_hash`、`split_state_hash` 与
   `forward_trace_hash`；
3. terminal lower 的 shape/dtype/device/layout/content digest；
4. 六层 lA 的 native preactivation identity、provider activation/preactivation identity、
   tensor shape/dtype/device/layout/content digest；
5. 每层 producer lineage：producer op ordinal、op type、output identity、preactivation shape、
   coefficient shape、dtype/device/layout；
6. lineage canonical hash 与整个 handoff canonical hash；
7. `terminal_lower_adjoint_handoff_count=1`、provider/fallback callback 全 0。

producer ordinal/name 从 normalized BoundFlow task graph 的 exact producer op 恢复；shape 从同一 producer
的 correlation-parent operator tensor/forward trace 恢复。若 producer 不唯一、topology 不全、shape/dtype/
layout 不一致、hash 错配、tensor 非 finite 或 handoff 被重复消费，必须 fail closed。

## 3. 执行合同

### 3.1 B3 control

B3 control 保持当前路径：10 次 optimizer CROWN + 1 次 terminal export CROWN + 3 次 KFSB child CROWN。
既有 B3 API、artifact replay 与默认 counter 语义不改。

### 3.2 B4-A candidate

- evaluation 0—8：继续使用当前 differentiable CROWN，随后执行 Adam、clamp 与 scheduler；
- evaluation 9：调用一次能同时返回 lower 与六层 lA 的同语义 CROWN，不执行 backward/update；
- 将 lower、lA、terminal state、forward trace 与 lineage 封装为 immutable typed handoff；
- terminal export 从 handoff 组装现有 `NativeBackwardExportV4`，不得进入 CROWN runner；
- KFSB、device live return、atomic commit、post packet 与 queue 完全沿用 B3。

禁止把 B4-B kernel、CUDA Graph、compile/JIT、batching、stream、allocator 或算法参数改变混入候选。

## 4. 预注册 correctness 门禁

单元与单次 integration 必须同时满足：

- evaluation/update=`10/9`，forward trace build=`4`，KFSB child batch=`3`；
- `terminal_export_crown_rerun_count=0`；
- `terminal_lower_adjoint_handoff_count=1`，且只消费一次；
- provider core/compute/update callback 与 fallback dispatch 全 0；
- B3 与 B4-A terminal lower、六层 lA、六组 intermediate bounds 均
  `allclose(atol=rtol=2e-4)`，sign exact；
- 9 次 update 后逐层 α/β、terminal split/topology、top-3 KFSB candidate/final decision、official
  post/queue/termination exact；
- B4-A lower 必须与 handoff lower 内容 hash 相同，不允许 assembly 后偷偷重算；
- handoff 的 topology/state/lineage/shape/dtype/layout/content/count 任一 outer-resigned tamper 均拒绝。

随后按冻结交替顺序运行 5 fresh B3/B4-A pair。10 个 worker 均为独立进程，raw-first，不恢复半成品；
5/5 direct pair、环境、代码与输入 digest、上述 semantic/counter/lineage 门禁全部通过后，才允许计时。

## 5. 性能门禁

正确性通过后，使用与 B3 相同的 fixed ResNet2B prop0、same-solver host、环境、warmup、同步、顺序与
统计协议进行 B3/B4-A 正式比较：

- B3/B4-A core geomean `>=1.03x`；
- 每个 query pair 的 `B3_latency / B4A_latency >=0.98x`；
- 同时报告 query geomean、terminal export wall、CUDA kernel/launch 差值与 peak allocated/reserved；
- 预期只消除一个完整 terminal export CROWN call，不把该局部收益外推成 B4 或 B0 parity；
- 未过 `1.03x` 时允许保留 correctness/机制证据，但不得累计为 B4 performance candidate。

## 6. Artifact、replay 与 tamper

正式 artifact 至少绑定 source/code blobs、B3 external closure、B4-0 manifest、five-fresh protocol、模型/
property/外部仓库/GPU/torch/CUDA identity、10 个 raw worker、typed handoff payload、lineage、counter、
semantic tensor digest、paired latency 与 root-derived summary。

replay 必须从 raw 重建 handoff/lineage/hash/counter/semantic/performance classification。outer-resigned
tamper 至少覆盖：terminal state、schedule/evaluation ordinal、native↔provider topology、producer ordinal/
name、shape/dtype/layout、lA tensor/content hash、handoff count、rerun count、callback/fallback count、raw
latency、worker swap/delete 与 summary classification。

## 7. 固定测试文件清单

为关闭 B4-0 外审 minor，B4-A exchange 的“B3/B4-A related tests”必须逐文件列出，不得只写数量：

- `tests/test_fsg4_b3_terminal_optimizer_schedule.py`；
- `tests/test_rvir_v4_native_backward_export.py`；
- `tests/test_fsg4_b3_device_live_return.py`；
- `tests/test_fsg4_b3_device_atomic_commit.py`；
- `tests/test_rvir_v4_native_kfsb.py`；
- `tests/test_fsg4_b4a_terminal_lower_adjoint_handoff.py`；
- 后续新增的 B4-A five-fresh/artifact/replay 测试文件（必须在 exchange request 中展开具体文件名）。

此外运行 full pytest、Black、Mypy、Pylint、`git diff --check`、B3 frozen replay、
`dol exchange validate` 与 `dol lint --soft`。

## 8. 实现顺序与停止条件

```text
typed lineage + immutable handoff schema
  -> evaluation 9 lower/lA producer
  -> no-rerun export assembly
  -> unit/integration/tamper counters
  -> one fresh GPU smoke
  -> clean-source five-fresh B3/B4-A
  -> correctness closure
  -> formal B3/B4-A timing
  -> replay/tamper/full regression
  -> external audit
```

任一 semantic/state/lineage/counter 门禁失败即停止，不测性能。five-fresh correctness通过但性能门禁失败，
B4-A以机制/reduced evidence关闭，B4-B是否启动必须另行依据B4路线门禁决定；不得调低阈值或复用同一
held-out反复调参。

## 9. 2026-08-16实现状态

typed producer/handoff/one-shot lease/no-rerun assembly与same-solver opt-in已实现；完整tensor content与
lineage/export digest移到query结束后的excluded audit，fresh worker另存base64 float32 raw以跨进程比较
lower/lA/intermediate。单元/相关门禁与独立GPU smoke通过，状态=
`IMPLEMENTED-B4-A-PENDING-CLEAN-SOURCE-FIVE-FRESH`。单pair约`1.02894x` core只作诊断，无性能claim。
下一唯一动作是clean-source five-fresh artifact；正式计时、B4-B/TIR仍关闭。

## 10. Five-fresh关闭更新

source=`43d4117`完成10/10 fresh worker、5/5 direct pair与每pair 19个raw tensor比较；全局最大差=
`6.109476089477539e-06`，sign/discrete/lineage/counter、无本机路径及root replay全过。状态=
`INTERNALLY-VALIDATED-B4-A-FIVE-FRESH-CORRECTNESS`，`performance_claimed=false`。下一唯一动作是独立
B3/B4-A正式计时，不得使用本轮latency形成claim；B4-B/TIR仍关闭。

## 10. 2026-08-18 正式计时关闭更新

source=`46a8493`的v5按独立24-process协议完成：correctness/environment/activation/profile、root replay
与14/14 tamper全部通过；core wall geomean=`1.0189949992x`未过预注册`1.03x`，query worst pair=
`0.9969470224x`通过`0.98x`。因此B4-A以
`INTERNALLY-VALIDATED-NO-GO-B4-A-PERFORMANCE-PENDING-EXTERNAL-AUDIT`关闭：机制与correctness保留，
但不得累计为B4 performance candidate。下一步只做外审；B4-B是否启动仍须外审后依据B4总路线另行
决定，不得降低阈值或复用held-out重跑。
