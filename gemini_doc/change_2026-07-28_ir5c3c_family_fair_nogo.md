# 变更记录：IR-5C3C architecture-held-out fair batching（VALIDATED-NO-GO）

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> runner 基线：`dedb67a`
> 工件：`artifacts/ir5/family-fair-v1-20260728`
> 判定：当前 IR-5 v1 性能门禁失败；IR-6 禁止启动

## 1. 正式执行范围

- calibration：2 个 MLP；
- held-out：2 个从未进入 calibration 的 chain-CNN；
- compiler candidates：reference、PyTorch dense、PyTorch chunked、TVM fused；
- fair baselines：fixed-single、ordinary typed batching、legacy batched-original；
- 每条 CUDA measurement：1 cold + 9 warm；
- 8 contexts × 6 policies = 48 outcomes；
- resource contexts 在执行前固定为 512 MiB / 64 MiB；
- manifest 精确绑定 `dedb67a5f3cbee6e27e00085135033bc5f006127`。

完整 SHA-256 integrity replay 与 semantic replay 通过：

- 8/8 held-out compiler measurements allclose；
- 2/2 batched-original 与 typed reference 全 batch allclose；
- 2/2 fixed-single 与 batch workload 第一 query allclose；
- Global 8/8 contexts 可行，无 unexpected OOM。

## 2. 公平结果：当前实现明确失败

| Policy | feasible | Oracle regret p50 | p90 | max |
|---|---:|---:|---:|---:|
| fixed-single | 8/8 | 306.450× | 311.501× | 311.501× |
| ordinary typed batching | 8/8 | 75.546× | 75.629× | 75.629× |
| batched-original | 8/8 | 1.000× | 1.000× | 1.000× |
| local greedy | 8/8 | 68.065× | 70.263× | 70.263× |
| global | 8/8 | 68.065× | 70.263× | 70.263× |
| oracle | 8/8 | 1.000× | 1.000× | 1.000× |

逐 workload 的 steady physical-batch / per-query mean latency：

| workload | typed reference | selected chunked | legacy batched-original |
|---|---:|---:|---:|
| gray CNN | 152.942 / 38.235 ms | 137.795 / 34.449 ms | 2.024 / 0.506 ms |
| color CNN | 153.613 / 38.403 ms | 142.713 / 35.678 ms | 2.031 / 0.508 ms |

Global 在 512 MiB 和 64 MiB 下均选择 chunked，`any_multi_budget_global_switch=false`。
内存也没有补偿性 Pareto：

- gray：chunked `10,313,216` vs batched-original `10,283,008` bytes；
- color：chunked `11,997,184` vs batched-original `11,910,144` bytes。

TVM held-out compile/setup 为 `499.922` / `500.586` ms，steady per-query 仍为
`35.289` / `36.253` ms，不能改变判定。

## 3. 为什么不是 batching 口径错误

- one box 明确定义为 one query；
- physical batch wall time 除以 exact batch size=4；
- compile/setup 收取一次且不除以 batch；
- fixed-single 用同权重/同第一输入，并对 batch 第一 query 逐 tensor 验证；
- legacy original 与 typed compiler 使用相同 BFTaskModule、InputSpec、dtype/device；
- p50/p90/p99、TTV、peak 与 feasibility 对所有 policy 使用同一 evaluator。

因此不能再用逐节点或 batch=1 baseline 隐藏成熟 batched-original。

## 4. host overhead 归因

一次额外的 `cProfile` 诊断（不作为 artifact wall-time 数字）显示：

- typed Schedule：约 1,984,753 Python calls；legacy：约 2,964 calls；
- 8 次 `build_backend_dispatch_key` 累计 profile time `0.301 s`；
- `PlanTemplate/PlanInstance/BoundModule/TaskIR` 的 validate、stable-hash、canonical JSON
  在单 query 内重复几十至上百次；
- typed path profile total `0.547 s`，legacy `0.004 s`。

`cProfile` 会放大 Python 开销，因此上述秒数只用于函数归因；正式性能数字以 artifact 的
CUDA synchronized samples 为准。代码与 profile 一致指向：静态 IR legality/hash 工作错误地
位于 query hot path，而不是只在 prepared plan/cache population 时执行。

## 5. 门禁判定

IR-5 原门禁中：

- architecture-family held-out：通过；
- fair fixed/local/global/oracle/ordinary/batched-original：通过；
- correctness/feasibility/TTV/tail/peak 报告：通过；
- p90 regret ≤1.20×：**失败（70.263×）**；
- 多预算选择不同计划：**失败**；
- 跨层规划优于单纯 batching：**失败**；
- latency–memory Pareto：**失败**。

故当前 IR-5 v1 为 **VALIDATED-NO-GO**，C2 不能升级为 paper-level performance claim，
ASPLOS-ready 仍为 NO。IR-6 cached specialization 的前置条件不成立，禁止启动。

## 6. 唯一合理的补救切片

下一步若继续工程，只允许 **IR-5D prepared execution capsule**：

1. Plan/Task cache population 时一次性完成静态 validate/hash 与 backend dispatch key；
2. query hot path 仅做 query payload、state version/capability 与 dynamic budget validity；
3. 不删除 verifier，不弱化 stale/collision rejection；
4. 先用本次失败 workload 做 profile/calibration，不回写本 final artifact；
5. 最终必须消费新的 frozen CNN/residual split；
6. fair Global p90 regret 仍高于 1.20×或无 memory Pareto，则停止该 ASPLOS 路线。

不得以增加 JIT、扩大 TVM candidate 或删除 fair batched-original 绕过本 No-Go。
