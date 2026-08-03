# 变更记录：IR-5C3B fair batching evaluator/runner contract

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`aa63a14`（IR-5C3A independent CNN family）
> 状态：公平 baseline/runner 完成；正式 9-sample artifact 仍 pending

## 改动

- 新增不破坏 IR-5C2 v1 artifact 的公平 evaluator v2：
  - `fixed_single`：batch=1 typed reference，按 query count 重复计入 TTV；
  - `ordinary_batching`：batch=K typed reference；
  - `batched_original`：相同 BFTaskModule/InputSpec 的 legacy plain-CROWN batch=K；
  - `local_greedy/global`：只能从显式 compiler candidate pool 选择；
  - `oracle`：在 baseline + compiler 的全部可行方案上实测取最优。
- 新增 physical-batching 测量：
  - batch wall time 明确除以 exact physical batch size 得到 per-query latency；
  - compile/setup 只收取一次，绝不除以 batch；
  - baseline 与 compiler 共享 memory budget、cache context、TTV/tail/peak/regret 口径；
  - legacy original 全 batch lower/upper 必须与 typed reference allclose。
- 新增 batch 语义门禁：
  - batch=1 fixed baseline 必须与 batch=K workload 的第一个 query final bounds 对齐；
  - 不能仅凭相同 seed 假定语义相同。
- 新增 architecture-held-out runner：
  - MLP calibration→chain-CNN held-out；
  - 2 calibration + 2 held-out、4 backend、3 baseline；
  - 固定 512 MiB / 64 MiB contexts；
  - 输出 raw measurements、batch checks、48 policy outcomes、summary 与 SHA-256 manifest。

## 验证

- fair evaluator synthetic contract 覆盖 cold/repeated/warm/low-memory、baseline infeasible 与
  compile amortization；
- CPU physical-batching contract 覆盖 typed/legacy allclose、per-query normalization、
  fixed/ordinary/original/compiler observation identity；
- 定向 2 passed；
- Mypy 0 issues；
- Pylint 10.00/10。

## pilot 边界

不入仓库的 warm=1 CUDA pilot 仅用于 runner smoke，所有 correctness/batch gates 通过，但显示
`batched_original` 明显快于 typed Schedule。该 pilot 不能写成正式数值，也不用于修改已冻结
workload 或 512/64 MiB budget。

下一步必须在本提交成为 HEAD 后执行 9-sample fresh artifact。若 fair batched-original 的优势
复现，则 IR-5 触发 No-Go，应归因 typed host/Schedule overhead 并停止 IR-6，而不是隐藏公平
baseline。

