# 变更记录：IR-5E residual-CNN final protocol freeze

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 父基线：`9b1144e`
> 状态：协议与 workload 已冻结；正式 final artifact 尚未生成

## 1. 目的

IR-5D 在已消费 chain-CNN 上证明 prepared-execution remediation 有效，但这些数据不能再次
作为 final。IR-5E 增加一个独立 residual-CNN family，并把 chain-CNN calibration →
residual-CNN final 的一次性协议固化到 runner。

本提交只冻结协议和验证代码路径，不执行最终两个 workload。正式生成必须基于本提交的
clean commit，artifact manifest 绑定该 commit。

## 2. 新 residual-CNN 语义

`build_residual_cnn_candidate` 构造：

`Conv(stem) → ReLU → Conv(residual) → Add(skip, residual) → ReLU → Flatten → Linear`

它与旧 two-convolution chain-CNN 的关键差异是：

- primal graph 有真实 fanout/merge；
- Bound IR 必须生成并执行 `add_backward`；
- Task/Schedule IR 必须保持残差两路 coefficient 的依赖与合并；
- backend candidate 仍只融合合法的 ReLU→Affine region，其余残差语义由 typed reference
  tasks 执行，不能把整图偷换成另一个 solver。

新增 `TypedResidualCNNWorkloadSpec`，其 family、shape、seed 和 MAC proxy 均进入 canonical
artifact record；measured/fair batching 接口接受 chain/residual 两类 convolutional
workload。

## 3. 冻结的 residual-final-v2

Runner：`scripts/run_ir5_family_fair_artifact.py --suite residual-final-v2`

### Calibration（允许用于模型）

- `calibration-chain-gray`：batch 4，1×16×16，channels 4→8，output 10，seed 7201；
- `calibration-chain-color`：batch 4，3×16×16，channels 8→12，output 10，seed 7202。

### Final（本提交未执行）

- `final-residual-gray-v2`：batch 4，1×14×14，block channels 5，output 12，seed 7401；
- `final-residual-color-v2`：batch 4，3×18×18，block channels 7，output 12，seed 7402。

最终常量是在 CUDA smoke 使用的旧 `7301/7302` 临时 workload 被明确废弃后重新生成；
`7401/7402` 没有参与代码调试、计时或阈值选择。

## 4. 公平协议

- suite 为 CUDA-only，CPU 在创建 artifact 目录前 fail closed；
- compiler candidates：reference、PyTorch dense、PyTorch chunked、TVM fused；
- calibration-only 模型在 residual final measurement 前拟合；
- legacy baseline 固定为 `batched-original-from-forward-trace`；
- typed 与 legacy timed region 都从预计算 forward trace 开始，只计 CROWN backward；
- physical batch wall time 除以 exact query count，compile/setup 只收一次且不除；
- fixed-single 必须与 batch 第一 query final bounds 一致；
- 输出 8 contexts × 6 policies、目录级 SHA-256 manifest 和 semantic replay。

新增显式硬字段：

- `global_p90_regret_lte_1_20`；
- `compiler_latency_memory_pareto_all_workloads`；
- `any_multi_budget_global_switch`；
- 每个 workload 的非支配 latency-memory frontier。

Pareto 定义为：compiler candidates 中至少存在两个互不支配、且 latency/peak memory 点不同
的方案；仅同一点重复或单一方案支配全部候选不算 tradeoff。

## 5. 预提交验证

- residual CPU reference/dense final bounds 对齐，Task IR 含 `add_backward`；
- 临时 residual CUDA probe（非 final 常量）四后端 allclose，最大差值
  `6.1035e-05`；
- 临时 CUDA suite smoke（已废弃 `7301/7302`）完成 8 calibration、8 held-out compiler、
  2 baseline、2 batch checks、48 outcomes，并通过 semantic replay；
- 旧 `family-fair-v1-20260728` manifest replay 继续通过；
- 定向测试 `9 passed`；
- Mypy 4 source files 零问题。

上述 smoke 只证明 runner 可执行，不是 final evidence，任何 smoke 数值不得进入 claims。

## 6. 下一步与一次性约束

提交本 protocol 后：

1. 检查 worktree clean，并记录 commit SHA；
2. 用 `warm_samples=9` 第一次且一次性生成 `7401/7402` artifact；
3. 完成 integrity replay 与 semantic replay；
4. 根据 artifact 原样判定：
   - Global p90 regret 必须 `≤1.20×`；
   - 两个 workload 都必须存在 compiler latency-memory Pareto；
   - multi-budget switch 单独报告，不得隐藏；
5. 失败则停止 ASPLOS system-performance 路线，不在 final 数据上继续调参。
