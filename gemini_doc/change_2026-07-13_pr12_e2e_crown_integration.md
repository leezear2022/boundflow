# 2026-07-13：PR-12 fused CROWN 端到端 runtime integration

## 目标与阶段结论

将已验证的 fused ReLU+Linear/Conv2d TIR 从孤立 kernel 接入真实 plain-CROWN backward。
本切片完成 **PR-12D correctness/mechanism foundation**，但没有运行 formal Pareto 或冻结的
final held-out，因此 PR-12 整体仍为 **IN PROGRESS**，PR-13 继续阻塞。

## 接口与语义边界

新增 `boundflow/runtime/fused_crown.py`，明确区分：

- boundary representation：当前 fused region 为 `DENSE`；
- internal materialization：`ELIDE_RELU_SCALED_A`；
- backend：TVM fused TIR 或 Torch dense reference。

`FusedCrownExecutionStep` 由窄 region planner 对 forward `Affine → ReLU` 生成，记录被消费的
op index/output。backward runtime 只执行显式 schedule，不在热路径重新猜 pattern。

`FusedCrownExecutor` 隔离 solver 与 TVM：request/result 只包含四项 CROWN 系数/偏置语义、
relaxation、affine 参数、shape 和 attrs。TVM executor 使用 DLPack view 输入输出，热路径不经过
CPU/NumPy；Torch reference 可在同一个 host solver 中替换 executor 做 oracle。

## Legality 与 fallback

TVM v1 仅接受：

- plain CROWN、无梯度；
- 无 α、β、split state；
- static contiguous FP32 CUDA；
- 已冻结的 Linear/Conv2d 属性子集。

不满足 capability 时返回原有 ReLU backward + affine backward，不执行 fused candidate。
execution step 与当前 task graph 不一致则报确定性错误，避免 stale plan 静默误执行。

## Correctness 与 interop 证据

新增 `tests/test_phase7a_pr12_e2e_crown_runtime.py`：

- Linear chain 最终 lower/upper bound 对齐；
- stride-1/stride-2 chain CNN 最终 bound 对齐；
- residual block 与 stride-2 downsample mini-ResNet-like block 最终 bound 对齐；
- unsupported executor 确定性 fallback；
- Torch dense region 与 TVM fused region 使用同一 solver；
- DLPack round trip `data_ptr` 相同，证明 Torch/TVM storage alias。

数值门禁为 0 failure、0 NaN/Inf、`lower <= upper`，CUDA network-level tolerance 不高于
`8e-5`。当前验证结果：

```text
PR-12 integration/CNN/DAG/ReLU 专项：71 passed
全量：284 passed、1 skipped
新 runtime integration 文件：8 passed
fused_crown.py mypy：success
fused_crown.py pylint：10.00/10
Black check / git diff --check：通过
```

## Kernel foundation 冻结

在集成前的 `6e08917` 建立本地 annotated tag `pr12-kernel-foundation`，未推送。轻量 manifest：

`artifacts/phase7a-pr12/kernel-foundation-20260713/`

它记录 48/276 基线、codegen/spill 摘要、latency calibration 和
`final_heldout_consumed=false`，用于区分 kernel、interop/runtime 与 Planner 回退。

## 尚未完成

- runtime-contract 的 allocation/dispatch/compile/cold/warm 分层；
- dense、structured eager、fused 的公平 latency-memory Pareto；
- candidate cost fields 与 Planner auto-selection；
- frozen `pr12-final-heldout-v1` 与 compile amortization；
- 至少一项非 toy 系统价值 gate。

因此下一提交应是公平 runtime/network benchmark，而不是扩 kernel、α/β 或启动 PR-13。
