# BoundFlow activation-BaB β-aware terminal TIR 实现记录

status: implemented-and-locally-validated
date: 2026-09-01
branch: feat/rvir-v4-production-state-ownership-v1
external-audit: not-requested
performance-claimed: false

## 1. 本轮目标

把已经在 PyTorch full-owner oracle 中闭合的 terminal 事务真正 lower 到 TVM/TIR：

```text
incoming A
  → ReLU coefficient selection（compressed α）
  → sparse β scatter/subtract
  → Linear coefficient + bias
  → output A / bias
```

backward 只发布 terminal incoming、compressed α 和 sparse β 的 VJP；intermediate lower/upper 继续是
frozen state，不进入 optimizer ownership。

## 2. 新 TVM/TIR ABI

新增 `boundflow/backends/tvm/bab_terminal_linear.py`：

- shape-specialized template，但不包含模型名、node id 或捕获路径；
- 独立表达 `spec/domain/feature/alpha/beta` 轴；
- β 公式固定为
  `post_beta_A = relu_A - scatter(beta_value * beta_sign, beta_location)`；
- forward 直接在 Linear reduction 中消费 β，不发布 `relu_output_A` 或 `post_beta_A`；
- backward 发布 incoming VJP、完整两侧 compressed α VJP 和 sparse β VJP；
- bound gradient 不在 ABI 中。

当前 production capture 每个 domain 只有一个 active sparse-β slot，因此 v1 fail-closed 固定
`beta_count=1`。这不是 node 特判；多 slot 是后续 ABI 版本而不是在 v1 中静默 fallback。

## 3. 第一次 lower 失败及修正

第一版试图把 Linear adjoint reduction 直接嵌入 incoming/α/β 三个输出 reduction，TVM 在 TE
construction 阶段拒绝：reduction 只能位于 compute 顶层。

最终设计显式生成一个内部 `terminal_linear_adjoint[1,6,100]`：

- 大小 `600 × 4 = 2,400 B`；
- 只在 backward module 内部存活；
- 不是 forward coefficient，也不跨 custom-autograd 边界保存；
- incoming/α/β 三个 VJP 共用它，避免把 1024-feature reduction 重算三次；
- workspace inventory 如实进入 compiled receipt。

因此本轮没有为了写“0 workspace”而隐藏必要的内部 reduction state。

## 4. current-stream runtime

新增 `boundflow/runtime/bab_terminal_tir.py`：

- DLPack pointer identity 强制 exact；
- `torch.cuda.current_stream` 与 `tvm_ffi` raw stream 双向核对；
- persistent output/VJP arena；
- shape/device/dtype/contiguous/value legality fail closed；
- 非法 β location/sign、负 β、nonfinite 在 launch 前拒绝；
- 无 eager fallback 路径。

full owner 现在允许注入 terminal TIR executor。outer custom backward 重算时，terminal forward 和
backward 均走 TIR，仍只存在一个 solver-visible differentiable owner。

## 5. 复用既有编译工作

本轮没有重写 residual/projection TIR。现场核对后确认两者原本就按 `spec/domain` 构造，能直接处理
activation-BaB 的 `spec=1, domain=6`：

- residual：forward 最大误差 `3.0e-8`，bias `3.87e-7`，α VJP约 `2.05e-8`；
- projection：forward 最大误差 `4.47e-8`，bias `1.19e-6`，α VJP约 `2.48e-8`；
- 两者均已作为 executor 注入同一个 full owner。

所以当前 full owner 的四段状态是：

| 段 | backend |
|---|---|
| terminal ReLU+β+Linear | 新 β-aware TVM/TIR |
| residual | 复用既有通用 TVM/TIR |
| projection residual | 复用既有通用 TVM/TIR |
| input Conv+L∞ concretization | 仍为 PyTorch correctness oracle |

这直接利用了此前工作，没有重新造一套 IR 或 per-site executor。

## 6. 真实 10/9 correctness

对 production capture 的 10 evaluation / 9 backward：

| 项目 | 结果 |
|---|---:|
| terminal forward/α/β VJP 最大绝对误差 | `1.210719347000122e-6` |
| lower/gradient sign | 全部 exact |
| forward/backward launch | `10 / 9` |
| DLPack exact pointers | `264 / 264` |
| fallback | `0` |
| terminal internal workspace | `terminal_linear_adjoint[1,6,100]` |

compiler identity：

- template hash：`7d284b5723800774408ad70599afbda414924543d947f391ac7f94b9a0e758de`；
- unscheduled TIR hash：`4bc9633c67efe002c7372c55795d8589ccabdd1ecdf31efd3e7b81256993b65c`；
- scheduled TIR hash：`7f9f39a7cdb915a8a80c85056d69044e6a24e609c30ae33b3f696315b33689aa`；
- device source hash：`79c309a9e37dc4bc347ad48db6bc4f20c993ff1c5bd2ed2acae33a15e7be8d8b`。

## 7. isolated 诊断计时

在同一捕获 evaluation 上做 200 次 wrapper-inclusive forward+backward CUDA event 诊断：

- candidate terminal TIR median：`0.331792 ms`；
- PyTorch terminal oracle median：`0.461824 ms`；
- native/candidate：`1.391908x`。

这只是单 terminal、单形状、同进程 micro-diagnostic，`performance_claimed=false`。它不能替代完整
region、same-solver 或 query 性能，也没有据此修改最终 claim。

## 8. 当前边界与下一步

当前还不能计 full-region 性能，因为 input Conv+L∞ 仍在 PyTorch，且既有 residual/projection TIR
仍包含多个内部 dense scratch。下一刀是：

1. 为 `spec=1, domain=6` 生成新的 input streaming TIR；
2. 局部生成 input coefficient 后立即完成 `A*center-|A|*radius`，不保存 dense input A；
3. 把第四段注入当前 owner，重跑 10/9 correctness；
4. 再对 residual/projection 的内部 scratch 做 lifetime/streaming 收缩；
5. 四段均 compiled 后才开放 full-region timing。

本轮未请求或执行外审。

## 9. 验证

- activation-BaB owner/TIR 专项：`10 passed`；
- BaB/root 相关专项：`90 passed`；
- 全量：`2198 passed, 3 skipped`；
- mypy：5 个相关文件 clean；
- pylint：`10.00/10`；
- Black：clean；
- `git diff --check`：PASS；
- DocOps lint：PASS。

3 个 skip 均为既有 TVM 重复编译或冻结 VNN-COMP checkout 边界。
