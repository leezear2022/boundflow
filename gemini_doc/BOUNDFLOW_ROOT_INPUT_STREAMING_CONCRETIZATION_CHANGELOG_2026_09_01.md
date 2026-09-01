# BoundFlow 根输入卷积流式 concretization 实现与性能记录

status: implemented-and-locally-validated
date: 2026-09-01
branch: feat/rvir-v4-production-state-ownership-v1
external-audit: not-requested
performance-claimed: false

## 1. 本轮解决的问题

上一阶段已经把 terminal Linear、末端 residual 和 projection residual 接到一个累计 TIR owner，
但在最早的 `/input-4 → /input → /input-1` 事务仍回到 PyTorch：先用 `conv_transpose2d`
生成完整输入系数 `A[3,1,3,32,32]`，再单独执行 L∞ concretization
`sum(A*center-|A|*radius)`。这既物化 dense A，也把本可消费的局部卷积值写回显存。

本轮把该事务接入同一个 production owner，并实现类似 FlashAttention 的“生成即消费”：

```text
compressed alpha + incoming A + ReLU interval
  -> 局部生成一个 input coefficient
  -> 立即累加 A*center-|A|*radius
  -> 不保存完整 input A
  -> backward 按需重算局部 coefficient
  -> 直接返回 incoming VJP + compressed-alpha VJP
```

BaB、branch、termination、optimizer step 和 host solver ownership 均未改变。

## 2. 真实生产事务捕获

捕获点来自 ResNet2B property 0 的真实 `stage_solve.update_bounds_core`：

- start node：`/49`；
- 拓扑：`/input-4` ReLU → `/input` stride-2 Conv → `/input-1` L∞ root；
- 1 次 optimizer transaction；
- 5 次 forward evaluation、4 次 backward mutation；
- incoming A：`[3,1,8,16,16]`；
- 原生外部 dense A：`[3,1,3,32,32]`；
- sparse alpha：`[2,3,1,164]`；
- concrete lower：`[1,3]`。

独立 PyTorch 闭合公式同时复算 forward 和完整 VJP，最大绝对误差
`3.5762786865234375e-7`，全部梯度符号一致。

## 3. TIR 设计及失败版本

第一版从 TE 直接 lower，虽然正确，但仍保留完整 coefficient 和 coefficient-gradient scratch，
形成 11 个 kernel；100 次中位数仅为 native 的约 `1.073x`。该版本没有进入 production。

最终版本使用显式 TensorIR：

- forward：每个 spec 一个 CUDA block，局部 coefficient 生成后立即进入 L∞ reduction；
- backward：按每个输出元素重算所需 coefficient，只产生 incoming 与 compressed-alpha 梯度；
- forward/backward 各一次 launch；
- workspace 只有标量 local buffer 和 `[2,128]` shared partial；
- `dense_input_coefficient_externalized=false`；
- 内部也不存在 `[3,1,3,32,32]` coefficient scratch；
- DLPack pointer identity 全部 exact，fallback 为 0。

最终源码的 100 次局部复测：candidate `0.516096 ms`，native PyTorch oracle
`0.682496 ms`，为 `1.32242x`。此前两次同机制测量为 `1.24901x` 和 `1.24194x`，
因此合理口径是局部约 `1.24–1.32x`，而不是挑最高一次。

最终 compiler identities：

- template hash：`c9fb7dea61cb3dfc2b17dafbcbb30f233ae439a71405f7d4279d5ea03056bedf`；
- unscheduled TIR hash：`025f6efb6dd76580b39e7baa7904f315129cfe053b063c4d89e50d9d09858d84`；
- scheduled TIR hash：`7b1f8ab86b069bc58c0775c444e2effddac19a2bfc163be9163e02a8418b1434`；
- device source hash：`3a5b6ff5a5064972a2178ccce2caf84adb836158cdb884c1662d1ffdd4e02909`。

## 4. 累计 production owner

新增累计 owner 将 terminal、residual、projection 和 input-domain 四个 prepared module 放在
一个 custom-autograd 边界后。projection 只暂存 A 与 bias，不提前发布 autograd owner；input Conv
消费该边界并返回已 concretize 的 lower marker。host 的 `concretize_root` 只消费结果，不再 reshape
或物化 dense A。

真实 solver receipt：

- input ReLU/Conv/concretize replacement：`5/5/5`；
- input TIR launch：forward `5`、backward `4`；
- cumulative owner definition：`1`；
- dense input A external materialization：`0`；
- fallback/eager：`0/0`。

## 5. same-solver 三对 fresh 结果

control 为原始 αβ-CROWN/PyTorch executor；candidate 为同一 solver、同一模型/property、同一
branch/termination/state trajectory 下的四段累计 BoundFlow owner。三对顺序为
control→candidate、candidate→control、control→candidate。

| scope | 三对几何平均 | 最差 pair |
|---|---:|---:|
| complete query | `1.120226492x` | `1.043639862x` |
| root incomplete | `1.214470721x` | `1.102312953x` |
| optimizer transaction | `1.303123033x` | `1.153551392x` |
| autograd backward | `1.329084410x` | `1.281001465x` |

离散语义全部一致，lower 最大绝对误差 `1.1920928955078125e-6`。第三对 control 明显比前两对
慢，因此 `1.1202x` geomean 可能被抬高；必须同时披露最差 `1.0436x` 和中间 pair `1.0830x`。

与接入 input-domain 前的三对结果相比，query geomean 从 `1.066130076x` 提高到
`1.120226492x`，即当前协议下累计倍率再提高约 `1.05074x`。这仍未达到 10x 总目标，也不能升级为
ASPLOS 最终性能 claim。

## 6. 验证

- independent production oracle：最大误差 `3.5763e-7`，gradient sign exact；
- final TIR probe：最大误差 `7.1526e-7`，sign exact，forward/backward 115/115；
- targeted root TIR：`71 passed`；
- full suite：`2184 passed, 3 skipped`；
- mypy：12 个相关文件 clean；
- pylint：`10.00/10`；
- `git diff --check`：PASS。

3 个 skip 均为既有环境/冻结 checkout 边界。本轮未请求、未执行外部审计。

## 7. 限制与下一步

当前 input TIR 是 ResNet2B P-anchor 的静态 specialization：geometry 与 164 个 alpha coordinate
被 fail-closed 冻结。它证明了 verification-specific generate-and-consume/recompute 机制可以传播到
complete query，但不是通用 Conv lowering。

下一刀应继续按真实 trace 扩大同一 owner，而不是增加外审流程：捕获并评估下一段
`/input-8 → /input-4` 的 Conv/residual 事务，先算 dense A materialization、kernel share 和所需
region speedup；只有 Amdahl 可达且独立 oracle 闭合时才扩 TIR。与此同时应增加 5-pair 或更多
系统噪声诊断，解释第三个 control 的重复性慢点。
