# 2026-07-13：PR-12 fused ReLU+Conv2d TIR foundation

## 目标与判定

在 Linear mechanism foundation 之后，实现 plain CROWN、无梯度、static FP32 CUDA 的
ReLU+Conv2d backward fused task。当前结论是 **kernel-level correctness/mechanism PASS，PR-12
整体仍 in progress**；end-to-end CROWN、真实 CNN/mini-ResNet Pareto 和 final held-out 未完成。

## 冻结的语义与 layout

```text
A/A_l:  [domain, spec, out_channel, out_h, out_w]
W:      [out_channel, in_channel, kh, kw]
A_prev: [domain, spec, in_channel, in_h, in_w]
```

原始 Conv input shape 是 signature 必填字段。forward output shape 必须由 input/kernel/stride/
padding/dilation 精确推出；ConvTranspose `output_padding` 再由原始 input shape 推导。stride-2
奇偶输入的 `output_padding=0/1` 都有 contract/CUDA 测试，禁止只从 output shape 猜测。

## 实现

- output-centric gather：每个 thread 负责一个 `A_prev[d,s,ci,hi,wi]`，对 `co/kh/kw`
  做合法位置 reduction，无 atomics；
- sign selection、slope application 和 Conv weight contraction 内联在 gather load 中；
- bias kernel 直接合并 ReLU intercept 与 Conv bias contraction；
- 一个 candidate/PrimFunc 当前 lower 为 4 个 CUDA kernel（upper/lower coefficient 与
  upper/lower bias），不追求形式上的单 kernel；
- 支持 1×1/3×3、stride 1/2、padding 0/1、groups=1、dilation=1、bias 有/无；
- groups>1、dilation>1、channels-last、非 FP32/静态 CUDA 等均显式拒绝。

## Correctness matrix

CUDA kernel-level 覆盖：

- domain batch 1/2/8，spec 1/3/9/32；
- 2 次幂和非 2 次幂 channel；spatial 4/7/8/14/16；
- 1×1/3×3、stride 1/2、padding 0/1、bias 有/无；
- 正、负、零、混合 coefficient；upper/lower coefficient 与 bias 四项输出；
- stride-2 奇偶 input 与 output-padding 0/1。

所有点与 deterministic PyTorch `conv_transpose2d` dense reference 对齐，当前最大 sanity
绝对误差为 `3.43e-5`。

## 机制证据

工件：`artifacts/phase7a-pr12/codegen-sanity-v1-20260713/`。

- Linear、stride-1 Conv、stride-2 Conv 三个代表 task 均为 4 kernels；
- pre/post schedule intermediate allocation 为空；CUDA source 中 `A_scaled`/`im2col` 为 0；
- PTX `.local` declaration/load/store 均为 0；
- ptxas 的 stack frame、spill load/store 均为 0；
- 最大 registers/thread 分别为 40、40、48。

这证明三个代表 shape 没有显式 workspace 或 spill，不等价于所有 shape 的完整 Nsight 结论。

## 小型 latency sanity

工件：`artifacts/phase7a-pr12/fused-sanity-v1-20260713/`，仅 calibration，warmup 5、repeat 20，
未消费 `pr12-final-heldout-v1`。

| Case | compile | fused / PyTorch dense eager median | scaled-A bytes avoided |
|---|---:|---:|---:|
| Linear D2/S8/I16/J12 | 612.9 ms | 0.172× | 2,048 |
| Linear D4/S32/I64/J48 | 350.6 ms | 0.470× | 65,536 |
| Conv s1 D1/S3/Ci5/Co4/H7 | 502.0 ms | 0.716× | 4,704 |
| Conv s2 D2/S8/C8/H16 | 476.3 ms | 1.717× | 65,536 |

前三点没有数量级退化；stride-2 medium 点当前慢于 PyTorch dense eager，必须保留为 schedule
limitation。该 sanity 使用预分配 TVM outputs，对照为会构造中间 tensor 的 PyTorch dense eager；
它既不是 structured-eager 公平 benchmark，也不是论文性能结论。

## 下一门禁

1. 将 fused candidate 接入真实 plain-CROWN backward region；
2. chain CNN、residual block、mini-ResNet end-to-end 四输出/最终 bound 对齐；
3. 统计真实 launch、allocator peak、logical bytes 与 compile/cold/warm；
4. 完成后才运行冻结 final held-out/Pareto，且不得回写 PR-11 profile。

## 收尾验证

```text
PR-12 专项：48 passed
全量：276 passed、1 skipped
Mypy：6 个 PR-12 core/script files success
Pylint：Linear/Conv/candidate 与 3 个 PR-12 scripts 逐文件 10.00/10
Black check / git diff --check：通过
```
