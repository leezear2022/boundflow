---
status: validated-math-reduction-tir-pending
updated: 2026-08-25T04:11:00+08:00
type: changelog
topic: boundflow
slug: r3-1b2-p-alpha-vjp-math-reduction
stage: s01
---

# R3-1b2 P-α VJP 数学归约与实现门禁记录

## 1. 结论

R3-1b2 的 P-anchor compressed α 梯度可以不依赖 `torch.autograd.grad`、不读取 native
output adjoint、也不从 forward 保存任一 dense coefficient A。针对冻结的 ResNet2B
`25/Conv_8`、one-evaluation、mutation=0 workload，独立闭合公式与 native autograd 的结果为：

- compressed gradient shape=`[2,1,6,86]`；
- max abs diff=`4.470348358154297e-08`；
- sign exact；
- nonzero=`281/281`；
- `atol=rtol=2e-4` 通过。

这只关闭“数学可归约性”门禁，不表示 compiled VJP、custom backward、memory gate 或 R3-1
已经关闭；`timing_recorded=false`、`performance_claimed=false`。

## 2. 独立公式

记 R3-1b1 full-lower recurrence 得到的 input lower coefficient 为 `A_input`，每个 ReLU 的
incoming lower coefficient 为 `A_k`。先按 `A_input` 的符号选择 input box corner `x*`，再沿
production topology 正向重放有效 affine/ReLU 值：

```text
x* -> Conv0 -> ReLU17[A17] -> Conv2 -> ReLU19[A19]
   -> {Conv4, Conv5} -> add -> ReLU23[A23] -> Conv8 -> z25
```

其中 stable-active/stable-inactive ReLU 使用固定斜率，ambiguous ReLU 按 incoming coefficient
符号选择 lower α 或 upper relaxation；upper intercept 同时进入有效值。对目标
`loss=-sum(lower)`，P-alpha 的 dense 梯度恰为：

```text
d loss / d alpha25 = -A25 * z25
```

但只在 `A25>=0` 且 preactivation-25 ambiguous 的位置成立；其余为零。最后依据冻结的
`alpha_flat_indices[86]` 直接 gather 到 production compressed layout，direction 1 保持零。

该公式由新测试中的独立 PyTorch 闭合实现计算；native autograd 只作为 oracle，不被 candidate
实现调用。

## 3. 对 compiled schedule 的约束

下一实现不得把上述 PyTorch oracle 当成 candidate。TIR backward 必须在 backward 内部完成：

1. 复用 b1 的两个 coefficient scratch，重放必要的 lower coefficient checkpoint；
2. 将 `A_input/A18/A20/A24` 的选择信息压为 backward-ephemeral sign bitmap，不能保存 dense A；
3. 从 input corner 正向计算到 `z25`，residual branch 在 kernel 内合并；
4. 重放 `A26`，直接生成 86-entry compressed dα；
5. custom Function forward 的 ctx 只保存 plan/execution key、schema/ordinal 和 production leaf 引用；
6. warm path 不新增第三 coefficient arena，不调用 eager/native shadow/autograd。

这里的 sign bitmap 是 backward 内部临时调度元数据，不是跨 forward/backward 保存的 M1
certificate，也不开放 M1 claim。

## 4. 验证

执行：

```text
pytest -q tests/test_r3_p_alpha_vjp_oracle.py
mypy boundflow/runtime/r3_p_alpha_vjp_oracle.py
pylint boundflow/runtime/r3_p_alpha_vjp_oracle.py tests/test_r3_p_alpha_vjp_oracle.py
```

结果：`1 passed`、mypy clean、pylint `10.00/10`。测试显式检查 shape、allclose、sign、nonzero
与严格 max-diff 上限。

## 5. 下一动作

只实现 R3-1b2 的 checkpoint/sign TIR 与 mandatory custom backward 单 worker；在 lower+dα、真实
saved-tensor hook、scratch pointer/high-water、allocated/reserved memory 全部通过前，不生成
five-fresh、不计时、不开放 R3-1b3/R3-2A。
