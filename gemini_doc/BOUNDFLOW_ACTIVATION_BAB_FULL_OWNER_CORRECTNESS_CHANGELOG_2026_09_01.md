# BoundFlow activation-BaB 全段 owner 正确性实现记录

status: implemented-and-locally-validated
date: 2026-09-01
branch: feat/rvir-v4-production-state-ownership-v1
external-audit: not-requested
timing-recorded: false
performance-claimed: false

## 1. 本轮完成了什么

上一阶段只在真实 αβ-CROWN CUDA query 中捕获了 activation-BaB 的四段事务。本轮把该捕获落成
一个可执行的单 owner 正确性实现：

```text
terminal incoming coefficient
  → terminal ReLU
  → sparse β injection
  → Linear
  → residual block
  → projection residual
  → input Conv + L∞ concretization
  → final lower
```

新增 `boundflow/runtime/bab_full_region_owner.py`。它不改变 host solver，不接 live bridge，不计时；
职责是冻结完整事务的数学合同和 custom-backward ownership，为下一步 TVM/TIR lowering 提供独立
oracle。

## 2. 关键语义

### 2.1 β 注入位置

真实捕获独立确认的公式已经进入 owner：

```text
linear_incoming_A = relu_output_A - scatter(beta_value * beta_sign,
                                             beta_location)
```

β 不是 ReLU slope 的一部分，也不能在 Linear 后补偿。它在 ReLU 生成 coefficient 后、Linear 消费
前注入；因此 terminal 的 Linear coefficient 和 bias 两条路径都消费注入后的 A。

### 2.2 frozen bounds

activation-BaB 的 intermediate lower/upper 只作为本次 evaluation 的冻结输入：

- 不作为 differentiable owner；
- custom backward 不返回 bound gradient；
- 只返回 terminal incoming、6 组 compressed α 和 1 组 sparse β 的 VJP；
- static weight/bias、split location/sign、input center/radius 同样不进入 optimizer ownership。

### 2.3 不跨层保存 dense A

custom-autograd forward 只保存 8 个边界可微张量：

1. terminal incoming coefficient；
2. terminal α；
3. residual entry α；
4. residual inner α；
5. projection entry α；
6. projection inner α；
7. input α；
8. sparse β。

terminal、residual、projection 和 input dense coefficient 都只在 forward 内生成；backward 从上述边界
状态整段重算，不保存任何跨层 dense A。receipt 固定
`saved_dense_coefficient_count=0`、`frozen_bound_gradient_count=0`。

## 3. 真实捕获闭合结果

输入仍是：

`artifacts/bab-full-region-capture/resnet2b-prop0-v1/capture.pt`

在 production-shaped `spec=1, domain=6` 的 10 个 evaluation 上：

| 项目 | 本轮结果 |
|---|---:|
| 四段 forward 最大绝对误差 | `9.5367431640625e-7` |
| 6 组 α + 1 组 β VJP 最大绝对误差 | `2.086162567138672e-7` |
| lower/gradient 符号 | 全部一致 |
| Adam 9 次 mutation 后最大状态漂移 | `1.3709068298339844e-6` |
| owner lifecycle | `10 forward / 9 backward` |
| dense coefficient 跨边界保存 | `0` |
| frozen-bound gradient | `0` |
| fallback | `0` |

optimizer 重放使用真实策略：α `lr=0.01`、β `lr=0.05`、Adam
`betas=(0.9,0.999)`、`eps=1e-8`、每步 clamp、ExponentialLR `gamma=0.98`。状态漂移来自
独立 ConvTranspose reduction 顺序；逐步 VJP 符号 exact，累计误差低于现有 `2e-5` production gate
一个数量级。

## 4. 本轮没有宣称什么

- 当前 owner 的内部数学执行是 PyTorch correctness oracle，不是最终 TVM/TIR candidate；
- 没有接入真实 activation-BaB live return；
- 没有测 kernel、region、query 或 queue 性能；
- 没有把历史 root-CROWN `1.120226492x` 写成本轮收益；
- 没有请求或执行外部审计。

## 5. 为什么这一步不是重复旧工作

已有 S4/RVIR owner 对应前半段 root optimizer；旧 root full pipeline 对应 `spec=3, domain=1`、
`5/4` 生命周期、无 active β，并允许 root bounds 参与 VJP。本轮对应后半段 activation-BaB：

- `spec=1, domain=6`；
- `10/9` 生命周期；
- active sparse β；
- frozen intermediate bounds；
- 完整四段只保留一个 differentiable owner。

因此复用了旧事务的 Conv/Residual/Projection 数学，但没有错误复用其 ownership 合同。

## 6. 下一步

下一刀进入 activation-BaB TVM/TIR lowering，不再做新的 PyTorch 旁路：

1. 新建版本化 terminal TIR，把 sparse β scatter 融入 ReLU→Linear coefficient consumer；
2. 复用 residual/projection 的通用 shape lowering，但 custom backward 只发布 compressed α VJP；
3. 为 `spec=1, domain=6` 生成 input Conv + L∞ streaming TIR，继续禁止 dense input A；
4. 四段接到本轮同一个 owner ABI，重放 10/9 correctness；
5. TIR correctness 通过后才接 live bridge并计时。

后续状态：terminal β-aware TIR 已实现，既有 residual/projection TIR 已复用并注入同一 owner；详见
`BOUNDFLOW_ACTIVATION_BAB_BETA_TERMINAL_TIR_CHANGELOG_2026_09_01.md`。当前唯一未 compiled 的四段
成员是 input Conv + L∞ concretization，full-region timing 仍关闭。

## 7. 验证

- 新 owner 专项：`4 passed`；
- root/BaB 相关专项：`84 passed`；
- 全量：`2192 passed, 3 skipped`；
- mypy：2 个新增文件 clean；
- pylint：`10.00/10`；
- Black：clean；
- `git diff --check`：PASS；
- `performance_claimed=false`。

本记录完成的是 correctness architecture，不是性能 closure。
