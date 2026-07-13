# 2026-07-13：PR-12 起点冻结与 fused ReLU+Linear TIR foundation

## 目标

从只读 `pr11-validated-reduced` tag 启动 PR-12，先冻结新候选与 unseen held-out 合同，再实现
不写回完整 ReLU-scaled coefficient 的 Linear fused task。此切片不声称 PR-12 完成，也不修改
PR-11 Planner/profile。

## 起点与证据隔离

- 分支：`feat/pr12-fused-crown-task`，从 tag `pr11-validated-reduced` 创建；
- tag 基线全量：228 passed、1 skipped；
- 起点工件：`artifacts/phase7a-pr12/baseline/`，含 `manifest.json`、`tests.txt`、
  `planner_freeze_ref.json`、`heldout_split.json`；
- split：`pr12-final-heldout-v1` 在 kernel 实现前冻结，7 个 PR-11 backend-gap case 仅属于
  motivation/development set；final 包含 unseen spec/domain/width、Linear/Conv/mini-ResNet 配置。

## 候选合同

新增 placement/backend 二维候选与显式 capability filtering。当前 capability 只开放 static FP32
CUDA、无梯度 plain CROWN 的 contiguous Linear；Conv、α/αβ、split、training、dynamic shape 和
非 CUDA 都给出明确 rejection reason。

## Linear fused task

`fused_crown_linear.py` 输入 upper/lower coefficient、upper/lower slope/intercept、Linear weight
与 bias，直接输出 upper/lower `A_prev` 和 bias delta。sign selection、slope scaling、ReLU
intercept 与 Linear bias contraction 均内联在 reduction 中：

- PrimFunc 不定义/分配 `A_scaled[D,S,I]`；
- deterministic CUDA schedule 为每个输出元素分配一个 thread，feature 维串行 reduction；
- thin Relax wrapper 只包含一个 `call_tir`；
- compile key 显式包含 D/S/I/J、dtype、target 与 schedule id；
- CUDA calibration shape `D=2,S=8,I=16,J=12` 的四个输出与 PyTorch dense reference 对齐，
  同时覆盖正、负、零 coefficient。

## 当前边界

这是 correctness/mechanism foundation，而非性能 closure：尚未接入 end-to-end CROWN runtime，
尚未实现 Conv2d，未运行 final held-out，也未证明 launch、peak memory、latency 或 compile
amortization 门禁。当前 capability 因此不能宣称 Conv 支持，Claims Map 只能标记 partial。

## 验证

```text
PR-12 专项：18 passed
Linear CUDA：4 outputs allclose，0 A_scaled allocation
Relax：1 Relax function、1 TIR function、1 call_tir
```
