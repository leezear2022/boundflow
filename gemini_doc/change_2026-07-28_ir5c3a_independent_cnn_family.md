# 变更记录：IR-5C3A independent CNN workload family

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`d9ac9cd`（IR-5C2 typed MLP CUDA PARTIAL）
> 状态：独立 architecture-family foundation 完成；fair batching/artifact 仍 pending

## 改动

- 正式 typed workload builder 新增 deterministic two-convolution chain CNN：
  - Conv3×3→ReLU→stride-2 Conv3×3→ReLU→flatten→linear；
  - seed、batch、channels、image size 与 output dimensions 全部显式；
  - 与 MLP 共用 BoundModule→PlanTemplate→PlanInstance→TaskIR→ScheduleIR lowering。
- measured benchmark 新增 `TypedCNNWorkloadSpec`：
  - `family=chain_cnn` 明确进入 JSON；
  - 以两层 convolution + head 的 MAC proxy 作为跨 architecture calibration feature；
  - calibration MLP 与 held-out CNN 可通过同一 typed measurement/evaluator 接口；
  - 原 IR-5C2 MLP spec/JSON schema 保持不变。
- candidate dispatch 按 spec 类型选择 MLP/CNN builder，不允许把 CNN shape 塞入 MLP 字段。

## 验证

- CPU typed reference/dense CNN final lower/upper 对齐；
- CUDA 小型 probe 中 reference、dense、chunked、TVM fused 全部对齐：
  - dense/chunked max diff `1.9073486328125e-06`；
  - TVM lower/upper max diff `4.76837158203125e-07` /
    `1.9073486328125e-06`；
  - trace 确认执行 `backend:tvm_fused_tir:fused`；
- MLP calibration→CNN held-out observation contract 通过；
- 定向 4 passed，Mypy 0 issues；
- IR-5C2 v2 integrity/semantic replay 继续通过。

## 边界与下一步

本切片只证明新的 architecture family 能经过完整 typed compiler/backend 路径，不是新的
held-out 性能结果。Residual/concat 目前有 reference semantics，但 v1 fused selector 只选择
一个连续 ReLU→Affine pair，因此不把整段 residual block 宣称为联合 fused candidate。

下一切片 IR-5C3B 将 ordinary batching 与 fair batched-original 定义为同一 workload、
相同 physical batch、相同 query count 和相同数值容差下的正式 baseline，再生成
MLP-calibration→CNN-held-out artifact。

