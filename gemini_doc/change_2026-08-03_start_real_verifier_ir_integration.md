# 变更记录：启动真实 Verifier IR 集成路线

> 日期：2026-08-03
> 分支：`feat/real-verifier-ir-integration-v1`
> parent：`d457b22`

## 主要改动

- 新增 `real_verifier_ir_integration_contract_v1_2026_08_03.md`；
- 将新路线与已封存的 IR-5 system-performance 路线隔离；
- 记录真实 ResNet mismatch 的逐层根因与 external-bounds + adaptive-policy 复核结果；
- 冻结 plain-CROWN external semantics 与 activation-BaB external exact call 两类 IR 路径；
- 明确 fused replacement coverage、typed IR admission coverage 与性能 claim 三者不能混写。

## 复核事实

- 冻结 external lower 在 CPU 上可重现；
- 本地 IBP trace + adaptive slope：max diff 约 `810.805`；
- external 6 组 ReLU pre-activation bounds + adaptive slope：max diff `2.15e-6`，sign `9/9`。

本提交只冻结契约与根因，不宣称代码实现已经完成。
