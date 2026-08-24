---
status: documented
updated: 2026-08-24T11:30:00+08:00
type: changelog
topic: boundflow
slug: r3-structured-owner-custom-backward-redesign
stage: s01
---

# BoundFlow R3 结构化所有权重设计修改记录

## Summary

- 新增 R3-SO-CVJP 设计预注册，回应“结构化表示保持到自定义 backward、不跨层保存 dense A”。
- 明确 R3 是独立新路线，不复活 B4-C2，也不开放实现或性能 claim。
- 新增可直接交给其他模型的 standalone external-review Prompt，包含 GitHub/PR/commit/代码链接。

## Changes

- 从代码重建 B4-C2 的双重生命周期根因：
  - `dense_lower_once=True` 在六个 ReLU 把 lazy sign-split operator 转成 dense Tensor；
  - `_CIBCDenseExactTIRFunctionV3` 通过 `ctx.executor` 间接强引用输入和工作区。
- 选择 region-level single custom VJP，拒绝 layer-level Function 链。
- 冻结 `StructuredLowerRegionTemplateV1` / `InstanceV1`、DAG node、closed-region ownership、
  start-node keyed α/β 和 Template/Instance 缓存边界。
- 冻结不跨层保存 dense A 的精确定义：允许 kernel 内 tile/两个复用 scratch，不允许 Python/autograd
  输出、saved tensor、ctx/executor 或 per-layer persistent buffer。
- 冻结 M0 rematerialization 默认方案；M1 bit-packed sign certificate 保持 CLOSED；dense checkpoint 禁止。
- 冻结 saved-tensor、allocator、external scratch 三层 liveness evidence 和 raw 字段。
- 冻结 R3-0—R3-7 阶段 DAG、correctness/memory/performance gate 与统一 kill condition。
- 将 B4-A 最终状态统一为已外审的 performance NO-GO，而非模糊的 reduced。
- 将父恢复计划、README、执行 memo、current status 和 claims map 指向新设计与 Prompt。
- 收录外部模型对失败门禁恢复计划的只读评审，保留其 R1 target/时钟校准等后续 finding。

## Validation

- 文档内引用的本地路径、fenced-code 配对、零绝对本机路径检查：PASS。
- GitHub repo/PR/commit 与 PyTorch/TVM 一手资料链接 HTTP 可达：PASS；代码 commit identity=
  `f87f737cebffaf10827957682e3196063e4c78ed`。
- `git diff --check`：PASS。
- `dol lint --soft`：PASS；`dol validate` 只报告 `ev009180/ev009388/ev010862` 三个历史重复 ID，
  已核对它们在本轮基线 HEAD 中均各出现两次，不是本轮引入。
- 本轮为文档与设计变更，不修改 production code，不运行性能 benchmark。

## Decisions

- 当前 executable engineering next 仍是 `preregister-cibc-g1-optimized-graph-attribution`；R3 只是未来
  α-CROWN 恢复路线的 design-review track。
- Region forward 只返回最终 lower，不返回逐层 A；upper path 暂留 native。
- Function context 不保存带 Tensor 的 executor/instance；所有必要 Tensor 只经 `save_for_backward`。
- v1 允许 bounded transient dense scratch，但禁止 dense A 跨 layer/autograd lifetime。
- 六 site累计 `>=1.05x` 且 memory `<=1.0x` 才允许 same-solver B4-D；不降低历史门槛。

## Follow-Ups

- 把 external-review Prompt 发给至少两个独立模型；合并 blocker/major 建议。
- 若评审通过，另开 R3-0 contract-only 分支和 DocOps exchange；不得直接写 TIR。
- CIBC 主线仍按恢复计划完成 R0/G1；外部评审提出的 R1 target T、clock-domain calibration 等问题
  在对应预注册中关闭，不在本设计提交里偷跑。

## Links

- plan: `BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`
- external review prompt: `BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`
- parent review: `external_review_failed_gates_recovery_plan_2026_08_24.md`
