# 变更记录：生成 PR-14A 真实 Verifier Query Traces

## 修改

- runner 增加 recorder-off baseline-first 对照，manifest 自动比较 status 与 visited domains；
- runner 在执行外部 verifier 前审计 BoundFlow ONNX frontend，不支持的模型会把全部 query
  profile fail closed；
- 补充 instance-level observer 恢复测试和 frontend precondition rejection contract；
- 在本地 ignored artifact 目录生成 MLP、CNN、VNN-COMP ResNet-2B 三组真实 query/profile/
  coverage/manifest；
- 新增 `pr14a_real_query_coverage_2026_07_19.md`，冻结 coverage 结果、限制与 PR-14B 窄化决策。

## 判定

- 540 个真实 `compute_bounds` 调用中 143 个 capability eligible；
- initial phase 143/146 eligible，activation-BaB phase 0/394 eligible；
- PR-14B 仅对 initial plain-CROWN NARROW GO，activation-BaB backend replay NO-GO；
- observer-on/off 三组 status 与 visited-domain count 一致；ResNet 独立运行 lower 差约 1.2e-7；
- PR-14A 仍为 VALIDATED-PARTIAL：尚无 external payload fixed replay 与 parent lineage。

## 验证

- PR-14A contract：6 passed；
- 原始工件 schema/profile accounting validation 通过；
- MLP 377 query、CNN 1 query、ResNet 162 query，0 duplicate/loss；
- raw artifacts 保持 `.gitignore`，Git 只提交代码、contract 与审计文档。
