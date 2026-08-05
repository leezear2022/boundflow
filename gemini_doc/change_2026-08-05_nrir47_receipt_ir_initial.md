# 2026-08-05 NRIR47 Receipt IR 初始实现

- 在新增 NRIR47 模块中实现“构建未二次选择的 exact Program”，旧核心 compiler 文件及 public
  full-validation API 保持原样；早期对旧文件的重构曾触发 NRIR33/34 frozen revision 测试，已在正式
  artifact 前撤回；
- 新增 typed target admission receipt、两阶段 Task IR 与 Schedule IR；
- 新增 single-pass Program/compiler：production validation 消费 exact receipt，不重调 selector；
  `validate_full` 仍执行原始 target selection；
- prepared capsule 以向后兼容的 v1/v2 schema 显式绑定 receipt hash，legacy v1 serialization/hash
  保持不变；
- 新增 6 条 receipt/compatibility/tamper/cross-program/full-replay 测试；与既有 prepared 测试合计
  `10 passed`。

预注册计数口径同步澄清：candidate 每条 queue 的 child compile selector/reselection 目标为
`30/0`，另有 root source + 30 child 共 `31` 份 receipt；既有 runtime semantic selector 仍为 `30`；
full replay selector=`31` 且不计入 production timing。Phase A/B timing 门槛未修改。
