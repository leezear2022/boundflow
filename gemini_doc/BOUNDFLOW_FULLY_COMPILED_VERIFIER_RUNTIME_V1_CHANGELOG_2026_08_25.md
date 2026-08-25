---
status: implemented
updated: 2026-08-25T19:10:00+08:00
type: changelog
topic: boundflow
slug: fully-compiled-verifier-runtime-v1-changelog
stage: s01
---

# 全编译验证器运行时 v1 修改记录

## 本轮变化

- 新增全编译验证器运行时 v1 架构，正式记录“逐步移除 production 热路径 PyTorch”的目标；
- 将编译范围从孤立 bound-op 扩展到 relaxation、bound propagation、custom backward、optimizer、
  branch score/top-k、split/materialization、queue commit；
- 冻结 semantic IR、tensor program IR、execution graph、memory/arena plan、queue scheduler 五层结构；
- 冻结 FCR-0—FCR-5 迁移路线、correctness/performance/kill gate；
- 明确 MR7 是 FCR-0 物理账本，用于选择第一个多算子 compiled region，而不是最终优化边界；
- 保留 PyTorch 作为迁移期 oracle/fallback，不把“最终剔除热路径 PyTorch”误写成已完成事实。

## 动机

MR5/MR6 证明当前三站点 bridge 的 framework crossing 和 ownership boundary 会淹没局部 TIR 收益；
而 B4-B2/CIBC 又证明合法的局部和整图编译可以显著快于 PyTorch。因此下一路线必须扩大编译所有权，
同时继续由 same-solver formal 门禁约束 claim。
## 验证

- `git diff --check`；
- DocOps lint；
- 本轮仅文档，不形成新性能 claim。
