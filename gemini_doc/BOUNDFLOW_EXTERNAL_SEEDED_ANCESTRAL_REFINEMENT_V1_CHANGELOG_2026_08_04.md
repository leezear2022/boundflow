# BoundFlow External-Seeded Ancestral Refinement v1 修改记录（NRIR-23）

日期：2026-08-04
分支：`feat/external-seeded-ancestral-refinement-v1`
状态：`VALIDATED-REDUCED`

## 当前记录

- 从 `main@030ed1f` 建立 NRIR-23 分支；
- 复核 NRIR-17 external hard-clause 根 bounds 与 NRIR-22 local ancestral bounds 的数量级差异；
- 确认现有 queue admission 明确禁止 per-child refinement 与 raw external override 共存；
- 冻结 typed external constraint seed、Plan/Task/Schedule 显式依赖、root/parent lineage 和三模式
  hard-clause 固定实验契约。

## 完成内容

- 新增 `ExternalIntermediateConstraintSeedIR`，绑定 external provider/ownership、图、输入、ordered
  external bounds、effective local constraints、source manifest/payload、model/property/objective-set；
- public builder 对 raw external 与 local forward 先求可行交集，Plan/Task/Schedule/action trace 显式
  消费并哈希 effective seed；seed 与 parent refinement source 严格互斥；
- queue 新增 `external_seeded_ancestral_carry_v1`：root 只消费 typed seed，六个 non-root 节点只
  消费 validated parent execution；queue/refinement/Plan/semantic/final hash 逐节点闭环；
- 新增 hard-clause generate/replay runner、冻结 artifact 和 8 条新增 contract/artifact/tamper tests。

## 执行中修正

- 首版实验误把“六个 ReLU 合计约 96 targets”实现为“每 ReLU 最多 96”；未采信该结果，已恢复
  NRIR-22 同预算的每 ReLU 16、chunk 8 并重新生成；
- 首次 semantic replay 发现 refinement payload 保留 `elapsed_ns`，使纯时钟差异触发整对象不等；
  已从 semantic refinement 序列化中删除该字段，顶层 timing 仍明确仅作单次诊断。
- 发布前 scope 审核将 NRIR-17 已独立冻结的完整 objective-candidate 表替换为逐节点
  Plan/Task/Schedule/trace hash 投影；artifact 从约 4.4 MB 降至 1.94 MB，完整 semantic replay
  仍通过。

## 固定结果

固定 ResNet2B property 0 clauses `0/2/4`，三模式均使用 objective branch、25-step adaptive
optimizer、每 ReLU 16 targets、chunk 8、7 nodes/depth 2：

| clause | external baseline worst leaf | seeded root-global | seeded ancestral | ancestral-root | ancestral-baseline |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.319798589 | -0.319109678 | -0.318286777 | +0.000822902 | +0.001511812 |
| 2 | -0.426609159 | -0.425480783 | -0.425476611 | +0.000004172 | +0.001132548 |
| 4 | -0.504676342 | -0.504142046 | -0.504142046 | 0 | +0.000534296 |

三条 ancestral 均不弱于 root-global，两条严格改善，因此按预设门禁关闭为
`VALIDATED-REDUCED`。所有 terminal leaves 仍负，三条 fixed-tree status 均为 unknown；不升级
完整 property、GPU、multi-workload、competitor、性能或 ASPLOS-ready claim。

## Artifact 与验证

- artifact：
  `artifacts/external-seeded-ancestral-refinement/vnncomp21-resnet2b-prop0-hard3-cpu-v1/`；
- generate/replay evidence hash：
  `9f52b99a74dab448626061f5b8f060f3b8c43b6c03f6deb0899d9fe91883d9f7`；
- focused：`33 passed`；全量：`766 passed, 37 skipped`；
- Black check、Mypy 4 files、Pylint `10.00/10`、`git diff --check` 通过；
- 首轮未激活 hook 的全量 collection 缺 TVM；真实 `conda activate boundflow` 后全量通过，未计为
  代码回归。

## 下一门禁

冻结 external-seeded ancestral 的 depth/node convergence 曲线；先回答增加 7→15→31 nodes、
depth 2→3→4 是否持续缩小 clauses `0/2/4` proof deficit，再决定 dynamic bound policy、更多
refinement passes 或停止该算法路线。单次 CPU timing 继续不是 performance claim。
