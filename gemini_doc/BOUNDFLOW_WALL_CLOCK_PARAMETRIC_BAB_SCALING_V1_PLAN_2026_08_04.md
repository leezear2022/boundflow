---
status: completed
updated: 2026-08-04T13:02:00Z
type: plan
topic: boundflow
slug: wall-clock-parametric-bab-scaling-v1
stage: s01
---

# Wall Clock Parametric BaB Scaling v1 Plan

## Goal

- 把 NRIR-28 的 parametric compiler/runtime 收益重新投入搜索：在相同模型、property、optimizer、
  branching、batching 与 `60 s` query deadline 下，将 bounded BaB 从 `7/depth2` 扩到
  `31/depth4` 和 `127/depth6`，判断系统加速能否转化为更多 verified clauses、更紧 terminal
  lower 或完整 property closure。
- 将 search budget、workload、repeat、fresh-process task 与交替执行次序编译为一等
  Plan/Task/Schedule；不得用临时循环或只报最快样本替代证据。

## Scope

- 固定 MNISTFC、CIFAR10 ResNet2B、OVAL21 三个 NRIR-28 workload，optimizer steps=5、search
  steps=4、torch threads=8、同一 parametric v2 runtime；唯一策略变量为 node/depth budget。
- 预算点预注册为 `7/2`、`31/4`、`127/6`；每 workload/budget 三个 fresh process，budget 顺序
  交替，保存 raw execution/E2E、completed/pending/unresolved、verified clause、逐 clause logical
  domains、terminal lower、cache/instance 与 Plan/Task/Schedule digest。
- v1 允许 deterministic best-first prefix 随预算扩展；跨预算以 split-state logical identity 校验
  nesting，不以 execution serial 冒充语义身份。公共 domain lower 允许 `1e-5` 数值容差。
- 不修改 NRIR-28 frozen compiler/runner；不声明 CUDA、αβ-CROWN speedup、完整 verifier parity 或
  ASPLOS-ready。

## Tasks

1. [x] 定义 search-scaling Budget/Plan/Task/Schedule IR，冻结三 budget、三 workload、三 repeats、
   fresh-process timing boundary、60 秒 deadline 与交替次序。
2. [x] 新增 additive worker/runner，执行 parametric complete query 并导出逐 clause domain/leaf、
   verdict、compiler cache 与 phase evidence；source/IR/log/manifest 可 fresh replay。
3. [x] 增加 IR lowering、schedule order、budget tamper、domain nesting、lower monotonic、deadline/
   accounting 与 artifact semantic replay 负向测试。
4. [x] 先完成 `7/31/127 × 3 workload` 全矩阵，再按预注册门禁判定；不得观察正式结果后改预算或
   过滤失败 repeat。
5. [x] 运行 focused/full pytest、Black、targeted Mypy、Pylint、NRIR-28/29 artifact replay、diff 与
   DocOps gate，更新 claims/status/memo/changelog。

## Validation

- 每个 workload/budget 三次均必须完成相同 clause 集，且 `completed=9`、`pending=[]`；若 127 在
  deadline 下丢 clause，则保留为资源边界，不得删除样本。
- `domains(7) ⊆ domains(31) ⊆ domains(127)` 按 clause/split-state 成立；公共 domain lower 漂移
  `≤1e-5`，verified clause 不得随预算增加而回退，所有 verdict/leaf accounting 继续 sound。
- 只有至少一个 workload 的 verified clause 数严格增加，且所有 workload 无 completed-clause/
  verified-clause 回退，才以 search-coverage `VALIDATED-REDUCED` 关闭；若仅执行更多节点但无
  closure/tightness 增益，则 `VALIDATED-NO-GO`。
- 性能/资源数字至少三次 raw + median/p90；不把预算不同的运行计算成 speedup，也不与历史外部
  competitor 数字直接相除。

## Rollback

- 本阶段只新增 experiment IR/runner/tests/artifact；删除新增文件即回到 NRIR-28。任何预算、source、
  domain nesting 或 replay 门禁失败时直接拒绝 artifact，不修改 production v2 runtime 兜底。

## Result

- 27/27 fresh-process tasks 完成，所有 workload/budget/repeat 都是 `completed=9,pending=[]`；三次
  同预算 semantic signature 一致，`7⊂31⊂127` logical domains 成立，公共 domain lower 最大
  漂移为 `0.0`。
- MNISTFC verified clauses 从 `6/9` 提升到 `8/9`，31 与 127 nodes 结果相同；ResNet2B 保持
  `0/9`，OVAL21 保持 `8/9`。因此按预注册门禁以 search-coverage `VALIDATED-REDUCED` 关闭。
- 127-node median execution：MNISTFC `2.515 s`、ResNet2B `58.566 s`、OVAL21 `2.287 s`；这些
  是资源曲线，不计算跨预算 speedup。artifact evidence hash=
  `e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214`。
- 下一路线不得继续盲目增加同类节点；应把 remaining hard clauses 编译成 typed escalation，按
  clause 选择更强 native intermediate refinement/branch policy，再在固定总 deadline 下复测。

## Links

- changelog: `gemini_doc/BOUNDFLOW_WALL_CLOCK_PARAMETRIC_BAB_SCALING_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_PARAMETRIC_DYNAMIC_BATCH_COMPILER_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
