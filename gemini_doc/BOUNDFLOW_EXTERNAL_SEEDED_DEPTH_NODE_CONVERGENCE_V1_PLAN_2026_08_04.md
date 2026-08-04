# BoundFlow External-Seeded Depth/Node Convergence v1 计划（NRIR-24）

日期：2026-08-04
分支：`feat/external-seeded-depth-node-convergence-v1`
基线：`main@c6a7998`（PR #34）

## 1. 起因

NRIR-23 已把 external-owned intermediate constraints、native root refinement 与 native ancestral
carry 连成一条 typed、可重放路径。在 ResNet2B property 0 hard clauses `0/2/4` 的七节点、深度二
树上，ancestral worst-leaf lower 相对 external baseline 分别改善约 `0.001512/0.001133/0.000534`，
但三条 lower 仍为负，只能关闭为 `VALIDATED-REDUCED`。

当前未知量不是 IR provenance，而是收敛性：沿用完全相同的语义和优化预算，仅增加完整二叉树的
深度/节点数，proof deficit 是否持续缩小，以及缩小速度是否足以支持继续这条算法路线。

## 2. 目标与研究问题

冻结 `7/15/31 nodes × depth 2/3/4` 的嵌套曲线，回答：

1. 三个 hard clause 的 worst terminal lower 是否随预算单调不降；
2. 深度三、四相对前一预算是否有可复现的严格改善；
3. 是否有 clause 在固定树内达到 `lower >= 0`；
4. 若仍未闭合，单位新增节点带来的 deficit 缩减是否快速衰减，从而触发算法门禁切换。

## 3. 固定协议

- workload：VNN-COMP 2021 CIFAR10 ResNet2B property 0，CPU；
- clauses：`0, 2, 4`；
- 唯一模式：NRIR-23 的 `external_seeded_ancestral_carry_v1`；
- objective branch：NRIR-17 冻结 policy；
- optimizer：adaptive alpha，25 steps；
- refinement：每 ReLU 最多 16 targets、chunk 8、单 pass；
- queue batching：expansion batch 2、evaluation batch 4；
- 预算：`(7,2)`、`(15,3)`、`(31,4)`，均为完整二叉树上限；
- threshold：`0.0`；时间只作运行诊断，`performance_claimed=false`。

除 `max_nodes/max_depth` 外不得改变 source artifact、model/property/objective、external seed、
branch/optimizer/refinement policy 或 batching。三个预算使用稳定 run identity，使小预算节点必须是大
预算节点的语义前缀。

## 4. Artifact 与断点恢复

- 每个 `clause × budget` 在独立 Python 进程中建立 context 并执行；
- 每完成一个单元即原子写入一个 digest-bound shard；已存在且通过严格校验的 shard 可恢复复用；
- shard 保存 queue/refinement/objective-branch 的逐节点语义 hash 投影、lineage、lower、terminal
  reason 与 summary，不复制 NRIR-23 已冻结的完整 IR payload；
- assemble 对九个 shard 做 source/protocol/coverage、嵌套逻辑域、单调性和 claim-boundary 校验后才写
  manifest；
- replay 必须在 fresh processes 中重算九个 shard 并逐对象比较，不能只检查文件 digest；
- 篡改 source、budget、node lower/hash、父子 lineage、status 或 performance claim 必须 fail closed。

## 5. Acceptance criteria

- **AC1 fixed matrix**：九个单元完整，唯一变量为 node/depth budget；所有 source/policy digest 一致。
- **AC2 nested semantics**：`7 ⊂ 15 ⊂ 31` 按 `split_state_hash` 匹配的公共逻辑域必须完整包含，
  parent split lineage、branch literal/selection 与去除执行序号后的 refinement semantics 一致；
  lower/upper/priority 使用 runtime 已冻结的 `NATIVE_REEXECUTION_ATOL=1e-5`。best-first 会改变跨预算
  的生成顺序、node ID、batch ID 和 objective Plan/Task/Schedule execution hash，因此这些执行身份不
  冒充跨预算逻辑域身份。
- **AC3 convergence accounting**：逐 clause 报告 root、worst/best terminal lower、proof deficit、
  相邻预算 delta、closure 和每新增节点的 deficit reduction。
- **AC4 replay-grade**：断点恢复、manifest digest 与 fresh-process semantic replay 均通过；关键字段
  篡改被拒绝。
- **AC5 regression**：focused tests、全量 pytest、Black、mypy、pylint、`git diff --check` 与
  DocOps lint 全部通过。

## 6. 预注册判定

- 若所有 clause 的 worst terminal lower 均随预算不降，且至少一个 clause 在 depth 3 或 4 严格改善，
  关闭为 `VALIDATED-REDUCED`；
- 只有某个预算的全部 terminal domains 均 `lower >= 0`，才可声明该固定 clause/budget 的
  bounded-tree closure；不得外推完整 property 或 verifier；
- 若出现超过 `1e-6` 的非单调退化，关闭为 `VALIDATED-NO-GO` 并定位 optimizer/refinement/queue
  原因；
- 若三条 clause 从 15→31 nodes 的 worst-lower 改善均不超过 `1e-6`，即使无退化也判定“深度扩展
  饱和”，下一路线切换到 dynamic refinement budget/multi-pass，不继续盲目堆树深；
- 无论结果如何，不形成 performance、GPU、multi-workload、competitor 或 ASPLOS-ready claim。

## 7. 提交边界

本轮优先新增实验 runner、artifact contract tests、冻结 artifact 与权威文档；除非实验暴露真实
契约缺陷，不修改既有 IR/runtime，也不修改 third-party 子模块。
