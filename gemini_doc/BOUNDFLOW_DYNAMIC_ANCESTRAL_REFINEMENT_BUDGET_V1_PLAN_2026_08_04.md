# BoundFlow Dynamic Ancestral Refinement Budget v1 计划（NRIR-25）

日期：2026-08-04
分支：`feat/dynamic-ancestral-refinement-budget-v1`
基线：`main@47ca159`（PR #35）

## 1. 起因

NRIR-24 证明 external-seeded ancestral 在 7→15→31 nodes 上持续改善，但 depth-4 proof deficit
仍为 `0.282360/0.401845/0.459939`。继续按所有节点固定 16 targets/ReLU、单 pass 堆树，会把相同
refinement 预算花在风险不同的域上，也没有形成一等预算决策 IR。

本轮不增加总 node budget，也不宣称多 pass 已解决；先回答更窄的问题：同一 31-node/depth-4
预算下，把每个 generated batch 的固定 target cap 动态分给风险更高的父域，是否比均匀分配得到
更强 worst terminal lower。

## 2. 一等预算 IR

新增 immutable typed policy 与 decision：

- policy 冻结 `parent_lower_generated_batch_v1`、base/high/low cap=`16/24/8`、risk order、tie 行为、
  conservation 与适用 strategy；
- root 或只有一个 unique parent 的 group 固定分配 base 16；
- 两个 parent 的 generated batch 中，parent lower 更低的一组 children 分配 high 24，另一组 low 8；
  parent lower 在 `1e-6` 内相同则全部 base 16；
- 每个 decision 绑定 policy hash、group ID/hash、node/split/depth、parent identity/lower/rank、派生 cap
  和 conservation totals；缺字段、非有限 lower、group 重复/不闭合或总 cap 不守恒均 fail closed；
- 派生的 `NativeIntermediateRefinementPolicyIR` 进入每个 node 的 refinement Plan，Task target coverage
  与 Schedule action chain 继续由实际 Plan 决定；budget decision hash 同时进入 queue evaluation/
  refinement trace，不能只作为旁路日志。

旧 fixed policy 在 dynamic policy 缺席时保持 payload 兼容。

## 3. 固定对照

- workload：VNN-COMP 2021 CIFAR10 ResNet2B property 0，clauses `0/2/4`，CPU；
- source：NRIR-23 typed external seed；strategy：`external_seeded_ancestral_carry_v1`；
- queue：31 nodes/depth 4、expansion batch 2、evaluation batch 4；
- objective branch、25-step adaptive optimizer、chunk 8、单 refinement pass 全部冻结；
- `fixed16`：所有节点 max 16 targets/ReLU；
- `dynamic8_24`：按第 2 节分配，root/single-parent group 为 16；
- 每组 planned cap sum 必须等于 `node_count × 16`，全树也必须等于；实际 selected target count 单独
  披露，不冒充同耗时；时间只作诊断，`performance_claimed=false`。

两组 branch decisions 允许因 refinement bounds 改变而产生不同 logical tree；必须报告按
`split_state_hash` 的 domain overlap，不能把不同树的 node serial 当作同一域。公平性主口径是相同
node/depth、相同 planned refinement cap 和相同其余 policy 下的 worst terminal lower。

## 4. Artifact

- 六个 `clause × mode` fresh-process shards，逐单元原子 checkpoint、严格 resume；
- shard 冻结 source、预算 policy/decision、派生 refinement Plan/Task/Schedule hash、queue、branch、
  lower 与实际 target count；
- assemble 重算 group/tree budget conservation、IR cross-link、mode comparison、logical overlap 与
  claim boundary；
- replay 必须 fresh-process 重算六个 shard 并逐对象比较；source、budget、decision、Plan linkage、
  lower、status 或 claim tamper 必须 fail closed。

## 5. Acceptance criteria

- **AC1 typed policy/decision**：policy 与逐 node decision typed、stable-hashed、严格校验；旧 fixed
  路径条件兼容。
- **AC2 lowering**：assigned cap 精确进入 refinement Plan policy，Task/Schedule/execution 与 queue
  decision hash 交叉绑定。
- **AC3 conservation**：逐 group 与全树 planned cap exact conservation；root/tie/single-parent 走
  base，two-parent risk order 走 24/8；篡改 fail closed。
- **AC4 fair fixed evidence**：三 hard clauses、两 mode、同 31-node/depth-4 与同 planned total cap；
  报告 worst/best terminal lower、proof deficit、selected targets 和 domain overlap。
- **AC5 replay/regression**：semantic replay、focused/full pytest、Black、mypy、pylint、diff check 与
  DocOps lint 全过。

## 6. 预注册判定

- dynamic 三条 worst terminal lower 均不弱于 fixed（容差 `1e-6`），且至少一条严格改善：
  `VALIDATED-REDUCED`；
- 任一 clause 退化超过 `1e-6`，或没有任何严格改善：`VALIDATED-NO-GO`；
- 只有某 mode 的全部 terminal domains 达到 lower≥0，才声明该固定 clause/budget bounded-tree
  closure；不得外推完整 property；
- 无论结果如何，不形成 latency/performance、CUDA、multi-workload、competitor 或 ASPLOS-ready
  claim。multi-pass 仅在本轮动态 cap 有正向证据后另立门禁。

## 7. 提交边界

一个逻辑提交包含预算 IR、queue lowering、正负向测试、六 shard artifact、权威文档与 DocOps；
不修改 third-party 子模块。
