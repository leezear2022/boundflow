# BoundFlow Typed Multi-Pass Refinement v1 计划（NRIR-26）

日期：2026-08-04
分支：`feat/typed-multipass-refinement-v1`
基线：`main@78ffa6b`（PR #36）

## 1. 起因

NRIR-25 在相同 31-node/depth-4、planned target cap 下，用 parent-lower 24/8 动态分配使 ResNet
clauses `0/2/4` 的 worst lower 一致改善，但幅度仅为
`3.86e-4/2.33e-4/2.72e-4`，三条 bounded tree 仍为 unknown。继续增加固定树深会重复 NRIR-24
已经暴露的长尾；直接把 `passes=2` 打开也不成立，因为现有实现只是对同一组初始 targets 重复
CROWN，没有一等 pass 重选、预算归属或停止语义。

本轮回答更窄的问题：在每个 node 的总 planned target cap 不变时，把一次 target selection 拆成
两个顺序 pass，并在 pass 1 tightening 后依据更新 width 重选未覆盖 targets，是否能提高 fixed hard
clauses 的 worst terminal lower。

## 2. First-class multi-pass IR

新增 immutable typed multi-pass policy/selection decision：

- `maximum_passes=2`，`total_cap_partition=equal_two_pass_v1`；总 cap 必须为正偶数，逐 pass cap
  精确为 `total/2`；
- `reselection=updated_width_excluding_prior_targets_v1`：objective influence 在 node 初始 clause state
  计算并冻结，各 pass 使用当前 tightened width 重新打分；已选 `(relu_input, neuron_index)` 不得重复；
- `termination=no_unseen_eligible_targets_v1`：如果某 pass 没有未覆盖 ambiguous target，则该 pass
  产生 typed stopped decision，后续数据流只做 sound passthrough；不使用开发后调出的数值阈值；
- 每个 pass decision 绑定 policy/Plan、pass index、input bounds hash、prior-target ledger hash、selected
  target hash/count、per-pass/total cap、continuation 与 termination reason；顺序、重复、超 cap、ledger
  或 hash 漂移全部 fail closed；
- Plan 冻结总预算与选择规则；Task/Schedule 对每个 pass 显式 lower
  `enumerate→select/decide→backward→intersect→propagate`，不能由 Python 隐式循环冒充 compiler IR；
  execution trace 与 action outputs 绑定实际逐 pass target identities 和 bounds lineage。

multi-pass policy 缺席时，旧单 pass 与历史 `passes>1` 重复-target payload/hash 必须保持兼容。

## 3. 固定公平对照

- workload：VNN-COMP 2021 CIFAR10 ResNet2B property 0，clauses `0/2/4`，CPU；
- queue：31 nodes/depth 4；external seed、ancestral carry、parent-lower dynamic 8/16/24、objective
  branch、25-step optimizer、batching 全部冻结；
- `single_pass_dynamic8_24`：每 node 一次使用完整 assigned cap；
- `split_two_pass_dynamic8_24`：assigned cap 平分为 4+4、8+8 或 12+12，第二 pass 排除第一 pass
  targets；
- 两 mode 的每 node/每树 planned total cap 必须相同；`backward_chunk_size=4` 两侧一致；实际
  selected count、stopped pass、target overlap、logical-domain overlap 与 CPU timing分别披露；timing
  只作诊断，`performance_claimed=false`。

两 mode 可以因 tighter bounds 改变 branch tree；主比较为固定 node/depth/总 target cap 下的 worst
terminal lower，不按 execution serial 强配不同 logical domains。

## 4. Artifact 与 replay

- 六个 `clause × mode` fresh-process shards，atomic checkpoint 与 strict resume；
- shard 冻结 source、dynamic budget、multi-pass policy/decisions、每 pass target ledger、refinement
  Plan/Task/Schedule、queue/branch/lower；
- aggregate 重算逐 node/树总 cap、pass disjointness、lineage、mode comparison 与 claim boundary；
- replay 必须在 fresh process 重编译/执行六 shards 并逐对象相等；policy、decision、target、pass
  bounds、Plan/Schedule linkage、lower、status 和 claim tamper 均 fail closed。

## 5. Acceptance criteria

- **AC1 typed policy/decision**：total partition、reselection、termination 与逐 pass decision typed、
  stable-hashed、严格 admission；legacy payload/hash 条件兼容。
- **AC2 compiler lowering**：每 pass enumerate/select/decision/backward/intersect/propagate 是一等
  Task/Schedule action，实际 target 与 bounds lineage 进入 replay-grade trace。
- **AC3 budget/selection**：逐 node 与全树 total cap 精确相同；pass target identities disjoint；无
  unseen target 的 stop/passthrough 正负路径可测且 tamper fail closed。
- **AC4 fixed evidence**：三 hard clauses、两 mode、同 31-node/depth-4、同 dynamic assigned total
  caps；报告 worst/best/proof deficit、actual targets、pass stop 与 tree overlap。
- **AC5 validation**：六 shard fresh replay、focused/full pytest、Black、Mypy、Pylint、diff check 与
  DocOps lint 全过。

## 6. 预注册判定

- split-two-pass 三条 worst terminal lower 均不弱于 single-pass（容差 `1e-6`），且至少一条严格
  改善：`VALIDATED-REDUCED`；
- 任一 clause 退化超过 `1e-6`，或无严格改善：`VALIDATED-NO-GO`；
- 只有某固定 mode/clause 的所有 terminals 达到 lower≥0，才声明该 bounded-tree closure；不得
  外推完整 property；
- 无论结果如何，不形成 latency/performance、CUDA、multi-workload、competitor 或 ASPLOS-ready
  claim。

## 7. 提交边界

一个逻辑提交包含 multi-pass IR、compiler lowering、runtime/trace、正负测试、六 shard artifact、
权威文档与 DocOps；不修改 third-party 子模块。
