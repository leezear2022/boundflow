---
status: completed
updated: 2026-08-04T11:50:00Z
type: plan
topic: boundflow
slug: production-prepared-verifier-v1
stage: s01
---

# Production Prepared Verifier v1 Plan

## Goal

- 把已经分别验证的 prepared root、optimized ReLU-split queue、objective branching、祖先约束与
  intermediate refinement 接成一条可计时的 production complete-verifier 路径；动态 node 不再
  构造逐 action audit hash chain，也不再执行 selected-native 双执行。
- 以相同模型、property、搜索预算、数值语义和 verdict 比较 audit/production；先建立 CPU
  cold/warm E2E 与 phase breakdown，再把同一冻结协议迁移到可用 CUDA 主机。只有公平重复矩阵
  通过后才允许讨论 speedup 或 ASPLOS performance claim。

## Scope

- 新增 typed production execution mode、prepare/execute phase 和 clause/node lineage；旧 audit
  mode、NRIR-1—26 artifact/hash/default behavior 保持不变。
- v1 允许动态 split、batched siblings、parent alpha/beta warm start、external/native-refined
  intermediate semantics；每个新 semantic scope 仍需一次 fail-closed compilation/validation，
  但 steady execution 不生成 replay-grade逐 tensor hash，也不重复 selected-native oracle。
- 首轮 correctness gate 覆盖 toy MLP 与 frozen CIFAR10 ResNet2B hard clauses；E2E gate 覆盖
  MNISTFC、ResNet2B、OVAL21 三类真实拓扑。当前主机无 CUDA driver，因此 GPU 结论明确 pending。
- competitor 数字只在相同 device、timeout、输入与完整性披露下报告；BoundFlow 的 bounded
  7-node search 与 αβ-CROWN complete BaB 不得直接计算 speedup。

## Tasks

1. [x] 冻结 production verifier Plan/Task/Schedule IR：prepare root、evaluate node batch、branch/
   prune、refine、emit verdict 均有 typed action、依赖、execution mode 与 phase boundary。
2. [x] 实现 production optimized-node evaluator：复用已验证 optimizer program，调用 prepared
   executor，跳过 audit action hash chain和 selected-native re-execution；保留一次性 identity/
   scope validation与数值有限性门禁。
3. [x] 接入 complete-query clause scheduler；root verified/unsafe short-circuit，unresolved clause
   才进入 prepared dynamic queue，deadline/pending/accounting 与旧路径一致。
4. [x] 增加 audit/production semantic parity、identity/tamper、action-order、deadline 与 fallback
   测试；证明 lower/upper、selected state、logical queue、clause status 和 final verdict 对齐。
5. [x] 生成至少 3 组交替次序的 CPU artifact，分离 import、prepare、cold first、warm execution、
   refinement、queue 与 verdict；在三种真实拓扑上报告重复样本和限制。
6. [x] 根据数据继续优化 compile cache、node batching 或 algorithmic budget；若不能稳定缩短公平
   E2E，则以 VALIDATED-NO-GO 关闭该具体策略，不把内部 audit-removal 写成竞品 speedup。

## Outcome

- NRIR-27 以 `VALIDATED-REDUCED` 关闭。三种真实拓扑的 clause-0 相同算法 audit→production
  fresh-process median speedup 分别为 `1.3663×/2.4723×/1.4511×`，每个 workload 均有三组
  交替次序样本和 semantic parity；这是 BoundFlow 内部 execution-mode claim，不是竞品 speedup。
- production full-query median 分别为 `14.834 s/60.754 s/11.964 s`；三者仍为 unknown。
  ResNet 三次均完成 `9/9` clauses，而历史 deadline-bound audit 只完成 `2/9`，但该差异不升级为
  complete-property claim。
- phase evidence 显示 production execution 的 `59%–65%` 仍位于四个 runtime action 之外；
  ResNet full-query median 的该部分约 `37.994 s`。下一路线冻结为 parametric dynamic-batch
  `PlanTemplate/PlanInstance` 与 compile-cache ownership，而非继续增加静态 refinement pass。
- artifact evidence SHA256=
  `7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b`；GPU、公平 complete
  competitor、verified/unsafe closure 与 ASPLOS-ready 均保持 pending。

## Validation

- production 与 audit 在固定种子下逐 clause lower/upper allclose、status exact、logical queue
  signature exact；任何 objective/input/split/intermediate/policy/program identity 漂移 fail closed。
- production trace 明确不含 audit hash chain且不执行 selected-native oracle；Plan/Task/Schedule
  action 数与真实 runtime phase 一一对应，不能以空 replay 代替执行。
- 性能比较至少 3 个独立 group，交替顺序，保存 raw samples、median/p90；cold preparation 与
  warm steady-state分开，超时和 pending 计入结果。
- focused/full pytest、Black、targeted Mypy、Pylint、artifact fresh replay、`git diff --check`、
  `dol validate` 与 `dol lint --soft` 全过。

## Rollback

- production mode 为显式 opt-in；删除新增 IR/runtime/runner 后，现有 audit verifier、prepared
  root 与 NRIR-1—26 artifact 均保持可 replay。数值或身份门禁失败时必须回退 audit mode或拒绝，
  不允许静默输出 production verdict。

## Links

- changelog: `gemini_doc/BOUNDFLOW_PRODUCTION_PREPARED_VERIFIER_V1_CHANGELOG_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_TYPED_MULTIPASS_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
