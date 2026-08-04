---
status: completed
updated: 2026-08-04T21:30:14Z
type: plan
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1
stage: s01
---

# Objective Branch Production Cost Attribution v1

## Goal

- 解释 NRIR-39 fixed-31-node 改善与 NRIR-40 global-deadline NO-GO 的表面矛盾，区分两个因果轴：
  objective branch 的 frontier order 是否在相同 accepted-node 前缀仍更强，以及 scoring wall-time 是否
  导致关键最后一个 sibling pair 无法提交。
- 只做归因，不在看到结果后修改 candidate policy、top-k、slice、node/depth、optimizer、refinement、
  cache 或 deadline。

## Scope

- 基线：`main@9befc51`，NRIR-39 pilot hash=
  `dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`，NRIR-40 formal hash=
  `d69b56d4d82ad5bf8d30883258c15a39e5a45f1fac9dbc8eb35e91fda9f6a492`。
- workload 固定 VNN-COMP 2021 `cifar10_resnet:000` property 0 的 clauses 2/3；CPU、8 torch threads；
  control=`widest_unsplit_ambiguous_relu`，candidate=`objective_bound_impact`。
- deterministic prefix 只消费 NRIR-39 frozen 31-evaluation rows，按 parent/child lineage 重建
  `21/23/29/31` accepted-node active frontier；不得重排、插值或从 NRIR-40 结果反推缺失节点。
- wall-time 使用 3 个 fresh subprocesses；每轮对两个 clauses 各执行 frozen widest/objective fixed
  `31/depth4` queue，fresh cache per variant，顺序按 `W→O/O→W/W→O` counterbalance。计时用真实
  `monotonic_ns`，不使用 logical clock；绝对时间只作内部成本归因，`performance_claimed=false`。
- cProfile 只另跑 1 个 diagnostic worker，映射 production exact call path 的 candidate enumerate、child
  materialize、bound evaluation 与完整 branch-program cumulative time；profiled 时间不得混入三轮 wall
  median 或形成系统 speedup claim。
- 不修改 NRIR-39/40 frozen runtime、runner 或 artifact；只新增 attribution IR/runtime/script/test/artifact。

## Tasks

- [x] 新增 first-class attribution Plan、TaskModule、Schedule、prefix row、phase row 与 Decision IR；Task
  顺序固定为 source admission→prefix reconstruction→unprofiled paired execution→diagnostic profile→
  causal decision→emit。
- [x] 从 frozen rows 独立重建两个 policy 在 `21/23/29/31` 的 active set、worst/median lower，验证
  parent closure、odd-node atomic prefix 与 31-node frozen summary exact。
- [x] 运行 3 fresh unprofiled paired workers，记录 source/root parity、31-node coverage、whole/queue/
  source wall time、cache miss/hit 与 execution hashes；再运行独立 cProfile diagnostic。
- [x] 生成 replayable artifact；replay 重算 prefix、三轮 median/MAD、phase share 与 Decision，且同步
  重哈希 source/prefix/phase/decision tamper 仍 fail closed。
- [x] 只按下述门禁选择下一单变量，不根据 timing 数字现场改阈值。

## Validation

- correctness gate：frozen source digest、两条 clause/两 policy/四 prefix 全覆盖；prefix active set 由
  parent lineage 唯一重建；三轮 paired execution 都达到 31 nodes/15 sibling groups，root/source/
  final lower 与 NRIR-39 exact，cache 每 variant 恰好 1 miss，其余 exact hit。
- `frontier_order_retained`：在 clauses 2/3 的 `21/23/29/31` 四个相同节点前缀上，candidate
  worst-active lower 均不弱于 control，且 31-node improvement 保持各 `>=+1.0`。
- `scoring_cost_dominant`：三轮 objective queue wall median / widest queue wall median 在两条 clause 均
  `>=1.20`，且 diagnostic 的 objective branch program cumulative time占 candidate queue wall 至少
  `20%`。两项只决定下一工程方向，不升级 performance claim。
- 两项都成立：下一阶段只允许优化 scorer ownership/复用；frontier gate 失败：冻结 objective branch
  production 路线；frontier 成立但 cost gate 失败：转查 deadline/atomic-tail scheduling，不优化 scorer。
- artifact generate/replay/tamper、targeted/full pytest、Black、mypy、Pylint、`dol validate` 与
  `dol lint --soft`。

### Closure

- `frontier_order_retained=true`：clauses 2/3 在四个前缀的 worst-active improvement 分别为
  `[+2.171364,+2.416264,+2.947929,+2.043362]` 与
  `[+4.988102,+6.255299,+6.350922,+5.641768]`。
- widest/objective queue median 为 clauses 2 `10.515292/18.387675 s`、clause 3
  `10.619606/18.591097 s`，ratio=`1.748660/1.750639`；对应 MAD 为
  `0.020595/0.266792 s` 与 `0.002217/0.242127 s`。
- cProfile 的 branch-program share=`21.9371%/21.9139%`；31 次 branch program 内外实际触发
  341 次 candidate enumeration。两项门禁均成立，Decision 自动选择 `optimize_scorer_ownership`。
- formal hash=`fe67b77197905a8a4d7f92ad5eac686892243dfb0e7d7b7c7434861aaa794834`；本阶段以内部
  causal attribution `VALIDATED-REDUCED` 关闭，`performance_claimed=false`。

## Rollback

- 删除本分支 additive NRIR-41 文件即可回到 `main@9befc51`；NRIR-39/40 代码与 artifact 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
