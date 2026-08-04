---
status: completed
updated: 2026-08-04T21:16:26Z
type: plan
topic: boundflow
slug: BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1
stage: s01
---

# Objective Branch Whole Query Formal v1 Plan

## Goal

- 把 NRIR-39 已通过的 objective-bound-impact branch 接入真实 nine-clause multi-clause anytime runtime，
  在同一 global 60 秒下做 three fresh process formal，回答 tightness 收益能否覆盖 branch scoring 成本。
- 同时保留 widest NRIR-37 frozen formal 作为既有 production control；本阶段不调 branch policy、top-k、
  slice、optimizer、refinement、cache 或 deadline。

## Scope

- 基线：`main@331086d` / NRIR-39 pilot hash
  `dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`；widest formal hash
  `9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`。
- workload 固定 VNN-COMP 2021 `cifar10_resnet:000` property 0；CPU、8 torch threads、3 个 fresh
  subprocesses、每个 whole query 单一 60 秒 global start。
- floor、rank `[2,3,4,5,0,8,6,7,1]`、selected `[2,3]`、dynamic equal-remaining allocation、steps5、
  cap128 ancestral refinement、shared cache、best-first queue、31/depth4 safety cap 与 atomic sibling commit
  固定；唯一算法变量为 widest→NRIR-39 frozen objective branch policy。
- objective scoring 必须计入 slice/global wall time；不允许 logical clock、不允许在计时外预计算 control/
  branch scores、不允许为每 clause 重置 deadline。
- 不修改 NRIR-39 frozen files/artifact；新增 raw production composition 层与 formal runner。
- 不形成 speedup、GPU、competitor、multi-workload 或 ASPLOS-ready claim。

## Tasks

- [x] 新增 objective-branch multi-clause production composition：直接执行 branch-aware shared queue，不为
  fixed-budget comparison 额外运行 widest control；保留 exact branch evidence 与 source lineage。
- [x] 新增 three-process worker/formal artifact，逐轮记录 floor、dispatch/slice/cutoff、accepted nodes、
  branch execution count、cache、active frontier、aggregate 与 final nine-clause verdict。
- [x] replay 必须从 payload 独立重算 rank/allocation/original ordinals、atomic pairs、branch coverage、cache
  ownership、aggregate 与 gate；同步重哈希 tamper 仍 fail closed。
- [x] 按下述冻结门禁关闭为 production-admission GO/NO-GO，不根据结果调常数。

## Validation

- correctness gate（三轮都必须成立）：floor 完成 9/9 original clauses；rank/selected 固定；aggregate 保留
  9 个 ordinals；所有 committed queue、branch Plan/Task/Schedule/selection、refinement lineage 和 sibling
  group valid；恰好一个 query-owned template miss，其余 exact hits；final sound。
- production-admission GO gate（三轮都必须成立）：
  1. clauses 2/3 各至少提交 31 nodes/15 sibling pairs；
  2. 两条 worst-active lower 相对 frozen widest 各至少改善 `+1.0`；
  3. whole cooperative elapsed 不超过 `70 s`（允许 deadline 前开始的 atomic pair 收尾）；
  4. 不出现 partial pair、deadline reset、cache recompile 或 branch evidence omission。
- correctness 失败为 validation failure；correctness 成立但任一 production gate 失败为
  `VALIDATED-NO-GO`；全部成立才是 objective-branch global-budget `VALIDATED-REDUCED`。
- artifact generate/replay/tamper、targeted/full pytest、Black、mypy、Pylint、`dol validate` 与
  `dol lint --soft`。

### Closure

- 三轮 correctness 全过，rank/selected 均固定；节点为 `[[29,23],[29,21],[29,21]]`，branch evidence
  与 accepted nodes 一一对应，每轮恰好一次 template miss。
- clauses 2/3 worst-active lower 分别稳定在 `-48.315041` 与 `-43.299690/-44.731468`，既没有达到
  `31/15` coverage，也没有超过 frozen widest `+1.0`；production gate 三轮均失败。
- whole cooperative elapsed=`[63.357098,63.161128,62.485366] s`，formal hash=
  `d69b56d4d82ad5bf8d30883258c15a39e5a45f1fac9dbc8eb35e91fda9f6a492`。本阶段以 objective-branch
  global-budget `VALIDATED-NO-GO` 关闭；fixed-budget NRIR-39 结论保留，但不得升级为 production claim。

## Rollback

- 删除本分支 additive NRIR-40 文件即可回到 `main@331086d`；NRIR-39 fixed-budget evidence 不变。

## Links

- changelog: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1_CHANGELOG_2026_08_05.md`
- predecessor: `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SHARED_EVALUATOR_V1_PLAN_2026_08_05.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
