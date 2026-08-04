---
status: validated-reduced
updated: 2026-08-04T07:24:00Z
type: changelog
topic: boundflow
slug: ancestral-constraint-refinement-v1
stage: s01
---

# BoundFlow Ancestral Constraint Refinement V1 Changelog

## Summary

- NRIR-22 已完成：ancestral constraint carry-forward 以一等 Plan/Task/Schedule source 输入接入
  per-child refinement；固定 ResNet clauses 0/1 的 worst depth-2 leaf lower 相对 independent
  提升 `+73.615173/+75.022095`，相对 root-global 提升 `+72.767212/+74.085449`。

## Changes

- `NativeIntermediateRefinementPlanIR` 新增条件序列化的 source constraints、source refinement
  Plan 与 source semantic trace hash；materialize-forward Task/Schedule 显式消费
  `refine.source_intermediate_constraints`。
- compiler 只接受已验证的 parent `NativeIntermediateRefinementExecution`，不接受裸 mapping；Program
  重算 local/constrained forward 并验证完整 interval state，Execution action trace 绑定 source hash。
- optimized queue 新增 `ancestral_constraint_carry_v1` strategy；root 无 source，child record 的
  source parent/final/Plan/semantic trace 必须与已完成 parent 一致，consumption 固定为
  `sound_constraint_only`，exact-state reuse 保持 false。
- 保留 `independent_exact_split_v1` 与所有默认旧 payload：无 source 时 Plan/Task/Schedule hash 和
  NRIR-21 independent queue 序列化结构不增加字段。
- 新增三模式 generate/replay runner、artifact contract/tamper tests，并同步权威文档。

## Validation

- focused/cross-generation artifact：`42 passed`。
- fresh generate：status=`validated_reduced`，evidence hash=
  `72c0c2a66b82cea425bf7486817c0ce39ae186ef2961cc1271acb31cb7a31b6f`。
- source-to-IR fresh replay 通过；ancestral semantic result hash=
  `67d1e7a733ff4941a6e4bd7136c9d7661832fa3266b7f1edd3dd1b6439f9ce95`、
  `d6c50d902fa7fb3ad70fb68f55248fdb4e5ef844efb8cd9d21b31f806f22d11d`；
  frozen root-global/independent semantic hashes保持精确不变。
- 全量回归：`758 passed, 37 skipped`；37 个 skip 均为 CUDA/TVM 环境边界。
- targeted Mypy clean、Pylint `10.00/10`、Black 与 `git diff --check` 通过。

## Decisions

- 不复用 parent refined bounds 作为 child exact result；它只作为 child local forward 的 sound
  intersection constraint，之后必须重新 propagation 与 objective-directed refinement。
- 保留 NRIR-21 independent strategy 作为同代码路径 baseline，不改写其 NO-GO 工件。
- clauses `0/1` 三模式 worst leaf：root-global=`-413.739044/-591.944275`，independent=
  `-414.587006/-592.880920`，ancestral carry=`-340.971832/-517.858826`；三模式 root lower
  均为 `-417.292480/-602.551392`。
- 结果验证了原因归因：child exact recomputation 本身没有错，问题是丢失祖先 selected-CROWN
  tightening；把 proven constraints 先与 child local forward 交集并传播，能够恢复且显著超过
  root-global reuse。该结论只覆盖固定 bounded-tree tightness。

## Follow-Ups

- 完成 DocOps closure、PR 与 merge。
- 下一门禁不再重复 intermediate shortlist plumbing；应测更深/更多 hard clauses 的收敛曲线，
  并决定推进 complete closure（更深 BaB/动态 budget）还是回到可用 CUDA 主机上的公平 E2E。

## Links

- plan: `gemini_doc/BOUNDFLOW_ANCESTRAL_CONSTRAINT_REFINEMENT_V1_PLAN_2026_08_04.md`
- predecessor: `gemini_doc/BOUNDFLOW_PER_CHILD_OBJECTIVE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
- artifact: `artifacts/ancestral-constraint-refinement/vnncomp21-resnet2b-two-clause-cpu-v1/`
