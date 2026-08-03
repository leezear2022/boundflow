---
status: completed
updated: 2026-08-04T08:05:00+08:00
type: changelog
topic: boundflow
slug: BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1
stage: s01
---

# Native Alpha/Beta Optimization State v1 Changelog

## Summary

- NRIR-10 在 NRIR-9 合并后启动；目标是把 legacy alpha/beta optimizer output、beta split
  constraint 与 warm-start validity 接入 native compiler stack。

## Changes

- 创建执行计划并冻结“不把 runtime optimizer loop 冒充 compiled optimizer”的完成边界。
- Bound IR 新增 optimized ReLU attrs 和 7-port contract：四元 affine state 加 exact split/alpha/
  beta typed inputs；alpha/beta shape/dtype/device/range/content/version/linkage 全部 fail closed。
- plain-CROWN builder 以 additive 参数生成 `ALPHA_BETA_CROWN` module；默认无优化状态路径的 API/
  hash 保持兼容。structured rewrite 保留三类 ReLU side inputs。
- reference interpreter 和 Task executor 实际以 alpha 替换 ambiguous lower slope，并把
  `-beta * split` 加到 lower dual coefficient；Plan capability/workload/provenance 显式声明
  alpha/beta state 与 runtime-owned optimizer control flow。
- 新增 scope/state/policy hash、exact/refinement/rejected warm-start classifier；只有 exact same
  scope 可复用 exact state，单调新增 split 只可作为 alpha/beta initialization。
- 新增 optimizer adapter、五层 IR compile/execute wrapper、toy tests、fixed ResNet artifact runner、
  manifest、replay 与 payload/warm-start/IR-hash/claim tamper tests。

## Validation

- fixed ResNet 首个 widest branch 为 ReLU input `31`、neuron `93`；parent→child 为 monotonic
  refinement，parent/child state hash 独立。
- 6 ReLU 对应 18 个 state inputs，加 objective 共 19 inputs；source/execution optimized ReLU ops
  均 6，Task/Launch/trace events 均 21。
- native/legacy lower、upper max diff 均 `0.0`；beta sum `0.04999999701976776`，相对 zero-beta
  lower improvement `0.34039306640625`；所有 10 个 compiler layer hash 随 beta payload 改变。
- artifact generate/replay exit 0，evidence hash
  `302f536685885e75248582698589d49f667d7709ca3258c043310e02278e6884`。
- 聚焦 `50 passed`；全量 `591 passed, 37 skipped`；Black/Mypy clean、Pylint `10.00/10`、
  diff check 通过。

## Decisions

- parent optimized state 在 child split refinement 下只能作为合法初始化，不能作为 child exact state。
- beta 必须被 native lower-dual 方程实际消费；只增加 hash/capability 不算完成。

## Follow-Ups

- 下一门禁为 native alpha/beta optimizer-step Task/Schedule control v1：把 iteration/gradient/update
  state transition 变成可审计控制程序；完成前不得把 frozen-state execution 写成 compiled optimizer。

## Links

- plan: `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_PLAN_2026_08_04.md`
