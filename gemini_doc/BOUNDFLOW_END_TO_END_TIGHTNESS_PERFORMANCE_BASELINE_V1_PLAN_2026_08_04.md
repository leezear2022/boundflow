---
status: completed
updated: 2026-08-04T02:41:43Z
type: plan
topic: boundflow
slug: end-to-end-tightness-performance-baseline-v1
stage: s01
---

# End-to-End Tightness and Performance Baseline v1 Plan

## Goal

- 建立 NRIR-14 后第一个不可粉饰的真实端到端诊断基线：同时量化固定 ResNet 九子句的
  proof tightness、candidate/queue/verdict wall time、validation-mode 双执行成本，以及
  external αβ-CROWN intermediate semantics 是否贯穿 optimized queue。
- 基线只用于定位和选择优化，不自动构成 ASPLOS performance claim；只有预注册重复、
  公平 production path 与同算法/竞品口径满足后才允许升级。

## Scope

- 固定输入继续使用 VNN-COMP 2021 ResNet-2B `prop_0_eps_0.008`、既有 model/VNNLIB/
  intermediate payload digest，禁止按结果换 workload。
- tightness 至少比较：external initial-CROWN、native local-IBP-intermediate queue、native
  external-intermediate queue；逐 clause 记录 lower、concrete best、threshold、proof gap、
  verified/unknown。
- timing 分离 setup、candidate search、optimized queue、verdict 与 complete query；明确区分
  audit/validation mode（scheduled result + selected native re-execution + hashes）和未来
  production mode，禁止把双执行路径当作最终系统性能。
- CPU 先产生 reproducible diagnostic artifact；CUDA/最快竞品端到端计时属于后续正式门禁，
  当前设备不可用时必须 fail closed。

## Tasks

1. [x] 冻结 benchmark schema、source identity、phase boundaries、warmup/group/repeat/order 与
   correctness gates；所有 latency comparison 至少 3 个独立 group。
2. [x] 增加 external-intermediate optimizer bridge：local interval environment 保留，六组
   external ReLU pre-activation intervals 经 exact ordinal/shape/dtype/hash binding 后进入
   optimizer Plan、state scope、selected native compiler 和 queue child batches。
3. [x] 增加 adaptive α initialization（与 frozen initial-CROWN lower-slope policy 对齐）；默认
   constant 初始化与 NRIR-10—14 hash/artifact identity 保持不变。
4. [x] 实现 phase/tightness runner；交替 local/external variant 顺序，输出 raw samples、median/p90、
   clause rows、execution-mode disclosure 与 manifest/replay。
5. [x] 用 baseline 决策下一优化：若 external bridge 关闭主要 proof gap，优先 dynamic optimizer/
   branching；若 wall time 主要为 compile/hash/re-execution，先做 prepared production fast path；
   不得未测量就同时旋转两条路线。

## Validation

- external-intermediate plain-CROWN control 必须继续与 frozen external lower allclose、sign 9/9。
- default local path 的 NRIR-10—14 frozen artifacts/hashes 与全量测试不得漂移。
- external path 必须在 objective、split、intermediate source/hash 或 tensor schema 篡改时
  fail closed；child 仍只允许 monotonic-refinement warm initialization。
- artifact replay、focused/full pytest、Black、Mypy、Pylint、`git diff --check`、
  `dol validate` 与 `dol lint --soft` 全过。
- 任何 speedup claim 必须满足 DocOps `runs>=3`；CPU audit timing 默认只称 diagnostic。

## Rollback

- external semantics 为 opt-in typed context；失败时保持 NRIR-14 local path 与 artifact 不变，
  状态仍为 9/9 unknown，不回写或删除失败 baseline。

## Links

- predecessor: `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_PLAN_2026_08_04.md`
- changelog: `gemini_doc/BOUNDFLOW_END_TO_END_TIGHTNESS_PERFORMANCE_BASELINE_V1_CHANGELOG_2026_08_04.md`
- roadmap: `gemini_doc/asplos_execution_memo_v1_0.md`
- artifact: `artifacts/end-to-end-tightness-performance/vnncomp21-resnet2b-prop0-cpu-v1/`
