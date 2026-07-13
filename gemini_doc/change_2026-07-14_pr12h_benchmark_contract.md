# 2026-07-14：PR-12H benchmark contract freeze

## 目标

冻结 PR-12G reduced evidence，并把 kernel、region-runtime、end-to-end final-bound 三层测量从
文档约定提升为机器可读合同，防止后续把不同 inclusion/allocation/同步口径直接比较。

## 修改

- tag `44f87ae` 为本地 annotated tag `pr12g-validated-reduced`；
- 新增 `boundflow/benchmarks/contracts.py`，定义三个 contract、validation、JSON payload 与 hash；
- 新增 `docs/pr12_benchmark_contract.md` 作为规范；
- 现有 fused-sanity 明确标记为 allocation 不公平的 legacy kernel calibration；
- 现有 runtime-Pareto 明确标记为 Planner/region matching 不在 timed call 的 legacy
  final-bound candidate evidence；
- 新增中长期完成计划和跨会话执行状态，PR-13 继续阻塞。

## 判定边界

PR-12H 不生成新性能 claim，也不修改 canonical3 数值。它只冻结证据解释边界：历史结果继续
有效，但不能被标为新三层合同 compliant。PR-12I 必须生成新的 structured eager/TVM-unfused
公平 baseline 工件。

## 验证

```text
PR-12H/legacy-runner focused:  7 passed
全量：                           321 passed、1 skipped
mypy：                          5 source files success
pylint：                        3 core/script files 10.00/10
Black / git diff --check：      通过
```

全量的 9 条 warning 均来自上游 deprecation/future warning；唯一 skip 是 TVM 已可用时避免重复
编译的 allow-no-tvm smoke。
