---
status: validated-internal-pending-external-audit
updated: 2026-08-23T07:00:00Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-2-sparse-linear-tir
stage: s01
---

# FSG4/B4-B2 B2-2 S-anchor sparse-source Linear TIR

## Summary

- B2-2 已完成 S-anchor sparse-source fused forward/backward correctness，状态=
  `VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；
- 5 raw、20 metrics、31,590 elements 全部 allclose/sign exact，max diff=
  `8.642673492431641e-07`；
- 本轮不计时，不开放 P-anchor、B2-3/B2-4/B2-5 或 B4-B3。

## Changes

- 新增 first-class sparse Template/Instance/Schedule/Module/Projection/Launch receipts；
- 将 27 个 alpha feature index、6 个 beta location/sign 固化为 PlanTemplate 常量并纳入 template/
  cache hash；
- runtime 只接收 compressed alpha=`[6,27]` 与 compressed beta=`[6,1]`，不接收或构造 native dense
  alpha/beta；
- CUDA/TIR forward 直接执行 compressed-alpha sign-select、active-beta pre-add 与 Linear contraction；
- CUDA/TIR backward 直接返回 compressed alpha/beta gradient，不生成 native dense gradient；
- projection receipt 绑定 native oracle、compressed gather、candidate gradient 与 scatter-back hash，
  并区分 mapping exact、数值 tolerance pass 与 nonzero sign exact；
- scheduled TIR 只保留 `adjoint_matmul`、`output_bias_delta` workspace；禁止的 `native_alpha`、
  `native_beta`、`relu_lower_a`、`scaled_a` global workspace count=`0`；
- 新增 11 项测试，包含 mapping/ABI/receipt 篡改、dtype/device/nonfinite/range、A==0、clamp endpoint、
  custom stream 异常恢复、fallback 计数、higher-order 与 projection 门禁。

## Validation

- runner status=`validated-b2-2-sparse-source-linear-correctness`；
- run/metrics/elements=`5/20/31,590`，max diff=`8.642673492431641e-07`；
- template=`adddcb6a5daa7ebf8a8dcc34cc0e08b1f2a30dd6ad43503f2ab7f3df2b9bf56f`；
- schedule=`b8fe0a7d2f859ada4f1bf3293b80ba6783003861ed66a16ad0a5542cc2350d57`；
- module receipt=`7f6ab5cbfceaaa8b29529d0624e9238f1f52386ad1279ee24d52b31d6f842679`；
- cache=`miss,hit,hit,hit,hit`，forward/backward=`1/1` per run，fallback/eager=`0/0`；
- projection 每 run 均为 alpha/beta mapping exact、numerical pass、nonzero sign exact、unowned native
  zero exact；
- targeted B2-0/B2-1/B2-2=`34 passed`；B4-B related=`88 passed`；
- full=`1448 passed, 3 skipped, 6 warnings in 474.91s`；3 skip 均为既有环境边界；
- `test_env=3 passed`；TVM rebuild 完成且增量树无需重编译；Black、Mypy、Pylint 10.00、diff
  check 全过。

## Decisions

- B2-2 S-anchor sparse-source correctness 内部门禁通过，只开放独立外审；
- `sparse_source_admitted=true`只表示本 ABI 直接消费 compressed state，不是 performance admission；
- `performance_claimed=false`，没有 timing API、speedup、memory ratio、P-anchor 或 same-solver claim；
- 外审批准前 B2-3/P-anchor、B2-4/B2-5、B4-B3 全部关闭。

## Links

- plan: `gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`
- IR: `boundflow/ir/differentiable_lower_sparse_linear_tir.py`
- backend: `boundflow/backends/tvm/differentiable_lower_sparse_linear.py`
- runtime: `boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py`
- runner: `scripts/run_fsg4_b4b2_sparse_linear_tir_correctness.py`
- tests: `tests/test_fsg4_b4b2_sparse_linear_tir.py`
- audit handoff: `gemini_doc/BOUNDFLOW_FSG4_B4B2_B2_2_EXTERNAL_AUDIT_HANDOFF_2026_08_23.md`
