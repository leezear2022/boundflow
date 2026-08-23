---
status: validated-internal-pending-external-audit
updated: 2026-08-23T04:55:26Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-1-dense-linear-tir
stage: s01
---

# FSG4 B4-B2 B2-1 S-anchor dense Linear TIR

## Summary

- B2-1已完成S-anchor dense Linear/Gemm TIR forward/backward correctness，状态=
  `VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；
- 5份冻结raw、20项metric、36,750元素全部allclose/sign exact，最大差=
  `8.642673492431641e-07`；
- 本轮只证明dense semantic ABI，不计时、不主张sparse-source融合或性能。

## Changes

- 新增S-anchor专属Template/Instance/Schedule/Module/Launch IR，绑定`[6,1,100] @ [100,1024]`、
  active beta、operator bias、双symbol、TVM/FFI commit与完整tensor inventory；
- 新增CUDA/TIR forward：lower sign selection、α clamp、active β signed pre-add、Gemm contraction、
  ReLU/operator bias reduction；
- 新增CUDA/TIR backward：真实output A/bias adjoint输入，直接返回native dense α/β gradient；
- 新增custom autograd与23个DLPack view pointer ledger，forward/backward各恰一次module launch；
- 新增五份fresh runner、11项B2-1测试，并保留B2-0 13项回归；
- 关闭外审minor：fallback/eager改为executor真实计数器；异常路径显式验证device、current stream、
  deterministic policy不漂移。

## Validation

- runner=`validated-b2-1-dense-linear-correctness`，run=`5`、metrics=`20`、elements=`36,750`；
- max abs diff=`8.642673492431641e-07`，allclose/sign exact=`true/true`；
- template=`d96bb8d62eb2e112e4f9ac5e98bc971cb41122cd97273ebb3fc1c4fc5c0a0be4`；
- schedule=`989c3eae7fcefed3a6399b000c51eb222c5e5ba2a31a220ef42db5d86ca5de4b`；
- module receipt=`e99121435e5db022c02f1d1610ffb9d4048397e09168f91f6857e425ad80801a`；
- cache=`miss,hit,hit,hit,hit`；每run launch=`1/1`、fallback/eager=`0/0`；
- α=0/1 clamp端点与A=0离散导数、P-anchor越界、higher-order、resigned receipt、
  custom-stream异常恢复均通过；
- targeted B2-0+B2-1=`23 passed`；B4-B相关=`76 passed`；full=
  `1437 passed, 3 skipped, 6 warnings`；
- `rebuild_tvm.sh`完成且增量树无需重编译；Black、Mypy、Pylint 10.00与diff通过。

## Decisions

- B2-1内部门禁通过，只允许提交外审；
- dense materialization、adjoint/output-bias workspace均显式保留，因此不形成融合或memory claim；
- B2-2 sparse-source fused forward/backward在B2-1外审批准前保持关闭。

## Follow-Ups

- 冻结clean source并生成B2-1外审交接；
- 外审批准后唯一下一工程动作=B2-2 S-anchor sparse-source fused forward/backward；
- probe/raw stdout正式artifact仍按预注册保留到B2-5。

## Links

- plan: `gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`
- IR: `boundflow/ir/differentiable_lower_dense_linear_tir.py`
- backend: `boundflow/backends/tvm/differentiable_lower_dense_linear.py`
- runtime: `boundflow/runtime/fsg4_b4b2_dense_linear_tir.py`
- runner: `scripts/run_fsg4_b4b2_dense_linear_tir_correctness.py`
- tests: `tests/test_fsg4_b4b2_dense_linear_tir.py`
- audit handoff: `gemini_doc/BOUNDFLOW_FSG4_B4B2_B2_1_EXTERNAL_AUDIT_HANDOFF_2026_08_23.md`
