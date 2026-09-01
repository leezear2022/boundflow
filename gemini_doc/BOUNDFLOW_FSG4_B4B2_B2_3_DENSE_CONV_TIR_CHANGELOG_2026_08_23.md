---
status: validated-pending-external-audit
updated: 2026-08-23T11:33:28Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-3-dense-conv-tir
stage: s01
---

# FSG4/B4-B2 B2-3 P-anchor Dense Conv TIR

## Summary

- 状态=`VALIDATED-B4-B2-B2-3-P-CONV-DENSE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`；
- P-anchor 5 raw、20 metrics、92,190 elements 全部 allclose/sign exact，max diff=
  `2.384185791015625e-06`；
- 本轮只关闭 dense correctness，不计时、不开放 B2-4/B2-5/B4-B3。

## Changes

- 新增 P-anchor first-class Template/Instance/Schedule/Module/Launch receipts；
- ABI 冻结 input/result=`[6,1,16,8,8]`、weight=`[16,16,3,3]`、stride/padding/dilation=
  `(1,1)`、output padding=`(0,0)`、groups=`1`；
- empty compressed beta=`[6,0]`由 template 固化，runtime/TIR 不接收 beta tensor，receipt 强制
  `beta_gradient_present=false`，不以零 tensor 冒充 absent；
- CUDA/TIR forward 实现 ReLU lower slope/intercept、ConvTranspose contraction 和 bias reduction；
- CUDA/TIR backward 直接返回 native alpha 与 incoming-A gradient，包含 Conv adjoint、operator-bias
  VJP 与 selected-intercept 对 incoming-A 的 VJP；higher-order/fallback 均 fail closed；
- scheduled TIR 结构遍历 `Block.alloc_buffers`，只准入 `adjoint_conv=[6,1,16,8,8]` 与
  `output_bias_delta=[6,1]`；不再依赖 scheduled script 子串作为唯一 workspace 门禁；
- 新增 8 项 P-anchor 测试，覆盖 five-fresh oracle parity、receipt round-trip、结构 workspace、
  dtype/device/nonfinite/range/interval、custom stream 异常恢复、fallback、higher-order 与 claim 篡改；
- 补齐 B2-1 dense Linear 的 dtype/device/nonfinite 专项拒绝测试，关闭 B2-2 外审 info finding。

## Correctness Evidence

- runner status=`validated-b2-3-p-anchor-dense-conv-correctness`；
- run/metrics/elements=`5/20/92,190`；
- max diff=`2.384185791015625e-06`，allclose=`true`，sign exact=`true`；
- template=`950f20535ab55120e497401c7d17513c5f2118fd65401e4e87d3a081567c4dc2`；
- schedule=`1de607ad7faf39ff1b45ee81b90013e3cc841c69e97fd3aabba0f135893cc7ec`；
- module receipt=`4511fbc51159cea516e568f025636fa9fee0cf97225f032ddf877f8239dbad79`；
- cache=`miss,hit,hit,hit,hit`；每 run forward/backward=`1/1`、fallback/eager=`0/0`、
  DLPack=`19/19`、beta gradient absent；
- structural workspace check=`true`，observed inventory 与 schedule exact。

## Validation

- B2-0/B2-1/B2-2/B2-3 targeted=`43 passed`；
- B4-B related=`97 passed`；
- full=`1457 passed, 3 skipped, 6 warnings in 473.26s`；3 skip 均为既有环境边界；
- `test_env=3 passed`；TVM incremental rebuild=`ninja: no work to do`；
- Black、Mypy 4 source clean、Pylint=`10.00/10`、`git diff --check`通过。

## Claim Boundary

- 允许的内部 claim 仅为 P-anchor dense Conv forward/backward correctness 待独立外审；
- `performance_admitted=false`、`performance_claimed=false`；没有 timing API、speedup、memory、
  B0 parity、whole-core/query 或 ASPLOS-ready claim；
- 下一唯一动作=B2-3 外部审计；B2-4 sparse-source schedule search、B2-5 formal artifact/timing、
  B4-B3 全部保持关闭。

## Links

- plan: `gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`
- IR: `boundflow/ir/differentiable_lower_dense_conv_tir.py`
- backend: `boundflow/backends/tvm/differentiable_lower_dense_conv.py`
- runtime: `boundflow/runtime/fsg4_b4b2_dense_conv_tir.py`
- runner: `scripts/run_fsg4_b4b2_dense_conv_tir_correctness.py`
- tests: `tests/test_fsg4_b4b2_dense_conv_tir.py`
