---
status: externally-approved-b2-0
updated: 2026-08-23T03:10:59Z
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-0-identity-tir
stage: s01
---

# FSG4 B4-B2 B2-0 lowering receipt identity TIR

## Summary

- B4-B2的B2-0 ABI门禁已在RTX 4060 Laptop GPU上通过，状态=
  `VALIDATED-B4-B2-B2-0-ABI-PROBE`；
- 本轮只证明first-class lowering/receipt、identity CUDA/TIR、DLPack/current-stream与一阶custom
  autograd机制，不证明region数学、融合或性能。

## Changes

- 新增`DifferentiableLowerTIRTemplateV1/InstanceV1/ScheduleV1/ModuleReceiptV1/LaunchReceiptV1`：
  canonical JSON、stable hash、round-trip parse、static/dynamic ownership与fail-closed验证；
- 新增双symbol identity TIR：forward与backward均由独立CUDA PrimFunc执行，schedule冻结为1D
  thread binding、无workspace、candidate ordinal=0；
- 新增PyTorch custom `autograd.Function`：当前stream上forward/backward各恰一次launch，higher-order
  gradient显式拒绝，无eager backward/fallback；
- 新增显式module cache与receipt：绑定unscheduled/scheduled TIR、device source、TVM/FFI Git SHA、
  Torch版本、symbol inventory与cache key；
- 新增DLPack pointer、stream、alias、cache与launch ledger；功能默认关闭且
  `performance_claimed=false`；
- 新增独立GPU probe脚本和12项正/负向测试。

## Validation

- 独立probe=`probe-passed`，GPU=`NVIDIA GeForce RTX 4060 Laptop GPU`，sm_89；
- template=`f927994b5dd02dd37269aa956d4a59645712a5dd451d52aea4245114ac2ea0fe`；
- schedule=`3bc85e3022e5262884bae856421c7c3be2d1968110c55bb340b6a3c3a1dd1a42`；
- module receipt=`ba765577a70b7a1cab9dbfc0b51861663767be38f02948b84d3b22bc4cfc1474`；
- cold cache miss→warm hit；forward/backward=`1/1`，fallback/eager backward=`0/0`；
- 四个DLPack round-trip pointer exact，forward/backward current stream exact，output/input-gradient
  均无alias；first-order output/gradient bit-exact；
- targeted=`12 passed`；B4-B1+targeted=`44 passed`；B4-B相关=`66 passed`；
- full=`1426 passed, 3 skipped, 6 warnings`（3 skip均为既有环境边界）；
- `rebuild_tvm.sh`完成且增量树无需重编译；Black、4-file Mypy、5-file Pylint 10.00、
  `git diff --check`通过。

### External audit

- `APPROVE`，0 blocker/0 major、2 minor+3 info；
- auditor现场重跑GPU probe，三项receipt hash逐位一致；
- 最终状态=`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-0-ABI-PROBE`；
- audit：`gemini_doc/external_audit_b4b2_b2_0_identity_tir_probe_2026_08_23.md`。

## Decisions

- B2-0通过，解除“ABI失败即停止”门禁，只开放B2-1 S-anchor dense forward/backward correctness；
- identity ABI不得进入timing；不得把identity copy描述成region TIR或CUDA融合；
- dense B2-1仍只证明语义机制，sparse-source B2-2之前不得主张消除dense materialization。

## Follow-Ups

- 下一唯一动作：B2-1 S-anchor (`semantic-active-beta-gemm-14`) dense semantic TIR
  forward/backward，5个B4-B1 raw instances correctness；
- B2-2 sparse-source、P-anchor、schedule timing、B4-B3 exact-call继续关闭。

## Links

- plan: `gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`
- IR: `boundflow/ir/differentiable_lower_tir.py`
- backend: `boundflow/backends/tvm/differentiable_lower_identity.py`
- runtime: `boundflow/runtime/fsg4_b4b2_identity_tir.py`
- probe: `scripts/run_fsg4_b4b2_identity_tir_probe.py`
- tests: `tests/test_fsg4_b4b2_identity_tir.py`
