---
status: implemented-pending-clean-source-formal-run
updated: 2026-08-24T00:35:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b2-b2-5-formal-microphysics-implementation
stage: s01
---

# FSG4/B4-B2 B2-5 Formal Microphysics Implementation

## Scope

本轮实现B2-5正式测量基础设施，不修改B2-0—B2-4语义或冻结12项candidate ledger：

- production P-anchor public-PyTorch sparse reconstruction + lower region + first-order VJP baseline；
- sparse-source TIR custom-autograd wrapper-inclusive candidate；
- 真实CUDA kernel静态inventory，明确`module_call_count`不等于kernel count；
- 12项独立calibration raw与winner derivation；
- S/P各5个独立correctness worker；
- 6个`AB/BA/AB/BA/AB/BA` timing worker，每侧10 warmup、30 measured pairs；
- CUDA event timing、allocated/reserved、GPU temperature/power/clock、semantic/cache/fallback证据；
- source/code/capture绑定artifact、root replay、TIR/device-source独立重编译；
- 8类outer-resigned tamper probe。
- clean-source检查只豁免DocOps hook自动追加的`.docops/ev.jsonl`事件；任何其他tracked diff或
  staged diff仍fail closed。

## Development Evidence

- wrapper baseline与P0 TIR：12,810元素，max diff=`7.152557373046875e-07`，allclose/sign exact；
- generated CUDA真实kernel=`3 forward + 3 backward`；module call仍为`1/1`；
- shared-memory/vector/half token均为0；
- 12项calibration开发跑完成，winner ordinal=`11`，median约`1.51104 ms`；
- 单worker开发测量显示v1明显慢于public-PyTorch baseline，但该数字不是clean-source formal claim。
- 首次formal tamper probe有7/8拒绝，暴露`--no-recompile`路径未把timing worker module hash与
  calibration winner交叉绑定；已补schedule/module双绑定与专项回归，旧artifact降级为失败证据并
  从新clean source重生成，禁止沿用旧manifest形成结论。

## Boundary

当前只表示runner/artifact/replay/tamper实现候选完成。必须提交clean source后生成正式artifact，才能按
冻结`1.05x`/bootstrap/worst/memory门禁关闭B2-5。若v1物理NO-GO，仅关闭当前6-kernel/12-schedule
实现，不否定后续CIBC-parity horizontal fusion/autotuning v2。
