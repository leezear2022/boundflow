---
status: implemented-b4-c0-core-timing-runtime-prep
updated: 2026-08-24T05:05:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4c0-core-timing-runtime-prep
stage: s01
---

# FSG4/B4-C0 Core Timing Runtime Prep Changelog

为累计core计时移除两类只属于correctness observer的hot-path同步：

- dense exact executor不再对每个evaluation的7个live tensors逐项执行GPU `isfinite().item()`；
  finite所有权继续由上游optimizer state与正式semantic runs保证，shape/dtype/device/layout/stream仍
  fail closed；
- exact observer新增`record_local_parity`开关，correctness默认保持开启，timing worker关闭，避免在
  evaluation 0中途同步。

这不是性能结论，只是把measurement perturbation移出被测路径。下一步实现预热、交错、30-group、
6-fresh累计core worker/artifact。
