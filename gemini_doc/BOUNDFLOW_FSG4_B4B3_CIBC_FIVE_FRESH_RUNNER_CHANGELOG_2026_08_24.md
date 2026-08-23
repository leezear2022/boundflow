---
status: implemented-b4-b3-cibc-five-fresh-runner-pending-formal
updated: 2026-08-24T04:25:00+08:00
type: changelog
topic: boundflow
slug: fsg4-b4b3-cibc-five-fresh-runner
stage: s01
---

# FSG4/B4-B3 CIBC Five-fresh Runner Changelog

新增5个独立进程的B3/B4-B3 semantic pair artifact/replay：冻结source code blobs、source capture、
model、B4-B1 reference manifest、`BC/CB/BC/CB/BC`顺序、13项terminal lower/α/β metrics、
evaluation-0 local parity及exact-call receipt。root replay从raw重算全部门禁并校验manifest/hash链。

wall timing只以`timing_diagnostic_only=true`披露，不进入本轮判定；five-fresh通过后只开放另行预热、
交错、重复的累计core timing。

下一步：提交clean source，然后生成正式artifact并root replay。

实现验证时发现fresh subprocess若覆盖`PYTHONPATH`会丢失conda activation注入的TVM Python路径；
runner已改为在继承值前置repo root，保持进程隔离且不破坏TVM环境。失败发生在任何raw worker写出前，
不构成部分证据。
