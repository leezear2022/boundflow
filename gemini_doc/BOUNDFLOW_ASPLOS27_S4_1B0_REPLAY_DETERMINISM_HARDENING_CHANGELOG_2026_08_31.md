---
status: replay-hardening-ready-pending-publication
date: 2026-08-31
stage: s04
performance-claimed: false
---

# S4-1B0 replay fresh-process determinism 加固记录

## 发现

formal 工具链第一次临时演练中，`coefficient_binary_and_index` 只翻转首个 coefficient 的最低有效位。
该位翻转没有改变 coefficient 符号，因此 selector 与 selected output 仍满足逐元素派生语义；同时旧 replay
没有再次绑定 5 个 positive fresh process 的 sidecar 全同关系，导致该攻击被接受，演练结果为 `9/10`。

这不影响 endpoint 实现正确性，也没有产生正式 artifact，但说明 replay 少了一条生成器已经执行的
fresh-process determinism 门禁。

## 修正

- replay 新增 5 个 positive sidecar SHA256 全同检查；
- summary 的 `positive_sidecar_sha256` 和 `positive_sidecar_byte_count` 必须回绑 raw；
- coefficient 篡改改为翻转 IEEE-754 sign bit，使该攻击明确触发 selector 派生语义不一致；
- coherent full-resign E0 边界、timing/performance false 均不改变。

正式 artifact 仍未生成。修正提交并推送后，必须重新通过 formal activation gate，并从空临时目录
完整演练 11-process generation、stdlib replay 和 10 类篡改；不得沿用第一次演练数据。
