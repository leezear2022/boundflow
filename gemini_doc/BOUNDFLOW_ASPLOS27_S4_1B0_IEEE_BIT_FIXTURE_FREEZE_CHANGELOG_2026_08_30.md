# S4-1B0 IEEE fixture freeze 修改记录

date: 2026-08-30
stage: s04
performance-claimed: false

## 改动

- 新增 machine-readable IEEE raw-bit fixture；
- 冻结 16 个 pack 与 16 个 select case；
- 冻结两个 midpoint reassociation 反例和 shape guard 期望；
- 记录当前 GPU/TVM observation 与 future production inventory边界；
- 新增 fixture 消费与版本升级规则。

## 边界

- 仅新增 `gemini_doc` 设计资产；
- production、tests、scripts、artifact 均未修改；
- S4-1B0 implementation/formal/timing仍关闭；
- 没有性能或 production correctness claim。
