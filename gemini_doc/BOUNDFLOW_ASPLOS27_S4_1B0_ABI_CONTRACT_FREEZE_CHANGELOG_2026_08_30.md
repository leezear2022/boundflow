# S4-1B0 ABI contract freeze 修改记录

date: 2026-08-30
stage: s04
performance-claimed: false

## 改动

- 新增 machine-readable build/schedule/compiled/module/cache ABI；
- 冻结prepared probe descriptor和warm launch receipt；
- 绑定 IEEE fixture 与 negative contract SHA256；
- 明确generic storage公式、production fixture账和formal observation边界。

## 边界

- 仅新增设计文档；
- production/test/script/artifact均未修改；
- implementation/formal/timing/performance仍关闭。
