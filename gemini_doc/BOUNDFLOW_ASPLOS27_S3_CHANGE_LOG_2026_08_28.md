---
status: active
updated: 2026-08-28T13:20:00+08:00
type: change-log
topic: boundflow
slug: asplos27-s3-change-log
stage: s03
---

# ASPLOS'27 S3 修改记录

## 2026-08-28：冻结 S3 optimizer/runtime 大批次

- 新增 S3 预注册，固定 P-anchor 10/9 local wrapper 的 N/D/P 三方语义与六全排列性能协议；
- 明确 direct custom VJP，不复用历史 autograd Function registry；
- 冻结 `3.00x/2.50x/1.50x` research gate 与 reduced/no-go 分支；
- 明确 10/9 仅为冻结 artifact trajectory，host policy 保留，禁止升级为通用 optimizer IR；
- 保留现有 S1+S2 pending external exchange，外审按用户要求延后到下一轮；
- 当前只完成预注册，代码、性能与 claim 均未完成。

### 验证

- 文档格式与 DocOps lint：待运行；
- 代码/性能：明确 deferred，待后续提交。

