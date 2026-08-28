---
status: active-documentation-only
date: 2026-08-28
type: change-log
topic: boundflow
slug: asplos27-s4-change-log
stage: s04
performance-claimed: false
---

# ASPLOS'27 S4 修改记录

## 2026-08-28：完成same-solver接入前的production coverage普查

- 正式创建并交付DocOps exchange `asplos27-s3-optimizer-runtime-20260828`，状态=
  `ready_for_audit/r001`，exchange validate PASS；
- 亲读RVIR live exact-call、native optimizer、pre-state、terminal export、KFSB与atomic commit路径；
- 从冻结production optimizer raw独立统计每step为6条α、8,496个α元素，以及1条`[6,1]` active β；
- 核对S3只动态拥有P-anchor `alpha/%2Finput-24/%2F49:[2,1,6,86]`的1,032元素，占α state元素
  `12.1468926554%`，且其P β为`[6,0]`；
- 确认S3只返回terminal P α/lower，不足以直接组装whole-core lA/intermediates/KFSB/12-path commit；
- 冻结S4为all-mutable-state compiled evaluation → existing host production policy → terminal handoff →
  existing KFSB/commit的组合路线；禁止P-only whole-core包装；
- 新增S4-0—S4-4 correctness/replay门禁与21类fail-closed reason；timing另设S4-P，当前关闭；
- 本轮只有文档/流程变更，无S4代码、GPU运行或性能claim。

### 验证

- production raw inventory stdlib解析：10/10 steps、六α、8,496 α元素、P=1,032元素、active β path=
  `beta/%2Finput-28/0/value`，PASS；
- RVIR native optimizer/mutation/live return、FSG4 same-solver合同与S3 pipeline targeted：`44 passed`；
- S4 source mapping与`git diff --check`：PASS；
- DocOps change/validation与lint：最终落账后执行。

## 2026-08-28：完成六路α/active β compiled VJP可行性审计

- 从冻结capture逐site恢复六个logical shape、compressed width、β owner及active β location；
- 对照R3 bounded-arena trace，确认site31/25分别有exact sparse Linear/Conv单siteTIR，其余site不能直接串
  现有B4-B2 wrapper；
- 亲读R31B1、R31B2、D1C、D2B与S2/S3代码，确认整图forward早已消费六α与active β，当前缺口仅为P-only
  gradient output ABI；
- 冻结all-state VJP物理方案：一次完整sign pass、一次六site effective-value pass、一次coefficient重算并在每个
  ReLU即时压缩gradient；
- 复用D1C/D2B residual stage scratch导出site25/site19 incoming coefficient，保持跨层saved dense A=`0`；
- 新增独立可行性文档并把S4-1细分为1A ABI、1B effective values、1C emitters、1D evaluator closure；
- 本轮仍无S4代码/GPU执行/性能claim，S3外审门禁不变。
