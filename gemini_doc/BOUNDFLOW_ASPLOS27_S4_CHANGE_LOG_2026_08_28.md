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

## 2026-08-28：冻结compressed evaluator ABI与terminal handoff

- 纠正α计数口径：六source为8,496 stored元素，lower-only optimizer-active为4,248，preserved direction为4,248；
  P为1,032 stored/516 active，coverage比例不变；
- 确认RVIR native optimizer使用37,464-element dense α和同shape dense β，仅作为provider-independent oracle更合适；
- 冻结S4 candidate直接优化production compressed lower-α/sparse β，terminal才一次性展开dense state供existing
  KFSB/commit；
- 把optimizer接入从“给existing函数塞callback”修正为“抽出sealed policy driver，只允许native oracle与compiled
  candidate两个exact evaluator”；
- 独立复核B4-A terminal handoff六lA共37,464 float32/149,856 bytes，handoff count=1、rerun=0；
- 冻结effective-value/terminal-lA phase-tagged slot复用方案与10类新增fail-closed reason；
- 明确existing KFSB仍执行3次batch-24 child CROWN、共72 child lower，S4-P必须单列，不能隐藏为host overhead；
- 新增S4 evaluator ABI/terminal handoff实施蓝图；仍无代码、GPU执行或性能claim。

### 验证

- stdlib-only重算：六α`stored/active/preserved=8496/4248/4248`，P=`1032/516`，B4-A terminal lA=
  `37,464 float32 / 149,856 bytes`，handoff/rerun=`1/0`，KFSB=`3 candidates × batch 24 = 72 child lower`，PASS；
- RVIR pre-state、atomic copy-out、native KFSB、B4-A terminal handoff与S3 optimizer targeted：最终
  `24 passed in 11.17s`；
- 验证过程先后暴露两项shell装载问题：`source env.sh; pytest`因当前shell未含conda env bin而找不到命令；
  只用固定Conda解释器则因未加载`env.sh`而有3项`import tvm`失败。最终按项目约定执行
  `source env.sh`后再用`/home/lee/miniconda3/envs/boundflow/bin/python -m pytest`全部通过；均为环境装载问题，
  不是代码回归；
- 主计划、S4预注册、可行性、ABI蓝图与三份权威状态文档的口径统一；`git diff --check`：PASS；
- 本批DocOps change/validation、exchange validate与lint在提交前执行。
