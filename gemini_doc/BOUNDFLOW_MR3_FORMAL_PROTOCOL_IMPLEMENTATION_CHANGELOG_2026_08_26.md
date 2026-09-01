# 修改记录：MR3 Five-Pair Formal Protocol Implementation

> 日期：2026-08-26
> 状态：协议实现完成，formal raw尚未运行

## 改动

- 新增MR3 formal semantic derivation，机械校验5 pair/10 fresh process的固定`PB/BP/PB/BP/PB`
  顺序、真实10/9 bridge receipt、P-region value、10-step loss、9-step gradient/Adam/clamp轨迹与final
  provider state；
- 普通数值门禁固定`atol=rtol=2e-4`，optimizer trajectory固定`atol=rtol=2e-5`，符号与离散
  identity exact；provider/candidate允许size-1维的等价contiguous stride不同；
- 新增formal orchestrator，每个worker独立进程、raw-first、缺失即整体失败；绑定implementation source
  `baddf7c`、formal code blob SHA256、外部repo commit与model/property digest；
- 新增evaluation-5异常注入的atomic rollback replay门禁；
- 新增18类fully re-signed tamper，覆盖source/order/verdict/count/value/loss/gradient/moment/lr/clamp/
  final state/atomic pointer；
- 新增synthetic正向、optimizer漂移与全tamper拒绝测试。

## Claim边界

当前提交只实现协议和门禁，不包含formal raw，不形成correctness closure；生成时必须确保全部code path
clean，并从pair-0完整运行。全程`timing_recorded=false`、`performance_claimed=false`。
