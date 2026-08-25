# 修改记录：MR2 Production CROWN Subgraph/Owner Inventory

> 日期：2026-08-26  
> 状态：预注册完成，尚未执行

## 修改

- 在MR1-S `0/51` full-graph eligibility后，冻结真实CROWN subgraph/owner inventory路线；
- 固定P-anchor Conv与S-anchor Linear两个既有production-derived site，不追加synthetic候选；
- 冻结七层readiness ledger和不按历史性能选择site的机械route；
- 明确本轮无GPU、无solver、无timing、无production code变更；
- 通过最多开放单site production exact-call bridge correctness预注册。

## 待验证

- 实现self-contained input snapshot、site ledger、gap matrix、replay与tamper；
- targeted/typing/lint/full regression；
- 根据机械route只开放合法后继。

