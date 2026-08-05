# NRIR-43 发布记录

## 变更

- `feat/cross-axis-verification-batch-schedule-v1@00b82c2` 已推送；
- PR #54 已合入 `main@2d245d6271517f16bbc153b643c92150784c35a6`；
- 当前状态、执行备忘、README 与总账统一切换 integration base；
- `VALIDATED-NO-GO`、`performance_claimed=false` 与 Phase-B gated-off 边界保持不变。

## 下一步

从该 merge commit 预注册 NRIR-44 Root-Projection Floor Schedule，先验证 consumer/liveness
投影能否在不改变 root lower、rank、selected clauses 与 soundness 的前提下消除冗余深层 floor work。
