# 2026-08-16 FSG4/B4-A terminal lower/lA handoff 预注册

## 改动

- 在B4-0外审批准后只开放B4-A，不提前实现B4-B/TIR；
- 冻结第10次optimizer evaluation同时产出terminal lower与六层lA、terminal export零CROWN重跑的
  producer/consumer合同；
- 将terminal state、graph/split/forward trace、native↔provider topology、producer op ordinal/name、
  preactivation/coefficient shape、dtype/device/layout/content hash纳入typed lineage；
- 将B4-0审计的kernel shape缺失minor转为correlation-parent operator恢复与fail-closed门禁；
- 冻结10/9 optimizer、4 forward、3 KFSB、handoff=1、rerun=0及provider/fallback=0物理计数；
- 冻结5 fresh correctness先于性能，以及B3/B4-A core geomean `>=1.03x`、query worst pair
  `>=0.98x`；
- 明确列出B3/B4-A related pytest文件，关闭B4-0外审的测试集合不确定minor。

## 状态

状态=`PREREGISTERED-B4-A-NOT-IMPLEMENTED`。当前只允许实现typed handoff与no-rerun assembly；无B4-A
correctness、speedup、B0 parity、memory或ASPLOS-ready claim。

权威计划：
`gemini_doc/BOUNDFLOW_FSG4_B4A_TERMINAL_LOWER_ADJOINT_HANDOFF_PLAN_2026_08_16.md`。

