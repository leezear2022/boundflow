# RVIR-v4 V4-3E Five-Fresh Correctness 修改记录

日期：2026-08-13

## 修改

- 新增fresh worker，以独立进程运行original provider whole core或BoundFlow candidate whole core；
- 观察并区分root-domain初始化`domains.add`与目标child post-add，只冻结后者的before/input/accepted/
  pruned/after count、depth、threshold与lower/upper identity；
- 新增formal orchestrator，严格执行`O,C,C,O,C,O,O,C,O,C`十个fresh进程，并按
  `(0,1)/(3,2)/(5,4)/(6,7)/(8,9)`组成五个counterbalanced correctness pairs；
- 每对比较完整451-tensor core/post树、α/β state、lA/intermediate/child lower、branch decision、
  queue admission、visited domains、termination与status/success；
- original固定24次provider bound lineage；candidate固定provider core/compute/update callback与fallback
  全零；
- 新增raw artifact replay和sequence/result/accounting/callback tamper probe。

## Capture-ready诊断

- 一个独立original与一个独立candidate smoke pair均完成RTX 4060 BaB round；
- original/candidate queue均为before/input/accepted/pruned/after=`0/6/6/0/6`，depth全1；
- status/success/visited=`verified/true/[6]`，final decision exact；
- core/post比较覆盖451 tensors、213,060 signs，最大差`1.0669231414794922e-05 <=2e-4`；
- original provider call count=`24`，candidate provider callback/fallback=`0/0/0/0`；
- synthetic five-pair contract测试覆盖10 runs/5 pairs、2255 tensor comparisons与1,065,300 signs；
- 相关脚本mypy clean、Pylint=`10.00/10`。

## 当前边界与下一动作

本段记录当时的`IMPLEMENTED-FIVE-FRESH-HARNESS / FORMAL-RUN-PENDING`状态；现已被
`change_2026-08-13_rvir_v4_five_fresh_formal_closure.md`中的十进程正式结果取代。

当前下一步以formal closure为准：预注册并执行FSG3/B2 same-solver timing；
`performance_claimed=false`直到正式measurement完成。
