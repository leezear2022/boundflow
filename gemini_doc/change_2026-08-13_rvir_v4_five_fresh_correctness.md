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

当前状态为`IMPLEMENTED-FIVE-FRESH-HARNESS / FORMAL-RUN-PENDING`。重复同一smoke payload的contract
测试不能替代十个fresh进程，不构成V4-3E关闭。

下一步固定runner source commit，随后按冻结顺序实际运行十个fresh GPU processes、static replay与
tamper suite。只有5/5独立pairs全部通过，才能设置`five_fresh_correctness_admitted=true`、关闭V4-3并
将B2 same-solver timing从gated改为admitted；`performance_claimed=false`。
