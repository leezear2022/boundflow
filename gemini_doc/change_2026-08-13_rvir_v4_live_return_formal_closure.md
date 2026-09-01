# RVIR-v4 V4-3D Live Return Formal Closure

日期：2026-08-13

## 结论

V4-3D以`VALIDATED-LIVE-RETURN`关闭，V4-3E five-fresh correctness准入。

本阶段证明BoundFlow在真实RTX 4060 αβ-CROWN进程中独立执行pre-state恢复、10 evaluation/9 update
optimizer、backward export与三候选KFSB，随后原子提交真实provider-owned α/β和host packet，构造完整
`UpdateBoundCoreReturn`并被未修改的official `update_bounds_post`、domain queue与termination链路消费。

它尚未完成预注册的5个fresh original/candidate interleave pairs，因此V4-3整体尚未关闭，B2 same-solver
timing与任何performance claim继续关闭。

## 正式身份

- source commit：`dc7038a93353531f3c5e126b1c0b603db0026027`；
- artifact：`artifacts/rvir-v4-live-return/resnet2b-core-v1`；
- manifest file SHA256：`272ac92c41efee212b6ede5e55cc5a535a697fa51158416b9ed146aeed932d10`；
- manifest internal hash：`4fb6518b6e68a55317f685efa39a6f26cc2452afcc97ff85f02d9616c4adbd98`；
- live result SHA256：`5559ce5b830e8746e692b027409e82377c29d967e4917ceb2f6afa04ad395a69`；
- summary file/internal SHA256：`014ee936ddc54e794d4e8e158d2c54be86825f7d58cb1a195f640103c2cdad00` /
  `4caeb541ec2f8d21fbdb7706c1c52ee20e247ff5e2b29acff0e8cb0846f39a6c`；
- tamper report SHA256：`1e4acb650b77d4ba5ef8a5cc070ab79ad50d43e1cd87546e856e46abdc81ddb1`；
- frozen V4-3A truth/manifest SHA256：`d0126427…d0e9` / `0e6ed721…9818`。

## 正式结果

- native `lb`、三组candidate child lower和六层lA的source device全部为`cuda:0`；
- 12条真实provider-owned α/β路径联合host packet原子提交，12/12 committed、7条changed；
- provider core/`compute_bounds`/`update_bounds` callback=`0/0/0`，fallback=`0`；
- official post/queue接受完整packet，`visited_domains=[6]`，solver status/success=`verified/true`；
- final decision exact：`[[5,27],[5,32],[5,90],[5,90],[5,32],[5,90]]`；
- 与独立V4-3A provider truth对照覆盖451 tensors、213,060个float signs；shape/dtype/device、sign与
  离散结构exact，最大绝对差`1.0669231414794922e-05 <=2e-4`；
- formal replay重新启动fresh GPU candidate，semantic parity、decision、commit/path count、callback和
  solver accounting全部通过。

## 事务与篡改门禁

- unit contract覆盖tensor写失败、host写失败与post-verify失败时的live tensor+host联合回滚；
- lA、intermediate、candidate lower、α state、branch decision、core accounting、provider callback、
  atomic flag八类攻击均重签raw truth hash、receipt/assembly hash、summary、replay stdout、file digest和
  outer manifest；8/8全部fail closed。

## 验证

- related targeted：`23 passed`；
- full：`1196 passed, 3 skipped`；三个skip均为既有TVM重复编译或frozen VNN-COMP checkout边界；
- Black、相关source mypy clean、Pylint=`10.00/10`；
- formal generate、static replay、fresh GPU replay与tamper suite均通过；
- DocOps validate/lint在最终交接前执行。

## Claim边界与下一动作

`whole_core_replacement_admitted=true`只针对固定ResNet2B property 0、固定生产配置和一次live core。
`five_fresh_correctness_admitted=false`、`b2_same_solver_timing_admitted=false`、
`performance_claimed=false`保持不变。

下一步只启动V4-3E：按冻结顺序`O,C,C,O,C,O,O,C,O,C`运行五个fresh original/candidate correctness
pairs，对call/core lineage、bounds/state、branch、accepted/pruned domains、visited nodes、termination和
status/success做exact或`2e-4`门禁。5/5通过后才能关闭V4-3并恢复B2 correctness-gated timing。
