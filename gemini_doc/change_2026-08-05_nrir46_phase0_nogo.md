# 2026-08-05 NRIR46 Template/Instance Phase 0 NO-GO

## 变更

- 新增 `scripts/run_intermediate_refinement_template_instance_phase0.py`，用三个 fresh CPU8 worker
  对 frozen NRIR45 child compile 做 inclusive/exclusive attribution；
- 新增 `tests/test_intermediate_refinement_template_instance_phase0.py`，覆盖动态 target selection
  保留与缺失 timer fail-closed；
- 生成 digest-locked Phase 0 artifact，generate/replay 均验证 frozen NRIR45 source、60 个 exact
  Program、计时计数、formal payload 与同步外层重哈希篡改拒绝；
- 未修改 NRIR45 production runtime，也未实现 NRIR46 Template/Instance IR。

## 结果

- compile total=`5.356892/5.366369/5.452290 s`，median=`5.366369 s`；
- strict static topology=`1.071197/1.062492/1.071704 s`，median=`1.071197 s`；
- ownership-convertible ceiling=`2.097255/2.102134/2.109857 s`，median=`2.102134 s`；
- 每轮 target selection observed/semantic=`124/60`，冗余=`64`，估计冗余成本=
  `1.026058/1.039642/1.038153 s`；
- 三轮保持 selected `[2,3]`、nodes `[31,31]`、60/60 capsules/full replay；60 个 target identity
  与 table hash 全部互异，primal graph、Task/Schedule topology 各只有 1 种；
- formal hash=`712ce359501a010a197797909ab71fb127ebda43329dd3a7a8e21b6dbb4cf846`，
  replay/tamper 通过，`performance_claimed=false`。
- targeted `2 passed`，全量 `986 passed, 37 skipped`；Black、mypy 与 Pylint `10.00/10` 通过。

## 判定

预注册 strict static-shareable gate 要求 whole-query median 至少 `1.5 s`；实测中位数只有
`1.071197 s`。NRIR46 因此以 `VALIDATED-NO-GO` 关闭，Phase A/B gated off，门槛不事后修改。

下一路线不能继续假设 target ledger 可共享，而应把已观测的 64 次冗余 target reselection 收窄为
single-pass exact target admission receipt：production compile 只选一次，显式 full replay 仍从源输入
重算并逐项比较。该路线必须作为 NRIR47 独立预注册，不继承 NRIR46 的通过状态。
