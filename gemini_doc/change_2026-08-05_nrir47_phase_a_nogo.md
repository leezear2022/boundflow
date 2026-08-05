# 2026-08-05 NRIR-47 Phase A NO-GO 关闭

## 起因与目标

NRIR-46 发现每轮 60 个 child compile 存在 64 次重复 target selection，估计中位成本约
`1.038153 s`。NRIR-47 预注册的唯一变量是 single-pass exact target admission receipt：每个 child
只选择一次 targets，production validator 消费 typed receipt；显式 full replay 仍从 exact
bounds/objective/policy 重选并比较。60 个动态 target ledger 不共享，算法、bound、branch、queue、
budget 与 deadline 均冻结。

## 实现

- 新增 `NativeTargetAdmissionReceiptIR`、Task IR 与 Schedule IR，绑定 graph/input/split/bounds、
  effective policy、objective/influence、ordered targets 和 selection count；
- 新增 additive single-pass compiler、prepared Program/capsule binding 与 candidate production route；
- legacy compiler 默认 full-validation 语义及其源文件保持不变；
- 新增 contract tests、three-fresh-process Phase A generate/replay、typed reconstruction、full selector
  replay 与 synchronized outer-rehash semantic tamper probe。

实现中曾短暂重构 legacy `native_intermediate_refinement.py`，导致 NRIR-33/34 frozen code revision
测试 5 项失败。正式测量前已把旧文件恢复为与 `main@ca0bcf3` 一致，并把构建逻辑移入新增 NRIR-47
模块；冻结测试随后 `10 passed`，全量回归未再出现漂移。

## Phase A 正式结果

- correctness/parity：PASS；clauses 2/3、三轮 control/candidate 的 branch、score、state、ancestry、
  refinement、bounds、worst lower 与 31-node queue exact；
- ownership：PASS；每条 candidate queue compile selector/reselection=`30/0`，runtime selector=`30`，
  root+child receipt=`31`，production timing 外 full replay/replay selector=`31/31`；共 replay 186 份
  typed receipt；
- compiler timing：FAIL；control/candidate median=`2.739226/2.563922 s`，ratio=
  `0.936003 > 0.85`，虽改善超过 pooled MAD，仍未过预注册绝对门槛；
- clause 2 queue timing：FAIL；control/candidate median=`10.099396/10.212559 s`，ratio=
  `1.011205 > 0.97`；
- clause 3 queue timing：FAIL；control/candidate median=`10.056289/10.250753 s`，ratio=
  `1.019338 > 0.97`；两条 queue 改善均未超过 pooled MAD；
- formal hash=`a7561e5187a6e396905d261e739280e39f2c3480e83ba2af0fbe6e3b1ec042ce`；
  replay 输出 `typed_receipts_replayed=186`、`outer_rehash_tamper_rejected=true`。

## 验证

- focused integration/frozen compatibility：`55 passed`；
- full suite：`992 passed, 37 skipped`；37 项均为 CUDA/TVM 环境边界；
- Black clean；mypy 9 files clean；Pylint `10.00/10`；`git diff --check` clean；
- artifact：`artifacts/single-pass-target-admission/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1/`。

## 判定与下一步

NRIR-47 以 `VALIDATED-NO-GO` 关闭。receipt correctness/ownership 机制成立，但减少一次 compile
reselection 只带来约 6.4% compiler median 改善，并未转化为端到端收益；candidate 不默认启用，
Phase B 按预注册 gated off，不能声称 performance、property closure、GPU、competitor、multi-workload
或 ASPLOS-ready。

下一步不继续细拆 validation ownership，而是对两条约 10 秒 top-2 production queue 做 execution
math/queue phase attribution，区分 forward/materialization、selected-CROWN、optimizer execute、branch、
queue bookkeeping 与 Python/dispatch overhead；只有先识别稳定 dominant cost，才能预注册下一项
stronger-bound、kernel/backend 或 queue fusion 单变量。
