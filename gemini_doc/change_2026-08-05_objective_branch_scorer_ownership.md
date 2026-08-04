# 2026-08-05：Objective Branch Scorer Ownership v1

## 起因

NRIR-40 在 single-global-60s 下只能提交 `29/21—23` nodes；NRIR-41 证明 frontier 顺序没有退化，
但 31 个 branch programs 会触发 341 次 candidate enumeration，objective queue 比 widest 慢约 1.75×。
本轮只改变 candidate-table/validation ownership，不改变 policy、optimizer、refinement、node/depth、
slice、cache 或 deadline。

## 实现

- 新增 `NativeValidatedBranchProgramCapsuleIR`，绑定 objective、ReLU bounds、split、selected α/β state、
  optimizer/branch policy、candidate table、Plan/Task/Schedule 与 semantic token；
- 新 scorer Task IR 的第一阶段显式读取 `branch.plan.candidates`；compile 唯一调用历史 enumeration，
  execute/materialize/evaluate/reduce/select 只消费不可变候选表；
- 新增 additive shared production queue 和 multi-clause anytime composition，不修改 NRIR-39/40 frozen
  runtime/artifact；
- Phase-A replay 将 serialized Plan/Task/Schedule/capsule 重建为类型对象，重算 candidate/score/selection/
  token/call/timing gate；Phase-B replay 绑定 Phase-A formal、31 capsules/slice、global deadline 与 cache。

## 结果

- Phase A clauses 2/3 三 fresh counterbalanced runs：historical/new enumeration=`341/31`，new
  compile=`31`、execute=`0`；六组 31-node branch、score、child lower、queue lower/upper、split、α/β、
  refinement exact；
- new/old queue median ratio=`0.706888/0.698486`，median 节省=`5.468696/5.680614 s`，均大于 MAD；
- Phase B three fresh whole queries：selected 均 `[2,3]`，accepted nodes 均 `[31,31]`，每条
  15 sibling groups/31 capsules，whole=`57.175184/57.697757/58.114412 s`；
- worst-active lower=`-35.530926/-30.258448`，相对 NRIR-37 widest 改善
  `+2.043362/+5.641768`；final status 仍 unknown；
- Phase-A formal hash=
  `0d310c2ffc96844648a83f9921bc7f353ec8425986bccb36f75e6d1cd2b25b58`；Phase-B formal hash=
  `7274e834b3bf08a9e138fa3284b70222620cf3c571395331e1a87ed5fee7d759`；
- targeted `10 passed`；全量 `958 passed, 37 skipped, 7 warnings`；Black/mypy clean、Pylint 10.00/10；
  artifact replay 与 synchronized capsule/score/call/deadline tamper tests 通过。

## 判定与边界

NRIR-42 以 fixed ResNet2B property 0、CPU8、global-60s objective-branch production admission
`VALIDATED-REDUCED` 关闭，并在这一窄范围内取代 NRIR-40 的 production NO-GO。它不是 complete property、
GPU、multi-workload 或公平竞品性能结论；`performance_claimed=false`，ASPLOS-ready 仍为 NO。

下一工程单变量应是 cross-clause/node/candidate batch Schedule：把当前顺序执行的 floor objective、
sibling nodes 和 scorer candidates lower 为联合 batch，同时保持本轮 exact capsule semantics。
