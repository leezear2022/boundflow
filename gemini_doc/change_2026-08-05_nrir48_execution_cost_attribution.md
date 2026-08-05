# 2026-08-05 NRIR-48 Top-2 Production Execution Cost Attribution 关闭

## 目标与冻结边界

NRIR-47 消除 target reselection 后未获得端到端收益，因此 NRIR-48 不再猜测优化点，而是对 frozen
NRIR-45 default production 的 ResNet2B property 0 clauses 2/3 两条 31-node queues 做互斥成本归因。
NRIR-47 candidate 保持禁用；算法、IR/runtime、cap、optimizer、branch、node/depth、queue 与 deadline
全部冻结。

## 实现

- 新增 additive three-fresh-process paired control/profile runner，不修改 production runtime；
- 七类顶层成本精确闭合到 `queue_elapsed_ns`：child refinement compile/execute、optimizer
  prepare/execute、branch bind/score、materialize/commit、queue-control residual；
- child execute 进一步闭合为 fast validate、runtime target select、selected-CROWN、propagate-forward、
  hash/trace residual；
- artifact 重建 source/input/code digest、category derivation、median/share/MAD/dominance decision；同步改
  category 并保持总和的 tamper 仍被 raw timer derivation 拒绝。

## 正式结果

- 6/6 control/profile branch/score/state/ancestry/refinement/bounds/worst lower/31 nodes exact；
- clauses 2/3 profile/control median ratio=`1.023199/1.020221 <=1.05`，插桩扰动门禁通过；
- child refinement execute 两条均 3/3 排第一：median=`3.816002/3.704755 s`，queue median share=
  `32.1966%/31.1640%`，share range=`2.8559/0.4758` percentage points；
- child compile 次之，median share=`25.6893%/26.1902%`；branch bind/score 为
  `25.0044%/25.2853%`；optimizer execute 仅 `8.1484%/8.9610%`；
- child-execute 内 selected-CROWN 为唯一过 `>=30%` 的子类：median=
  `2.663321/2.694436 s`，parent share=`71.7725%/72.7291%`；runtime target selection 仅
  `15.5807%/13.4686%`，hash/trace residual=`11.7893%/12.3367%`；
- formal hash=`571c2e47c0c8906d2486e5e19e8152eb1ef0d3024b08cf561e25ed4f71d177a4`；
  6 profile rows replay 与 synchronized category tamper 拒绝通过。

## 验证与判定

- focused：`4 passed`；
- full suite：`996 passed, 37 skipped`；
- Black clean；mypy 2 files clean；Pylint `10.00/10`；
- artifact：`artifacts/top2-production-execution-cost-attribution/`
  `vnncomp21-resnet2b-property0-three-repeat-cpu-phase0-v1/`。

NRIR-48 以 attribution `VALIDATED-REDUCED` 关闭。该结论只证明 fixed workload 的 dominant execution
cost 已可靠缩小到 selected-CROWN，不是 speedup、property closure、GPU、competitor、multi-workload
或 ASPLOS-ready claim。

下一步另立 NRIR-49 selected-CROWN execution 预注册：先冻结 call/shape/chunk/backend decomposition 与
公平 timing gate，再只改变一个执行变量；不得重新优化已被证明非 dominant 的 validator、optimizer 或
queue bookkeeping。
