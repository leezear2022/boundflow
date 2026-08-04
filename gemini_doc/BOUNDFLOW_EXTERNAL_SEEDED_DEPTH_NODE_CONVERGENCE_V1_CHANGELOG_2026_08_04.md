# BoundFlow External-Seeded Depth/Node Convergence v1 修改记录（NRIR-24）

日期：2026-08-04
分支：`feat/external-seeded-depth-node-convergence-v1`
状态：`VALIDATED-REDUCED`

## 当前记录

- 确认 PR #34 已合并到 `main@c6a7998`，并从该基线建立 NRIR-24 分支；
- 复核 NRIR-23 的 typed external seed、ancestral queue 与 7-node/depth-2 固定结果；
- 冻结 ResNet2B property 0 clauses `0/2/4` 上的 `7/15/31 nodes × depth 2/3/4` 嵌套曲线；
- 冻结 fresh-process、逐单元原子 checkpoint、严格 resume、semantic replay 与篡改拒绝要求；
- 明确本轮只测正确性/收敛性，CPU timing 不是 performance claim。

## 实现进展

- 新增 checkpointed convergence runner：每个 clause/budget 由 fresh Python worker 执行并原子写
  shard；严格校验通过的已有 shard 可恢复复用；
- shard 冻结 source、typed seed、全部 policy、queue evaluations/decisions、refinement lineage、
  objective-branch IR hash 投影、summary 与 semantic hash；
- aggregate validator 重算九单元 coverage、`7→15→31` 公共 logical domains、worst lower、
  proof deficit、相邻 delta、饱和与 closure 门禁；
- clause 0 的 7-node/depth-2 worker 首次实跑通过，semantic hash=
  `80ac273993350fa7f26b0ba40a831e37492d69fe577ae4463fbfa32c1dae701f`。

## 首轮 assemble 修正

- 九单元首轮全部生成，但旧 validator 把 best-first 的 evaluation list 误当作 ordered prefix，使
  clause 2/4 的 15→31 检查错误报 NO-GO；
- 独立定位确认：15-node 的全部 15 个 logical split domains 都存在于 31-node 树中。深树会先扩展
  priority 更低的 depth-3 domain，再回来生成其他 depth-3 domain，故序号型 node/batch identity
  必然不同；这不是树语义漂移；
- validator 改为按 `split_state_hash` 做集合包含与 parent-split lineage 校验，并比较 branch
  selection、去执行序号的 refinement semantics；公共域 lower/upper/priority 使用 runtime 既有
  `1e-5` tolerance。观测到最大公共 lower 漂移约 `1.2e-6`，来自不同 batch composition；
- 此项作为 artifact contract bug 明示保留，不采信首轮 `ab38620e…` provisional evidence hash；
  修正后必须重新 assemble、完整 fresh-process replay 与 tamper tests。

修正后九个 shard 原样 resume 并重新 assemble，状态为 `VALIDATED-REDUCED`，evidence hash=
`db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6`。新增 5 条 frozen
artifact/curve/logical-domain/tamper/resume tests，focused `5 passed`。

## 固定结果

| clause | 7 nodes / depth 2 | 15 nodes / depth 3 | 31 nodes / depth 4 | 15→31 delta |
|---:|---:|---:|---:|---:|
| 0 | -0.318286777 | -0.299506187 | -0.282359719 | +0.017146468 |
| 2 | -0.425476611 | -0.413456440 | -0.401844978 | +0.011611462 |
| 4 | -0.504142046 | -0.479104042 | -0.459939480 | +0.019164562 |

三条曲线均单调严格改善且未触发饱和门禁，因此本轮收敛趋势为 `VALIDATED-REDUCED`；但
depth-4 proof deficit 仍为 `0.282360/0.401845/0.459939`，没有任何 fixed bounded-tree closure。
九单元 fresh-process replay 全部逐对象相等，最终 evidence hash=
`db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6`。

## 验证

- focused NRIR-24 + seed/refinement/optimized-BaB：`38 passed`；
- 全量：`771 passed, 37 skipped`，7 warnings；37 个 skip 均为 CUDA/TVM 环境边界；
- Black check 通过；Mypy 1 source file clean；Pylint `10.00/10`；`git diff --check` 通过；
- 九单元 fresh-process semantic replay 通过；5 条 artifact/tamper/resume tests 通过。

## 下一门禁

冻结 dynamic ancestral refinement budget/multi-pass：在同一 31-node tree 上，把新增 refinement
预算集中到 proof deficit 最大的 logical domains，和固定 16-target 单 pass 做同预算/同树对照。
纯 fixed-depth 扩展不再是主路线；不升级 complete property、CUDA/performance、multi-workload、
competitor 或 ASPLOS-ready claim。
