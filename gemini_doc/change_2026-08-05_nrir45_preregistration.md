# NRIR-45 Prepared Intermediate Refinement Capsule v1 预注册记录

> 历史预注册记录：NRIR-45 已按冻结门禁完成并以 `VALIDATED-REDUCED` 关闭；正式结果见
> `gemini_doc/change_2026-08-05_nrir45_prepared_refinement.md`。下文保留开工前合同，不再代表当前状态。

## 起因

NRIR-44 已把 floor 从约 24.2 秒降到约 9.9 秒，但 whole trace 仍约 44.1 秒。对单条 31-node
production queue 的 cProfile 显示，per-child refinement 是最大累计成本；`_select_targets` 246 次中
186 次由同一 Program 的递归重复 validation 触发。

## 路线前探针

只读 monkeypatch 保证每个 exact refinement Program/Execution 仍完整验证一次，只省重复调用；clause 3
queue trace 从约 `12.85 s` 降到 `9.761678 s`，31 nodes 与 worst-active lower exact。该探针只用于
选择路线，不作为正式性能证据。

## 预注册

- 唯一变量：prepare-once intermediate-refinement validation ownership；
- 不改 refinement/optimizer/branch/queue/预算/精度/deadline；
- Phase A 要求 clauses 2/3 exact、重复 target/validation work 收敛、queue ratio `<=0.80`；
- Phase B 要求 whole trace/measured 每轮 `<=40/50 s`，相对 NRIR-44 median ratio `<=0.90/0.85`；
- 当前没有实现、artifact 或新 claim，ASPLOS-ready 仍为 NO。
