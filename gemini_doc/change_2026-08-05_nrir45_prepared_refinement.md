# 2026-08-05 NRIR-45 Prepared Intermediate Refinement Capsule v1 关闭

## 起因

NRIR-44 已把九子句 ranking floor 降到约 8.6 秒，但 top-2 production queues 仍占主要成本。只读
cProfile 显示每条 31-node queue 有 246 次 `_select_targets`，其中 186 次来自递归
`NativeIntermediateRefinementProgram.validate()`，而不是新的语义工作。

## 修改

- 新增 typed prepared capsule、Task、Schedule 与 execution Trace；
- 每个 child refinement 在 prepare 阶段完整验证一次，runtime 用 exact owner、容器身份和 Tensor
  mutation version receipt 快速准入；显式 full replay 仍重跑原始完整 validator；
- prepared `hashes()` 只返回 capsule 中已完整准入的 source Plan/Task/Schedule digest，避免 aggregate
  阶段反复序列化同一大 target table；
- 新增 additive per-child/shared production queue 与 NRIR-44 projected-floor global composition；历史
  NRIR-42/44 runtime 不改；
- 新增 4 个 capsule/receipt 契约与篡改单测，以及 1 个 exact 31-node queue parity 测试；
- 新增 Phase A/B formal generator、typed replay、manifest 与 outer-rehash tamper probe。

## 正式证据

- Phase A：clauses 2/3 三轮反平衡 exact；target selection `246→98`、full Program validation
  `186→38`、full hash `217→39`。两条 median ratio=`0.727519/0.736603`，均通过 `<=0.80` 与
  pooled-MAD 门禁；formal hash=`be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439`。
- Phase B：floor=`8.625022/8.583826/8.628565 s`；whole trace=
  `31.262521/31.319772/31.470078 s`；measured wall=
  `36.396631/36.513683/36.611709 s`。相对 NRIR-44 trace/measured median ratio=
  `0.710268/0.615738`，每轮 `[31,31]` nodes、60/60 capsules full replay；payload hash=
  `4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8`。
- 两阶段 replay、typed reconstruction、tamper probe、Black、mypy、Pylint `10.00/10`、全量
  `984 passed, 37 skipped` 通过。

## 判定与边界

NRIR-45 以 fixed ResNet2B property 0 CPU8 internal production admission `VALIDATED-REDUCED` 关闭。
final 仍为 sound 9/9 unknown，`performance_claimed=false`；不得外推公平竞品 speedup、GPU、
multi-workload、property closure 或 ASPLOS-ready。

## 下一步

先对最终约 31.3 秒 execution trace 做 residual phase attribution，再预注册一个单变量 NRIR-46。
优先量化 remaining target selection、selected-CROWN、optimizer/branch 与 aggregate 验证；不重开已
NO-GO 的 CPU scorer batching，也不通过降低 cap/nodes/depth 或改变 policy 换速度。
