# BoundFlow ASPLOS27 S4 → RVIR exact-call 接入变更记录

status: implemented-and-observed
date: 2026-09-01
performance-claimed: false
external-audit: deferred-by-user

## 1. 本轮目标

把已经在独立 production-shaped region 上测得约 `3.05x` 的 S4 compact
α/β optimizer 接入真实 αβ-CROWN host solver，并测量收益从 optimizer region 向
RVIR exact call、core 和 complete query 的传播情况。

本轮不改变 host solver 的 branching、termination、KFSB、queue、postprocess 和 atomic
commit；只替换既有 B4-A terminal optimizer/lower-adjoint producer。

## 2. 实现内容

### 2.1 exact-call bridge

新增 `boundflow/runtime/asplos27_s4_exact_call_bridge.py`：

- 预编译并绑定 D1C、compact coefficient、selector、six-site value、compressed gradient；
- 从真实 RVIR snapshot、mapping 和 live α/β owner 建立 S4 evaluator；
- 执行固定 `10 evaluation / 9 mutation` optimizer；
- 把 compact terminal α/β 写回完整 native state，未归 S4 所有的 dense state 保持原值；
- 生成 B4-A 兼容的 terminal lower/lA handoff，后续仍走既有 backward assembly、KFSB、
  atomic commit 和 queue；
- receipt 明确记录 setup/optimizer/handoff 分段、0 provider callback、0 fallback、
  0 exact-call 内编译。

### 2.2 预编译对象注入

S4 selector、compact coefficient、six-site value 和 gradient runtime 现在允许注入已经编译
并校验过的 module，避免在 exact call 内重复 TVM compile。

### 2.3 Prepared S4 region

第一版直接接入虽然语义通过，但每个 core 都重新构造 Plan、Trace、Executor、DLPack view 和
evaluator。分段实测为：

| 阶段 | 未 Prepared 耗时 |
|---|---:|
| IR/trace | 约 109 ms |
| executor/view | 约 133 ms |
| state/buffer | 约 57 ms |
| evaluator 首次初始化 | 约 997 ms |
| S4 10/9 optimizer | 约 50 ms |

因此新增 one-shot `PreparedS4ExactCallRegionV1`：

- query 前构造 Plan、Trace、persistent tensors、executor、buffer 和 evaluator；
- query 内通过已验证的 `CorePlanInstanceV1` 绑定动态 instance；
- 使用固定地址 persistent storage，动态值只做 D2D copy；
- compact α/β 只更新七个实际 optimizer owner；
- prepared region 一次消费，重复调用 fail closed。

真实 query 中 prepared setup 降至约 `4.555 ms`：IR guard `0.603 ms`、persistent input
rebind `0.320 ms`、compact state rebind `0.064 ms`、evaluator readiness `0.002 ms`。

## 3. 真实 same-solver 观察

协议：同一 ResNet2B/property、同一 αβ-CROWN host solver、seed、branch、termination、
max BaB iteration；对照为 B4-A，候选为 S4 prepared region。三组 fresh process 交替顺序的
观察值如下：

| pair | core speedup | complete-query speedup | 双方环境准入 |
|---|---:|---:|---|
| 0 | 1.1269x | 1.1643x | 是 |
| 1 | 1.1506x | 1.1595x | 否（S4 环境门禁） |
| 2 | 1.1385x | 1.0273x | 否（S4 环境门禁） |

三组观察性几何平均：

- core：`1.1386x`，最差 `1.1269x`；
- complete query：`1.1152x`，最差 `1.0273x`。

唯一双方均通过现有环境门禁的 pair 为：

- core：`249.097 ms → 221.042 ms`，`1.1269x`；
- query：`1420.919 ms → 1220.449 ms`，`1.1643x`。

这些数字仍是观察值而不是 formal claim：现有 worker 的环境计数窗口从静态准备前开始，S4
进程会额外执行约 `6.58 s` 的 compile/warm/prepare，导致后两条 S4 虽然 query 计时排除了
静态准备，环境门禁仍把准备阶段的功耗计数包含进来。后续 formal 必须把双方的环境窗口统一
放在 static prepare 之后，不能简单忽略门禁。

## 4. 语义结果

三组 pair 均满足：

- final decision、depth、history、split、visited domains 和 queue 计数完全一致；
- lower 最大绝对差 `2.6226043701171875e-06`；
- lower sign 一致；
- provider callback、fallback 和 exact-call 内 compile 均为 0。

专项测试同时比较了 dynamic bridge、prepared bridge 与原 B4-A handoff，当前为 `3 passed`。

## 5. 当前结论与限制

本轮已经证明：S4 局部收益可以通过真实 RVIR exact call 传播到 host solver core，Prepared
ownership 是必要条件。它还没有证明稳定的 complete-query `>=1.15x`：三组 query 抖动明显，
只有一个环境准入 pair。

另外，当前 benchmark worker 使用冻结 production capture 完成 shape-specific warm/prepare；这适合
验证 Prepared 机制，但不是最终通用 API。正式实现还需要：

1. 从 PlanTemplate/shape signature 构造 persistent region，不依赖冻结 capture 文件；
2. 将 TVM module 走 AOT/disk cache，披露约 `6.58 s` cold setup 与 break-even；
3. 在 static prepare 后重置双方一致的环境计数窗口；
4. 跑至少五组 fresh process paired formal，再决定 complete-query claim；
5. 继续归因 core 中约 150 ms 的 backward assembly、KFSB、commit 和 receipt/hash 成本。

## 6. Claim 边界

- 本文只记录实现和观察值，`performance_claimed=false`；
- 没有把三组中的最好结果写成最终结果；
- 没有把独立 region 的 `3.05x` 写成 complete-query speedup；
- 没有升级 ASPLOS-ready 或 10x 系统 claim；
- 本轮按用户指令不启动外部审计。
