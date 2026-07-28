# 变更记录：IR-5D prepared execution capsule 与公平计时校准

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 父基线：`3c0399c`
> 判定：remediation 实现完成；旧 IR-5C3 正式 No-Go 不撤销，新 final artifact 尚未执行

## 1. 问题与边界

IR-5C3 的正式 artifact 显示 typed Global 相对 fair batched-original 的 p90 regret 为
`70.263×`。额外 profile 把主要问题定位到 query hot path 内反复执行静态 IR
validate、stable hash、canonical JSON 和 dispatch-key 构造。

本切片只实现 No-Go 文档允许的 prepared-execution remediation：

- 静态 legality、模型指纹、IR hash 和 primary/fallback dispatch key 在 prepare 时计算一次；
- query hot path 复用冻结的程序与 dispatch capsule；
- 保留 query payload、Schedule query binding、state version/capability 和 identity 检查；
- 不修改、不删除 IR verifier；
- 不回写 `artifacts/ir5/family-fair-v1-20260728`，也不把已消费 workload 当成新 held-out。

## 2. 实现

### 2.1 Prepared Bound/Task program

- `PreparedPlainCrownBoundIRProgram` 一次性验证 Bound IR、旧 Task module 与 primal
  fingerprint，缓存 Bound hash/value index；
- tensor 参数在 prepare 时 `detach().clone()`，形成静态模型快照；外部后续原地修改旧
  binding 不会污染已准备执行；
- `PreparedTaskIRExecution` 固定 Task/Bound/Plan/legacy module 的对象身份、Task/Schedule
  hash，以及每个 primary/fallback task 的 dispatch key；
- 动态 Schedule 只允许 query IDs、BatchLoop slices 和 EmitResult query binding 按同一
  静态结构重绑定；buffer、action、backend、state 或输出结构漂移仍 fail closed。

### 2.2 Audit 与 production trace 分离

- `TaskTraceMode.AUDIT` 仍是默认值，保留逐 Task tensor SHA-256，适合 replay/审计；
- `TaskTraceMode.PRODUCTION` 保留 Task/Schedule/dispatch identity 与事件序列，但不在 timed
  query path 计算中间 tensor hash；
- measured workload runner 显式使用 production mode；最终 lower/upper hash 和
  cross-backend allclose 仍在计时区外统一计算。

这不是删除正确性证据：legacy batching baseline 本身也不在 timed region 生成逐 Task
hash，production mode 使两端计时责任一致。

### 2.3 Runtime/cache 与公平 baseline

- typed compiler query plan cache 现在同时保存 lowered Task/Schedule 和 prepared capsule；
- static PlanTemplate hash 增加 exact-object cache，state store 复用 session 的已验证 Bound
  hash，并校验 session/module 对象身份；
- 新增 `batched_original_from_forward_trace`：forward IBP trace 在计时外预计算，timed
  region 只执行 CROWN backward，与 typed workload 从预计算 `relu_pre` 开始的责任对齐；
- 原 `batched_original` API 保留，旧 artifact 的语义和 replay 不变。

## 3. 验证

- 全量：`476 passed, 1 skipped`；
- prepared/query/fair batching 定向回归：`15 passed`，后续 state identity 修订后
  `14 passed`；
- Mypy：7 个修改的 source 文件 `Success: no issues found`；
- Pylint：修改 source/tests 在禁用跨模块重复代码报告后 `10.00/10`；
- Black：9 个修改 Python 文件均无需改写；
- `git diff --check`：通过。

新增回归覆盖：

- prepared capsule 重复执行和 backend cache hit；
- AUDIT/PRODUCTION 数值一致，production 不生成中间 tensor hash；
- 合法动态 query Schedule 重绑定与非法结构漂移拒绝；
- prepared 参数快照不受旧 binding 原地修改影响；
- from-forward-trace legacy baseline 的 variant、计时与语义字段。

## 4. Calibration-only CUDA 诊断

在已被 IR-5C3 消费的两组 chain-CNN 上做 20 warm samples 诊断，设备为
RTX 4060 Laptop GPU。两端都从预计算 forward trace 开始，以下为 median
per-query latency 及 typed/legacy 比值：

| workload | legacy from-trace | reference | dense | chunked | TVM fused |
|---|---:|---:|---:|---:|---:|
| gray CNN | 0.43875 ms | 0.894× | 0.884× | 0.942× | 0.880× |
| color CNN | 0.44382 ms | 0.902× | 0.896× | 0.950× | 0.940× |

这些数字只证明 host-overhead remediation 方向有效。它们不是 fresh held-out、没有目录级
manifest/replay，也没有重新运行完整 fixed/local/global/oracle evaluator，因此：

- 不升级 C2 performance claim；
- 不把 IR-5 状态改成 validated；
- 不撤销 IR-5C3 的 `70.263×` 正式 No-Go；
- 不解锁 IR-6。

## 5. 下一硬门禁

下一步只能：

1. 在任何 final measurement 前冻结新的 residual-CNN workload schema、seeds、资源
   contexts、policy pool 和 `from-forward-trace` fair baseline；
2. 用 calibration-only 数据决定模型/阈值，final residual split 一次性消费；
3. 产出目录级 manifest、integrity replay、semantic replay；
4. 重新检查 Global fair p90 regret `≤1.20×`、多预算选择和 latency-memory Pareto。

若新的 frozen residual split 仍不能同时满足性能与 Pareto 门禁，则停止当前 ASPLOS
system-performance 路线；不得继续在 final workload 上迭代调参。
