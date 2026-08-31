# BoundFlow S4 Prepared Request 与查询性能修改记录

status: internally-validated-three-fresh-diagnostic
date: 2026-09-01
external-audit: deferred-by-user
performance-claimed: false

## 1. 结论

本轮没有继续外审，而是直接处理 S4 完整 same-solver 查询的剩余开销。新增的 Prepared Verification
Request 把模型、property、runtime spec 与 VNNLIB handler 的静态构造移到查询外，并用
copy-on-prune 保持每次查询的可变状态隔离。

在 RTX 4060 Laptop GPU、ResNet2B/property 0、同一 αβ-CROWN host solver 上，3 组独立进程、交错
`B4-A/S4-PREP` 的诊断结果为：

- complete-query speedup geomean：`1.242124410016909x`；
- complete-query worst pair：`1.2353494935603082x`；
- core speedup geomean：`1.1100949852653137x`；
- core worst pair：`1.099719557052884x`；
- lower max absolute difference：`3.0994415283203125e-06`；
- 三组 lower sign 与离散 solver 结果一致；
- 平均查询节省：`278.191408 ms`。

该结果超过既有 `1.15x complete-query` 研究门槛，但当前只有 3 组诊断，不是冻结 formal artifact，
因此不升级 claims map，`performance_claimed=false` 保持不变。

## 2. 归因结果

新增闭合时间账把查询拆为：solver 构造、constraint 解析、verify 前段、S4 core、verify 后段与最终同步；
并把 verify 内部进一步拆为 incomplete verifier、complete verifier、first-decision、BaB preprocess/solve/
postprocess。

代表性普通 S4 样本：

| 阶段 | 时间 |
|---|---:|
| complete query | `1253.917 ms` |
| constraint 首次解析 | `50.712 ms` |
| incomplete verifier | `664.206 ms` |
| complete verifier | `478.320 ms` |
| S4 core | `222.727 ms` |
| VNNLIB handler 构造 | `50.379 ms` |
| post-core | `1.737 ms` |

因此此前“只优化 S4 BaB core”确实不完整。当前最大剩余对象已经从猜测变成实测：root incomplete
verifier/α-CROWN 约占查询 53%，应是下一主线。

## 3. 实现

### 3.1 Prepared Verification Request

新增 `boundflow/runtime/prepared_verification_request.py`：

1. 在 warm query 外准备 constraint、Torch model、runtime spec、VNNLIB handler；
2. query 内复用 model，clone runtime spec；
3. 只读的 parsed VNNLIB 和 initial BatchedSpecs 共享；
4. handler 一旦 prune，原 αβ-CROWN 代码通过字段替换产生 query-local tensor/state；
5. `rhs_offset` 或 sanity-check 等可能原地修改共享 state 的模式 fail closed；
6. receipt 分别记录 static preparation、reuse/clone 次数与热路径耗时。

copy-on-prune 将 handler 热路径 clone 从约 `20.9 ms` 降到约 `0.013 ms`。

### 3.2 查询归因

`scripts/run_fsg3_same_solver_timing.py` 现在保存：

- `diagnostics.query_phase_timing`：exact-closure 查询账；
- `diagnostics.host_phase_timings`：solver 主阶段墙钟；
- `diagnostics.prepared_verification_request`：静态准备与 warm reuse receipt。

`S4-PREP` 是当前诊断候选；旧 `B4-A` 和 `S4` 保留，便于三方直接比较。

## 4. 冷启动与摊销

3 个 candidate 进程中：

- S4 TIR/static region prepare 平均：`6.412840 s`；
- request prepare 平均：`0.170791 s`，其中一组为 `0.290142 s`；
- 合计平均 cold setup：`6.583631 s`；
- 按平均每 query 节省 `0.278191 s`，break-even 约 `23.67 queries`。

所以当前结果只能表述为 cached/AOT warm-query 收益；单次一次性查询把全部 compile/setup 算入后仍不占优。

## 5. 本轮证据边界

- 原始诊断目录：`/tmp/boundflow-b4a-s4prep-pairs.9hfLnD`；机器清理或重启后可能消失；
- 3 组 order：`B4-A→S4-PREP`、`S4-PREP→B4-A`、`B4-A→S4-PREP`；
- 每个 worker 为 fresh Python process；
- 未做外审、未生成 formal replay/tamper artifact；
- 不据此声明 10x、ASPLOS-ready 或跨模型泛化。

## 6. 下一步

下一性能动作不是再审当前 1.24x，而是处理实测最大的 root incomplete verifier：

1. 给 root `LiRPANet.build/init_alpha/CROWN-Optimized` 建立更细的 GPU/kernel/materialization 账；
2. 复用已完成的 CIBC、initial-CROWN external-bounds 与 S4 structured state 资产；
3. 先判断 external CIBC intermediate bounds 能否减少 root 初始/优化 CROWN 的工作量；
4. 若只能省 IBP 小项，则停止该接法，转向 root α-CROWN 的 structured coefficient + compiled custom
   backward；
5. 候选通过单样本 correctness 后，再做 B4-A/S4/S4-PREP/root-candidate 同批 formal。
