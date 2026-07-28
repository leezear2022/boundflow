# BoundFlow ASPLOS Claims Map

> 本表是动态证据账本。`planned` 不代表已经实现；只有代码、测试和工件均存在时才能改为
> `validated`。当前执行基线为 PR-12 validated-reduced；PR-13 已以
> `VALIDATED-REDUCED` 关闭；PR-14B 真实 replay 为 `VALIDATED-NO-GO`，C3 已降级为 C1/C2
> 基础设施，不再主张 non-toy verifier acceleration。

| Claim | 当前状态 | 代码/设计落点 | 必需测试 | 必需工件 |
|---|---|---|---|---|
| C1：显式物化语义的 Structured Bound-Operator IR | IR-1 reference semantic closure + IR-3 Task/Schedule reference integration validated；production backend pending | typed Bound IR + lowering + dense/structured interpreter + explicit cast/materialize rewrite + per-task stepping | 25 个 Bound tests；MLP/CNN/residual/concat final bounds 对齐；structured materialize 进入 Task trace | deterministic dump/hash + synchronous artifact v2 已有；production backend E2E 尚缺 |
| C2：Method/Autograd/Memory-Aware Materialization Planner | IR-4 closure；IR-5C3 fair architecture-held-out VALIDATED-NO-GO | typed PlanTemplate/PlanInstance、Task/Schedule lowering、adaptive/fair evaluator、calibration-only CUDA runner | Global 8/8 feasible，但相对 fair batched-original p50/p90 regret 68.065×/70.263×，无多预算切换/Pareto | IR-5C3 有 MLP→CNN split、48 outcomes 与 replay；当前性能 claim 失败，只允许 prepared-execution remediation |
| C3：Verification Query Runtime Infrastructure | downgraded after PR-14B | query/validity + batcher + capability routing + real observer | 保留 reduced correctness；真实 coverage/replay 作为 limitation | activation 0/394 eligible；ResNet bound-equivalence fail；不作 acceleration claim |
| BoundFlow Schedule IR | IR-3 synchronous reference closure validated-reduced；production driver pending | typed ScheduleModule + memory/batch/transfer/event/state/retry/replan + Task launch linkage；旧 topo loop 仅作历史对照 | 12 个 lowering/control/trace/artifact tests；OOM bounded、stream happens-before、query accounting、tamper rejection | deterministic dump/hash/trace + artifact v2 fresh-process semantic replay 已有 |
| BoundFlow Task IR | IR-3 per-task semantic closure validated-reduced；production backend pending | TaskIRModule/Unit + typed op/shape/parameter/external/state/memory/backend refs + stateful Bound stepping | 12 个 tests（含 4 graph families、structured materialize、skip/reorder rejection） | per-task output hashes 与 final bound hashes 已入 fresh-process artifact v2 |
| backend 执行 typed Planner/Task 结果而非定义核心抽象 | IR-4 validated-reduced；IR-5C3 fair performance No-Go | composite typed registry + query adapter + real fused/unfused/fallback；当前 query hot path 重复 validate/hash | MLP→CNN held-out correctness 通过；typed Global 对 batched-original p90 regret 70.263× | IR-5C3 artifact 可 replay；backend correctness 成立但 production performance 不成立 |
| 相同浮点语义下保持 reference bound computation | reduced-only；non-toy fail | dense reference + planned paths | allclose、gradient、auto_LiRPA、replay | MLP pass；ResNet max diff 796.765，PR-14C blocked |

### 2026-07-20 IR-first claim 纠偏

历史 PR-10/11/12 的数值、OOM、held-out 和 backend 证据不撤销，但其 claim 范围必须与代码对象
层级一致：

- runtime `LinearOperator` 证明结构化表示机制，不自动证明一等 Bound IR；
- `MaterializationPlan`、`MaterializationPlacementPlan` 和 `ExecutionCandidate` 证明局部决策机制，
  不自动证明统一 Plan IR；
- TaskGraph 拓扑序和 `FusedCrownExecutionStep` 不自动证明 Schedule IR；
- PR-13 batching 证明保持 ordinary batching 收益，不自动证明 adaptive runtime 贡献；
- cached specialization/JIT 在新 break-even 证据出现前只属于 planned hypothesis。

新的升级门禁见
`gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。

### 2026-07-28 IR-1A 进度

- `boundflow.bound_ir/v1.0` 已新增 typed value/type/spec/domain/op/graph/module；
- graph verifier 已覆盖 SSA/use-def、类型/极性、batch axes、representation change、method state；
- module verifier 会把 input/spec bind 与 concretize ID 交叉解析到 typed VerificationSpec；
- module 已有 canonical JSON 与 SHA-256 stable hash；
- Bound IR 源模块不依赖 runtime、backend、PyTorch 或 TVM；
- 旧 `DomainState` 兼容路径保留；
- builder、reference interpreter、CROWN lowering 和 IR-driven E2E 尚缺，因此不升级完整 C1。

实现与测试边界见
`gemini_doc/change_2026-07-28_bound_ir_v1_schema_foundation.md`。

### 2026-07-28 IR-1B 进度

- `BoundAffineStateRef` 显式表示 `A_u/b_u/A_l/b_l`，不再把真实 CROWN state 压成单值；
- residual/concat backward route 和 fanout compose 已成为 typed BoundOp，并验证 bias-once 语义；
- `boundflow/frontends/plain_crown_bound_ir.py` 已把单任务 plain-CROWN Task/trace lower 为
  validated `BFBoundModule`；
- `boundflow/runtime/bound_ir_interpreter.py` 已独立执行 dense Bound IR，不 import CROWN oracle；
- identity/multi-spec MLP、chain CNN、residual/concat fanout 的 final lower/upper 已与现有
  `run_crown_ibp_mlp` 对齐；
- stale parameter/objective、缺失 ReLU trace fail closed；
- 专属测试 20 passed，全量 392 passed、1 skipped；
- materialize/representation rewrite、structured execution、生产 runtime 迁移和 IR-driven artifact
  尚缺，因此仍不升级完整 C1。

实现与门禁边界见
`gemini_doc/change_2026-07-28_bound_ir_v1_plain_crown_lowering.md`。

### 2026-07-28 IR-1C / IR-1 closure

- affine-state verifier 禁止 Linear/Conv/ReLU/Reshape/route/compose 隐式改变 representation；
- 新 verified rewrite 在 affine region 入口插入 dense→structured cast，在 ReLU/concretize
  dense boundary 前插入 materialize；
- reference interpreter 已执行 structured LinearOperator region 和显式转换；
- multi-spec MLP、chain CNN、residual/concat fanout 的 dense/structured rewrite final bounds 对齐；
- 非法隐式转换和重复 rewrite fail closed；
- 专属测试 25 passed，相邻 47 passed，全量 397 passed、1 skipped；
- IR-1 契约的最小 reference semantic closure 门禁已通过；
- 完整 C1 仍需 IR-2/3/4 的 Plan/Schedule/backend integration 和 IR-driven E2E artifact。

实现与门禁边界见
`gemini_doc/change_2026-07-28_bound_ir_v1_representation_rewrite.md`。

### 2026-07-28 IR-2A 进度

- 新 `boundflow.plan_ir/v1.0` 已区分 `PlanTemplate` 静态候选空间与 `PlanInstance` 动态选择；
- region/representation/materialization/backend/domain-spec-sample batch/storage/state 成为独立
  typed candidate/decision；
- cross verifier 已检查 Bound hash、partition coverage、capability、transition、memory、
  storage lifetime/alignment/alias、state version 和候选全量记账；
- template/instance canonical JSON/hash 已有，instance strict JSON replay 已拒绝 noncanonical 和
  tampered selection；
- PR-11/12 六类旧对象已有 adapter/partial/unsupported 代码级迁移表；
- 专属 12 passed，相邻 88 passed，全量 409 passed、1 skipped；
- reference template builder、query-time selector、多预算选择和 artifact 尚缺，因此 IR-2/C2
  均不升级为 complete。

实现与门禁边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_schema_and_legacy_migration.md`。

### 2026-07-28 IR-2B 进度

- 新增 typed evidence → `PlanTemplate` reference builder，自动推导 Bound IR region boundary、
  storage lifetime/alignment 和 capability rejection；
- 新增有界 deterministic selector，memory/deadline 改变时产生不同且完整记账的
  `PlanInstance`，无可行计划时 fail closed；
- 新增不可变 Bound/Template/Instance artifact API、逐文件 SHA-256、精确 replay 与 tamper
  rejection；
- Plan IR 专属 11 passed，连同 migration 共 16 passed；相邻 92 passed；全量
  413 passed、1 skipped；
- 尚缺旧 PR-11/12 真实 artifact 批量 assembly/report、query-time state-validity 和独立 replay
  CLI，因此 IR-2/C2 仍不能升级为 complete。

实现与门禁边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_reference_builder_selector.md`。

### 2026-07-28 IR-2C / IR-2 closure

- `PlanInstance.state_validities` 和 `StateAction.REUSE` 已把 query-time exact cache validity 纳入
  canonical verifier/hash/replay；stale state 转 recompute，伪造 valid stale state fail closed；
- legacy migrations 可原子组装到同一 template，accepted/unsupported/rejected 形成稳定报告；
- reference artifact 已有 fresh-process generate/replay CLI；
- 对当前 `artifacts/` 扫描 58 个 JSON/JSONL、4,911 个 JSON objects，三种 PR-11/12
  planner raw schema 记录均为 0；因此只关闭对象族级 migration，不声称历史逐记录迁移；
- 专属 21 passed，相邻 97 passed，全量 418 passed、1 skipped；
- IR-2 最小 reference contract 关闭为 `VALIDATED-REDUCED`；C2 仍需 IR-3 Schedule IR、
  runtime/backend migration 和 IR-driven E2E，不能升级为 paper-level complete。

实现与 closure 边界见
`gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`。

## PR-10 子阶段

| 子阶段 | 状态 | 完成证据 |
|---|---|---|
| PR-10A Materialization instrumentation | validated | `25225e5`；ReLU barrier opt-in trace |
| PR-10A.1 Trace schema v1 | validated | `boundflow.materialization/v1`、schema contract tests、164 passed |
| PR-10B.1 workload characterization | validated | `8f2c998`；180/180 clean GPU profile；mini-ResNet s128/d32 |
| PR-10B.2 真实 BaB fixed-domain replay | superseded | 不再执行；由 PR-14A/B 真实 verifier trace/replay 取代 |
| PR-10C.1 Dense/gradient reference oracle | validated | 显式 `A_u/A_l/b_u/b_l` oracle；独立 α sign-gradient；170 passed |
| PR-10C.2 Dense/structured 双路径 oracle | validated | local/full/gradient、plain/α/αβ、真实 solve_bab 搜索等价 |
| PR-10D.1 Exact SignSplit operator | validated | exact dense/gradient；composition 包裹而不下推 sign；26 passed |
| PR-10D.2 Structured ReLU 主路径 | validated | main coefficient 不永久 dense；ephemeral bias；operator dump；177 passed |
| PR-10E 全路径回归与 benchmark | validated（guarded） | 360 rows；354 ok/6 structured OOM；179 passed；dense 默认 |

## 当前 Gate 0 证据

- PyTorch 2.12.1+cu132、CUDA 13.2、LLVM 20.1.8、TVM 与单一内嵌 tvm-ffi 已完成现场验证；
- MLP/CNN reduced artifact 已生成：small matrix、warmup 3、iters 10，2 行均通过 correctness；
  它是 Gate 0 回归，不替代论文要求的至少 5 次独立重复；
- Gate 0 已冻结在本地提交 `4e0e059`，全量验证为 162 passed、1 个预期 skip；
- Gate 0 已完成；PR-10 已在 `263ea81` 结项，ReLU structured path 为 feature-gated，dense 默认。

## PR-10 第一版 profile claims

- `C1-E1a` validated：persistent ReLU logical bytes 在固定结构下随 spec×domain 线性放大；
- `C1-E1b` validated：mini-ResNet αβ s128/d32 为 939,524,096 logical bytes、3.45 GB
  trace-off peak allocated；
- `C2-M1` partial：query axes 会改变 materialization 规模，但尚未证明不同计划各有最优 regime；
- 详细口径与限制：`gemini_doc/pr10_materialization_profile_summary_2026_07_12.md`。

## PR-10 完成判定

- `C1-E2` validated：local/full/gradient、CROWN/α/αβ/solve_bab 与 dense reference 对齐，
  360 行矩阵中 0 correctness failure；
- `C1-E3` validated：代表性 plain CROWN 大点 structured peak 降低约 29.8%；
- `C1-L1` validated limitation：同一点 structured latency 增加约 9.17×，不适合默认启用；
- `C1-L2` validated limitation：α/αβ structured 显存恶化，并在 6 个大点 OOM；
- `C2-M1` validated motivation：不存在跨 method/grad/memory regime 的统一最优表示；
- `C2-H1` planned hypothesis：最优可行计划必须感知 method、differentiation stage、capability
  与 memory budget；PR-10 数据只能作为动机/校准数据，尚不是 Planner 有效性证据；
- PR-10 状态：**complete, feature-gated**；默认 dense，structured 由环境开关启用；
- 对照证据：`gemini_doc/pr10_dense_structured_comparison_2026_07_12.md`。

## PR-11 内部门禁

- 0 bound/gradient correctness failure，0 unexpected OOM；
- 若任一合法候选可运行，Planner 应找到可运行计划；α/αβ structured 不得被误选；
- workload-family held-out 上 median latency regret 相对 Oracle 研发目标不超过 20%，并报告 p90；
- 至少选择 dense 与 structured 两类计划；
- 至少一个预算下，让 Always Dense OOM 的 plain CROWN case 成功运行；
- 与 Always Dense、Always Structured、Method-Only、Memory-Threshold、Local Greedy 和 Oracle
  公平比较。

## PR-11 子阶段

| 子阶段 | 状态 | 完成证据 |
|---|---|---|
| PR-11A Context/capability/action/plan dump | validated | `materialization.py`；真实 CROWN shape-derived context；JSON plan |
| PR-11A.1 Runtime guard | validated | CROWN 显式 plan；α/αβ structured capability 拒绝；reduce-batch re-plan signal |
| PR-11A.2 Per-case measured Oracle | validated foundation | fastest observed feasible action；capability/OOM 不可绕过 |
| PR-11B Cost model calibration/held-out | validated foundation | calibration + validation/refit + final mini-ResNet held-out；method/action linear model |
| PR-11C Local/Global benchmark matrix | partial | 1728 rows；Global 239/239、0 unexpected、median/p90 1.0；但与 Memory-Threshold 相同 |
| PR-11C.1 Multi-barrier placement mechanism | validated foundation | synthetic Local re-plan vs Global mixed feasible；两 ReLU mixed execution 与 dense 对齐 |
| PR-11C.2 Measured barrier-level held-out | partial | shuffled calibration 56 rows + held-out mini-ResNet 128 rows，184/184 correct；one-shot Global 未过 feasibility gate |
| PR-11C.3 Global Retry held-out replay | validated reduced | 7/7 feasible、0 unexpected、median 1.159×、p90 1.562×；仅一个 held-out query |
| PR-11D Host OOM retry | validated reduced | 380 MiB cap；dense real OOM→structured success，3/3 独立重复；仅 plain CROWN 单配置 |
| PR-11D.1 Bounded stratified retry | validated reduced | s32/d8 与 s128/d8 均 7/7、0 unexpected；median 1.159×/1.171×；最多 3/5 次；真实 OOM 3/3 |
| PR-11D.2 Scheduler reduce-batch execution | planned | 当前 reduce-batch 仍主要返回 host re-plan signal |
| PR-11E Independent-topology held-out | failed gate | branched ResNet 128/128 correct、9/9 feasible、0 unexpected，但 median/p90 regret 1.976×/4.494×；需 static topology/liveness cost |
| PR-11E.1 Static topology/liveness cost | validated reduced | 不读取 candidate trace；显式 shape/FLOPs/bytes/reuse/batch axes；3× replicated 1,416/1,416 correct |
| PR-11E.2 Ridge/factor LOO calibration | validated reduced | topology-density v3；6-family/36-budget LOO 选择 ridge=.001、factor=1.30；manifest 固化 |
| PR-11E.3 Replicated held-out | validated reduced | 聚合后 23/23 feasible、0 unexpected；median 1.000×/1.194×/1.880×；p90 1.747×/1.194×/2.377× |
| PR-11E.4 Production candidate foundation | validated foundation | static summary→model load→candidate generator→plain-CROWN bounded runtime；真实 OOM v3 3/3 |

## PR-11 冻结 Claims

- `C2-E1` validated-reduced：三组 replicated held-out 共 23/23 产生可行计划，0 unexpected OOM；
- `C2-E2` validated-reduced：380 MiB CUDA cap 下 dense OOM 后 structured recovery 3/3；
- `C2-E3` partial：mini s32/s128 median regret 为 1.000×/1.194×；
- `C2-L1` validated limitation：branched topology median regret 仍为 1.880×；
- `C2-L2` validated limitation：9 个 regret>=1.5 case 全部首先归因为 bounded candidate set
  未包含 measured oracle；7 个仅带待验证的 backend-gap flag；
- `C2-S1` pending：full-scale same-solver BaB 与 time-to-verify 尚未验证。

归因细节见 `gemini_doc/pr11_regret_attribution_2026_07_13.md`。PR-12 只验证 fused backend
是否改善 Pareto frontier，不改写 PR-11 历史 Planner/profile 结论。

## PR-12 当前证据

- `C1-E4` validated kernel foundation：fused ReLU+Linear/Conv PrimFunc 在 reduction 中内联
  sign/slope/bias，pre/post schedule 0 intermediate allocation，不写回完整 `A_scaled`；
- `C2-E4` validated foundation：placement/backend 已拆分，Linear/Conv capability 对
  grad/α/β/split/dtype/device/dynamic shape 和不支持的 Conv 属性显式拒绝；
- `C2-E5` partial sanity：4 个 calibration 点中 3 个快于 PyTorch dense eager，stride-2 medium
  为 1.717× slowdown；尚无正式 latency-memory Pareto、end-to-end 或 final held-out；
- `C2-E6` validated correctness closure：显式 single-consumer Affine→ReLU step、graph/contract
  runtime validation、fanout safe fallback、后端无关 executor、DLPack zero-copy storage alias、
  TVM-FFI custom-stream bridge，以及 chain/residual/multi-block mini-ResNet 最终 bound 对齐；
  尚不等价于正式性能验证；
- `C2-L2` validated current limitation：只支持 static FP32 CUDA plain CROWN、Linear 与
  groups=1/dilation=1 的有限 Conv 子集；
- `C3-M1` pending：compile amortization 与 repeated-query stream 尚未测量。

PR-12E/F 正式证据更新：

- `C2-E7` validated mechanism/Pareto：calibration 12/12、frozen held-out 24/24 candidate rows
  correctness 通过；default/custom stream 均用同 stream CUDA Events，无 timed global sync；
- `C2-E8` validated memory frontier：5 个 held-out 的 fused peak 全部低于 eager；64 MiB
  memory-sensitive Linear 中 eager 68.599 MiB、fused 29.282 MiB，只有 fused 满足预算；
- `C2-E9` guarded Planner：5/5 预算可行、0 unsafe、median/p90/max regret
  1.000×/1.262×/1.262×；fanout fallback 1/1，但 profitable 或 budget-required 仅 3/5；
- `C2-L3` validated limitation：unseen Conv 与三 block mini-ResNet warm speedup 仅
  0.792×/0.968×，memory-sensitive Linear 0.238×；当前 schedule 不能作为 latency headline；
- `C3-M1` partial：warm-faster 点 compile break-even 约 2.2k–7.4k queries；尚未接真实
  repeated-query runtime/BaB stream；
- 工件链：`artifacts/phase7a-pr12/pr12e-calibration-v1-20260713/` →
  `pr12f-final-heldout-v1-canonical-20260713/` → `pr12ef-report-v1-canonical-20260713/`。

PR-12G 多后端证据更新：

- `C2-E10` validated reduced：新增 `pytorch_chunked_r512`，每次只物化有限 query rows 的
  scaled-A，并复用 cuBLAS/cuDNN；Linear/Conv、default/custom stream 和真实 CROWN execution
  step backend contract 均有回归；
- `C2-E11` validated reduced Planner：全新 v2 split 上 calibration 48/48、held-out 36/36
  candidate rows 正确；5/5 budget feasible、0 unsafe、exact Oracle 3/5、median/p90 regret
  1.000×/1.054×，eager/chunked/TIR 各选择 1/2/2 次；
- `C2-E12` validated budget Pareto：memory-sensitive Linear 中 chunked 2.217 ms / 54.08 MiB，
  eager 3.284 ms / 65.69 MiB，64 MiB 下只有 selected candidate 可行；
- `C2-L4` validated limitation：selected geomean 仅为 eager 的 1.081×，尚无 structured eager/
  TVM-unfused 完整正式对照；TIR long-reduction schedule 仍不是 latency headline；
- authoritative 工件链：`pr12g-multibackend-v2-freeze-20260713/` →
  `pr12g-multibackend-v2-calibration-canonical3-20260713/` →
  `pr12g-multibackend-v2-final-canonical3-20260713/` →
  `pr12g-multibackend-v2-planner-replay-canonical3-20260713/` →
  `pr12g-multibackend-v2-report-canonical3-20260713/`。

PR-12H benchmark contract freeze：

- `C2-M2` validated evidence boundary：机器可读合同区分 preallocated kernel、region-runtime 与
  complete final-bound 三层 inclusion/allocation/synchronization；
- `C2-L5` validated limitation：PR-12 fused-sanity 的 PyTorch/TVM allocation contract 不同；
  PR-12E/G candidate timing 又把 region matching/Planner 放在 timed call 外，二者均标记
  `compliant=false`，历史数据不得冒充正式三层合同；
- freeze tag：`pr12g-validated-reduced` → `44f87ae`；规范见
  `docs/pr12_benchmark_contract.md`，持续状态见 `gemini_doc/pr12_execution_status.md`；
- `C2-E13` validated baseline：PR-12I 新合同下 72 rows 为 54 ok、18 N/A、0 correctness
  failure；structured eager 只在 complete final-bound 比较，TVM-unfused 在 region/E2E 都显式
  物化 scaled-A；default/custom stream 均通过；
- `C2-E14` validated attribution/limitation：TVM fused E2E geomean speedup 仅 0.546× eager，
  但 median peak ratio 为 0.512 且 3/3 Pareto；TVM-unfused 为 0.481×、0/3 Pareto，说明显存
  收益来自 fused materialization elimination，但当前 latency 不能成为 headline；
- `C2-L6` validated limitation：`torch.compile(fullgraph=True)` 在 3 workloads×2 streams 均因
  final-bound host path 的 `ContextVar.set` 无法 capture，结构化记录为 N/A，没有改写 workload；
- PR-12I 工件：`pr12i-baseline-v2-20260714/` →
  `pr12i-baseline-report-v2-20260714/`；下一门禁为 PR-12J compile/load/cache amortization。

PR-12J compile/cache 证据更新：

- `C3-M1` validated measurement：cache key 覆盖 signature/target/code schema/TVM ABI，`.so` 与
  manifest SHA 校验；3/3 workload 的 fresh compile、memory hit 与独立进程 disk hit 数值正确，
  worker 0 hidden recompile；
- `C3-E1` partial regime：mini-ResNet fused warm 6.847 ms vs eager 7.234 ms，fresh/disk-first/
  process restart break-even 为 4668/1062/4450 queries；均超过 Q=1024，且不优于 chunked
  6.513 ms；
- `C3-L1` validated limitation：Linear/Conv fused warm 分别 8.557/3.301 ms，均慢于 eager 与
  chunked，因此严格为 `not_amortizable`；3 个 workload 在 Q≤1024 内 0 个可对 eager 摊销；
- v1 tuple/list manifest bug 与 v2 warm-path SHA 污染保留；authoritative 工件为
  `pr12j-amortization-v4-20260714/` → `pr12j-amortization-report-v4-20260714/`。

PR-12K profiler 证据更新：

- `C2-E15` validated activity profile：6 workload×5 backend 共 30/30 complete final-bound rows
  correct；raw Chrome trace、kernel/API activity CSV、图与 SHA manifest 闭合；
- `C2-E16` validated mechanism boundary：fusion 对 TVM-unfused 每个 eligible region 只减少
  2 launch，六点最大整体 launch 降幅 1.96%；按 5% CUPTI device-time 阈值为 3/6 退化、
  1/6 改善、2/6 中性；
- `C2-L7` validated tooling limitation：Nsight Compute 2026.1.1 实测 `ERR_NVGPUCTRPERM`，
  禁止 SpeedOfLight、bandwidth/cache、occupancy 和 stall claim；不根据缺失 counter 猜测；
- `C2-D1` validated decision：PR-12L 唯一分支为 `E_STOP_OPTIMIZING_TIR`；保留 fused 为
  Planner candidate，但停止无 counter 支撑的孤立 schedule 调优；
- authoritative 工件：`pr12k-cupti-v3-20260714/` →
  `pr12k-cupti-report-v4-20260714/`。

PR-12L 止损决策：

- `C2-D2` validated scope freeze：唯一选择 `E_STOP_OPTIMIZING_TIR`，PR-12 closure 不再增加
  Linear tile、CUDA Graph、chunk-size family 或 Conv capability；
- `C2-D3` validated backend boundary：不删除 fused backend；PR-12M 仍可在预算或 amortized
  latency 合适时选择它，避免把局部负结果误写成后端全面失败；
- `C2-L8` validated evidence limit：如果未来获得硬件 counter 或新 workload，必须用新假设/
  新 split 重新开启，不能回写 PR-12K 或消费冻结 final held-out。

PR-12M compile-aware Planner 证据：

- `C2-E17` validated-reduced：capability→budget→risk→amortized latency 决策显式使用 expected
  reuse、memory/disk cache probability 与 fresh/disk setup；
- `C2-E18` validated held-out isolation：v3 split 在 final 未消费时冻结，calibration/final 各
  25/25 correct，fit/replay model SHA 完全一致；
- `C2-E19` validated multi-regime：75 decisions 中 72 个存在可行 candidate，Planner 72/72
  选到可行 backend、0 unsafe；feasible median/p90/max regret 1.000×/1.000×/1.016×；
- `C2-E20` validated nontrivial selection：总选择 eager/chunked/structured/fused 为 47/12/3/13；
  fused 从 cold/mixed 各 1 次增至 warm Q1024 的 11 次，32 MiB 下四类 backend 都出现；
- `C2-L9` validated capacity limit：memory-heavy Linear 在 16 MiB 下 3 个 policy 均无实测可行
  candidate；单独报告，不用不可行区 regret 污染/美化 feasible gate；
- authoritative 工件链：`pr12m-compile-aware-v3-freeze-20260714/` → calibration → model-freeze
  → final-heldout → `pr12m-compile-aware-v3-replay-v2-20260714/` → report。

PR-12N closure：

- `C2-CLOSE` validated-reduced：H–M 门禁、hash、失败与限制已审计，closure tag 为
  `pr12-validated-reduced`；
- 不能升级 `VALIDATED`：Q≤1024 compile amortization 0/3、硬件 counter unavailable、收益仅限
  部分 regime、尚无真实 BaB/VNN-COMP；
- 不降级 `MECHANISM-ONLY`：non-toy mini-ResNet/Conv E2E Pareto、预算可行性、自动多 regime
  selection 与独立 held-out 已成立；
- PR-13 gate 为 GO/READY，但尚未启动；closure audit 与 Artifact Appendix 分别见
  `gemini_doc/pr12_closure_audit_2026_07_14.md`、
  `gemini_doc/pr12_artifact_appendix_2026_07_14.md`。

## PR-13A Query/State Contract 证据

- `C3-M2` validated foundation：`BoundQuery` 显式覆盖 parent、model/weight/input/spec/split、
  method/stage、α/β/cuts、dtype/device/numeric policy 与 requested outputs，canonical JSON 确定；
- `C3-M3` validated foundation：完整 `QueryCompatibilityKey` 分组；αβ/split 强制
  `alpha_beta_dense_split` capability，不会误选 PR-12 plain-CROWN fused TIR；logical
  `QueryBatch` 拒绝 mixed key，并验证 pack/unpack order/result restoration；
- `C3-M4` validated foundation：state validity 对 graph/kernel/planner/intermediate/α/β/cuts/final
  显式返回 EXACT/CONDITIONAL/WARM_START/INVALIDATE；父 β/final 不可 exact reuse；
- `C3-E2` validated smoke：真实 `solve_bab_mlp` driver 产生 8-query 父子流，8/8 replay、
  max abs diff 0、0 query loss、0 duplicate；
- `C3-L2` validated limitation：工件为 CPU two-ReLU smoke，尚无 dynamic batch、OOM split、
  same-solver multi-backend、non-toy/TTV/tail-latency，不能作为性能或完整 C3 claim；
- 工件：`artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714/`；持续状态：
  `gemini_doc/pr13_execution_status.md`。

## PR-13B Dynamic BatchManager 证据

- `C3-M5` validated foundation：exact-key buckets、budget first-fit、fill/timeout/deadline wakeup、
  deterministic OOM bisection 与 ID-based order restoration；
- `C3-M6` validated foundation：physical αβ executor pack/unpack center/spec/split/α/β，并继续强制
  dense split capability；perturbation 与 execution-options 进入 compatibility；
- `C3-E3` validated smoke：真实 8-query stream 动态形成 3 batches，8/8、max diff 0、0 loss/
  invalid；deadline flush 与 queue-wait 分位数字段存在；
- `C3-E4` validated fault path：显式 OOM fault 触发 8→4+4→2+2+2+2，3 events/splits，最终
  8/8、0 loss；
- `C3-L3` validated limitation：CPU、逻辑 clock、fault OOM；尚无 same-solver live adapter、真实
  GPU OOM、non-toy throughput/TTV；
- 工件：`artifacts/phase7a-pr13/pr13b-dynamic-batch-v7-20260714/`。

## PR-13C Same-Solver Adapter 证据

- `C3-M7` validated foundation：原 `solve_bab_mlp` 继续拥有 branch/heap/node order/termination；
  optional adapter 只替换 single/batched bound-call execution；
- `C3-M8` validated foundation：runtime result 携带真实 α/β tensors，solver 可继续 warm
  start/cache；comparison 强制 state tensor 数值对齐，exact content hash 仅作诊断；
- `C3-E5` validated smoke：αβ steps=3、batch=4 下 original/runtime query IDs 7/7，per-query
  bounds/branch/αβ state 7/7，status/node counters/best bounds 一致，0 loss；
- `C3-E6` validated capability guard：forged plain-CROWN capability 在 αβ physical executor 0 次
  调用时拒绝；alpha-only serial adapter 也与原 solver 对齐；
- `C3-L4` validated limitation：toy CPU smoke，单次 wall time non-authoritative；尚无 non-toy
  fixed-tree/E2E、TTV、真实 GPU OOM/stream、plan/cache ablation；
- 工件：`artifacts/phase7a-pr13/pr13c-same-solver-v5-20260714/`。

## PR-13D/E Reduced GPU 与 Closure 证据

- `C3-E7` validated-reduced：RTX 4060、5 repeats、16-query fixed stream，runtime 相对 per-node
  96.52×，相对 batched original 1.024×，16/16 correctness；
- `C3-E8` validated-reduced：同一 solver hard E2E 16 nodes，runtime 相对 per-node 9.93×，相对
  batched original 0.980×；三 variant status/node count 一致；
- `C3-M9` validated foundation：custom CUDA stream event-only test、dispatch cache 1 miss/4 hits、
  query loss/invalid 为 0；
- `C3-L5` validated limitation：收益主要来自 ordinary batching；easy root 为负收益；
  `compiled_plan_cache_applicable=false`、`pr12_planner_dispatches=0`；
- `C3-L6` validated limitation：chain-CNN 16 nodes，不是 VNN-COMP/non-toy；真实 GPU OOM、
  branch/prune/GPU-active 分解未完成；
- 工件：`artifacts/phase7a-pr13/pr13d-bab-runtime-v5-20260714/`；closure：
  `gemini_doc/pr13_closure_audit_2026_07_14.md`。

## PR-14A/B 真实 Verifier Coverage 与 No-Go

- `C3-E9` validated coverage：官方 MLP/CNN 与 VNN-COMP ResNet-2B 共 540 个真实
  `compute_bounds`；initial 143/146 region-level eligible，activation-BaB 0/394；
- `C3-M10` validated foundation：external observer 可撤销，on/off status 与 visited domains
  一致；exact Box perturbation 保留 VNNLIB per-element clipped bounds 与 query identity；
- `C3-E10` validated narrow equivalence：simple MLP 的 external replay 与 BoundFlow
  eager/chunked/TVM lower 全部 max diff 0；但 external lower-only、BoundFlow lower+upper，公平
  performance 为 N/A；
- `C3-L7` validated non-toy limitation：ResNet nominal forward 对 ONNX max diff `1.67e-6`，但
  BoundFlow whole-query lower 对 external max diff `796.765`，符号只一致 3/9；
- `C3-CLOSE` validated-no-go：activation route 与 initial whole-query replacement 均未过门禁，
  PR-14C blocked；C3 降级为 C1/C2 基础设施，禁止 verifier acceleration claim；
- 证据：`gemini_doc/pr14a_real_query_coverage_2026_07_19.md`、
  `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`。

该段 PR-11 early evidence 当时为专项 21 passed、全量 200 passed/1 skipped；其“Global 与
Memory-Threshold 决策相同”的历史限制已由后续 PR-11E 和 PR-12G 证据分别补充，不能再读作
当前全量状态。PR-12G 收尾全量为 318 passed、1 skipped。

第三切片与 profiler 完成后全量为 208 passed、1 skipped。Global 已在 multi-barrier 合成案例中做出非阈值式
mixed placement，但在真实 held-out workload 上尚无 barrier-level cost/Oracle 证据，C2 状态
仍为 `partial`。有界分层 retry 已把第二 query scale 的最坏 56 次 replay 限制到 5，并在两个
reduced held-out query 上通过 median/feasibility 门禁；证据仍局限于一个 architecture family，
不足以把 C2 整体标记 validated。

有界分层 retry 切片收尾验证：全量 216 passed、1 skipped；Mypy 11 files success；Planner 与
PR-11 脚本逐文件 Pylint 10.00/10；`git diff --check` 通过。

独立 branched-ResNet topology 明确否决了当前 v1 aggregate cost model：feasibility 成立但 regret
门禁失败；同时 evaluator 仍依赖 candidate-specific trace logical bytes，属于 profile-guided replay。
C2 保持 partial，下一实现切片改为 static topology/liveness-aware cost summary。
加入独立 topology contract 后最新全量为 217 passed、1 skipped，profiler Mypy/Pylint 与 diff
check 通过。

Static-v3 已消除 candidate-trace feature 依赖，并显式覆盖 shape/FLOPs/bytes/reuse/batch axes。
3× replicated profiles 共 1,416/1,416 correct；聚合后三组 held-out 全部通过 feasibility/median
门禁，p90/max 最坏为 2.377×/3.160×。Production candidate foundation 与真实 OOM 3/3 已成立；
C2 标记 validated-reduced，不能解释为论文级 complete。
