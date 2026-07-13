# BoundFlow ASPLOS Claims Map

> 本表是动态证据账本。`planned` 不代表已经实现；只有代码、测试和工件均存在时才能改为
> `validated`。当前执行基线为 PR-10 complete、PR-11 validated-reduced，下一工程切片为 PR-12。

| Claim | 当前状态 | 代码/设计落点 | 必需测试 | 必需工件 |
|---|---|---|---|---|
| C1：显式物化语义的 Structured Bound-Operator IR | validated foundation（PR-10 guarded） | `boundflow/runtime/linear_operator.py`、`crown_ibp.py` | dense/operator 数值与 gradient 对齐；materialization trace | count/bytes/reason/lifetime JSONL |
| C2：Method/Autograd/Memory-Aware Materialization Planner | validated-reduced（PR-11） | static topology/liveness summary、global candidate model、bounded runtime；CROWN/α/αβ capability 接口 | 3× replicated correctness、LOO、held-out/Oracle、真实 OOM | 1,416 executions→472 aggregate patterns；23/23 feasible；manifests |
| C3：BaB-Oriented Repeated-Query Runtime | partial（现有 node batch/cache） | `boundflow/runtime/bab.py` 等；QueryState 接口待 PR-13 | same-solver executor 对齐、state validity | TTV、solved/timeout、p90/p99、batch fill |
| TVM 后端执行 Planner 结果而非定义核心抽象 | partial | `boundflow/backends/tvm/`、`runtime/tvm_executor.py` | Python/TVM/unfused/fused 对齐 | compile/cold/warm、launch、bytes |
| 相同浮点语义下保持 reference bound computation | partial | dense reference + planned paths | allclose、gradient、auto_LiRPA、replay | correctness fields 与失败记录 |

## PR-10 子阶段

| 子阶段 | 状态 | 完成证据 |
|---|---|---|
| PR-10A Materialization instrumentation | validated | `25225e5`；ReLU barrier opt-in trace |
| PR-10A.1 Trace schema v1 | validated | `boundflow.materialization/v1`、schema contract tests、164 passed |
| PR-10B.1 workload characterization | validated | `8f2c998`；180/180 clean GPU profile；mini-ResNet s128/d32 |
| PR-10B.2 真实 BaB fixed-domain replay | planned | 当前仅 `synthetic_fixed_domain_batch`，不得描述成 BaB 结果 |
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
