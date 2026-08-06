---
status: fsg0-post-audit-validated
updated: 2026-08-06T13:41:43Z
type: plan
topic: boundflow
slug: BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1
stage: s01
---

# BoundFlow Full-Stack GPU Baseline and Attribution v1 Plan

## 0. Route Correction

NRIR49A 的正式数据与 artifact 保持有效，但其判定作用域必须收窄为：

```text
VALIDATED-NO-GO(selected-CROWN-only incremental optimization)
```

`s_selected=0.070986` 给出的

```text
1 / (1 - s_selected) = 1.0764x
```

只是假设当前 selected-CROWN region 变为零耗时的 deletion-only Amdahl 上限。它不约束 BoundFlow
从算子、Bound/Graph IR、Plan/Schedule、跨阶段融合、JIT/CUDA Graph、runtime batching/streams 到
arena/buffer reuse 的累计全栈收益。正式 G1 artifact 的 `next_route=gpu-winner-reselection` 是冻结历史
机器输出，不修改其 payload、manifest 或 hash；当前路线由本文档取代。

此前路线的结构性错误不是 G1 数据错误，而是把一个单区域 gate 放在全栈公平 baseline 之前，并让
单区域 share 承担了系统级 kill gate。单区域归因今后只决定工程优先级或关闭该区域专属实现，不再
关闭 BoundFlow full-stack 路线。

## 1. Goal

- 建立官方 αβ-CROWN host solver 内 original executor 与 BoundFlow replacement executor 的
  same-solver GPU 公平基线；
- 对 operator→graph/IR→JIT→runtime schedule→allocator/memory 的完整 critical path 做互斥闭合；
- 用累计消融与 leave-one-out 量化各层增量及交互，而不是把单个现有 region 的 share当作系统上限；
- 最终以 complete-query、TTV、solved verdict、nodes/s、peak allocated/reserved 和公平 E2E 决定
  BoundFlow GPU 系统主张。

当前总状态为 `UNMEASURED / NOT-YET-AUDITABLE`：尚无合格的 BoundFlow-vs-original GPU
same-solver speedup。本文完成不等于 replacement executor、TIR、JIT或性能结果已经实现。

## 2. Scope and Non-Goals

允许：

- 新增 profiling schema、RVIR payload、对称 worker/RPC、artifact/replay和测试；
- 在不改变数学语义的前提下接入 BoundFlow replacement executor；
- 独立实现并累计启用 operator/cross-stage fusion、IR/graph、Plan/Schedule、JIT、runtime和memory层；
- 对当前 ResNet2B与公开可solve workload做read-only baseline和分层归因。

禁止：

- 用 BoundFlow自有 `solve_bab_mlp` 对比官方 αβ-CROWN并称 same-solver；
- 把 RVIR 当前“validate后exactly-once调用原compute_bounds”的transport壳称为replacement；
- 用不同算法、branch、timeout、iteration、batch或termination产生headline speedup；
- 只测kernel/device sum后外推complete-query；
- 把 profiler运行自身的wall time作为headline；
- 用人为调低软件显存budget伪造physical-memory claim；
- 修改NRIR49A冻结artifact来追认本次路线纠正。

## 3. Existing Evidence and Hard Gaps

已具备：

- G0 post-reboot：RTX 4060、两套Torch CUDA、TVM CUDA TIR、TVM-FFI stream与跨环境digest通过；
- RVIR typed `Bound→Plan→Task→Schedule` external exact-call transport与CPU correctness/replay；
- official αβ-CROWN checkout `e5c7e17`、auto_LiRPA `5a098e8`和独立`.venv`；
- NRIR49A CUDA event、CUPTI、allocator、fresh worker、artifact/replay与GPU parity基础；
- 一个双方均可在冻结timeout内产生`verified`的公开`mnistfc:2` qualification workload。

当前硬缺口：

- `execute_external_verifier_call()`仍执行原external callable，不是BoundFlow replacement；
- PR13C `SameSolverQueryRuntime` 的host是BoundFlow自有solver，不是官方αβ-CROWN；
- activation-BaB replacement需要完整live alpha/beta/split/cuts、intermediate bounds、requested polarity、
  optimizer policy与mutation ownership；当前RVIR identity记录不能直接执行这些数值状态；
- competitor Python 3.11/Torch 2.11环境当前不能import本机TVM，BoundFlow环境则是Python 3.12/
  Torch 2.12；headline必须建立combined env或对称RPC，不能使用非对称IPC；
- 当前没有original/candidate共同的GPU full-stack hierarchical raw trace。

## 4. Fair Comparison Boundary

### 4.1 Primary A/B

唯一headline主对照：

```text
same official alpha-beta-CROWN host solver
  control: original auto_LiRPA/alpha-beta-CROWN batched executor
  candidate: RVIR typed request -> BoundFlow replacement executor
```

两侧必须固定同一repo/config/model/property/input、seed、branch policy、alpha/beta iteration、split/cuts、
domain/spec batch、node budget、timeout、termination与requested bounds。逐call核对sequence、parent、state
version、payload digest、call order、bounds、branch、node accounting和最终verdict。

首选同process combined environment。若依赖冲突不可解，只允许：

- control与candidate都跨同一种worker/RPC边界；
- 输入序列化、IPC、同步和错误处理成本计入双方；
- worker持久性、cache与warmup规则对称；
- 跨env torch数值差异单列，不把非对称transport时间归给某一backend。

### 4.2 Diagnostic Comparisons

以下只能诊断，不能headline：

- BoundFlow自有complete verifier vs official αβ-CROWN（算法不同）；
- current BoundFlow runtime的内部chunk或单region sweep；
- TVM unfused、Torch eager、TorchInductor、source-level BoundConv oracle；
- profiler on运行、单次microbenchmark和历史PR-12 fused artifact。

### 4.3 Replacement Maturity Ladder

RVIR接入必须显式记录replacement mode，禁止把transport层存在误判为BoundFlow已经执行：

| mode | 物理行为 | 可用于什么结论 |
|---|---|---|
| A0 `original_provider` | 直接执行official original callable | B0 control |
| A1 `rvir_passthrough` | typed validate后仍exactly-once执行original callable | transport/contract开销 |
| A2 `shadow_only` | original给solver结果，BoundFlow旁路执行并比对 | correctness与覆盖率，不能timing |
| A3 `nested_region` | official call内部一个合法region由BoundFlow物理替换 | 首个可准入局部performance的模式 |
| A4 `whole_call` | 整个external bound call由BoundFlow执行 | 完整replacement与headline候选 |

A2即使数值exact，也不构成replacement speedup；A3/A4必须证明candidate结果没有回调original provider、
没有未披露fallback，且所有mutation ownership已闭合。B2最初可在A2完成correctness，但进入FSG3 timing
前必须提升为A3或A4。

## 5. Hierarchical Attribution Contract

每个span/event必须同时记录四个正交轴，禁止只用一个扁平“winner”标签：

| 轴 | 冻结取值 |
|---|---|
| stack layer | `solver_control / adapter_transport / ir_graph / plan_schedule / backend_compile_jit / operator_execution / graph_boundary / runtime_schedule / memory_allocator / unclassified_residual` |
| solver phase | `setup / initial_crown / selected_crown / alpha_optimize / beta_split / intersect / forward_propagate / branch_score / queue_commit / termination / unclassified` |
| resource | `host_thread / cuda_stream / cuda_runtime_api / memory / ipc` |
| cache state | `cold_compile / process_hit / disk_hit / warm_execute / not_applicable` |

时间口径分离：

- `host_wall_scope_ns`：外层真实wall；
- `gpu_union_ns`：所有stream device区间并集；
- `gpu_sum_ns`：kernel/device事件总和，只用于诊断；
- `critical_path_ns`：按stream event、host wait与dependency重建的关键路径；
- `exclusive_critical_path_ns[layer,phase]`：互斥归属，必须闭合到critical path；
- compile、IPC、allocator、H2D/D2H、sync、launch gap不得塞进operator residual。

正式闭合门禁：

- exclusive critical-path sum与scope误差`<=1%`；
- 未归类 residual `<=3%`，且单项列出；
- GPU多流重叠显式记录，禁止把`gpu_sum_ns`除以wall产生share；
- profile/control paired wall中位ratio `<=1.05`，否则timing verdict=`NOT-AUDITABLE`；
- 每个scope都保存raw events、normalized spans、dependency edges与closure table。

只有baseline-side critical-path share可用于机会判断；candidate-side“已经变小的share”不能反向抹掉
BoundFlow已有贡献。跨层融合会改变边界，其收益进入interaction ledger，不强行分摊给单一层。

### 5.1 Physical Feature Activation Ledger

每轮配置同时冻结feature activation ledger，至少包括：

- Bound/Graph IR是否真的驱动执行，而不是仅在执行后生成对象；
- Plan是否在执行前编译、Task/Schedule是否真的发起物理dispatch；
- backend identity、physical backend dispatch数、fallback dispatch数；
- cold/process-hit/disk-hit/warm cache状态；
- stream/event/wait计数与依赖边；
- storage plan是否被allocator实际执行；
- A0—A4 replacement mode。

声明某层收益时，该层要求的activation字段必须为真；对象存在、hash存在或post-hoc lowering都不能替代
物理激活证据。若candidate声称B3—B7但ledger缺少对应feature，artifact必须fail closed。

## 6. Cumulative System Ablations

冻结累计链：

| ID | 配置 | 作用 |
|---|---|---|
| B0 | original αβ-CROWN batched executor | 公平control |
| B1 | B0 + RVIR typed transport，仍执行original callable | 量化adapter/IR壳开销 |
| B2 | B1 + BoundFlow replacement executor，reference operators | replacement correctness与基础执行 |
| B3 | B2 + Bound/Graph IR + Plan/Schedule复用 | IR/graph与compile/cache层 |
| B4 | B3 + operator及跨阶段fusion | TIR/kernel/materialization boundary |
| B5 | B4 + conditional JIT/CUDA Graph | compile amortization与launch graph |
| B6 | B5 + runtime batching/streams/branch pipeline | 调度、并行和critical path |
| B7 | B6 + arena/buffer reuse | allocator、workspace与physical peak |

每个ID是可独立关闭的功能集合，并进入Plan/Schedule/manifest hash。正式结果同时报告：

- cumulative `B0→Bi`；
- incremental `B(i-1)→Bi`；
- leave-one-out `B7→B7-minus-i`；
- interaction residual。若交互显著，禁止声称单层收益可线性相加。

实现仍遵循“一次提交一个可审计变量”，但系统benchmark从B0起始贯穿所有阶段；单层未过局部收益门禁
只关闭该实现或降级优先级，不杀死未测试的其它层。

## 7. Execution DAG

```text
FSG0 scope correction + schema + replay contracts
  |
  v
FSG1 official control full-stack trace + perturbation/closure
  |
  v
FSG2 RVIR-v3 executable payload + mutation/ownership + replacement correctness
  |
  v
FSG3 paired same-solver B0/B1/B2 baseline (5 fresh AB/BA)
  |
  +--> correctness fail: stop replacement, keep control attribution
  v
FSG4 B3-B7 cumulative + leave-one-out implementation/ablation
  |
  v
FSG5 multiworkload complete-query/TTV/solved/peak external audit
```

### FSG0 — 当前切片

- 修正所有权威文档中的G1作用域；
- 新增full-stack layer/phase/resource/cache schema、critical-path closure与joint Amdahl聚合器；
- 新增结构、overlap、double-count、residual、tamper和interaction测试；
- 不运行性能，不修改production。

### FSG1 — Original Control Attribution

- 在official competitor env内hook compute_bounds、solver phase、CUDA runtime/profiler与allocator；
- 固定ResNet2B prop0和`mnistfc:2`，各5 fresh paired control/profile；
- profiler用于分类，unprofiled control用于wall；
- 产出B0 hierarchical baseline，不能称BoundFlow speedup。

### FSG2 — Replacement Correctness

- RVIR-v3 capture executable tensors/state，而非只记录identity/hash；
- state mutation采用copy-in/copy-out receipt或明确external-owned alias合同；
- BoundFlow backend必须独立执行，不得回调original callable；
- targeted initial/alpha/beta/split、lower/upper、batch/ragged、reject/fallback测试；
- 正式逐call、branch、parent、node、verdict exact后才准入timing。

### FSG3—FSG5

- FSG3只建立B0/B1/B2 same-solver性能基线；
- FSG4逐层实现B3—B7，所有candidate均从同一B0比较；
- FSG5扩到至少两个held-out family和一个双方同timeout非unknown公开workload。

## 8. Gates

### 8.1 Correctness

- query/state/parent/order/call count exact；
- shape/dtype/device/requested polarity exact；
- all raw floats finite；lower/upper按预注册`atol=rtol=2e-4`，离散branch/verdict exact；
- soundness方向单独检查，不允许allclose掩盖更乐观的错误bound；
- baseline/candidate solver nodes、termination reason与timeout accounting一致；
- tamper即使同步改payload digest也必须被语义重算拒绝。

### 8.2 Measurement

- 正式性能至少5 fresh repeats，AB/BA反平衡；
- GPU process排他、driver/clock/power/temperature/background process入manifest；
- cold compile、warm execute、cache hit分开；报告break-even query count；
- profiler perturbation、closure、failure rows任一失败则不形成speedup；
- 所有ratio使用paired raw，报告median、range、MAD与geomean，不只报最佳值。

### 8.3 System Outcome

最终门槛只施加到B7 vs B0累计系统结果：

- 31-node queue geomean speedup `>=1.20x`；
- complete-query geomean speedup `>=1.15x`；
- 任一合法workload不得退化超过`5%`；
- 至少一个公开held-out workload双方产生相同非`unknown` verdict；
- memory claim另要求至少两个natural memory-bound workload peak allocated下降`>=25%`且latency
  ratio`<=1.05`；
- 若只有固定kernel有收益，保留backend claim并降级compiler/planner claim；
- 若B7未过，只能关闭已完整测试的累计配置，不得用一个region结果外推未实现层。

## 9. Artifact and Replay

正式artifact最少包含：

```text
manifest.json
environment.json
workloads.jsonl
paired_runs.jsonl
raw_events.jsonl
normalized_spans.jsonl
dependency_edges.jsonl
closure.json
ablation.json
summary.json
failure_rows.jsonl
replay_stdout.txt
README.md
```

manifest绑定三仓commit、两环境lock、model/property/config、GPU、功能集合B0—B7、代码revision和全部文件
SHA256。replay从raw events重建union/critical path/exclusive closure、paired ratios、cumulative/
leave-one-out与最终decision；同步修改summary/manifest digest不得绕过语义重算。

## 10. Initial Workloads

- `cifar10_resnet:000`：固定ResNet2B prop0，用于真实residual/activation-BaB call与31-node queue；
- `mnistfc:2`：双方已`verified`，用于非unknown verdict与complete-query smoke；
- 第二held-out family在FSG1前冻结，不从candidate结果反选；AveragePool等frontend缺口独立处理，
  不混入性能PR。

## 11. Tasks

1. [x] FSG0：文档作用域纠正、schema/aggregator/tests；
2. [ ] FSG1：official control hook、五fresh baseline artifact/replay；
3. [ ] FSG2：RVIR-v3 executable state与BoundFlow replacement correctness；
4. [ ] FSG3：B0/B1/B2 paired same-solver baseline；
5. [ ] FSG4：按B3—B7逐层实现、累计与leave-one-out消融；
6. [ ] FSG5：multiworkload E2E、solved/TTV/peak与外部审计。

## 12. Validation

- unit：schema、nested/exclusive span、overlap union、critical-path closure、joint Amdahl、interaction、
  tamper、failure rows；
- targeted：official control observer on/off、RVIR exactly-once与replacement no-fallback；
- GPU：五fresh paired、CUPTI分类与unprofiled timing分离；
- static：Black、mypy、Pylint；
- regression：`pytest tests`；
- DocOps：每次code/doc变更`dol ch add`，确定性验证`dol va add`，性能至少5 repeats，handoff前
  `dol lint --soft`。

### FSG0 Closure（2026-08-06）

- 新增 `boundflow/runtime/gpu_attribution.py`：typed layer/phase/resource/cache span、exclusive
  critical path、feature activation ledger、A0—A4 replacement mode、GPU interval union、joint Amdahl
  与累计/leave-one-out interaction聚合；
- 新增 `scripts/run_full_stack_gpu_baseline_attribution.py`：contract-only artifact generate/replay，
  绑定raw/summary/code/file digest并从raw语义重算；`performance_claimed=false`；
- 新增 `tests/test_full_stack_gpu_attribution.py`：20项schema、cycle、overlap、closure、residual、
  activation、Amdahl、interaction与digest-synchronized tamper测试；
- targeted=`20 passed`；全量激活环境=`1079 passed, 3 skipped`；Black、三个新文件mypy clean、
  Pylint=`10.00/10`、`git diff --check`通过；
- 外部审计`APPROVE-WITH-MINOR`；枚举命名、测试mypy覆盖和git provenance三项minor均已修复；
- FSG0不含GPU timing、official control trace或production替换，故没有性能claim。下一步仅为FSG1 B0
  official-control full-stack attribution。

## 13. Rollback

- FSG0只新增schema/tests/docs，可独立删除；
- FSG1 observer必须可逆，off时调用顺序与结果exact；
- FSG2 replacement默认关闭，任何错误fail closed回到original executor；
- B3—B7每层独立feature gate，不修改B0；
- 历史NRIR49A artifact与hash不改。

## 14. Links

- changelog: [Full-stack GPU changelog](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_CHANGELOG_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
- selected-only history: [NRIR49A G1 plan](BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- RVIR contract: [Real Verifier IR integration](real_verifier_ir_integration_contract_v1_2026_08_03.md)
