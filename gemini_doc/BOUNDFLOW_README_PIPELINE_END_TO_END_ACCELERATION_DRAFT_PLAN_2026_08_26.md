# BoundFlow README 主流水线端到端接通与加速计划（用户审阅稿）

status: draft-for-user-review
date: 2026-08-26
execution-authority: false
code-change-open: false
external-audit: deferred-by-user
performance-claimed: false

## 0. 一句话结论

README 给出的架构方向是对的：

```text
PyTorch / ONNX
  → BoundFlow IR
  → Global Planner
  → BoundTasks
  → TVM Backend
  → Optimized GPU Kernels
```

项目当前的主要问题不是“没有这些层”，而是这些层存在**多条历史实现路径，尚未收束为一个真实生产
入口**。接下来应停止把某个 capture、某个 kernel 或某份 schema 当成独立主线，改为连续完成两个纵向
切片：

1. `E2E-A`：用已经证明有收益的 CIBC/IBP 图打通完整 compiler pipeline，确认 IR/Planner/Task/
   Schedule/TVM/Runtime 没有吃掉 `2.45631x` 的已有收益；
2. `E2E-B`：把同一套 pipeline 扩到 αβ-CROWN production exact-call region，以 RVIR adapter 做
   same-solver 公平替换，逐步达到 B0 parity 和 complete-query 研究门槛。

GC0-1 capture/analysis 不删除，但降为 `Bound IR legality` 子阶段；它服务主流水线，不再单独决定项目节奏。

## 1. README 流程的工程展开

README 的简图对应到当前仓库，应展开为：

```text
PyTorch / ONNX
  │
  ├─ Frontend + Normalize
  ▼
Primal IR + Verification Spec
  │
  ├─ Bound semantics construction
  ├─ Verification legality analysis（GC0-1 属于这里）
  ▼
Bound IR / Verification Graph
  │
  ├─ candidate enumeration
  ├─ fusion / batching / representation / storage / backend selection
  ▼
PlanTemplate + PlanInstance
  │
  ├───────────────┐
  ▼               ▼
Task IR        Schedule IR
  │               │
  └───────┬───────┘
          ▼
TVM TIR / library / reference backend
          │
          ▼
Prepared Runtime + persistent arena/cache
          │
          ▼
αβ-CROWN / BaB host solver（termination、branch、queue 保留在 host）
```

这不是改变 README，而是把它已经声明的职责落实到当前一等 IR 对象上。

## 2. 当前做到哪里

### 2.1 已经具备的资产

| 层 | 当前资产 | 结论 |
|---|---|---|
| Torch/ONNX Frontend | 双前端、normalize、Primal IR、general DAG 子集测试 | 已能作为统一模型入口 |
| Primal/Spec/Bound IR | typed `BFBoundModule`、value/op/spec/domain、interpreter/hash | 语义骨架已存在，不应重写 |
| Verification schema | GC0-0 generic graph/effect/VJP/legality schema 已外审关闭 | 应作为 Bound legality 扩展，不应形成第二套主编译器 |
| Plan IR | `PlanTemplate/PlanInstance`、region/representation/materialization/backend/batch/storage/state decision | typed schema与selector已存在 |
| Task IR | `TaskIRModule/TaskIRUnit`、backend binding、state/external dependency | 可从 PlanInstance lowering |
| Schedule IR | allocate/materialize/transfer/launch/event/state/retry/fallback/emit action | reference verifier/executor已存在 |
| TVM backend | interval Relax/TIR、fused/unfused CROWN、CIBC、B4-B2 differentiable TIR | 快 kernel 已存在，缺统一生产绑定 |
| Backend dispatch | reference/PyTorch/TVM fused/unfused registry与typed dispatch key | 可复用，不应再写per-site旁路 |
| Runtime/solver | CROWN/α/β/BaB、RVIR exact-call adapter、state/trajectory/rollback证据 | solver控制已成熟，编译region尚未成为默认executor |
| Artifact/DocOps | raw-first/replay/tamper/external audit工作流 | 可直接复用于E2E closure |

### 2.2 当前断点

1. `planner/pipeline.py` 的老入口仍是 interval-only `PlanBundle → BFTaskModule`，与新的
   `Bound → PlanTemplate/Instance → TaskIR/ScheduleIR` 并存；
2. 新 typed IR 链有 reference execution 和部分 TVM dispatch，但没有一个用户可调用的统一
   `compile/prepare/run` API；
3. CIBC `2.45631x`、局部 CROWN TIR `4.89834x` 等收益来自专用 runner，没有全部经过 Global Planner、
   Task/Schedule 与 production runtime；
4. αβ-CROWN/BaB 的真实 state/optimizer/termination ownership 已有 RVIR 证据，但 compiler region 仍通过
   多个历史 bridge 接入；
5. Global Planner 的 typed candidate/selector存在，但 production cost evidence与真实compiled artifact
   还没成为同一个决策闭环；
6. 结果是“kernel 很快、wrapper/query 不快”：MR5约`0.8344x`，MR6只回收到`1.0331x`，B3相对B0
   query约`0.9100x`。

## 3. 本计划的北极星与边界

### 3.1 北极星

> 同一 solver、同一模型/property、同一 branch/termination/state trajectory 下，由 BoundFlow compiler
> 选择并执行 bound regions，使 complete-query time 降低，同时保持验证语义和fail-closed行为。

### 3.2 近期目标

- 只有一条 canonical compiler entry；
- 每个 production execution 都能回溯
  `Primal/Spec/Bound/Legality/Plan/Task/Schedule/Module` hash；
- Planner 选择的 TVM backend 真正执行，而不是trace里选TVM、runtime里仍走PyTorch；
- per-op Python/DLPack/allocation从compiled region消失；
- standalone图收益能传播到same-solver region，再传播到complete query。

### 3.3 不在近期范围

- 把整个 BaB 控制流编译到GPU；
- 通用动态shape或任意ONNX op；
- certified training；
- 没有实测crossover证据的“智能全局planner”claim；
- 同时推进多stream、CUDA Graph、autotuning、distributed solver；
- 仅凭kernel microbenchmark升级ASPLOS claim。

## 4. 总体执行顺序

```text
R0 统一入口与事实盘点
  ↓
R1 E2E-A：IBP/CIBC纵向闭环（先证明pipeline不吃收益）
  ↓
R2 Planner真实候选与Task/Schedule/TVM绑定
  ↓
R3 Prepared Runtime：persistent arena/cache/单region提交
  ↓
R4 E2E-B：CROWN structured region + custom backward
  ↓
R5 RVIR same-solver exact-call替换
  ↓
R6 Global Planner扩展、并行/内存/JIT按证据逐项开放
  ↓
R7 complete-query / queue formal
```

每一级有kill gate；前一级没有把收益传到自己的scope，后一级不开放。

## 5. R0：统一 compiler entry，不再新增旁路

### 5.1 目标 API

冻结一个生产入口，名称可在实现前确认，但职责固定：

```text
compile_boundflow(
    primal_program,
    verification_spec,
    hardware_profile,
    workload_profile,
    runtime_context,
) -> PreparedBoundProgram

PreparedBoundProgram.run(dynamic_state) -> BoundResult + ExecutionReceipt
```

内部必须按顺序产生：

```text
BoundModule
LegalityResult
PlanTemplate
PlanInstance
TaskIRModule
ScheduleModule
CompiledArtifactSet
PreparedRuntime
```

### 5.2 兼容策略

- 老 `PlanBundle` 作为 interval frontend compatibility adapter，不再是论文主Plan IR；
- 专用 CIBC/B4/R3 runner只保留oracle、benchmark和artifact生成，不直接成为production入口；
- `VerificationGraph`由Bound region legality pass消费，不与`BFBoundModule`竞争顶层owner；
- RVIR adapter只负责solver边界和state ownership，不拥有compiler内部planning；
- 所有后端通过`TaskBackendBinding + BackendDispatchKey`进入，禁止按site调用私有executor。

### 5.3 R0交付与门禁

交付：一份exact mapping table、统一API contract、旧入口deprecation表、端到端hash receipt schema。

GO：没有语义对象同时被两条production path拥有；所有历史组件有明确adapter或retire状态。

STOP：如果统一入口仍只是把原runner包一层、没有真实Plan/Task/Schedule hash传播，则不进入R1。

## 6. R1：E2E-A IBP/CIBC纵向闭环

### 6.1 为什么先用IBP

CIBC整图已有`2.45631x`，是当前最强且最清楚的“GPU kernel收益可传播到一张图”证据。先用它验证
compiler plumbing，比直接进入复杂α/β optimizer更容易定位新增开销。但它只是流水线资格证明，不是最终
solver claim。

### 6.2 固定路径

```text
Torch + ONNX同一模型
→ Primal IR exact-equivalent
→ interval Bound IR
→ PlanTemplate候选：PyTorch reference / TVM-CIBC fused
→ PlanInstance
→ TaskIR + ScheduleIR
→ TVM compiled artifact
→ Prepared runtime
→ lower/upper输出
```

输入copy、layout normalization、dispatch、result materialization必须全部计入wrapper计时；compile cold
cost单独披露，不混入warm headline。

### 6.3 R1 correctness gate

- Torch/ONNX进入后产生相同canonical Primal/Bound semantics；
- reference、direct CIBC、pipeline TVM三方lower/upper数值与sign通过冻结容差；
- residual/fanout与全部17-op graph不丢贡献；
- Plan→Task→Schedule→module hash逐层绑定；
- fallback/eager/native shadow可见且formal candidate为0；
- cache miss→warm hit行为可重放。

### 6.4 R1 structural gate

- 一个graph-level prepared invocation；
- compiled region内部per-op Python dispatch/DLPack crossing/dynamic output allocation=`0`；
- persistent input/output views与workspace复用；
- internal kernel数如实披露，不把一次submission伪写成一个kernel；
- old `TVMTaskExecutor`路径与new TaskIR dispatch结果一致，但production只保留一个owner。

### 6.5 R1 performance qualification

历史direct CIBC graph geomean=`2.45631x`只作为起点。冻结两条资格线：

1. pipeline/direct-CIBC geomean `>=0.90x`，即compiler plumbing最多消耗10% direct winner收益；
2. pipeline/PyTorch graph geomean `>=2.20x`，最差pair `>=2.00x`。

若正确性/结构通过但性能不过，先只拆compile entry、dispatch、copy/layout、allocation、launch和return；
不得直接进入CROWN或用CUDA Graph掩盖基本wrapper成本。

## 7. R2：Global Planner必须选择真实可执行候选

### 7.1 第一版候选空间

只开放物理上已存在的候选：

- backend：PyTorch reference、TVM CIBC fused；
- representation：paired lower/upper、必要的layout normalization；
- batch：domain/spec/sample三轴独立；
- storage：reference allocation、persistent static arena；
- compile/cache：AOT/disk hit、warm in-process hit；
- schedule：冻结winner schedule，不在formal中autotune。

不允许把未实现的candidate写进Plan后再fallback。

### 7.2 Planner evidence

每个candidate必须有：

- capability与legality reason；
- measured wrapper latency、kernel latency、peak allocated/reserved；
- compile/setup cost、cache状态、reuse count；
- Task/Schedule/module identity；
- rejected candidates及原因。

### 7.3 Planner claim边界

第一版只claim“能在给定hardware/shape/cache context下选择已测合法计划”。只有至少两个context出现
真实winner crossover，held-out regret `<=1.10x`，才开放`GPU-context plan selection`研究claim。
否则使用静态winner，不复活失败的通用global planner叙事。

## 8. R3：Prepared Runtime与内存/提交边界

### 8.1 目标

把Plan/Schedule中的storage与execution真正落到runtime：

- compile/admission时绑定module、parameters、arena和views；
- warm run只更新dynamic state/version；
- O(1) identity guards，不在per-op做同步value检查；
- fail before launch或commit，失败时rollback；
- graph/region submission与内部kernel分开计数。

### 8.2 先后顺序

1. persistent arena与zero-copy views；
2. compile/module/cache identity；
3. 单stream正确性和稳定submission；
4. CUDA Graph只在shape、pointer、schedule全部稳定后准入；
5. multistream只有profile证明可重叠bucket `>=10%`且event开销不吞收益时才准入。

### 8.3 禁止

- 先开CUDA Graph再解释wrapper为何慢；
- 为过memory gate刻意构造与query无关的OOM；
- 把reserved memory下降写成allocated下降；
- 让runtime重新做Global Planner已冻结的静态决策。

## 9. R4：E2E-B CROWN compiled region

### 9.1 结构选择

不得复活B4-C2 dense-retention路线。固定为：

```text
compressed α/β + split/history + incoming coefficient
→ Verification legality（GC0-1作为Bound pass）
→ structured region owner
→ fused TIR forward
→ custom backward / minimal saved state
→ compressed dα/dβ + compact lower/status
```

dense A只允许kernel内部scratch/recompute，不能跨层saved/persistent/external escape。

### 9.2 复用资产

- B4-B2 dense/sparse Linear/Conv TIR正确性；
- R3 structured owner、bounded arena、custom backward经验；
- RVIR-v4 production state/trajectory/atomic commit；
- MR7-R host boundary opportunity；
- GC0 schema和后续legality pass。

### 9.3 开放顺序

1. no-timing legality/correctness；
2. 单regionforward + custom backward；
3. 10 evaluation / 9 mutation trajectory；
4. wrapper-inclusive timing；
5. 多region扩展。

每一级不通过，后一级关闭。

MR7-R历史口径下，打到B0 parity所需region speedup曾为`1.91214x`；该数只用于设计警戒。R4必须用
新pipeline同scope的GPU share、wrapper overhead和目标T重新计算，不能把历史值直接当GO门槛。

## 10. R5：RVIR same-solver exact-call替换

主比较固定为：

```text
同一个 αβ-CROWN host solver
  control: original bound executor
  candidate: RVIR adapter → BoundFlow PreparedBoundProgram
```

保持model/property、branch、termination、timeout、α/β初始化、optimizer steps、seed、device、dtype一致。

### 10.1 数学路由

历史B3/B0 query=`0.910001x`：

- 到B0 parity，candidate相对B3至少需`1 / 0.910001 = 1.09890x`；
- 到`1.15x` complete-query研究门槛，至少需`1.15 / 0.910001 = 1.26373x`。

每次用真实same-solver share和候选region speedup重算：

```text
T = 1 / ((1 - share) + share / region_speedup) / integration_overhead
```

不得把独立IBP graph share或CPU share代入GPU query目标。

### 10.2 R5 gate

- correctness：verdict/lower/sign/trajectory/branch/termination exact或冻结容差；
- qualification：candidate/B0 complete-query geomean `>=1.00x`；
- research：complete-query `>=1.15x`；
- 最差pair不得隐藏；unknown workload只形成runtime claim；
- 至少一个baseline/candidate均能solve的公开property才形成TTV/solved claim。

## 11. R6：只有证据允许时才扩优化轴

优先级固定：

1. 扩大合法fused region和epilogue，减少framework crossing；
2. representation保持与minimal saved state；
3. arena/lifetime/recompute；
4. domain/spec batching；
5. static shape/schedule specialization；
6. CUDA Graph；
7. multistream/multi-branch并行；
8. JIT/autotuning。

准入规则：

- 新轴先做read-only attribution和Amdahl feasibility；
- `required_region_speedup >10x`时默认STOP，除非有新的物理机制证据；
- memory路线先证明目标硬件上peak/OOM可达；
- JIT以AOT/cache为主，只有预计reuse足以覆盖compile cost才触发；
- 并行必须保持state/effect/commit顺序，不以异步掩盖语义漂移。

## 12. R7：最终formal矩阵

### 12.1 Workload

- 至少两个held-out model family；
- Torch/ONNX至少在支持子集上双前端一致；
- 至少一个非unknown、双方都能solve的公开property；
- 一个memory-pressure workload，但不得仅为过门槛而脱离真实solver；
- ResNet2B保留为production unknown/runtime workload。

### 12.2 Baselines

- 原始αβ-CROWN/PyTorch executor；
- BoundFlow reference Plan；
- BoundFlow direct kernel winner；
- BoundFlow完整 compiler pipeline；
- 必要的fusion/storage/batching/runtime消融。

### 12.3 最终研究门槛

- queue geomean `>=1.20x`；
- complete-query geomean `>=1.15x`；
- correctness与验证强度不降；
- peak memory不恶化；如走memory claim则目标workload下降`>=25%`；
- compile/setup/cache/fallback全部披露；
- held-out不允许事后调参。

## 13. 建议的提交序列

用户批准本计划后，建议按以下短提交推进：

1. `docs: freeze canonical compiler entry and legacy adapter map`
2. `feat(compiler): add Primal Spec Bound compile request`
3. `feat(planner): bind CIBC reference and TVM candidates`
4. `feat(ir): lower selected plan to Task and Schedule IR`
5. `feat(runtime): prepare and execute compiled IBP graph`
6. `bench: close E2E-A CIBC pipeline propagation`
7. `feat(analysis): integrate verification legality as Bound pass`
8. `feat(runtime): add structured CROWN region and custom backward`
9. `feat(adapter): route RVIR exact call to prepared BoundFlow executor`
10. `bench: run same-solver three-way formal`

每个代码阶段独立correctness closure；不要求每个小提交都外审，但性能closure与claim升级必须有冻结
artifact和可复放证据。

## 14. 第一刀建议

如果你认可这份路线，第一刀不是继续GC0-1外审，也不是立刻写新的TIR kernel，而是：

> 冻结并实现 `R0 canonical compiler entry + legacy adapter map`，选定CIBC IBP为第一个真实
> `Primal→Bound→Plan→Task/Schedule→TVM→Prepared Runtime`纵向切片。

第一刀只解决ownership和接口收束，不计时；第二刀接通已有CIBC winner后才做性能资格。这样能最快
回答最关键的问题：**BoundFlow完整编译器能否保住已经存在的算子/图级GPU收益。**

## 15. 本稿与GC0-1的关系

- 本稿写作期间GC0-1 exchange已异步出现`approved`审计产物，但按用户“先不用外审”的指令，不读取
  findings、不执行respond/close，也不让该结论改变本稿；
- GC0-1文档中的capture/analysis算法仍可复用；
- 若用户批准本计划，GC0-1应改为R4之前的`verification legality Bound pass`，而不是下一立即主动作；
- 本稿在用户批准前`execution-authority=false`，不修改当前代码门禁或claims。
