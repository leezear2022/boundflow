---
status: superseded-by-full-stack-overlay
updated: 2026-08-06T09:06:51Z
type: plan
topic: boundflow
slug: gpu-compiler-acceleration-research-v1
stage: s01
claim_scope: research-only
revision: v1.2-full-stack-overlay
review_date: 2026-08-06
inspected_branch: feat/top2-production-execution-cost-attribution-v1
inspected_head: 849912d
---

# BoundFlow GPU 编译器加速诊断与执行计划 v1

> **v1.2（2026-08-06）full-stack overlay**：NRIR49A/G1 的数据与冻结 artifact 保持有效，
> 但它只关闭 **selected-CROWN-only incremental optimization**，不是 BoundFlow 从算子、
> Bound/Graph IR、Plan/Schedule、JIT/cache、runtime scheduling 到 allocator/memory 的全栈上限。
> 旧 G2—G4 保留为历史预注册与 gated 路线，不再是当前执行指令。当前唯一路线入口是
> [BoundFlow Full-Stack GPU Baseline and Attribution v1 Plan](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)：
> FSG0 schema/critical-path/replay合同已验证；当前下一步执行FSG1 official αβ-CROWN B0 full-stack
> baseline。这不是重新寻找另一个单点 winner，也不产生任何 BoundFlow 性能主张。
> 本文下方 G0—G8 正文作为 v1.0/v1.1 历史证据与门禁保留；与本修订冲突时，
> 以 full-stack plan 为准。
>
> **v1.1（2026-08-06）评审修订**：新增G1 Amdahl反解与`>10x/INFEASIBLE` kill gate、公开可solve
> workload、physical-memory reachability、RVIR same-solver主对照、GPU fallback、frontend coverage、
> G2 timebox与GPU-context selector边界。修订前外部审计见
> [审计报告](external_audit_gpu_compiler_plan_v1_2026_08_05.md)。
>
> **G0 执行状态（2026-08-06）**：pre-reboot admission 已把 blocker 从三项收敛到一项。独立
> αβ-CROWN 环境和公开 `mnistfc:2` 双方 `verified` solveability 均已关闭；ASUS dGPU enable 已 queued，
> 当前只等待重启后 CUDA smoke。详见
> [G0 admission](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md)。

## 0. 文档用途与一句话结论

这是一份供其他大模型、系统研究者和执行方共同审计的研究诊断与预注册草案。它回答四个问题：

1. 当前 BoundFlow 的编译器和求解器真正做到了哪里；
2. CUDA/TIR、JIT、流程级融合、物理内存和多分支并行中，哪一条最值得先做；
3. 历史负结果为何不能被遗忘、又为何没有永久否定 GPU 编译路线；
4. 如何从 kernel 逐层走到 same-solver complete-query，而不把微基准包装成系统收益。

**执行结论：应该继续深耕 GPU 编译器，但不应把四条路线平行铺开。第一主线应改为
“verification-aware selected-CROWN GPU compilation”：把真实 production 路径中的 selected objective、
ReLU relaxation、Linear/BoundConv 反向传播、lower/upper 输出和 split/alpha/beta 状态编译成一等
Bound/Plan/Task/Schedule IR region。**

优先顺序是：

1. 恢复 GPU，并在公平合同下找回/复现用户报告的 BoundConv `40x`；
2. 测量 GPU 上真实 selected-CROWN 的 shape、launch、同步、分配和时间占比；
3. 先资格审查现有 fused executor能否合法覆盖 selected-CROWN narrow case；能覆盖才接入，否则直接
   进入 sparse/ragged selected-objective TIR；
4. 再做物理 arena、跨 sibling packed batching 和真实 stream/event Schedule executor；
5. 只有复用率和 host-launch 占比过门禁后，才做 JIT/CUDA Graph；
6. 最后用相同求解算法、相同 property、相同节点和终止条件对比 auto_LiRPA/alpha-beta-CROWN。

当前没有任何新的 GPU 性能结果。本计划也**不声称** BoundFlow 已比 auto_LiRPA 快。本会话无法访问
NVIDIA driver/GPU，
且用户报告的 `40x` 实现尚未在当前分支中定位；这两项是 Phase G0 的明确输入，而不是可以略过的细节。

## 1. 北极星问题

建议把后续 ASPLOS 研究问题冻结为：

> 在不改变 verifier 数学语义、host solver、branch policy、deadline 和 termination 的前提下，
> BoundFlow 能否利用一等 Bound/Plan/Task/Schedule IR，把动态演化的 CROWN/alpha/beta/BaB query
> 编译为 verification-aware GPU execution plan，并相对公平的 auto_LiRPA/alpha-beta-CROWN batched
> executor 改善 latency、peak memory 与 time-to-verify 的 Pareto 前沿？

这个问题故意排除了三种较弱故事：

- 不是“又写了一个 CUDA Conv”；
- 不是“相对逐节点 Python baseline 得到很大的 batching 倍数”；
- 不是“只有 schema、hash 和 reference executor，没有物理 GPU ownership”。

最终 compiler contribution 至少要同时体现：

- verification-specific lowering：target、polarity、relaxation、split/alpha/beta provenance；
- plan choice：chunk、batch、fusion、storage、stream 和 cache state 中至少两个上下文选择不同计划；
- physical effect：真实 kernel、物理分配或并发执行发生变化；
- end-to-end evidence：至少到 31-node queue 和 complete query，而不是只停在 micro-kernel。

## 2. 证据标签和审计规则

本文所有重要结论使用以下标签，审计者不应把不同等级混为一谈。

| 标签 | 含义 |
|---|---|
| `MEASURED-CURRENT` | 在本次所检查代码/环境或最新正式 artifact 中可直接复核 |
| `MEASURED-HISTORICAL` | 历史 artifact 的正式数字；保留原 workload、设备和方法边界 |
| `CODE-FACT` | 当前源码静态事实，不自动等价于性能结果 |
| `INFERRED` | 从代码或历史数据推导出的机会/风险，尚未由目标 GPU 实测 |
| `USER-REPORTED` | 用户提供、当前仓库尚未独立复现的线索 |
| `PROPOSED-GATE` | 供外部审计后冻结的预注册门槛，不是已通过结论 |
| `EXTERNAL-PRIOR` | 外部系统、论文或官方文档已经覆盖的机制边界 |

性能 claim 必须遵守：

- 至少 3 个 fresh process；本计划建议正式性能用 5 个；
- control/candidate 顺序反平衡；
- cold compile、disk/process cache hit 和 warm execution 分开；
- CUDA event device time 与同步后的 host wall time同时记录；
- median、分位数、MAD/离散度和全部 raw rows一并保存；
- OOM、fallback、timeout 和不可用行不能从分母中静默删除；
- 只有 kernel、region、child、queue、complete-query 五层全部标清，才允许谈“快多少”。

### 2.1 Claim registry

下表是外部审计的最小逐项索引；审计者可以直接引用 ID，而不必接受整篇文档的总判断。

| Claim ID | 状态 | scope | 主要证据 | 下一升级门禁 |
|---|---|---|---|---|
| `F-ENV-01` | `MEASURED-CURRENT` | 本会话GPU不可用 | `nvidia-smi`、PyTorch/TVM smoke | G0四项CUDA smoke |
| `F-ENV-02` | `CODE-FACT` | 当前torch超出vendored auto_LiRPA声明范围 | 两侧版本/setup约束 | G0独立competitor env smoke |
| `F-IR-01` | `CODE-FACT` | typed IR/replay骨架已进入real path | RVIR closure、IR contract、当前源码 | 不需性能升级 |
| `F-EXEC-01` | `CODE-FACT` | physical stream/event/arena未闭环 | Schedule/storage reference runtime | G5/G6 physical artifact |
| `F-CPU-01` | `MEASURED-HISTORICAL` | fixed CPU8 selected-CROWN attribution | NRIR48 raw/replay与变更记录 | 不能直接升级到GPU |
| `U-40X-01` | `USER-REPORTED` | BoundConv约40x线索 | 尚无唯一仓库artifact | G0独立给出REPRODUCED/FAILED/NOT-AUDITABLE |
| `H-GPU-01` | `INFERRED` | GPU selected-CROWN可能仍dominant | CPU attribution + code shape | G1 share `>=20%`且latency required `<=10x` |
| `H-FUSE-01` | `INFERRED` | sparse/ragged region可降计算/物化 | dense one-hot源码与旧TIR限制 | G2/G3 semantic+region gate |
| `H-MEM-01` | `INFERRED` | physical arena可降peak/allocator | logical storage与per-chunk allocation | G1 reachability + G5 physical gate |
| `H-SCHED-01` | `INFERRED` | packed/stream Schedule可降queue latency | sibling availability与Schedule IR | G6 queue/peak gate |
| `H-JIT-01` | `INFERRED` | 重复shape下JIT/Graph可摊销 | cache机制与PR-12J反证 | G7 reuse/launch gate |
| `C-END-01` | `PROPOSED-GATE` | GPU-context selector系统贡献 | G8 RVIR same-solver + solved matrix | held-out Pareto/regret gate |

每个后续 artifact 应把对应 `claim_id`、evidence path/hash、replay command、known limitation 和
next gate 写入 manifest。未在表中的新 claim须先加 ID，再运行正式实验。

## 3. 本次检查快照

### 3.1 Git 与第三方版本

`MEASURED-CURRENT`

| 项 | 检查值 |
|---|---|
| 当前分支 | `feat/top2-production-execution-cost-attribution-v1` |
| 当前 HEAD | `849912d` |
| 远端 `origin/main` | `c0ccfb5`，已合入当前 NRIR48 feature head |
| auto_LiRPA submodule | `9d100ec070868440b48d34e2f1dd21b97aab9172` |
| TVM submodule | `6248b5db43505fbcfb13cc289d11877d5d2649e8` |
| tvm-ffi submodule | `438f6439148b059d424ce2cc2a348736923f6948` |

当前工作基线包含 NRIR48 cost attribution，但本计划的文档编辑发生在尚未新建/提交的工作树中。
本文没有假设一个尚不存在的 NRIR49 实现。

### 3.2 本机运行环境

`MEASURED-CURRENT`

| 项 | 检查值 | 解释 |
|---|---|---|
| Python | `3.12.12` | conda env `boundflow` |
| PyTorch | `2.12.1+cu132` | wheel 包含 CUDA 13.2 support |
| TVM | `0.23.dev0` | `tvm.runtime.enabled("cuda") == True` |
| NVIDIA driver/session | `nvidia-smi` 无法连接 driver | 可能是driver或会话GPU透传；当前不能产出正式GPU性能证据 |
| PyTorch CUDA | `torch.cuda.is_available() == False`，device count `0` | 与 driver 失败一致 |

`CODE-FACT`：vendored auto_LiRPA `setup.py` 声明 `torch>=2.0.0,<2.9.0`；当前 BoundFlow
PyTorch `2.12.1` 超出其声明支持范围并与依赖元数据冲突。公平 competitor 不能假定共用同一个 conda
env；G0 必须建立独立锁定的
auto_LiRPA/alpha-beta-CROWN env或container，并用同一物理GPU、driver、功耗/时钟、模型/property
digest、输入工件和计时合同运行。每条结果必须记录 `environment_id`；跨env差异作为显式限制和
ablation，不能通过强装不受支持的依赖来“统一环境”。

因此，“过去是否用过 GPU”和“当前能否现场重跑 GPU”必须分开回答：PR-12/PR-13 有历史 RTX 4060
GPU 工件；NRIR44—48 当前 production 路线是 CPU8；本次现场则无法访问 GPU。

### 3.3 当前正式 claim 边界

`MEASURED-CURRENT` / `MEASURED-HISTORICAL`

可以主张：

- `BFBoundModule -> PlanTemplate -> PlanInstance -> TaskModule -> ScheduleModule` 五层 typed stack、
  deterministic hash、verifier、reference execution、replay 和 tamper rejection 已经存在；external
  exact-call是其中的Bound op/backend语义，不是单独一层；
- IR 已进入固定 VNN-COMP ResNet2B 的 native production 路径，不只是 toy schema；
- RVIR external intermediate bounds 路径已有 CPU correctness：typed admission `394/394`、online
  execution `377/377`，initial-CROWN ResNet max diff约 `3.10e-6`、sign `9/9`；
- fixed real-graph logical storage plan 曾把 retain-all `1,860,912 B` 降为 lifetime-reuse
  `442,656 B`；这是**逻辑 liveness**，不是物理 CUDA peak-memory claim；
- NRIR48 把当前 CPU production 的最大 child-execution category 定位为 selected-CROWN。

不能主张：

- 不能声称当前 BoundFlow 比 auto_LiRPA/alpha-beta-CROWN 快；
- 不能声称当前 production 已有 GPU 结果；
- 不能把 logical storage byte reduction 当作 CUDA allocated/reserved/NVML reduction；
- 不能声称 fused kernel 已覆盖真实 alpha/beta/split query；历史 fused coverage 仍是 `0/394`；
- 不能声称 adaptive global Planner 已验证成功；IR-5 Global p90 regret `1.26160x > 1.20x`，结论是
  `VALIDATED-NO-GO`；
- 不能声称 property 已 closure；当前 fixed complete query 仍为 `9/9 unknown`；
- 不能把 `USER-REPORTED` 的 BoundConv `40x` 升级为仓库结论，直到 G0 完成公平复现。

## 4. 现有编译器路径：已经有骨架，但物理执行没有闭环

### 4.1 所有权图

```text
verification query / forward trace / split-alpha-beta state
                         |
                         v
                    Bound IR
       target + polarity + provenance + numeric policy
                         |
                         v
             PlanTemplate / PlanInstance
  backend + chunk + three batching axes + storage + costs
                         |
                         v
                     Task IR
             legal regions and exact dependencies
                         |
                         v
                   Schedule IR
    launch / transfer / allocate / event / stream / lifetime
                         |
       +-----------------+------------------+
       |                                    |
       v                                    v
PyTorch/native reference          TVM fused / compiled artifact
       |                                    |
       +-----------------+------------------+
                         v
               physical CUDA runtime
```

当前断点不在“有没有 IR 类”，而在最后两条边：

- selected-CROWN 还没有被 lowering 成专属 fused region；
- Schedule 的 stream/event/storage 还没有变成真实 CUDA stream/event/arena ownership。

### 4.2 当前组件事实表

| 层 | `CODE-FACT` 已有 | 关键缺口 |
|---|---|---|
| Bound IR | target/provenance、lower/upper、domain/spec 等 typed 语义 | 没有 sparse/ragged `SelectedObjectiveIR` |
| Plan IR | backend、chunk、三种 batching 轴、StorageBinding、compile/setup/peak cost | GPU selected-CROWN 候选未进入 production 选择 |
| Task IR | typed tasks、dependency、backend dispatch | selected gather→relaxation→Conv/Linear contraction 不是一等 region |
| Schedule IR | Launch/Transfer/Allocate/Free/RecordEvent/WaitEvent、verifier | reference lowering把 launch固定到 `stream_id="sync"` |
| Schedule runtime | deterministic reference trace | record/wait只记 trace；没有创建真实 CUDA stream/event |
| Storage runtime | last-use释放 Python env 引用、logical live-byte accounting | 没有按 arena/offset创建物理 CUDA view与复用 |
| fused runtime | DLPack zero-copy、TVM-FFI current/custom Torch stream bridge | capability只接受 plain static CUDA FP32，拒绝 split/alpha/beta |
| compiler runtime | Plan/Task/Schedule/prepared execution caches | query仍逐个执行，明确不声称 physical cross-query batching |
| dynamic batch | exact compatibility bucket、deadline、OOM二分、顺序恢复 | memory estimate只是 payload bytes，不是 allocator peak |

关键源码入口：

- `boundflow/runtime/native_intermediate_refinement.py:923`：当前 selected-CROWN；
- `boundflow/runtime/crown_ibp.py:2136`：通用 from-forward-trace 路径已暴露 fused 参数；
- `boundflow/runtime/fused_crown.py:279`：plain CUDA FP32 capability；
- `boundflow/backends/tvm/fused_crown_conv2d.py:28`：现有 Conv signature 与 schedule；
- `boundflow/backends/tvm/fused_crown_cache.py:44`：validated compiled module cache；
- `boundflow/ir/schedule.py:1240`：reference single-stream lowering；
- `boundflow/runtime/schedule_ir_executor.py:172`：reference Schedule executor；
- `boundflow/runtime/storage_plan_runtime.py:206`：logical storage runtime；
- `boundflow/runtime/query_batcher.py:164`：现有 query batch manager。

## 5. 当前最值得优化的真实 region

### 5.1 selected-CROWN 当前执行形态

`CODE-FACT`

`_run_selected_crown` 目前执行：

```text
targets 按 ReLU 分组
  -> 每组按 backward_chunk_size（默认 32）切块
    -> 为每个 chunk torch.zeros 物化 dense one-hot objective
    -> CUDA 情况下再构造 indices tensor
    -> 每个 ReLU、每个 chunk 单独调用通用 from-forward-trace CROWN
    -> 收集 lower/upper
  -> torch.cat
  -> 再构造 tensor 并与 intermediate bounds intersect
```

更关键的是，这个调用没有传入已经存在的 `fused_crown_executor`、`fused_crown_steps` 和 context。
被调用函数本身已经支持这些参数，所以第一个工程实验不必先发明新 kernel：应先打通现有 fused backend，
确认 capability 和 semantics 后测量边际。

### 5.2 CPU attribution 与 Amdahl 上限

`MEASURED-HISTORICAL`：NRIR48 在 fixed ResNet2B property 0、CPU8、clauses 2/3 上测得：

- selected-CROWN：每条 queue 约 `2.663 / 2.694 s`；
- selected-CROWN 占 child execute 约 `71.77% / 72.73%`；
- child execute 占各自 queue 约 `32.20% / 31.16%`；
- fixed whole trace median约 `31.320 s`。

`INFERRED`：把两条 queue 的 selected-CROWN 粗略相加，约为 whole trace 的 `17.1%`。如果这部分在
相同 CPU 路径上真能加速 `40x`，而其它部分完全不变，Amdahl 上限也只有约 `1.20x` whole-trace。
两条 child execute 合计约占 whole trace `24.0%`，即使无限加速，上限也约为 `1.32x`。

这不是否定 `40x` 的价值，而是明确论文证据必须逐层传播：

```text
kernel -> fused region -> selected-CROWN child -> 31-node queue -> complete property
```

同时，CPU share 不能代替 GPU share。GPU 上 cuDNN/GEMM、launch、同步和 allocator 的比例可能完全不同，
所以 G1 设有明确的 GPU opportunity gate。

### 5.2.1 G1 必须反推的量化可达性

`PROPOSED-GATE`：G1 不能只输出“占比大于20%”。对每个目标scope，令：

- `s = region_time / control_scope_time`：目标region的GPU实测占比；
- `r`：region speedup；
- `T`：要求的scope speedup。

Amdahl投影与反解必须写入artifact：

```text
projected_scope_speedup(s, r) = 1 / ((1 - s) + s / r)
required_region_speedup(s, T) = s / (s + 1 / T - 1)
```

若 `s + 1/T - 1 <= 0`，即使region无限快也达不到目标，记为 `INFEASIBLE`。G1分别计算：

```text
r_queue_required    = required_region_speedup(s_queue,    1.20)
r_complete_required = required_region_speedup(s_complete, 1.15)
r_latency_required  = max(r_queue_required, r_complete_required)
```

例子只用于检查公式，不代替GPU测量：`s_queue=25%`时达到`1.20x`需要`3x` region；
`s_queue=20%`时需要`6x`；若把CPU历史`17.1066%`错误地当作queue share，`1.20x`会需要约
`38.9x`并已接近`1.206x`理论上限；同一share达到complete-query `1.15x`则需约`4.21x`。

G1→G3 latency准入规则：

- 任一必要scope为`INFEASIBLE`，或 `r_latency_required > 10x`：**不开G3 latency路线**；
- 此时只有G1预判已准入的physical-memory路线可以继续，否则本selected-CROWN路线`NO-GO`/重选winner；
- `r_latency_required <=10x`时才允许进入G2/G3，并把该数值冻结为G3正式目标；
- G3的`1.3x`只是mechanism gate，不能替代由G1反解出的headline可达性门槛。

### 5.3 CUDA 热路径的额外风险

`INFERRED`：native refinement execution trace 会为 Schedule action计算 input/output content hash；
底层 tensor content hash 在 CUDA tensor 上会 `.cpu()`。如果 production timed region保留这一行为，
可能引入隐式 host synchronization。它必须被 profiler计数，然后再决定是否引入：

- `AUDIT`：完整内容 hash 和可独立 replay；
- `PRODUCTION`：typed identity/version/mutation receipt，完整 hash移出 timed path；
- formal replay：仍从源输入重新计算内容 hash，不能降低 tamper resistance。

这只是待证假设；在 profiler 没证明同步占比前，不先改 correctness ownership。

## 6. 历史证据：为什么不能简单“继续调旧 TIR”

| 工作 | `MEASURED-HISTORICAL` 结果 | 对新计划的约束 |
|---|---|---|
| PR-12I fair E2E | Linear `8.644 ms` vs eager `1.736 ms`；Conv `1.768` vs `1.386`；mini-ResNet `7.009` vs `7.234`；geomean speedup `0.546x` | 旧 fused path 不能作为默认 winner |
| PR-12J compile amortization | PR-12J compile phase约 `0.324/0.480/1.299 s`；fresh/disk/process break-even约 `4668/1062/4450` queries，超过 `Q<=1024` | JIT必须由 reuse gate触发，AOT/cache family优先 |
| PR-12K device activity | fused对unfused整体 launch 最大只降 `1.96%`；6 个 workload 中 3 regress、1 improve、2 neutral | 旧 region太小，孤立 kernel fusion不是 headline |
| PR-12L closure | 旧 plain-CROWN TIR路线 `E_STOP_OPTIMIZING_TIR` | 不能篡改旧结论；新路线必须是新 hypothesis/split |
| PR-13D GPU batching | 相对逐节点 `96.52x`，相对公平 batched original仅 `1.024x`；hard E2E约 `0.980x` | baseline必须是已有 batched executor |
| IR-5 | Global p90 oracle regret `1.26160x > 1.20x`，`VALIDATED-NO-GO` | 不复活broad global planner；只审GPU-context selector |
| NRIR43 CPU batching | scorer launch `31 -> 16`，queue反而慢约 4%–5% | 不重开相同 CPU scorer batch；GPU selected ragged batch是不同变量 |
| NRIR46 | static shareable median `1.071 s < 1.5 s` | 不重开 validator/template 常数优化 |
| NRIR47 | receipt correctness成立，但 queue慢约 1%–2% | correctness ownership不自动产生 speedup |
| NRIR48 | selected-CROWN 成为 child execute dominant category | 只给出调查对象，不等于 GPU winner |

### 6.1 为什么新路线不违反 PR-12L

PR-12 的负结果适用于：

- plain CROWN；
- static CUDA FP32；
- 旧的 output-gather 128-thread schedule；
- ReLU+单个 Linear/Conv 局部 region；
- 当时的 workload、shape 和 timing contract。

新路线研究的是：

- production 中真实出现的 split-constrained selected-CROWN；
- sparse/ragged target，而不是 dense full objective；
- selected seed、relaxation、polarity、Conv/Linear contraction 和 emit 的语义 region；
- sibling/domain/spec/target 轴和 storage/stream/cache 的联合选择；
- queue 与 complete-query 传播。

因此正确表述是“在新的 production dominant region 上开启新的有条件 GPU hypothesis”，而不是
“PR-12L 判断错误”或“继续 PR-12 调参”。

## 7. 用户报告的 BoundConv `40x`：第一线索，不是当前结论

`USER-REPORTED`：用户曾独立把 BoundConv 融合并观察到约 `40x`。

本次对当前分支、历史文档和已有 artifact 的只读检查没有找到足以独立重放该数字的唯一实现与合同。
G0 必须定位并冻结：

- repo/branch/commit 或未提交 patch；
- exact input/output tensor shape、stride、padding、channels、layout、dtype；
- CROWN/alpha/beta/split、grad/no-grad 和 lower/upper 输出合同；
- 对比 baseline 是 per-op、unfused TVM、PyTorch eager、auto_LiRPA batched 还是包含 Python/alloc；
- warmup、iteration、CUDA event、host synchronize、compile 和 allocation范围；
- GPU、driver、功耗/时钟、PyTorch/TVM/cuDNN版本；
- correctness tolerance、conservative lower-bound 和 branch/verdict 对齐。

复现后必须分别报告：

1. kernel-only；
2. BoundConv fused region；
3. 一个 selected-CROWN child；
4. 31-node queue；
5. complete query。

若 `40x` 在公平合同下缩小或消失，不删除结果；输出 root-cause audit，判断收益来自计算融合、避免
Python、避免 allocation、避免冷启动、不同 output contract，还是 baseline 本身不公平。

## 8. 四条技术路线的综合诊断

| 路线 | 单独新颖性 | 与 BoundFlow IR结合后的价值 | 风险 | 排序 |
|---|---:|---:|---|---:|
| CUDA/TIR 算子/region融合 | 中低 | 高：可消费 target、polarity、relaxation、state 和 batch axes | 变成单 shape 手写 kernel；旧 schedule已有负结果 | P1 |
| 流程级局部融合 | 中 | 很高：直接体现 Bound IR legality 和 materialization ownership | fanout、split、alpha/beta、external state使等价证明复杂 | P1 |
| 物理内存与多分支调度 | 中 | 很高：可形成 latency-memory-TTV Pareto | 多流争用、显存复制、arena/graph pool反增内存 | P2 |
| JIT specialization | 低 | 有条件中高：PlanTemplate/Instance 可决定 break-even | compile成本、shape explosion、cache invalidation | P3，严格 gated |

这些不是四条独立 feature branch。建议的统一决策变量是：

```text
context = (
  graph/model/weight-version,
  domain + spec + selected-target axes,
  method + split/alpha/beta state,
  shape/dtype/layout/device,
  memory budget + deadline,
  cache state + expected reuse
)

plan = choose(
  objective representation,
  fusion region,
  TIR schedule family,
  chunk/packed batch,
  storage arena/lifetime,
  single-stream/multi-stream,
  AOT/JIT/graph replay
)
```

### 8.1 H1：sparse/ragged selected-objective fusion

`CODE-FACT`：vendored auto_LiRPA `operators/convolution.py:70` 的 `BoundConv.bound_backward` 在收到
`OneHotC` 时明确调用 `onehotc_to_dense`，Tensor path再用 `F.conv_transpose2d`；同时它也有 `Patches`
表示路径。这使“selected objective不物化dense”成为一个具体、可与最强原生表示对照的研究问题，
但不能预设Patches/Tensor中的哪一条更快。G0/G1必须同时记录实际representation dispatch与fallback。

`INFERRED`：dense one-hot 不是 selected objective 的必要物理表示。可以引入
`SelectedObjectiveIR(indices, segment_offsets, polarity, output_ordinal)`，lowering 为：

```text
selected target gather / implicit seed
  -> ReLU relaxation selection
  -> coefficient sign/polarity selection
  -> Linear or ConvTranspose contraction
  -> residual add/concat contribution join and view transforms
  -> bias accumulation
  -> selected lower/upper emit
```

优先联合优化：

- 不物化 dense one-hot；
- lower/upper 共享 target index、slope、weight load；
- ragged target count 用 segment offsets，不让 padding lane参与 min/max/reduction；
- chunk size由 occupancy、workspace、target distribution和reuse决定；
- stride 1/2、1x1/3x3、channel/spatial regime使用不同 schedule family。

fixed ResNet2B 的 backward graph不只有 ReLU和单个affine op；当前 reference还处理 residual `add`
fanout/join、`flatten`、`reshape` 和 `concat`。因此 region legalizer必须为每个非affine op显式选择：

- `fuse_through`：公式、fanout和layout都已证明；
- `view_only`：仅在flatten/reshape不要求物理重排且stride/layout合同成立时；
- `materialize_boundary`：第一版无法合法跨越的 residual join、concat或fanout；
- `reject`：unsupported axis/layout/consumer topology，确定性回退reference。

跨 residual fanout时，所有 downstream coefficient contribution必须恰好累加一次；不能丢失、重复或因
执行顺序改变 lower/upper polarity。concat backward的slice ordinal/axis必须与reference一致。第一版
允许在residual join处显式停住，但必须把边界、物化bytes和launch记入artifact，不能把它藏起来后仍声称
实现了大region fusion。

现有 Conv TIR 只把 spatial loop fuse 后固定 128 threads，reduction基本串行。候选 schedule应包括：

- thread/warp级 reduction；
- `rfactor` 或分层 reduction；
- vectorized target lanes；
- shared/prepacked weight；
- upper/lower pair fusion；
- calibration以外的 held-out shape。

### 8.2 H2：split-constrained 与 frozen alpha/beta final evaluation

现有 fused capability会拒绝 split、alpha、beta和grad。不能通过伪装 `plain` 绕开。建议分两级扩展：

1. `selected_crown_split_constrained`：显式接受 frozen `relu_pre`/split constraints，只做 no-grad
   selected-CROWN；
2. `frozen_alpha_beta_final_eval`：optimizer仍保留 PyTorch/autograd；在 alpha/beta state冻结后，
   fused backend只执行最终 bounds evaluation。

每个 capability key必须含 method、grad policy、split provenance、alpha/beta state version、numeric policy、
shape/layout/dtype 和 exact operator限制。完整 autograd/optimizer fusion是最后路线，不能先做。

### 8.3 H3：物理 memory lowering

当前 logical StorageBinding 还没有产生物理 arena。建议实现：

- per-device/per-stream-class scratch arena；
- `arena_id + offset + dtype + shape` 创建 non-overlapping tensor view；
- event完成前禁止跨 stream复用；
- objective、index、lower/upper和intermediate scratch预分配；
- 去掉每 chunk 的 `zeros`、indices tensor和末尾 `cat`；
- concurrent Schedule 用重叠 lifetime峰值做预算，而不是各分支单独最大值；
- allocated、reserved、NVML/device footprint分开报告。

目标不是只让逻辑 byte表更好看，而是减少 allocator calls、真实 peak或在相同预算下救回 OOM。

### 8.4 H4：packed batch优先，多 stream有条件启用

同一 parent 的 sibling天然同时可用。第一选择应是把同类 selected-CROWN work按 domain axis物理 pack，
共享大 kernel；只有 profiler证明 kernel未占满 GPU且存在独立 critical-path work时，才尝试多 stream。

物理 Schedule executor需要真正实现：

- `stream_id -> torch.cuda.Stream`；
- `RecordEvent/WaitEvent -> torch.cuda.Event`；
- ready-set scheduler，而不是简单 topo `for`；
- stream-ordered scratch reuse；
- deterministic child/node commit和结果顺序；
- deadline、OOM retry、fallback和cancellation协议。

跨 clause 默认仍串行，因为当前 first-unsafe short circuit 和共享 global deadline具有语义。若将来做
`speculative_parallel`，必须是显式 policy，报告额外 work和显存，并保证未提交 clause不污染状态或 verdict。

### 8.5 H5：AOT/cache优先，JIT和CUDA Graph最后

PR-12J 已证明旧 workload的 compile break-even过高。因此：

- 先预编译常见 shape/schedule family；
- target的具体 index内容作为 runtime input，不进入 code identity；
- expected reuse 至少明显超过 measured break-even才后台 JIT；
- compile失败必须回退已qualified reference，不阻塞 correctness-critical search；
- CUDA Graph只用于 shape、arena地址和stream topology稳定的重复 region。

validated cache key至少绑定：

```text
Bound/Plan/Task/Schedule schema + hash
model graph + weight version
method + domain/spec/target shape class
split/alpha/beta state capability and version
dtype/layout/device/compute capability
chunk/padded shape/numeric policy
code schema + TVM/PyTorch/tvm-ffi ABI
stream topology + arena generation
```

不得直接把旧 `TVMTaskExecutor` 的短 hash/disk cache当论文主缓存；它需要达到现有
`fused_crown_cache.py` 的 manifest、library SHA256、atomic replace和failure handling强度。

## 9. 与外部系统的关系和新颖性边界

`EXTERNAL-PRIOR`

- [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) 已支持 backward CROWN、
  alpha-CROWN、beta-CROWN、split constraints、一般计算图和 GPU/multi-GPU；其官方README还明确列出
  convolutional backward bounds 的 memory-efficient GPU implementation。vendored源码已有
  `operators/convolution.py` 中的 `BoundConv.bound_backward` 和 `BoundConvTranspose.bound_backward`。
  仅仅“把 CROWN/BoundConv 放到 GPU”不是贡献，新TIR必须相对这条最强原生路径而非只相对eager或
  TVM-unfused。
- [TVM 架构文档](https://tvm.apache.org/docs/arch/index.html) 已包含 Relax `FuseOps`/`FuseTIR`
  和 TensorIR schedule。普通 operator fusion本身不是新颖点。
- [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
  已提供在真实硬件上测量候选 schedule 和复用数据库的通用机制；用了 MetaSchedule也不是贡献。
- [TVM DLight/customize optimization](https://tvm.apache.org/docs/how_to/tutorials/customize_opt.html)
  展示了快速默认 schedule与搜索式 tuning的取舍。BoundFlow应先做有限 schedule family，再决定是否搜索。
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) 已提供
  region compilation、guard和cache；BoundFlow需要证明 verification state/plan ownership带来额外价值。
- [PyTorch compile caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
  包含多层缓存和版本/硬件有效性问题；BoundFlow cache不能忽略 ABI和设备约束。
- [CUDA 异步执行文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
  说明 stream/event只表达潜在并发，实际 overlap受硬件资源和依赖限制；“用了多流”不等于更快。
- [CUDA Graph 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  说明 graph memory node、固定地址、stream-ordered allocation和memory pool约束；Graph可能反增 peak。
- [PyTorch graphed callables](https://docs.pytorch.org/docs/main/generated/torch.cuda.graphs.make_graphed_callables.html)
  也要求静态tensor地址、固定调用顺序和memory pool约束。

本项目的潜在新颖性应是：**verification-specific IR 使同一个 compiler能合法地联合选择 objective
representation、fusion region、chunk/batch、storage lifetime、stream concurrency和cached
specialization，并在不同 shape/state/budget/reuse context中选择不同计划。**

注意：本仓库 TVM 固定在 commit `6248b5d...`，当前官方 MetaSchedule/DLight/Relax API可能与本地 fork
不同。G0 必须做 compatibility probe；文档不能直接假定最新 API 可复制进来。

## 10. 分阶段执行 DAG

```text
G0 环境与 40x 证据恢复
  |
  v
G1 GPU selected-CROWN归因 + Amdahl反解 + memory/solveability admission
  |
  +---- bottleneck/latency可达性不成立 ----> memory-only / 重新选择winner / 关闭路线
  |
  v
G2 timeboxed现有fused backend资格审查
  |
  v
G3 SelectedObjectiveIR + verification-aware TIR region
  |
  +----------+
  v          v
G4 frozen    G5 physical arena / hot-path sync
alpha/beta        |
  +----------+----+
             v
G6 ragged sibling batch + physical stream/event Schedule
             |
             v
G7 conditional JIT / CUDA Graph
             |
             v
G8 same-solver multi-workload E2E / compiler contribution
```

### G0：恢复环境、找回 `40x`、冻结公平基线

目标：让任何后续性能工作有可复现起点。

只允许：环境修复、benchmark harness、trace和artifact；不改数学、不优化 kernel。

必须交付：

- `nvidia-smi`、PyTorch CUDA、TVM CUDA build/run、TVM-FFI custom stream smoke；
- GPU/driver/compute capability/clock-power policy、PyTorch/TVM/cuDNN版本 manifest；
- BoundFlow env与独立 competitor env/container 的 compatibility smoke、lockfile/image digest；两者共享
  同一GPU/driver、模型/property/input digest和timing contract；
- auto_LiRPA/alpha-beta-CROWN repo commit、submodule状态、完整solver config、启动命令和container/env
  digest；不能只记录包版本；
- 当前候选model family的frontend coverage表：逐op记录ONNX importer、Primal/Bound IR、backward、
  selected-CROWN、TVM lowering和external fallback所有权；PR-14历史AveragePool fail-closed必须在当前
  HEAD重新审计，不能假定已修复；
- source-level competitor oracle：记录 `BoundConv.bound_backward`/`BoundConvTranspose.bound_backward`
  的实际调用、tensor输入输出、conv/conv-transpose路径、物化/分配、launch与输出合同；
- 用户 BoundConv实现的 commit/patch/shape/合同；
- 同设备公平对照：auto_LiRPA batched、PyTorch eager、`torch.compile`（可 capture时）、TVM
  unfused、旧 TVM fused、用户 BoundConv；
- cold/warm、compile/load、CUDA event/wall、launch、allocated/reserved peak原始 JSONL；
- 至少 5 轮反平衡运行和独立 replay命令。

GPU环境fallback也在G0预注册：

- 本机恢复timebox为`1 engineer-day`或`2`次独立clean attempt（先到者为准）；
- 超时后转移到用户批准的备用主机/云实例，不允许无限期反复修环境；
- 最低配置：Linux x86_64、NVIDIA Ada/compute capability `8.9`优先、显存`>=8 GiB`、稳定独占或可证明
  无邻居干扰、支持锁定container/driver与5次fresh-process测量；
- G0/G1备用资源预算建议上限为`50 GPU-hours`或`USD 100`（先到者为准），任何实际租用仍需用户显式
  批准；若只能获得不同GPU架构，结果只作attribution/portability，headline仍需在冻结目标硬件复核。

G0输出两个彼此独立的 verdict：

1. **GPU/competitor infrastructure**：四项CUDA smoke、competitor env compatibility和跨env输入digest
   检查全部PASS；这是进入G1的强制门，失败则停止性能工作；
2. **`U-40X-01` evidence**：`REPRODUCED`、`FAILED-FAIR-REPLAY` 或 `NOT-AUDITABLE-SOURCE-MISSING`。
   后两者永久禁止传播40x claim，也禁止用该历史数字选择kernel，但**不阻止**使用其它公平baseline进入
   G1 GPU bottleneck profiling。

frontend coverage另有独立 verdict：G8矩阵中至少两个model family必须能走native typed executor且不把
目标region降级成external black box；未达到时，frontend op支持作为单独 prerequisite PR，不能混入
TIR speedup PR，也不能声称multi-family compiler结果。

### G1：NRIR49A GPU call/shape/chunk/backend decomposition

目标：验证 selected-CROWN 在 GPU 上是否仍是 winner。

冻结：fixed ResNet2B property 0、clauses 2/3、31/31 nodes、target ledger、split/branch/optimizer、
deadline、termination和numeric policy。

按 ReLU/child记录：

- target count、ragged segment、chunk count、objective shape和one-hot bytes；
- 完整backward op-type sequence、consumer/fanout degree、residual add/concat join、flatten/reshape layout、
  当前region/materialization boundary；
- Conv/Linear signature、stride/channel/spatial和backward call数；
- kernel count、CUDA event device time、host launch API time；
- allocator calls、allocated/reserved peak、workspace和materialization bytes；
- `.cpu()`/`.item()`/global synchronize来源；
- chunk `8/16/32/64/128` 的 timing-memory曲线；
- domain batch与公开workload的memory-pressure envelope：最小`B80_alloc/B80_reserved`（对应peak达到
  物理可用预算80%）、首次`B_OOM`、最大semantic-valid batch；allocated/reserved分别记录；
- child、queue、whole-query share。

G1严格是read-only attribution：chunk sweep只能通过harness override运行，不修改production默认chunk、
Planner policy或cache；所有门禁必须在查看candidate结果前冻结，G1结果不得被用作未声明的隐性调参。

`PROPOSED-GATE`：

- selected-CROWN至少占目标 GPU queue device/critical-path time的 `20%`；若未过，不得照搬CPU归因，
  应按GPU profile重新选择winner；
- 按§5.2.1计算并冻结`s_queue/s_complete/r_queue_required/r_complete_required/r_latency_required`；
- physical-memory路线只有在公开/自然workload或semantic-valid domain batch存在`B80_alloc`、
  `B80_reserved`、真实budget failure或`B_OOM`时准入；仅靠任意调低软件budget构造的压力只能支持
  mechanism claim，不能进入G8 memory headline；
- 若目标RTX 4060 Laptop/8GB上没有可达的physical-memory admission，则该硬件的G8 memory path预先
  记为`N/A`，不等到G5才发现。

`2026-08-06 FORMAL OUTCOME — HISTORICAL SCOPE CORRECTED BY v1.2`：五个fresh-process worker的
`s_queue/s_complete`中位=
`0.0709863183/0.0705232890`，paired perturbation中位=`0.999304/1.006747`，故测量有效但20%机会
门禁失败。queue `1.20x`与complete `1.15x`均超过Amdahl无限区域加速上限，required speedup为
`INFEASIBLE`；最大allocated/reserved仅物理显存`0.996%/1.353%`，最大合法domain batch=1且无OOM，
memory path=`N/A`。该结果以`VALIDATED-NO-GO(selected-CROWN-only incremental optimization)`
关闭旧 G1 单点路线；旧 G2—G4 对 selected-CROWN 保持 gated，只作历史路线保留。formal
summary hash=`7eefe6a7…ab50`，`performance_claimed=false`。

artifact 中已冻结的 `next_route=gpu-winner-reselection` 是当时机器输出，不改写 payload、
manifest 或 hash；但当前工程路线已由
[full-stack plan](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
取代。FSG0 schema/critical-path/replay已关闭；下一步进入FSG1 official B0 full-stack baseline；
不再重新挑选一个单点 winner，且FSG1不宣称 BoundFlow 性能。

### G2：资格审查并尝试接通现有 fused executor

目标：隔离“缺少 backend wiring”“现有数学不覆盖 split state”和“kernel本身不合适”。

`CODE-FACT`：当前 `_execute_fused_relu_affine_step` 不只是没有收到fused参数；它会硬拒
`split_state_present/alpha_enabled/beta_enabled`，并固定用 `relu_alpha=None` 生成relaxation。因此G2
首先是 legality/qualification，而不是把三个参数接上线就完成。

实施：

- 先形式化当前 selected-CROWN 中 frozen `relu_pre`、split constraints、relaxation、额外beta系数和
  output selection的数学所有权；证明现有kernel公式是否足以覆盖该narrow case；
- 若公式足够，再修改 runtime legality guard、relaxation lowering和prepared step/context创建，并为
  production selected-CROWN 显式传递 fused executor/steps/context；
- 只有语义证明与拒绝路径都通过后，才新增 `selected_crown_split_constrained` capability；禁止只改名称
  或伪装 plain；
- 若 frozen split仍需现有kernel未表示的额外系数，则G2直接记录qualification `NO-GO`，reference
  fallback保留，G3设计新region公式；
- Plan/Task/Schedule记录 backend choice、artifact hash、chunk、workspace和fallback；
- 保留 reference PyTorch dense/chunked候选；
- 加正向、拒绝路径、tamper、fallback和exact-call-count测试。

正确性 gate：公式/legality review PASS；target ordinal、lower/upper、intersection monotonicity、
branch/score/state/ancestry、31-node queue全部保持协议一致；数值按预注册 tolerance且 lower bound保持
保守；split/alpha/beta不支持的组合继续fail closed。

性能 go不是本阶段必需；若现有 fused path仍慢，保留证据并进入 G3 的前提是 G1 opportunity成立。

G2 timebox冻结为`2 engineer-days`或`1`个qualification PR（先到者为准）。超时必须提交可审计的
`NO-GO`：列出未被现有kernel公式表示的split/alpha/beta项、最小反例和reference fallback；若G1仍过
opportunity/可达性门禁，则直接进入G3设计新region，不允许qualification无限期卡住路线。

### G3：SelectedObjectiveIR 与 verification-aware TIR family

目标：消除 dense one-hot 和过小 fusion region。

实施：

- 定义 `SelectedObjectiveIR` 和 ragged segment语义；
- Task region覆盖 implicit seed/gather、relaxation、polarity、Conv/Linear contraction、bias和emit；对
  add/concat/flatten/reshape逐个记录 `fuse_through/view_only/materialize_boundary/reject`；
- verifier检查residual fanout contribution恰一次累加、concat slice ordinal/axis、view layout和每个显式
  materialization boundary；
- 支持有限且显式的 Conv capability，先不伪装通用；
- upper/lower pair共享读取；
- shape regime schedule family；
- calibration/held-out按 model family划分；
- 本地 TVM兼容时才引入 DLight/MetaSchedule，搜索数据库作为 artifact。

`PROPOSED-GATE`：

- held-out代表 shape相对最强公平 backend（必须包含qualified auto_LiRPA BoundConv原生路径），kernel
  geomean至少 `1.5x`；
- selected-CROWN region相对strongest qualified reference/backend（含auto_LiRPA source-level path）
  至少 `1.3x`；
- launch数或真实 materialized bytes至少下降 `30%`；
- peak allocated不恶化超过 `5%`；
- 改善必须大于 run noise/MAD；
- 全部 semantic、conservative bound和branch协议通过；
- residual/add/concat held-out graph的fanout contribution、target ordinal和boundary replay exact。

其中`1.3x`只关闭mechanism。若要继续latency headline，G3实测region speedup必须达到G1冻结的
`r_latency_required`；或者提交一个只由**已测**其它region构成的复合Amdahl预算并证明组合后同时达到
queue `1.20x`和complete-query `1.15x`。不得用尚未实现的G5—G7未来收益填补缺口。

未过则冻结 `NO-GO`，不能靠 calibration-only shape宣传。

### G4：frozen alpha/beta final evaluation

目标：让 fused region覆盖更真实 verification state，但不先重写 optimizer/autograd。

实施：optimizer仍用 PyTorch；alpha/beta/split state版本冻结后，compiled backend执行 final no-grad bounds。

`PROPOSED-GATE`：state version/provenance exact；所有 reference bounds和gradient boundary测试通过；在真实
alpha/beta final-eval region达到 G3 region gate。否则 fused capability继续保持 selected plain/split narrow。

### G5：production trace去同步与物理 CUDA arena

目标：让 Schedule memory ownership变成物理效果。

先 profile后分别实施：

- `AUDIT/PRODUCTION` trace模式，移除 timed CUDA path中非必要 `.cpu()`/`.item()`；
- formal replay继续从源输入重算内容 hash；
- static scratch arena、预分配 objective/index/output、event-safe reuse；
- `StorageBinding(arena_id, offset, lifetime)` lowering为物理 tensor view；
- concurrency-aware peak estimator。

`PROPOSED-GATE`：

- tamper/full replay能力不下降；
- production hot path没有未声明 global synchronization；
- trace去同步/速度路线只有在G1中host sync或allocator time占region `>=10%`时准入；
- physical arena/memory路线独立准入：control发生真实OOM/budget failure，或peak allocated/reserved
  任一达到冻结显存budget的`>=80%`；它不要求allocator时间占比；
- 上述memory准入必须对应G1冻结的`B80_alloc/B80_reserved/B_OOM`公开或自然workload；人为software budget实验单独标为
  `policy-budget mechanism`，不升级为目标GPU physical-memory headline；
- peak allocated至少下降 `20%`，或救回真实 OOM；若只降内存，latency退化不超过 `5%`；
- 若做速度 claim，region runtime至少下降 `10%`。

### G6：ragged sibling batching与物理 stream/event Schedule

目标：用 Plan/Schedule在 batch、串行复用和并发间做资源感知选择。

先实施 sibling ragged pack：

- 同一 parent的两个 child沿 domain axis打包；
- 每个ReLU保留独立 indices/segment offsets；
- padding lane完全mask；
- exact child/node ID unbatch和deterministic commit。

再根据 occupancy/critical path尝试 physical streams：

- ready-set scheduler；
- Torch CUDA stream/event；
- event-safe arena reuse；
- memory budget按并发lifetime峰值；
- OOM时可审计地降级到较小batch/单流。

两级control必须分别冻结：

- ragged packed 对比当前最强的 single-stream sequential-child/reference执行；
- multi-stream 对比已经qualified的最强 single-stream packed执行；
- 不允许multi-stream回头只对逐child弱baseline计算speedup。

`PROPOSED-GATE`：

- packed方案 physical selected-CROWN launch至少下降 `35%`，padding不超过有效rows `25%`；
- queue median `candidate/control <= 0.85`；
- peak memory不超过预注册budget；
- lineage/order/deadline/OOM/fallback和verdict协议一致；
- 多流若因occupancy争用变慢，则保留single-stream packed path并关闭并发claim。

### G7：条件式 JIT 与 CUDA Graph

JIT准入：

- measured expected reuse至少为 break-even的 `2x`；
- compile在后台，不在critical path；
- cache qualification、digest、dedup、failure fallback全部通过；
- 不满足则使用AOT/cached family并标记 JIT `NO-GO`。

CUDA Graph准入：

- 同一 `(plan hash, artifact, shape bucket, chunk, stream topology)` 至少重复 `10` 次；
- host launch/dispatch占region至少 `15%`；
- 动态分配已被稳定arena替换；
- capture不含compile、CPU控制、content hash或动态OOM retry。

`PROPOSED-GATE`：graph micro replay `candidate/control <= 0.80`、queue `<= 0.90`、complete query改善
超过噪声且peak受控。否则保留普通 stream launch。

### G8：same-solver multi-workload E2E 与 compiler contribution

目标：决定是否能升级为 ASPLOS 级系统主张。

#### G8 主比较模式：复用 RVIR exact-call 合同

主A/B不再另造host solver：

1. 固定同一alpha-beta-CROWN repo commit、完整config、property、seed、branch、deadline和termination；
2. control使用原始batched bound executor；
3. candidate复用RVIR typed exact-call request/receipt/replay合同，但把backend显式替换为BoundFlow executor；
4. 两侧逐call核对query identity、state version、输入digest、调用顺序、bounds、parent lineage与verdict；
5. RVIR历史证据只证明typed transport/exact-call ownership，**不证明**BoundFlow replacement executor已实现，
   该replacement仍是本路线的新代码与新门禁。

首选同process、同competitor env运行。若PyTorch依赖冲突使其不可行，只允许对control/candidate都使用
对称worker/RPC边界并把IPC计入两侧；非对称跨env/跨process计时不得作为headline，只能列为限制或诊断。

必须比较：

- same model/property/input/spec；
- same CROWN/alpha/beta/split算法和iteration；
- same branch policy、node budget、deadline、termination和verdict；
- auto_LiRPA/alpha-beta-CROWN 原有 batched executor，而非逐节点弱 baseline；
- BoundFlow best-fixed、GPU-context selector、local oracle和每项ablation。

`PROPOSED-GATE`：在至少两个held-out model family上，二选一满足：

- **latency path**：31-node queue geomean speedup `>=1.20x` **且** complete-query geomean speedup
  `>=1.15x`；任一合法workload的complete-query不得退化超过 `5%`；或者
- **memory path**：至少两个memory-bound workload的peak allocated各下降 `>=25%`，同时各自
  complete-query latency ratio `candidate/control <=1.05`；该path必须已通过G1 physical-memory
  admission，policy-budget-only结果不合格；
- 至少一个公开held-out workload必须在相同timeout内由control和candidate都产生相同的非`unknown`
  verdict；否则TTV/solved指标记为`N/A`，G8 latency/TTV claim不能关闭；
- 至少两个 workload/budget context选择不同合法计划；
- GPU-context selector只允许根据shape regime、cache state、显存压力、target/domain batch和expected
  reuse选择已qualified plan；它必须优于best fixed，或形成严格Pareto；
- 若升级`C-END-01`/C2 GPU-context selector claim，held-out p90 Oracle regret必须`<=1.20x`；未运行或
  未过则记为`N/A/FAIL`并降为narrow backend/memory claim；IR-5 broad global planner继续保持
  `VALIDATED-NO-GO`，不在本路线复活；
- complete property报告solved/timeout/unknown、TTV、nodes/s和peak，不能只报更快地产生相同unknown。

若收益全部来自固定手写kernel，允许保留backend贡献，但C2/adaptive compiler claim必须降级。

## 11. 公平 benchmark 矩阵

| 轴 | 最小正式取值 |
|---|---|
| model family | MNIST FC、CIFAR CNN、fixed ResNet2B、至少一个更宽/更深 residual family |
| frontend ownership | 至少两个family的目标region全程native typed；逐op记录import/Bound/backward/TIR/fallback |
| verdict coverage | 至少一个公开held-out workload两侧在同timeout产生相同非`unknown` verdict |
| query phase | initial CROWN、selected intermediate-CROWN、alpha-CROWN、beta/split node、complete BaB |
| domain batch | `1/4/16/设备可承受上限` |
| spec/target | `1/9/32/128/真实 ragged trace` |
| Conv | `1x1/3x3`、stride `1/2`、多组channel/spatial regime |
| memory budget | baseline peak 的 `25/50/75/100%` |
| memory reachability | natural/public workload的`B80_alloc/B80_reserved/B_OOM/max-valid-batch`；physical与policy budget分开 |
| cache state | fresh compile、disk/process hit、warm execution |
| hardware | 历史/目标 RTX 4060（G0重新确认型号与compute capability）；资源允许时增加一块较大显存GPU |
| backend | auto_LiRPA batched + source-level BoundConv oracle、PyTorch eager、TorchInductor、TVM unfused、旧fused、新BoundFlow |
| primary E2E harness | RVIR exact-call contract内 original batched executor vs BoundFlow replacement executor |
| repeats | 正式性能至少5次；顺序反平衡；DocOps最低仍为3次 |
| metrics | max diff、conservative sign/branch/verdict、device/wall、launch、alloc/reserved、TTV、nodes/s、solved/unknown/timeout |

workload split必须按 model family/shape regime留出，不允许把相邻 shape随机切分后称为泛化。

## 12. Artifact 与 replay 合同

每个正式 gate至少生成：

```text
manifest.json
environment.json
queries.jsonl
results_raw.jsonl
normalized.jsonl
summary.json
replay_stdout.txt
failure_rows.jsonl
README.md
```

G0同时生成`legacy_artifact_mapping.json`，明确现有`formal.json + manifest.json + shards/ + logs/`
如何映射到上述前向schema；不得因文件名不同丢失历史raw/replay lineage。

manifest绑定：

- git head、dirty diff digest和三项submodule SHA；
- model/ONNX/VNNLIB/input/spec/weight digest；
- command、environment、GPU、driver、clock/power；
- baseline/candidate code identity和compiled artifact digest；
- warmup/repeat/order/synchronization/allocation合同；
- frozen variables、allowed variables、acceptance/kill gate版本；
- raw payload、normalized table和summary digest。

replay必须从raw重新计算统计和semantic check，不能只验证summary自洽。同步修改payload和manifest digest的
tamper也应因语义重算失败。

## 13. 建议 PR/提交切分

不要在一个巨型“GPU optimization”分支中同时改数学、kernel、planner和benchmark。

| PR | 内容 | claim边界 |
|---|---|---|
| P0 | 本诊断、预注册、G0环境/40x复现协议 | research-only |
| P1 | G1 profiler、raw artifact、GPU opportunity判定 | attribution，不是speedup |
| P2 | G2现有fused wiring与typed capability | mechanism/correctness |
| P3 | `SelectedObjectiveIR`、reference lowering和拒绝路径 | IR correctness |
| P4 | selected-CROWN TIR family和held-out kernel/region evidence | narrow backend claim |
| P5 | physical arena、production trace mode | physical memory/sync claim |
| P6 | ragged sibling batch、physical stream/event executor | queue schedule claim |
| P7 | 条件JIT/Graph，若过准入门禁 | amortization/launch claim |
| P8 | same-solver multiworkload E2E、ablation和external audit | 论文claim升级或NO-GO |

建议每个PR都遵守“一项主要变量、一个变更记录、一组raw artifact、一个replay入口”。

## 14. 替代假设与 kill decisions

| 反方假设 | 诊断方式 | 结果如何改变路线 |
|---|---|---|
| `40x` 来自弱baseline/Python/allocation/cold-start | G0分层计时与同输出合同 | 保留根因，不把40x传播到solver |
| GPU上selected-CROWN不再dominant | G1 device/critical-path attribution | 关闭G2-G4，重新选GPU winner |
| G8门槛在Amdahl上不可达或需`>10x` region | G1反解`s/T/r_required` | 关闭latency G3；转已准入memory path或NO-GO |
| 目标GPU无自然`B80_alloc/B80_reserved/B_OOM` workload | G1 memory-pressure envelope | 该硬件memory headline记N/A，不人为造OOM升级claim |
| workload始终unknown | 公开held-out solved-workload admission | TTV/solved claim不关闭，只保留runtime mechanism |
| frontend不足以覆盖两个family | G0逐op ownership audit | 单独frontend prerequisite PR或降级multi-family claim |
| auto_LiRPA/cuDNN已覆盖相同计算 | 同shape/同method/batched competitor | 只保留真正额外的IR lowering |
| fusion浮点重关联改变branch/verdict | reference/tolerance/conservative/lineage replay | fail closed，不接受速度 |
| ragged padding吞掉收益 | target distribution与padding统计 | 改bucket或停止batch |
| 多stream增加contention和peak | occupancy、overlap、concurrent peak | 回退single-stream packed |
| CUDA Graph private pool反增内存 | allocated/reserved/NVML与graph pool | 关闭Graph |
| JIT复用不足 | fresh/disk/process break-even | 保留AOT，不实现JIT headline |
| target内容进入cache key导致爆炸 | cache cardinality/key audit | target仅作为runtime input |
| GPU-context selector仍不如fixed plan | held-out regret和Pareto | 降级为backend，不复活broad global planner |
| 更快但仍 `9/9 unknown` | complete-query verdict/TTV | 只能主张runtime mechanism，不主张solver effectiveness |

## 15. 对外审计请求

建议至少让不同模型分别扮演四种角色，不互相引用结论：

1. **代码所有权审计**：确认 selected-CROWN、IR、backend、Schedule、storage和cache断点是否属实；
2. **数值/replay审计**：从raw重算所有历史与后续数字，做tamper和拒绝路径探针；
3. **benchmark公平性审计**：检查baseline、同步、compile、allocation、batch、output合同和OOM分母；
4. **研究新颖性/反方审计**：对照auto_LiRPA、alpha-beta-CROWN、TVM、TorchInductor和CUDA Graph，
   判断贡献是否只是通用GPU技巧。

审计输出格式：

```text
verdict: approve | approve-with-minor | revise | reject
blocker: [...]
major: [...]
minor: [...]
claim-by-claim: PASS | FAIL | NOT-AUDITABLE
gate-change-request: [...]
counter-hypothesis: [...]
recommended-first-experiment: ...
```

审计者必须主动回答：

- GPU环境与40x能否独立重放；
- selected-CROWN在GPU上是否真是bottleneck；
- 新region是否消费verification-specific IR，还是固定手写调用；
- logical memory是否真的变成physical allocated/reserved改善；
- packed batch与multi-stream是否比较了最强单流batched baseline；
- JIT真实复用是否超过break-even；
- G1反解的`r_latency_required`是否可达且`<=10x`；
- 至少一个公开workload是否由两侧得到相同非unknown verdict；
- memory path是否在目标硬件存在自然`B80_alloc/B80_reserved/B_OOM`，而非人为budget压力；
- G8是否真正复用RVIR exact-call合同并保持同一alpha-beta-CROWN host solver；
- same-solver、same-property、same-timeout、same-tightness是否成立；
- PR-12、IR-5、NRIR43/46/47的NO-GO是否完整保留。

## 16. 立即下一步

【v1.2 当前路线覆盖】本节以下 G0/G1 指令是 v1.0/v1.1 的历史执行顺序，已完成其
selected-CROWN-only 判定使命。当前唯一下一步是
[full-stack plan](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
的FSG1 official B0 full-stack baseline；FSG0 schema/critical-path/replay合同已经验证关闭。FSG1只建立
可审计B0基线，不宣称性能。

本计划批准后，**只启动 G0，不直接写 TIR 或多流 runtime**：

1. 恢复/确认 NVIDIA driver或本会话的GPU透传；
2. 在timebox内失败则按已冻结规格切换备用GPU，不无限期修本机；
3. 找到用户的 BoundConv `40x` 代码、commit和原benchmark；
4. 完成frontend逐op coverage与至少一个可solve公开workload admission；
5. 建立同设备、同输出、同同步、同allocation的复现矩阵，并预判
   `B80_alloc/B80_reserved/B_OOM`；
6. 将G1 profiling字段、Amdahl反解公式和所有performance/kill gate在运行前冻结；
7. G1确认opportunity、`r_latency_required<=10x`或memory reachability后，再决定G2/G3是否开工。

若需要命名下一工程路线，建议使用：

> **NRIR49: GPU Selected-CROWN Opportunity and Verification-Aware Compilation**

它明确区别于旧 PR-12 plain-CROWN TIR，也避免在尚无GPU证据时承诺JIT、多流或完整alpha-beta融合。

## 17. 仓库证据入口

- [ASPLOS claims map](asplos_claims_map.md)
- [PR-12 closure audit](pr12_closure_audit_2026_07_14.md)
- [PR-13 closure audit](pr13_closure_audit_2026_07_14.md)
- [PR-12I fair baselines](change_2026-07-14_pr12i_fair_baselines.md)
- [PR-12J compile amortization](change_2026-07-14_pr12j_compile_amortization.md)
- [PR-12L stop decision](change_2026-07-14_pr12l_stop_tir_optimization.md)
- [IR-5 final NO-GO](change_2026-07-28_ir5h_residual_final_v3_nogo.md)
- [NRIR43 batching NO-GO](change_2026-08-05_nrir43_cross_axis_batch_nogo.md)
- [NRIR46 phase-0 NO-GO](change_2026-08-05_nrir46_phase0_nogo.md)
- [NRIR47 phase-A NO-GO](change_2026-08-05_nrir47_phase_a_nogo.md)
- [NRIR48 execution attribution](change_2026-08-05_nrir48_execution_cost_attribution.md)
- [compiler IR contract](boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md)
- [外部审计报告（v1修订前）](external_audit_gpu_compiler_plan_v1_2026_08_05.md)：
  `approve-with-minor`，M-1的PR-12J compile归属已在v1.1修正

## 18. 最终边界

本计划的完成意味着：代码事实、历史结果、研究假设、依赖顺序、benchmark合同和kill gate已形成一份
可反驳的草案。它不意味着：

- GPU已恢复；
- `40x` 已复现；
- selected-CROWN GPU bottleneck已确认；
- TIR/JIT/arena/multi-stream已实现；
- BoundFlow已有auto_LiRPA或ASPLOS性能主张。

在 G0/G1 之前，最诚实也最有价值的状态是：**GPU编译路线值得重开，但必须从证据恢复和真实GPU
bottleneck归因开始。**
