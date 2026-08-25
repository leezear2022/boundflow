---
status: proposed-and-frozen-for-measurement
updated: 2026-08-25T19:10:00+08:00
type: architecture
topic: boundflow
slug: fully-compiled-verifier-runtime-v1
stage: s01
---

# BoundFlow 全编译验证器运行时 v1 架构与研究路线

## 1. 起点与最终目标

当前 MR5/MR6 的失败不能推出“编译器不适合验证器”。它只证明：把三个 α-CROWN Conv
站点逐个替换成 TVM kernel，同时让 PyTorch/auto_LiRPA 继续拥有 tensor layout、autograd、optimizer、
调度和内存生命周期，无法把局部 kernel 收益传播到 production outer。

本路线的最终目标不是继续堆叠孤立 bound-op，而是把验证器热路径变成一个由 BoundFlow 拥有的、
可编译和可调度的张量程序：

- PyTorch 只保留为可选前端、独立 oracle 和迁移期 fallback，不再是 production 热路径 runtime；
- TVM/TIR 不只编译 Conv/Linear，而是编译合法的多算子 verification region；
- α/β、split/history、optimizer state、domain queue 和临时张量的所有权进入 typed IR/runtime contract；
- launch、stream、依赖、buffer lifetime 和 arena placement 由统一 runtime graph 决定；
- correctness receipt 从每个微操作的热路径检查，提升到 compile/admission 与 coarse commit 边界。

“最终剔除 PyTorch”是目标状态，不是未经门禁的即时承诺。迁移期间必须始终保留可重放 oracle，且任何
compiled region 都需先过 typed admission、数值/符号等价、状态轨迹和篡改拒绝门禁。

## 2. 为什么当前 bridge 会输给 PyTorch

已有证据同时成立：

- 单个 B4-B2 TIR 局部路径相对 PyTorch oracle 为 `4.89834x`；
- CIBC IBP 真实 Conv operator 几何平均为 `12.7951x`，完整 IBP 图为 `2.45631x`；
- MR5 production bridge 相对 native provider 只有 `0.83440665x`；
- MR6 删除多数热路径 value guard 后也仅比 full bridge 快 `1.03312564x`，相对 provider仍为
  `0.90300665x`。

因此矛盾不在“TVM kernel 一定慢”，而在当前 ownership boundary：一次 outer 中有 30 forward、
27 backward TVM launch，并产生约 540 次 DLPack view/pointer 往返；与此同时 PyTorch 仍执行 α 重建、
permute/transpose/contiguous、zero allocation、autograd graph、optimizer update 和结果 materialization。
孤立 kernel 只优化了局部计算，却额外引入 framework crossing，且无法跨算子消除中间张量。

## 3. 编译边界原则

### 3.1 不盲目编译 Python 控制流

第一阶段保留 host solver 的分支终止、规格 verdict 和策略选择；只把稳定、可类型化、重复执行的 tensor
region 编译掉。随着 state machine 和 queue contract 成熟，再把可证明的控制段迁移到 device runtime。

### 3.2 编译单位是 verification region，不是单算子

最小有意义的 region 应能同时拥有输入 layout、中间表示、backward 合同和输出 commit。候选包括：

1. relaxation slope/intercept 构造 + α 解压/投影；
2. ReLU sign selection + Conv/Linear bound propagation + bias reduction；
3. compressed α/β 与 split/history 驱动的 custom backward；
4. optimizer evaluate + gradient/update/project/clamp；
5. KFSB/branch score + reduction + top-k；
6. child split generation + bound/state materialization；
7. domain score + batching + queue commit；
8. state pack/copyout 与最终 coarse receipt。

### 3.3 dense A 不跨层保存

B4-C2 已证明跨层保留 dense adjoint/autograd history 会使内存变为 `1.3401x`，速度降至
`0.337–0.349x`。新路线只允许保存语义上最小状态，例如 compressed α/β、split/history、identity 和
必要的统计量；dense A 在 kernel/region 内生产、消费或重算，不成为跨层长期 state。

## 4. 五层系统结构

### L1：Verification Semantic IR

负责定义 operator、bound direction、domain、start-node、α/β、split/history、lineage、mutation 与
fail-closed admission。它回答“这段计算在验证语义上是否合法”。

### L2：Compiled Tensor Program IR

把多个合法 verification op 降为显式 tensor program，包含 shape/layout、reduction、epilogue、VJP、
alias 和 mutation effect。它必须支持跨 op fusion，而不是退化为逐 op 调 TVM。

### L3：Runtime Execution Graph

把 compiled regions、host decisions、events 和 state commits 组织成 dependency DAG。它拥有 CUDA Graph、
cross-site/cross-step launch amortization、异步 error ledger 和 coarse synchronization。

### L4：Memory/Arena Plan

根据 liveness、alias、reuse distance 和 stream dependency 为 persistent state、scratch、output 与 copyout
分配 arena。目标是消除重复 zeros/empty/contiguous 和 allocator traffic，并阻止 dense A 跨层存活。

### L5：Queue/Parallel Scheduler

按独立 domain、branch、property、site 和 shape signature 做 batching 与多 stream 调度；只在依赖满足时
并发，禁止为了 headline 制造无语义依据的 overlap。调度结果必须进入 receipt，可重放。

## 5. 优先融合模式

按潜在收益和语义风险排序：

1. **layout + relaxation + bound op + epilogue**：消除输入 permute/contiguous、zero bias、输出 transpose
   和 finite scan；
2. **forward + minimal-saved-state custom backward**：forward 不输出/保存 dense 中间量，backward按需
   重算；
3. **同 site 多 optimizer step**：固定 topology/shape 下持久化 buffer，合并反复的 admission、launch
   和 state projection；
4. **跨 site pipeline**：C2→C1→C0 采用统一 layout/arena，只有真实数据依赖允许的边界才同步；
5. **branch score + reduction + top-k**：避免把完整 score tensor 回送 host；
6. **split + child state materialization + queue commit**：直接在 device arena 生成下一批 typed state。

## 6. 运行时改进假设

### 6.1 Persistent buffer 与统一 arena

当前每次调用重新构造 output、bias、gradient 和 layout-copy tensor。统一 runtime 应按 signature 预分配，
用 epoch/lease 防止别名冲突，并由 memory plan 给出 peak 与复用证明。

### 6.2 Launch/FFI amortization

当前 57 次 launch、约 540 次 DLPack/pointer crossing 是明确的候选成本。可选机制为单次 region launch、
CUDA Graph replay、batched FFI descriptor 或 persistent command graph；选择必须由 MR7 share 决定。

### 6.3 多分支并行

独立 branch/domain 可以并行执行 bound region、score 和 materialization。调度器需要显式 stream/event
dependency、显存预算和公平性策略；必须报告真实 overlap 与 critical path，不能把 kernel-sum 当 wall。

### 6.4 AOT/cache 优先，JIT 仅作可摊销 fallback

固定 model family、shape 和 verification signature 先生成 AOT family/cache。只有预计 reuse 超过实测
break-even 时才后台 JIT；compile time、cache miss 与 eviction 都计入完整 query 账本。

## 7. 分阶段迁移

### FCR-0：边界真相与 MR7 归因

- 冻结 PyTorch op、TVM launch、DLPack crossing、materialization、allocator 和 sync 账本；
- 分开 host critical-path、CUDA kernel sum 和 overlap，不跨时钟域相加；
- 输出第一段可编译 region 的选择依据。

### FCR-1：单站点多算子 compiled region

- 选 MR7 最大且数学可达的一段；
- 输入/输出保持 device resident；
- 把 layout、relaxation、bound op、epilogue 和 custom backward 纳入同一合同；
- 相比当前 bridge 显著减少 57 launch/540 crossing，而不是只换 kernel schedule。

### FCR-2：device-owned optimizer

- 编译 evaluate、VJP、update/project/clamp；
- 验证 10-step mutation trajectory、terminal state 与 native exact-call 等价；
- optimizer step 间不把 tensor state 交还 PyTorch。

### FCR-3：branch/queue tensor path

- 编译 KFSB/score/top-k、split 和 child state materialization；
- host 只接收 compact decision/termination receipt；
- 开始评估多 domain/branch stream 并行。

### FCR-4：统一 execution graph 与 arena

- 跨 region 做 liveness、alias、persistent allocation 和 CUDA Graph；
- 形成 model/signature-keyed static schedule；
- PyTorch runtime 从 production tensor hot path 移除。

### FCR-5：same-solver formal closure

- 同一 αβ-CROWN host solver、branch、termination、seed 与 workload；
- native executor 对 fully compiled executor；
- 先过 correctness/state/receipt，再过 B0 parity、`1.15x` complete-query 和 `1.20x` queue 研究门禁；
- 至少两个 held-out model family，且至少一个 workload 在 timeout 内产生非 unknown verdict。

## 8. 门禁与 kill rules

每一阶段都必须满足：

- discrete/sign exact；数值误差使用预注册 tolerance；
- α/β/split/history/optimizer mutation ownership可重放；
- cache、schedule、stream、arena、launch 和 fallback receipt完整；
- fully re-signed semantic/state/topology tamper fail-closed；
- performance headline 至少三组 fresh、counterbalanced、unprofiled；
- profiler只做归因，不作为 headline timer；
- 如果由真实 share 反推所需 region speedup `>10x`，该切片直接NO-GO；
- memory 路线只有真实 peak/OOM 可准入，不能用人为小预算制造 claim；
- 局部收益不能自动升级为 query、queue、competitor 或 ASPLOS claim。

## 9. MR7 在本路线中的位置

MR7 不是“继续调一个 Conv kernel”的代号，而是 FCR-0 的第一份物理账本。它必须回答：

- 当前 bridge 的 CPU/FFI/materialization/launch/kernel critical path 分别有多大；
- 57 launch 和约 540 crossing 中哪些能由 persistent region 一次消除；
- 各 site 是否真的由 kernel schedule 主导，还是被 framework boundary 主导；
- 第一个 FCR-1 region 应是 per-site kernel、cross-site/step graph，还是 layout/arena/FFI 合并。

MR7 原冻结路由继续有效，但解释更新为：MR7-A/MR7-C 若成立，产物必须是多算子 compiled region/runtime
计划，不能只做一个更薄的 PyTorch↔TVM wrapper；MR7-B 只有在 device kernel share 和跨 run site winner
同时成立时才允许 schedule sweep。

2026-08-26历史执行注：MR7因一个CUPTI profile/control扰动样本超过`1.10`而正式INVALID；其host boundary
与device kernel数字仅作诊断，不能开放上述路由。下一步MR7-R只用unprofiled pair验证host ledger与
boundary；通过后最多开放FCR-1的compiled-region/arena/FFI ABI与correctness。

2026-08-26后续执行注：MR7-R已以10 fresh/5 pair正式通过，boundary median=
`20.333%/24.684 ms`、5/5过门禁，required parity region speedup=`1.91214x`。因此现只开放
GC-0/FCR-1 verification graph ABI与correctness预注册；timing、query和queue仍关闭。

## 10. 预期贡献边界

这条路线的研究主张不是“把 CROWN 放到 GPU”，而是：verification-specific IR 使编译器能在保持
α/β/split/history 语义和 fail-closed 证据的前提下，联合优化 tensor program、state mutation、launch、
memory 与 branch scheduling。它比孤立 TVM bound-op 有更高理论上限，因为它能消除 framework crossing
和中间 materialization，并利用跨 op/step/site 的结构；但能否达到系统门禁仍必须由 FCR-0—FCR-5
逐级证伪或验证。
