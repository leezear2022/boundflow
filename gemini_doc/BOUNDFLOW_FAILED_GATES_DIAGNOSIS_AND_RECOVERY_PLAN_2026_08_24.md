---
status: diagnosed-recovery-preregistration-required
updated: 2026-08-25T00:50:00+08:00
type: plan
topic: boundflow
slug: failed-gates-diagnosis-and-recovery
stage: s01
---

# BoundFlow 失败门禁诊断与恢复计划

> **2026-08-25 R0/R1执行更新**：R0 的3条新增mypy `arg-type`已修复，lazy runtime import的
> pylint `C0415`已限定并记录循环依赖原因；CIBC closure已补steady-state/cold边界和预注册
> `3e-4`/单-ULP量级解释；§12已补FSG3、NRIR49A、B4-C2 raw，B4-A唯一性能分类已澄清为
> externally approved NO-GO。R1 scope/clock/query-local协议已冻结但未实现/运行；下一动作只允许
> 实现clock/topology/schema及negative tests。独立IBP `G=2.45631`不得代填same-solver
> `G_query,k`。见`BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`。

## 0. 一句话结论

下一步不是继续给现有 `B4-C2` dense autograd 链打补丁，也不是立即宣称 CIBC 已经完成。
应先完成 **R0 审计卫生 + R1 协议/目标冻结**，再启动只读的 **CIBC-G1 optimized-graph
attribution**：在当前已经外审批准、整图 `2.4563x` 的 CIBC-IBP candidate 上，用
NVTX/CUPTI/CUDA Graph node 证据重新分解 Conv、Linear、ReLU、residual add、input copy 与
graph/runtime 的时间和关键路径。随后必须在 same-solver 路径实测 eligible-IBP query share；只有
把同一时钟域的 share 与同一层级的冻结目标代入可达性公式后，才允许选择
`Linear/elementwise 图融合`、`Conv 深度调优`或 `copy/runtime`，而不是凭直觉挑算子。

对 α-CROWN 路线，局部 TIR 已经证明能快 `4.89834x`，失败点不是“CUDA/TIR 本身不行”，而是
**集成所有权与 autograd 生命周期不对**。后续若恢复，必须从“结构化表示保持到自定义 backward、
不跨层保存 dense A”重新设计，不能复活 B4-C2。

## 1. 本文解决什么问题

本文把截至 2026-08-24 的状态分成四类，避免再次把不同结论混在一起：

1. **正式失败（NO-GO）**：候选已按预注册协议测量，未过冻结门槛；
2. **部分通过（VALIDATED-REDUCED）**：局部或相对上一基线成立，但完整目标未达到；
3. **尚未开放/尚未运行**：没有失败证据，不能写成 NO-GO；
4. **已经通过**：其作用域内成立，但不能自动外推到 auto_LiRPA、α-CROWN、BaB 或 ASPLOS 系统 claim。

目标不是为历史结果辩护，而是回答三个更重要的问题：

- 哪些门禁真没过，差多少；
- 失败来自物理上没机会、实现没做好，还是集成边界选错；
- 下一笔工程投入怎样最短地证伪或证实系统级加速路线。

## 2. 证据纪律与口径

### 2.1 事实来源

- 门禁数字优先读取仓库 formal artifact、replay 与外部审计，不使用聊天摘要替代 raw；
- `speedup = baseline / candidate`，大于 1 才表示 candidate 更快；
- “core”只表示冻结 solver prefix 内的核心区域，不等于 complete query；
- steady-state 与 cold/JIT 成本分开，不允许把编译成本悄悄排除后再泛化为端到端；
- local operator、whole IBP graph、same-solver core、complete query 是四层不同 claim；
- Amdahl 上限只约束被测区域，不约束未被包含的 BoundFlow 全栈。

### 2.2 当前最终系统门槛没有改变

- queue/BaB 端到端：相对 B0 geomean `>=1.20x`；
- complete query：相对 B0 `>=1.15x`；
- 任一 workload 不得回退超过 `5%`；
- 至少一个 held-out workload 在相同 timeout 内得到非 `unknown` verdict；
- memory claim 需要两个自然 workload 峰值下降 `>=25%`，且 latency 不超过 `1.05x`；
- ≥2 个 held-out model family，不能只在 ResNet2B property 0 上闭环。

这些门槛目前是 **尚未运行**，不是“已经失败”。

## 3. 总账：哪些门禁没过，哪些并没有失败

| 路线 | 实测结果 | 冻结门槛/目标 | 分类 | 能说明什么 |
|---|---:|---:|---|---|
| NRIR49A selected-CROWN 机会 | queue share `7.0986%`；无限加速上限 `1.0764x` | share `>=20%`；queue `1.20x` | NO-GO（仅该单区域） | 不能只靠 selected-CROWN 区域达到系统门槛 |
| FSG3/B2 reference replacement | B0/B2 query `0.9084x`；core `0.5168x` | 作为正确性/所有权基线 | 慢基线，不是最终候选 | typed reference 接管成本很高 |
| FSG4/B3 IR/graph/plan reuse | B2/B3 core `1.0716x`；query `1.0066x`；B0/B3 query `0.9100x` | full core `>=1.15x`；B0 parity `>=1.0x` | VALIDATED-REDUCED | 相对 B2 恢复 7.2%，仍比 B0 慢约 9% |
| B4-A terminal reuse | core `1.0190x`；query worst `0.99695x` | core `>=1.03x`；query worst `>=0.98x` | NO-GO | 正确但收益太小，不能累计进正式基线 |
| B4-B2 v1 TIR | geomean `0.42484x`；worst `0.37769x` | geomean `>=1.05x`；worst `>=0.98x` | NO-GO（v1 物理实现） | 6 个标量式 kernel 的实现方向不对 |
| B4-B2 v2 Triton | 对 PyTorch `2.83772x` | minimum `>=1.20x` | PASS | 水平融合和调度物理机会成立 |
| B4-B2 v2 manual TIR | 对 PyTorch `4.89834x`；对 Triton `1.68273x` | 对 PyTorch `>=1.20x`；达到 Triton `>=0.90x` | PASS | TVM/TIR 局部 kernel 并非瓶颈 |
| B4-C0 native-value bridge | core `0.94034x`；worst `0.93418x` | geomean `>=1.0x`；worst `>=0.98x` | NO-GO | native value 与 TIR gradient 双算抵消局部收益 |
| B4-C1 provider-owned lower | core `0.94815x`；worst `0.94547x` | geomean `>=1.0x`；worst `>=0.98x` | NO-GO | 拿到 value 所有权仍未拿到表示/生命周期所有权 |
| B4-C2 六层 dense retention | `0.337–0.349x`；allocated `1.3401x` | kill：明显回退或 memory `>1.05x` | HARD NO-GO | dense A 与 autograd history 跨层存活是错误边界 |
| CIBC-IBP 6 Conv | operator geomean `12.7951x`；worst `9.1423x` | `>=2.0x`；worst `>=1.2x` | EXTERNALLY APPROVED PASS | 四 Conv → 一个 center/deviation TIR kernel 有效 |
| CIBC-IBP ResNet2B 整图 | geomean `2.45631x`；worst `2.45091x` | `>=1.5x`；worst `>=1.2x` | EXTERNALLY APPROVED REDUCED PASS | 仅 IBP、单模型、steady-state、对 BoundFlow baseline 成立 |
| B4-D、B5、B6、B7、complete solve | 未运行 | 系统门槛 | CLOSED/UNTESTED | 没有证据写成失败，也没有证据写成成功 |

## 4. 对每个未过门禁的详细诊断

### 4.1 NRIR49A：把“一个区域的上限”误当成“系统路线的上限”

实测 selected-CROWN region 只占 queue `0.0709863`、complete trace `0.0705233`。即使把这一段
时间降为零，queue 上限也只有：

```text
S_max = 1 / (1 - 0.0709863) = 1.076410x
```

因此它没过 `share >=0.20` 是正确结论。但早期路线上的结构性问题是：我们曾把这个单区域机会
门禁描述得过于接近“BoundFlow 加速空间不足”。这不成立，因为 BoundFlow 原目标还包括 IBP、
CROWN 其他区域、图融合、JIT、调度、内存和多分支运行时。

可保留的证据：selected-CROWN-only 不值得单独投入；不可外推的结论：BoundFlow 全栈最多
`1.0764x`。

### 4.2 B2/B3：正确性架构成本没有被全额收回

B2 是 correctness-first 的 typed exact-call reference，B0/B2 query `0.9084x`、core `0.5168x`。
B3 通过 IR/graph/plan reuse 把相对 B2 core 提高到 `1.071617x`，query 没回退，但相对 B0 query
仍只有 `0.910001x`。

量化差距：

- 仅回到 B0 parity，B3 之后还需要 `1 / 0.910001 = 1.09890x` 的 query 改善；
- 达到最终 `1.15x` complete-query，B3 之后需要 `1.15 / 0.910001 = 1.26373x`；
- 因此只做 1%—3% 的局部 micro-optimization 不可能形成最终故事。

没做好的地方不是 B3 门禁，而是此前把 Python/receipt/typed-state/atomic-commit 机制接进热路径，
再期待一个 plan reuse 阶段全部回收。**receipt 是待测归因假设，不是已经证实的唯一瓶颈**：R1 与
same-solver profile 必须分别量出 validation、state assembly、dispatch、atomic commit 和 O(1)
identity/counter 的 exclusive/critical-path 成本。只有被 raw 证明为热路径成本的静态/结构检查，才
允许移到 compile/admission；运行时仍须保留 fail-closed identity 与必要的 O(1) 证据。

### 4.3 B4-A：目标区域太小，机制正确但 headline 不够

B4-A 消除 optimizer 第 10 次 evaluation 后的 terminal export 重复 CROWN call。core geomean
`1.018995x`，离 `1.03x` 门槛只差约 1.08% 的相对改善；query worst `0.996947x` 通过 no-regression。

这里不是实现完全失败，而是区域本身太小。它可作为未来 cumulative implementation 的一项，
但根据预注册纪律不能把 `1.9%` 偷偷累计进 B3 baseline。只有当新的全栈 candidate 独立重新跑
B0/B3/candidate 正式协议时，才可重新纳入。

### 4.4 B4-B2 v1：当时确实没有做到 CIBC 级算子融合和调优

v1 虽然把 allocated memory 降至 `0.474638x`，但物理 inventory 是 `3 forward + 3 backward`
真实 CUDA kernels；没有 shared-memory tiling、vectorization、half/packed load，也没有把中间量
从 global workspace 中完全消除。最终 geomean `0.424842x`，说明 wrapper、launch 和标量 reduction
开销大于节省的计算。

这正是“没想到/没做好”的明确案例。随后 v2 按 CIBC 的横向融合思路改为 exact `1+1` kernel、
零 global intermediate workspace，并通过 12 项 Triton search 与 manual TIR port，分别达到
`2.83772x` 与 `4.89834x`。所以 v1 NO-GO 只关闭 v1，不是否定算子级路线。

### 4.5 B4-C0：为了保住 Adam 数值轨迹，生产路径做了双份工作

B4-B3 exact-call 的 terminal semantics 正确，但累计计时中的 candidate 仍以 native path 计算 lower
value，同时用 TIR 计算 gradient。这个 bridge 是正确性脚手架，不是性能设计。结果 core
`0.940341x`，memory allocated `1.04818x` 仅勉强过 `1.05x`。

关键教训：候选要对一个 semantic value 只有一个 production owner。reference/native shadow
只能在 correctness/profile worker 中运行，不能留在 control 热路径。

### 4.6 B4-C1：所有权拿在了错误的表示边界

C1 移除了 native lower 双算，做到了 provider-owned value，但仍在结构化 production operator tree
本来无需 materialize 的位置提前生成 dense tensor，并承担 provider/custom-autograd 包装成本。
geomean `0.948150x`、worst `0.945475x`，说明局部 `4.90x` 没传播到累计 core。

这揭示了更深的边界：需要接管的不只是数值 `lower`，而是 **structured representation、消费点、
保存集合与释放时刻**。如果 native 路径用轻量 operator tree 延迟组合，而 candidate 提前 dense，
即使单 kernel 更快也会输。

### 4.7 B4-C2：扩大覆盖前没有先设计 autograd live set

C2 接管 6 个真实 lower materialization sites，每次 optimizer 有 60 次 receipt，正确性仍是
max diff `4.768e-7`、sign exact；但三轮 speedup 只有 `0.348761/0.337448/0.346003x`，peak allocated
增长到 `1.3401085x`。

直接根因是 6 层 dense coefficient 及其 autograd history 跨层存活。native structured path 保存的
是轻量 operator tree，candidate 却把“避免重复物化”变成“长期保留大 tensor”。这是设计层失败，
不能靠 threads、unroll 或再加一个 TIR schedule 修好。

恢复前必须先满足三个结构门禁：

1. forward 不把 dense A 保存到跨层 autograd graph；
2. backward 只保存 compressed α/β、split/history、shape/weight identity 等最小状态，或按需重算；
3. 每层消费完成后有可观测的 release/liveness receipt，六层峰值不得高于 native baseline。

## 5. CIBC 为什么能快，而当前实现还缺什么

### 5.1 当前真正完成的内容

当前 CIBC-IBP 实现不是“只改公式的假 benchmark”，已外审确认：

- 对 6 个 production Conv，把 lower/upper 的四次 PyTorch `conv2d` 正负权重分支改成一个
  center/deviation TIR kernel；
- baseline 与 candidate 都进入 CUDA Graph，输入 copy 都计时；
- 6 Conv operator geomean `12.7951x`，完整 ResNet2B IBP graph `2.45631x`；
- replay、10 类仓内 tamper 和审计方另造 3 类攻击均 fail-closed；
- 语义 sign exact，整图最大差 `2^-12`，在该量级是 float32 1 ULP。

### 5.2 当前调优很浅，不能等同于论文 CIBC

`boundflow/backends/tvm/cibc_ibp_conv.py` 当前本质是 one-thread-per-output + serial reduction，
schedule 只在 `64/128/256` threads 中三选一。它没有完成论文中的多层 tiling、split/reorder/fuse、
shared/local cache、compute-at/inline、vectorize/unroll、hardware cost model/evolutionary search。

当前三个 threads 候选的最好全局 schedule 是 128；按算子独立从三个候选取最好，geomean 只从
`12.7951x` 到约 `12.929x`，仅约 `1.05%`。这不证明深度调优没价值，只说明继续扩大同一个
“threads 数”旋钮的边际价值很小。

### 5.3 整图仍有明确未融合部分

ResNet2B IBP candidate 中只有 6 Conv 进入 CIBC path；2 Linear、6 ReLU、2 residual add、flatten
仍是 PyTorch/图内原算子。算子级 `12.8x` 被整图稀释到 `2.456x`，既有 Amdahl 稀释，也有 baseline
operator 处于 eager launch-bound、整图 baseline 已被 CUDA Graph 压缩的差异。

因此下一步不应直接猜“再调 Conv”或“先做 Linear”，而应在 **candidate 自身** 重新归因。

## 6. 从一手资料得到的可执行修正

以下不是泛泛 GPU 建议，而是与当前失败点一一对应：

1. Apache TVM MetaSchedule 官方教程说明，搜索空间可直接包含 loop tiling、vectorization、
   thread binding，并在目标硬件上用 evolutionary/XGBoost cost model 和 database 选择 schedule。
   这对应当前只有 3 个 threads 候选的缺口。
2. TVM 的 operator fusion 设计明确把消除中间 allocation 和 kernel launch 作为收益，并用
   post-dominator 处理 residual/diamond graph。这对应 ResNet residual add/ReLU 的图级机会。
3. Triton matmul教程显示 block tiling、L2-friendly program ordering 与 autotune 会造成显著差异。
   这支持先给 2 个 Linear 建独立 search space，而不是复用 Conv 的 threads schedule。
4. CUDA Graph 能降低 CPU launch overhead，但不会自动融合节点；Nsight Systems/CUPTI 可用 NVTX
   projection、graph node tracing 与 correlation id 把 runtime API、kernel 和 memcpy 对齐。
   这正是 CIBC-G1 attribution 所需证据。
5. CUDA stream 只保证同 stream 顺序；多 non-default streams 允许但不保证 kernel concurrency。
   因此 residual 多分支并行必须先证明 critical-path headroom，不能把“多流”当默认优化。
6. PyTorch autograd 会在 forward 构图并保存 backward 所需 tensor；custom `Function` 和
   `save_for_backward` 可以控制保存集合。AOTAutograd 的 joint graph/min-cut 思路说明“保存还是重算”
   是可优化的分割问题。这对应 B4-C2 的 dense live-set 爆炸。
7. α/β-CROWN 的 bound quality 与分支/求解结果耦合。任何 runtime 优化都必须保持 lower、sign、
   α/β 更新与最终 verdict，不能只比较 kernel latency。

## 7. 恢复路线

### R0：审计卫生闭环（2026-08-25 已完成，不改性能语义）

目标：让已经批准的 CIBC claim 没有模糊口径。

- 修复 `boundflow/domains/interval.py:83-85` 本轮新引入的 **3 条 mypy `arg-type`**；
- 处理或明确限定 `boundflow/domains/interval.py:74` 本轮新增的 **1 条 pylint `C0415`**；既有
  `DomainState` 的 8 条 `attr-defined` 不在本修复中偷换范围；
- 在 CIBC closure 补充 `3e-4` 来自正式运行前冻结且实测 `2^-12` 为该量级 1 ULP；
- 明确 operator/graph 数字是 steady-state，TIR compile 与 plan construction 不在计时区；
- 下一阶段额外记录 cold compile、plan construction 和 break-even，不把它们混进 steady-state；
- `.docops/ev.jsonl` 三个历史重复 id 作为独立 DocOps 维护任务，不与性能代码混交。

R0 已由 `BOUNDFLOW_R0_HYGIENE_R1_PREREGISTRATION_CHANGELOG_2026_08_25.md` 关闭：不重跑正式
性能、不改变任何阈值、不升级 claim。三个历史 DocOps duplicate id 仍作为独立维护项，不影响
`dol lint --soft`，也未与本轮性能/代码修复混交。

### R1：CIBC-G1 optimized-graph attribution（协议已冻结，runner/artifact 尚未实现）

目标：回答当前 candidate 的剩余 0.071–0.072 ms 究竟花在哪里。

以下摘要受独立预注册
`BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`约束。该计划把 R1
拆为 candidate graph attribution、same-solver eligible admission、exact query-local replay 和机械
route closure；本节未写出的 raw/schema/tamper 条款以独立预注册为准。

#### R1.1 必须分出的桶

- input lower/upper copy；
- 6 个 CIBC Conv TIR nodes；
- 2 个 Linear；
- 6 个 ReLU；
- 2 个 residual add；
- flatten/view；
- CUDA Graph/runtime/同步残差；
- 若 graph 中存在独立 residual branch，记录 critical-path 与可并发空隙，而非只加 kernel sum。

#### R1.2 证据机制

- 给每个高层节点加入稳定 NVTX range/ordinal；
- CUPTI activity 用 correlation id 恢复 kernel/memcpy owner；
- Nsight Systems 开启 CUDA Graph node trace，并把 NVTX range 投影到 GPU；
- control/profile 成对 fresh worker，headline 只读 control，profile 只做归因；
- profile/control 扰动必须预注册且 `<=1.05`，raw-first，不能从 summary 倒推；
- kernel sum、exclusive wall、critical path、overlap-adjusted wall 四个口径分开。

时钟域必须额外 fail closed：

- 用一个显式 CPU/NVTX 同步点绑定 CUPTI GPU timestamp 与 host monotonic timestamp，冻结映射参数；
- Nsight Systems export 必须生成 calibration receipt，记录 trace session、clock source、同步点、
  最大残差和无法关联的 event 数；receipt 缺失或残差超预注册阈值时，不得形成 share；
- CUDA Graph node、kernel、memcpy 只能在完成 correlation 和时钟校准后进入 critical path；
- 若图是单 stream 且没有真实 overlap，headline 采用 exclusive/critical-path wall；此时
  overlap-adjusted 结果必须退化为同一口径，不能把时钟域差或重复扣除包装成额外收益。

#### R1.3 量化路由公式

先冻结三个不同系统层级的目标，后续不得互换分母：

| 目标 | 冻结值 | baseline 与用途 |
|---|---:|---|
| `T_query_qualification` | `1.00x` | candidate/B0 complete-query parity，决定能否作为累计候选 |
| `T_query_research` | `1.15x` | candidate/B0 complete-query 研究门槛 |
| `T_queue_research` | `1.20x` | candidate/B0 queue/BaB 端到端研究门槛 |

`T_graph` 是每个 whole-IBP-graph 实验另行预注册的局部目标；当前 CIBC closure 的历史资格门槛仍是
相对 BoundFlow 四-Conv graph baseline `>=1.50x`，但它不等于 query 或 queue 目标。由此：

- graph 内 share `s_graph` 只能与同一 graph timing scope 的 `T_graph` 配对；
- complete-query share `s_query` 只能与 `T_query_qualification/research` 配对；
- queue share `s_queue` 只能与 `T_queue_research` 配对；
- kernel-sum share 不能直接代入 exclusive/critical-path 的目标。

对同一 timing scope 的目标增益 `T` 和实测区域 share `s`，需要的区域加速倍数为：

```text
r_required = s / (1 / T - (1 - s))
```

只有分母为正且 `r_required` 在该类 kernel 的物理可达范围内，才开实现分支；分母 `<=0` 表示该
单区域即使无限加速也无法到达目标。多区域 candidate 使用 measured overlap/critical path 重算，
不能简单相加 kernel time。

在把 CIBC graph 收益外推到 query 前，必须先从 **same-solver original executor vs RVIR adapter +
BoundFlow executor** 的 eligible 调用中分别实测两侧 share。下式使用的是待优化 B3/candidate 侧
`q_B3 = eligible_replaceable_IBP_wall_B3 / complete_query_wall_B3`，而不是 B0 share 或 profiler kernel
sum。以当前 B3/B0 query ratio `R_current=0.910001` 和已批准 CIBC whole-graph speedup `G=2.45631`
作为纯可达性上界，进一步假设该 graph speedup 能无接入成本覆盖同一 replaceable B3 region，则：

```text
R_new = R_current / ((1 - q_B3) + q_B3 / G)
q_B3_required(T) = (1 - R_current / T) / (1 - 1 / G)

q_B3_required(1.00) = 0.151798   # 回到 B0 parity 至少 15.18%
q_B3_required(1.15) = 0.351998   # 达到 query 研究门槛至少 35.20%
```

这两个数只是用现有 graph speedup 计算的乐观 feasibility bound，不是 query speedup claim；在
`q_B3,k`、op-type构成、adapter/wrapper成本、region identity、eligible coverage与exact production
signature的`G_query,k`未由raw冻结前，不得据此宣称B0 parity可达。真实传播必须使用
`delta_k=q_B3,k*(1-1/G_query,k)`或event-DAG critical-path counterfactual；测不到的`G_query,k`按1处理。

#### R1.4 same-solver 与 benchmark 准入（只读，不提前补前端）

- 冻结 eligible-IBP 的调用定义、计时边界和 `q`，同时披露 ineligible/fallback/unknown owner；
- 对目标公开模型做前端 op coverage 审计，列 unsupported op、shape/dtype 和最小实现范围；
- 至少预选一个 baseline/candidate 都能在同 timeout 内得到非 `unknown` 的公开 workload；
- 至少预选两个 held-out model family，并在看到 candidate 结果前冻结 hash、timeout 和排除理由；
- 本阶段只做 admission 与缺口清单，不因为缺 op 就先写 parser/backend，避免 benchmark 选择被实现
  结果反向污染。

### R2：CIBC-G2 全图编译优化（由 R1 数据选支路）

#### R2-A Linear horizontal fusion 与调优

若 2 个 Linear 的 overlap-adjusted share 足以影响目标：

- lower/upper 四 matmul → center/deviation fused TE/TIR；
- 建立独立的 block-M/N/K、warps、shared/local cache、L2 ordering search space；
- 与 PyTorch/cuBLAS、Triton oracle、manual TIR 三方比；
- 禁止把小 shape 上的 launch-bound 数字外推到大 shape。

#### R2-B Conv 深度 schedule search

若 Conv 仍占主要 critical path：

- 以 MetaSchedule/自定义 design space 扩展多层 tiling、cooperative fetch、shared/local cache、
  vectorize、unroll、software pipeline；
- 每个 production shape 建 workload key 与 tuning database；
- 比较 direct center/deviation、two-conv center/deviation、cuDNN grouped/batched 变体；
- compile/tune budget、search trials 与 winner receipt 全部冻结。

#### R2-C 图级 fusion

若 elementwise/launch 是主要残差：

- 优先 `Conv/Linear -> ReLU`，其次 residual `add -> ReLU`；
- 用 BoundFlow 高层 IR 表达多个 bound outputs 的共同 producer，再下沉 FuseTIR；
- residual diamond 使用 post-dominator/critical-path 规则，避免错误跨依赖融合；
- fusion 后必须证明中间 allocation 和真实 CUDA node 数下降。

#### R2-D Runtime/copy

若 input copy 或 launch/runtime 占比足够：

- 允许上游把输入写入 graph-owned static buffers，比较 copy-included/copy-elided 两个合法模式；
- PlanTemplate cache 按 shape/dtype/device/policy/code hash 锁定，记录 cold/warm 与 break-even；
- 多流只在 R1 显示 residual branch 有并发 headroom 时试验，并核对 overlap 而非仅 kernel 数。

### R3：α-CROWN 恢复路线（独立于 CIBC-IBP，不立即实现）

本节的初版方向已由独立详细预注册
`BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md` 取代。新方案不是
B4-C2 v2，而是 closed lower region 的 first-class DAG owner 与 region-level single custom VJP：

- Python/IR/autograd 边界不传 dense A；
- region forward 只产出最终 `[batch,spec]` lower；
- dense A 只能在 kernel 内或最多两个 plan-owned ping-pong scratch 中短暂存在；
- backward 默认从 compressed α/β、bounds、weights 与静态 plan 重算，不保存逐层 A；
- `ctx.executor`、逐层 custom Function、native+candidate 双算和 implicit `to_dense()` 全部禁止；
- 单 P-anchor、active-beta S-anchor、双 site、residual DAG、六 site逐级门禁后，才可能重开 B4-D。

该详细设计当前只是 `PREREGISTERED-DESIGN-REVIEW-ONLY`，不开放实现或性能 claim；配套外审 Prompt
为 `BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`。当前 executable next 是
R0 审计卫生与 R1 协议/目标冻结，随后才执行 CIBC-G1 attribution；不因 R3 文档完成而提前开放
R3-0。

### R4：JIT、调度、内存与多分支运行时

只有 R2 或 R3 出现累计 no-regression candidate 后才进入：

- JIT：AOT/cache-first，shape/signature specialization；只有满足
  `expected_reuse * expected_per_query_saving > compile_cost + cache_load + invalidation_cost` 才允许
  后台编译 fallback，不能用任意的“复用次数倍数”替代成本账；
- runtime：critical-path aware launch、CUDA Graph update、真正可并发的 residual branches；
- allocation：plan-owned arena、liveness reuse、避免 CUDA Graph private pool 被重复实例化；
- batching：query/domain/spec 三个轴分开，不能用一种 batch 承担所有语义；
- 当前证据只支持 **shape/signature-keyed static schedule specialization**：CIBC raw 的 per-op winner
  为 ordinal `0→256`、`2→128`、`4/5/8/10→64`，而 6-op formal global winner 是 `128`。这证明不同
  production shape 需要不同合法 schedule，但尚未证明 cache state、memory pressure 或动态负载下的
  adaptive planner；后者必须另做 context-changing raw 才能形成 GPU-context selection claim；
- 不复活已失败的通用 global planner 性能 claim。

### R5：系统闭环

- primary comparison：同一个 αβ-CROWN solver 内，original executor vs RVIR adapter + BoundFlow
  executor；branch、termination、timeout、输入与 GPU 相同；
- 把 B3、CIBC-IBP、恢复后的 CROWN、JIT/runtime/memory 作为 cumulative candidate，同时做
  leave-one-out attribution；
- 至少两个 held-out model family，且至少一个 workload baseline/candidate 都能得到非 unknown
  verdict；
- 报告 queue、complete query、TTV、solved、bound quality、peak allocated/reserved、compile/break-even；
- 只有通过第 2.2 节系统门槛才可形成 ASPLOS headline。

## 8. 建议的实际执行顺序

### 现在（一个短提交）

1. 完成 R0 的 3 条新增 mypy `arg-type`、1 条新增 pylint `C0415` 与
   steady-state/tolerance 披露；
2. 预注册 R1 artifact schema、三个 scope target、NVTX ordinal、时钟校准、control/profile 扰动与
   critical-path 口径；
3. R3 设计评审可并行继续，但 R3-0 实现保持关闭；不改 TIR schedule、不加第四个 threads 候选、
   不碰 α-CROWN production path。

### 紧接着（只读测量阶段）

4. 生成 5–6 fresh CIBC candidate-only attribution raw，先完成时钟/correlation admission；
5. 从 raw 冻结每个桶的 exclusive/critical-path share 和 cold/warm/break-even；
6. 在 same-solver 路径实测 eligible-IBP query share `q`，同时只读冻结前端 op coverage、两个
   held-out family 与至少一个可 solve workload；
7. 用同 scope 的 `r_required` 与 `q_required` 写出 R2-A/B/C/D 的明确 GO/NO-GO 排序。

### 归因之后

8. 只实现排名第一且数学上可达到 query qualification/research 目标的 R2 分支；
9. 单算子/子图通过后，必须回到完整 ResNet2B IBP CUDA Graph；
10. 以 B0/B3/cumulative candidate 三方 formal protocol 检验接入后 parity、query 与 memory；
11. R2 关闭后再决定是否开放 R3-0；若要提前转 R3，必须有显式 reprioritization 记录。无论何时
    实现，R3 都必须先完成 custom-backward/live-set 外部设计评审。

## 9. 预注册时必须回答的反证问题

- 当前 `2.456x` candidate 剩余时间中，哪一类节点占 critical path，而不是 kernel-sum 最大？
- 目标 `T` 对应的 `r_required` 是否低于该算子在同 shape 上的可信物理上限？
- schedule 搜索预算是否足以覆盖多层 tiling，而不是只改 thread count？
- fusion 是否真的减少 CUDA nodes/allocations，还是把工作搬进更慢的大 kernel？
- forward 保存了哪些 tensor，何时释放；是否存在 C2 式跨层 dense retention？
- control path 是否完全不执行 native shadow/reference？
- CUDA Graph、input copy、compile/plan 构造在两侧是否对称？
- 局部 bound 数值微差会不会改变 α/β 优化轨迹、branch 或 verdict？
- 若一个分支失败，kill condition 是否会阻止继续堆工程量？

## 10. 当前 claim 边界

现在可以说：

- 在 RTX 4060 Laptop/sm_89、ResNet2B property 0 的 BoundFlow IBP 路径上，6 Conv 横向融合
  相对 production 四-Conv baseline 的 operator geomean 是 `12.7951x`；
- 同一输入 copy 和 CUDA Graph 口径下，完整 IBP graph geomean 是 `2.45631x`；
- B4-B2 v2 manual TIR 局部 differentiable kernel 相对 PyTorch 是 `4.89834x`；
- B3 相对 B2 core 改善 `1.07162x`，但相对 B0 query 仍只有 `0.91000x`；
- 原 B4 纵向 dense-retention 集成以 NO-GO 关闭。

现在不能说：

- BoundFlow 已经比 auto_LiRPA/αβ-CROWN 快；
- 当前 CIBC 实现等同论文完整 auto-tuning/compiler；
- `12.8x` 是 whole-model、query 或 BaB speedup；
- B4-C2 证明所有 α-CROWN 融合都不可能；
- B5/B6/B7 或 complete-solve 门禁已经失败；
- 当前结果已经 ASPLOS-ready。

## 11. 外部资料

资料检索时间：2026-08-24；只使用项目内用户提供论文和官方/论文一手资料。

- 用户提供的 CIBC 论文：`docs/CIBC_for_DAC.pdf`（本机未跟踪文件，不由本文提交）；
- [Apache TVM MetaSchedule tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)；
- [Apache TVM TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)；
- [Apache TVM operator fusion](https://tvm.apache.org/docs/arch/fusion.html)；
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng)；
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)；
- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)；
- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-program-guide/02-basics/asynchronous-execution.html)；
- [NVIDIA CUPTI documentation](https://docs.nvidia.com/cupti/main/main.html)；
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)；
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)；
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://pytorch.org/assets/pytorch2-2.pdf)；
- [Beta-CROWN](https://papers.nips.cc/paper/2021/hash/fac7fead96dafceaf80c1daffeae82a4-Abstract.html)；
- [auto_LiRPA](https://proceedings.neurips.cc/paper/2020/file/0cbc5671ae26f67871cb914d81ef8fc1-Paper.pdf)。

## 12. 仓库证据入口

- CIBC 外审：`external_audit_cibc_ibp_horizontal_2026_08_24.md`；
- CIBC formal：`../artifacts/cibc-ibp-horizontal-formal/resnet2b-prop0-v1/summary.json`；
- FSG3 B0/B2 same-solver formal（B0/B2 query/core 来源）：
  `../artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5/summary.json`；
- NRIR49A selected-CROWN raw/summary：
  `../artifacts/nrir49a-g1-gpu-attribution/resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1/`；
- B3 formal：`../artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/summary.json`；
- B4-A formal：`../artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5/summary.json`；
- B4-B2 v1：`../artifacts/fsg4-b4b2-b2-5-formal-microphysics/resnet2b-prop0-v1/summary.json`；
- B4-B2 v2 TIR：`../artifacts/fsg4-b4b2-v2-cibc-tir-formal/resnet2b-prop0-v1/summary.json`；
- B4-C0 formal：`../artifacts/fsg4-b4c0-cumulative-core/resnet2b-prop0-v1/summary.json`；
- B4-C1 formal：`../artifacts/fsg4-b4c1-provider-owned-lower/resnet2b-prop0-v1/summary.json`；
- B4-C2 raw：
  `../artifacts/fsg4-b4c2-materialization-frontier-pilot/resnet2b-prop0-v1/run_00_BC.json`、
  `run_01_CB.json`、`run_02_BC.json`；closure：
  `BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`；
- R1 scope/clock/query-local attribution 预注册：
  `BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`；
- 外部评审 prompt：`BOUNDFLOW_FAILED_GATES_EXTERNAL_ADVISOR_PROMPT_2026_08_24.md`；
- 本文变更记录：`BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_CHANGELOG_2026_08_24.md`。

## 13. Rollback / Kill discipline

本文只改变路线与文档，不改变 production 默认路径。若 R1 归因无法达到 profile 扰动、closure、
correlation 或 critical-path 重算门禁，则删除该不合格 artifact，不据其选择优化分支；若 R2/R3
候选未过预注册 no-regression，则保留正确性/机制证据并关闭候选，不下调阈值、不挑样重跑、
不把局部数字累计成系统 claim。
