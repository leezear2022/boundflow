# BoundFlow 失败门禁与恢复路线外部评审 Prompt

> **2026-08-25 评审对象更新**：R0已完成，R1现有独立预注册
> `gemini_doc/BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`。请把它作为
> 当前主要审计对象；不要继续把“缺目标T/缺时钟校准”当未处理finding。重点检查按op type的
> `q_B3,k`、exact production `G_query,k`、独立graph `2.45631x`禁用规则和16类tamper是否足以防止
> same-solver分母失真。

请把下面整段连同仓库或相关文档交给另一个大模型。评审方应作为**怀疑型 GPU 编译器、神经网络
验证器和实验方法学审稿人**，不要把执行方摘要当事实。

---

你要独立审计 BoundFlow 的失败门禁诊断与下一阶段恢复计划。主文档是：

`gemini_doc/BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`

GitHub 入口：

- 仓库：<https://github.com/leezear2022/boundflow>
- Draft PR #60：<https://github.com/leezear2022/boundflow/pull/60>
- 当前分支：<https://github.com/leezear2022/boundflow/tree/feat/rvir-v4-production-state-ownership-v1>

建议同时读取：

- `gemini_doc/BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`
- `gemini_doc/BOUNDFLOW_R0_HYGIENE_R1_PREREGISTRATION_CHANGELOG_2026_08_25.md`
- `gemini_doc/external_audit_cibc_ibp_horizontal_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4B2_V2_CIBC_PARITY_FUSION_PLAN_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C0_CUMULATIVE_CORE_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C1_PROVIDER_OWNED_LOWER_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`
- `artifacts/*/summary.json` 中与 NRIR49A、FSG3/B3、B4-A/B/C、CIBC 有关的 formal raw/replay
- 用户提供的 `docs/CIBC_for_DAC.pdf`（若附件可用）

## 一、已知数字（必须从 raw 独立复算，不要直接采信）

1. selected-CROWN queue share 中位 `0.0709863183`，complete share `0.0705232890`，单区域
   deletion-only 上限约 `1.076410x`；
2. B3 相对 B2 core/query `1.071617x/1.006623x`，但 B0/B3 query `0.910001x`；
3. B4-A core `1.018995x`，未过 `1.03x`，query worst `0.996947x`；
4. B4-B2 v1 geomean/lower/worst `0.424842x/0.403157x/0.377693x`，真实为 3 forward + 3 backward
   kernels；
5. v2 Triton 对 PyTorch `2.83772x`；manual TIR 对 PyTorch `4.89834x`、对 Triton `1.68273x`；
6. B4-C0 core `0.940341x`，B4-C1 `0.948150x`；
7. B4-C2 三轮 `0.348761/0.337448/0.346003x`，peak allocated `1.3401085x`；
8. CIBC-IBP 6 Conv operator geomean/worst `12.795108x/9.142291x`，完整 ResNet2B IBP graph
   geomean/worst `2.456310x/2.450908x`，输入 copy 计入，两侧均 CUDA Graph；
9. 当前 CIBC Conv schedule 只有 threads `64/128/256`，主要是 one-thread-per-output + serial
   reduction，没有完成论文级多层 tiling/shared cache/vectorize/unroll/cost-model search；
10. B5 JIT、B6 runtime、B7 memory、complete query/solve 尚未运行，不能当成已经失败。

## 二、修订后冻结的协议（请审计，而不是把它当未解决建议）

1. 三个系统 scope target 已冻结：complete-query qualification=`1.00x`、complete-query research=
   `1.15x`、queue/BaB research=`1.20x`，baseline 都是 B0；局部 whole-graph 实验另用同 scope 的
   `T_graph`；
2. `r_required = s / (1/T - (1-s))` 只能使用同一 timing scope 的 `s/T`；分母 `<=0` 即该单区域
   物理不可达；
3. 把 CIBC whole-graph 收益传播到 query 前，必须先测 same-solver 两侧 eligible-IBP share。传播
   方程只使用待优化 B3/candidate 侧的 `q_B3`。当前 `R_current=0.910001`、`G=2.45631` 的乐观
   上界为 `R_new=R_current/((1-q_B3)+q_B3/G)`；parity/research 分别要求
   `q_B3>=0.151798/0.351998`；
4. CUPTI GPU timestamp 与 host/NVTX 必须有同步点和 Nsight export calibration receipt；单 stream、
   无真实 overlap 时 headline 使用 exclusive/critical-path wall，overlap-adjusted 必须退化一致；
5. R1 之后先做 same-solver share admission、前端 op coverage 清单和可 solve/held-out workload 冻结，
   数学可达才实现 R2；随后跑 B0/B3/cumulative candidate 三方 formal；
6. R3 设计评审可并行，但 R3-0 实现保持关闭，除非 R2 关闭或有显式 reprioritization。

## 三、请重点质疑的判断

### Q1：门禁分类是否正确

- 哪些是正式 NO-GO，哪些只是 reduced，哪些未运行？
- 有没有把“局部失败”错误外推成“路线失败”，或把“局部通过”错误外推成“系统通过”？
- CIBC claim 写成 reduced 是否仍然过强或过弱？

### Q2：根因是否抓对

- B4-B2 v1 的失败是否主要来自 6 kernel、workspace 和缺少 schedule optimization？
- v2 局部 4.90x 到 C0/C1 的 0.94–0.95x，是否能由 native shadow、provider wrapper、early dense
  materialization 解释？请从代码热路径验证，不要只接受文档叙述。
- C2 的 0.34x/1.34x memory 是否真由跨层 dense autograd live set 造成？还有没有 DLPack、stream、
  allocator、synchronization 或 receipt overhead 被漏算？

### Q3：下一步先做 attribution 是否合理

- 在当前 CIBC candidate 上做 candidate-only NVTX/CUPTI/CUDA Graph node attribution，能否可靠分解
  Conv、Linear、ReLU、add、copy 与 runtime？
- 应使用 exclusive wall、kernel sum、critical path 还是 overlap-adjusted share 作为路由依据？
- 修订后的同步点/calibration receipt、profile/control perturbation、correlation 与 CUDA Graph node
  ownership 是否足以让时钟域 fail closed？还缺哪项可机器检查字段？
- 是否有更短、更能证伪的第一个实验？

### Q4：CIBC 优化空间是否被低估

- 对当前 production shapes，MetaSchedule/Ansor 式多级 tiling、cooperative fetch、shared/local cache、
  compute-at/inline、vectorization、unroll、software pipeline 分别可能解决什么瓶颈？
- Linear horizontal fusion、Conv/Linear→ReLU、residual add→ReLU、two-conv center/deviation、cuDNN/
  Triton/manual TIR 三方比较，优先级应该怎样排？
- 请不要给“用 shared memory/多流/更多融合”这种泛泛建议；每条建议必须映射到具体文件、shape、
  已失败门禁和预期减少的 kernel/allocation/critical-path time。

### Q5：α-CROWN 应否恢复以及怎样恢复

- `structured owner + custom backward + recomputation/minimal saved state` 是否能避开 C2？
- forward/backward 各应保存什么，哪些 dense tensor 必须禁止跨层存活？
- 应如何用 `saved_tensors_hooks`、allocator snapshot、NVTX lifetime 与 receipt 证明 live set？
- 单 site→双 site→六 site 的门槛是否足够严格？如果这条路线数学上/工程上不值得做，请给出
  可证伪的 kill condition。

### Q6：系统目标是否可达

- 从 B3 的 B0-relative query `0.910001x` 到 parity 需要约 `1.09890x`，到 final `1.15x` 需要约
  `1.26373x`。请用 Amdahl/critical-path 模型计算各建议路线需要的区域 speedup；
- 哪些组合在 RTX 4060 Laptop 上物理可达，哪些在测量前就应 NO-GO？
- 请独立推导 `q_B3_required(1.00)=0.151798`、`q_B3_required(1.15)=0.351998`；公式、scope 或舍入是否有误？
- same-solver `q_B3,k` 应包含/排除哪些 adapter、receipt、copy、fallback 成本，才能避免把 graph
  share高估？exact production `G_query,k` 的wrapper边界是否公平？
- 若真实query的op构成/shape/state与独立IBP图不同，R1把未知`G_query,k`置为1、只保留
  `G_independent=2.45631`作历史敏感性的处理是否足够保守？
- memory path 在 8 GB 上如何构造“自然 workload”而不是人为放大 batch？

### Q7：执行顺序与替代路线是否正确

- R0/R1 protocol freeze → G1 attribution → same-solver share/workload admission → mathematically
  reachable R2 → B0/B3/candidate formal 的顺序是否存在越序或遗漏？
- 当前 CIBC per-op schedule winner 不同、global winner=128，只能支持 shape/signature-keyed static
  specialization；需要什么 context-changing raw 才能升级为 cache/memory-aware adaptive planner？
- JIT 的 admission 改为
  `expected_reuse*expected_per_query_saving > compile_cost+cache_load+invalidation_cost` 是否完整？
- receipt 热路径目前只是归因假设；请从代码指出哪些检查可安全移到 admission，哪些必须每次执行。

## 四、必须检查的工程卫生问题

- 外审指出 `boundflow/domains/interval.py:83-85` 有 3 条本轮新增 mypy `arg-type`，line 74 有 1 条
  新增 pylint `C0415`；既有 8 条 `DomainState attr-defined` 不属于本次修复；
- CIBC closure 应明确 `3e-4` tolerance 的 1 ULP 理由；
- steady-state operator/graph timing 排除了 compile/plan construction，需补 cold/break-even；
- `.docops/ev.jsonl` 有 3 个历史 duplicate ids，但不是本轮性能改动引入；
- 检查这些问题是否只是 minor，还是会污染性能/语义结论。

## 五、期望输出格式

请按以下结构输出，结论要可执行：

1. **总体 verdict**：approve / approve-with-changes / request-changes；
2. **门禁分类纠错表**：逐项列出你复算的数字、分类和任何不一致；
3. **根因审计**：哪些有代码证据，哪些只是推断，缺什么实验；
4. **遗漏机制**：最多 8 项，按 expected system value / risk / effort 排序；
5. **候选路线排名**：每条包含目标文件/IR 边界、预期物理变化、所需 share、`r_required`、
   correctness gate、memory gate、kill condition；
6. **R1 预注册修订稿**：审计 scope target、same-solver `q`、时钟校准 raw schema、fresh 顺序、扰动
   阈值、closure 方程与 tamper cases；
7. **两周执行计划**：每天或按 5 个工程阶段列出，不要把未通过阶段后的工作提前开放；
8. **论文 claim 边界**：现在能写什么、不能写什么、还缺哪三份核心证据；
9. **blocker/major/minor/info findings**；
10. **唯一下一动作**：限定为一个可以直接开工的提交或实验。

审计时请优先使用 formal raw、代码和一手资料。凡是从证据推断而不是直接观测的结论，请明确标注
“inference”。不要因为局部 kernel 有 4.90x/12.8x 就默认系统会快，也不要因为 B4-C2 失败就默认
编译器全栈没有价值。

---
