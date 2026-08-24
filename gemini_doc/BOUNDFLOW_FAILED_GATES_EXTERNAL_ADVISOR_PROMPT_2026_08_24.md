# BoundFlow 失败门禁与恢复路线外部评审 Prompt

请把下面整段连同仓库或相关文档交给另一个大模型。评审方应作为**怀疑型 GPU 编译器、神经网络
验证器和实验方法学审稿人**，不要把执行方摘要当事实。

---

你要独立审计 BoundFlow 的失败门禁诊断与下一阶段恢复计划。主文档是：

`gemini_doc/BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`

建议同时读取：

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

## 二、请重点质疑的判断

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
- profile/control perturbation、correlation、CUDA Graph node ownership 和时钟域该怎样 fail-closed？
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
- CIBC-IBP `2.456x` 加入 same-solver 后可能被什么比例稀释？需要哪项 raw 才能回答？
- memory path 在 8 GB 上如何构造“自然 workload”而不是人为放大 batch？

## 三、必须检查的工程卫生问题

- 外审指出 `boundflow/domains/interval.py:83-85` 有 3 条本轮新 mypy arg-type 错误和新增 pylint
  import-outside-toplevel；
- CIBC closure 应明确 `3e-4` tolerance 的 1 ULP 理由；
- steady-state operator/graph timing 排除了 compile/plan construction，需补 cold/break-even；
- `.docops/ev.jsonl` 有 3 个历史 duplicate ids，但不是本轮性能改动引入；
- 检查这些问题是否只是 minor，还是会污染性能/语义结论。

## 四、期望输出格式

请按以下结构输出，结论要可执行：

1. **总体 verdict**：approve / approve-with-changes / request-changes；
2. **门禁分类纠错表**：逐项列出你复算的数字、分类和任何不一致；
3. **根因审计**：哪些有代码证据，哪些只是推断，缺什么实验；
4. **遗漏机制**：最多 8 项，按 expected system value / risk / effort 排序；
5. **候选路线排名**：每条包含目标文件/IR 边界、预期物理变化、所需 share、`r_required`、
   correctness gate、memory gate、kill condition；
6. **R1 预注册修订稿**：给出你建议的 raw schema、fresh 顺序、扰动阈值、closure 方程、tamper cases；
7. **两周执行计划**：每天或按 5 个工程阶段列出，不要把未通过阶段后的工作提前开放；
8. **论文 claim 边界**：现在能写什么、不能写什么、还缺哪三份核心证据；
9. **blocker/major/minor/info findings**；
10. **唯一下一动作**：限定为一个可以直接开工的提交或实验。

审计时请优先使用 formal raw、代码和一手资料。凡是从证据推断而不是直接观测的结论，请明确标注
“inference”。不要因为局部 kernel 有 4.90x/12.8x 就默认系统会快，也不要因为 B4-C2 失败就默认
编译器全栈没有价值。

---
