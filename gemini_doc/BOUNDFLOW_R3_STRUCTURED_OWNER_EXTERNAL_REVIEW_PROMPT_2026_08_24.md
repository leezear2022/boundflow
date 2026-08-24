# BoundFlow R3 结构化 lower owner / custom VJP 外部评审 Prompt

请把下面整段直接交给另一个大模型。它是 standalone Prompt；如果模型能访问 GitHub，应优先读取
链接中的代码和文档。如果不能访问，请把主设计文档一并上传。

---

你现在是一位怀疑型的 GPU 编译器、PyTorch autograd、神经网络验证和实验方法学评审人。请独立
审计 BoundFlow 的 R3-SO-CVJP 重设计。不要把执行方摘要当事实；代码事实要从 GitHub 验证，数学
结论要自己推导，性能可达性要用 liveness/FLOPs/launch/Amdahl 模型说明。

## 1. GitHub 与评审入口

- 仓库：<https://github.com/leezear2022/boundflow>
- Draft PR #60：<https://github.com/leezear2022/boundflow/pull/60>
- 当前工作分支：<https://github.com/leezear2022/boundflow/tree/feat/rvir-v4-production-state-ownership-v1>
- 设计依据代码基线：<https://github.com/leezear2022/boundflow/commit/f87f737cebffaf10827957682e3196063e4c78ed>
- 主设计文档：<https://github.com/leezear2022/boundflow/blob/feat/rvir-v4-production-state-ownership-v1/gemini_doc/BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md>
- 失败门禁总诊断：<https://github.com/leezear2022/boundflow/blob/feat/rvir-v4-production-state-ownership-v1/gemini_doc/BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md>

关键代码：

- ReLU structured/C2 dense 分叉：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/crown_ibp.py#L1401-L1538>
- production reverse traversal：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/crown_ibp.py#L1945-L2293>
- native structured operators：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/linear_operator.py#L722-L1092>
- B4-C2 frontier owner：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/fsg4_b4c2_materialization_frontier.py#L1-L110>
- 当前 TIR executor/autograd：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/fsg4_b4b3_cibc_dense_tir.py#L20-L283>
- production α/β capture schema：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/fsg4_b4b_production_region_capture.py#L20-L330>

建议同时阅读仓库中的：

- `gemini_doc/BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C1_PROVIDER_OWNED_LOWER_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C0_CUMULATIVE_CORE_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4B2_V2_MANUAL_TIR_FORMAL_CLOSURE_2026_08_24.md`
- `artifacts/fsg4-b4c2-materialization-frontier-pilot/resnet2b-prop0-v1/`

如需理解 CIBC 背景，我会另行上传 `docs/CIBC_for_DAC.pdf`；它目前不是本 R3 设计正确性的依据。

## 2. 基础问题

BoundFlow 要把 α-CROWN 的 lower-bound coefficient propagation 编译到 CUDA/TIR。局部 P-anchor
differentiable TIR 对 PyTorch 已达到 `4.89834x`，但 production 累计接入失败：

- B4-C0 native-value bridge：core `0.94034x`；
- B4-C1 provider-owned lower：core `0.94815x`；
- B4-C2 六个 lower materialization frontier：`0.337—0.349x`，peak allocated `1.3401x`；
- 六个 frontier 是 `31/28/25/23/19/17`，每次 optimizer 10 次，共 60 次 materialization；
- correctness 仍通过，所以问题主要是表示与生命周期，而不是明显数学错误。

请从代码验证以下诊断是否成立：

1. native path 用 `SignSplitLinearOperator`/`Conv2dLinearOperator` 等延迟结构化组合；
2. C2 的 `dense_lower_once=True` 在每个 ReLU 把 `A_l` 物化、乘 slope 并包装为
   `DenseLinearOperator`，使 dense A 和 autograd history 跨层存活；
3. 当前 custom Function 又把含所有输入 Tensor 和工作区的 executor 挂到 `ctx.executor`，进一步
   延长 lifetime；
4. 因此继续调 block/thread 或扩大 C2 coverage 不会修好结构边界。

## 3. 提议的方法

新设计不是 B4-C2 v2，而是 **region-level structured owner + one custom VJP**：

1. 用不可变 `StructuredLowerRegionTemplateV1` 表示 SpecSeed、ReluLowerTransform、LinearRight、
   Conv2dRight、Add、Reshape、Slice、BiasSplit、InputConcretize 的 DAG；
2. `StructuredLowerRegionInstanceV1` 每个 optimizer evaluation 绑定 compressed α/β、bounds、
   weights、split/history、input/spec 和 plan-owned scratch，静态 Template 可缓存，动态 Instance 不可
   跨 mutation 复用；
3. coefficient value 在 Python/IR 边界只用非 Tensor handle；α/β、bounds、weights 等 compact/native
   leaves 仍作为明确的 Function Tensor inputs；production `to_dense()` fail closed；
4. closed lower region 只向 autograd 返回最终 `[batch,spec]` lower，upper 暂留 native；
5. 不允许逐层 Function；一个 region 只有一个 custom Function；
6. custom Function 的 ctx 只存无 Tensor 的 immutable plan key；所有必要 Tensor 只用
   `save_for_backward`，禁止 `ctx.executor` 或间接 Tensor closure；
7. 默认 backward 使用 `M0-rematerialize`：从 compressed α/β、bounds、weights 和 plan 重算 A/sign，
   用最多两个 ping-pong scratch 或 tile-local storage，返回全部 α/β gradients；
8. dense A 可在 kernel 内短暂存在，但不能作为 Function output、saved tensor、ctx/executor state 或
   六层 persistent buffer；
9. correctness worker 可在独立进程运行 native shadow，正式 control path 必须 single-owner；
10. bit-packed sign certificate 是尚未开放的 M1 variant；任何 dense checkpoint 在 v1 禁止。

## 4. 请做独立数学审计

请不要只看 API 形式，至少回答：

1. 对 lower ReLU，`A>=0` 选择 α_l、`A<0` 选择 α_u 的系数变换，以及 intercept/β bias reduction，
   region DAG 应怎样写成精确 recurrence？A=0 的端点所有权是什么？
2. backward 对每个 compressed α leaf 的 VJP 如何从 output adjoint、incoming A、bounds 和 index/lookup
   得到？是否需要保存或重算 sign？
3. active β 的 location/sign/pre-add 对 VJP 有何影响？empty beta 是否应返回 `None` 而不是零 tensor？
4. Conv2dRight 的 conv-transpose 索引、groups/stride/pad/dilation 和 weight layout 是否可由当前 IR
   无歧义表达？
5. residual Add/fanout 下怎样避免复制 coefficient 子树和重复 bias？`BiasSplit` 是否足够？
6. 一个 region 只输出 final lower 时，是否还有 solver 必需的中间 lower-A consumer 被设计遗漏？请从
   `crown_ibp.py` 的 diagnostic、branching/objective-influence 和 optimizer 路径区分 production 必需项。

请给出关键 recurrence/VJP 公式或伪代码，并标出任何无法从现有代码确定的假设。

## 5. 请做独立 autograd/liveness 审计

重点质疑：

- region-level single Function 是否真的消除 dense A 生命周期，还是把它藏进 registry/executor？
- weights、bounds、α/β 应哪些通过 `save_for_backward`，哪些可以只保存 identity；PyTorch version
  counter 和 in-place mutation 如何保证？
- `save_for_backward` 的 logical bytes 与 unique storage bytes 应怎样分别计算？
- M0 重算的时间复杂度、kernel 数和最小 scratch 是多少？两个 ping-pong buffer是否足够？
- 是否需要每层 sign mask；若需要，能否 bitpack，是否会改变 A=0 语义？
- CUDA memory snapshot 只看 PyTorch allocator，TVM/DLPack/scratch 怎样证明没有不可见 allocation？
- 一个 Python object 的 recursive tensor-reachability checker 能否可靠拒绝 `ctx.executor` 式间接引用？
- CUDA Graph capture 时 plan buffer、DLPack view 和 autograd context 生命周期怎样处理？

请给一个按 tensor 类别列出的 live-set 表：shape、dtype、是否已有 baseline owner、是否 saved、何时释放、
是否允许跨层，以及估算的 max-live bytes 公式。不要只说“用 checkpointing”。

## 6. 请审查阶段和门槛

提案阶段：

- R3-0 contract/validator only；
- R3-1 P-anchor single-site correctness：optimizer state 固定、无 `optimizer.step()`，但一次 custom
  backward 与 `dα` 对照是强制门禁；纯 no-grad 只能 smoke；
- R3-2A P-anchor 10/9 mutation trajectory correctness：逐 evaluation lower/gradient/mutation、final
  α/β、split/history 和 mutation order 通过前不计时；
- R3-2B 只在 R3-2A 通过后比较同 P-anchor、同 10/9 轨迹的 native single-owner wrapper，
  wrapper-inclusive geomean `>=1.20x`、worst `>=0.98x`、memory `<=1.0x`；
- R3-3 S-anchor active-beta correctness；
- R3-4 topology 选出的 closed two-site region，geomean `>=1.0x`、worst `>=0.98x`；
- R3-5 最小 residual/fanout DAG；
- R3-6 六 frontier，core geomean `>=1.05x`、worst `>=0.98x`、memory `<=1.0x`；
- 只有 R3-6 GO 才开放 R3-7 same-solver B4-D。

统一 kill：任何 dense A 进入 output/saved/ctx/persistent layer buffer、任何 shadow/fallback/to_dense、
语义或 optimizer mutation 漂移、node count 超线性、R3-2B single-site `<1.20x`、two-site
`<1.0x`、memory `>1.0x`，都停止当前 variant。

请判断：

1. 哪些 gate 太松、太严或无法可靠测量？
2. R3-1 “冻结 optimizer mutation、但 mandatory backward”是否是隔离 VJP/liveness 的正确最小实验？
   no-grad 是否应明确禁止关闭该阶段？
3. 把原 R3-2 拆成 2A trajectory correctness 与 2B wrapper timing 是否足以防止数值轨迹和性能
   归因互相污染？`1.20x` 是否合理，还是局部 share/重算成本使其数学上不可达？
4. R3-4 应如何从 topology/post-dominator 机械选择 pair？
5. R3-6 的 `1.05x` 足以开放 query 实验吗，还是必须先恢复 B0 parity？
6. 哪个最短实验能尽早证伪“region-level custom VJP 可行”？

## 7. 请比较替代方案

允许提出替代设计，但必须保持硬约束：不跨层保存 dense A、不回到 native+candidate 双算、不用局部
microbenchmark 替代累计证据。请具体比较：

- region-level custom Function vs AOTAutograd/min-cut；
- full rematerialization vs bit-packed sign certificate vs bounded checkpoint；
- multi-kernel bounded arena vs monolithic TIR kernel；
- 扩展现有 `LinearOperator` vs 新 first-class StructuredLowerRegion IR；
- TVM TIR vs Triton/torch.compile 作为 reference 或 production backend。

每个替代方案列 expected benefit、correctness risk、memory formula、工程量和 kill condition。不要给“试试
更多融合/共享内存/多流”这种没有代码边界的泛泛建议。

## 8. 期望输出格式

请严格按以下结构输出：

1. **总体 verdict**：approve / approve-with-changes / request-changes；
2. **代码事实核对表**：每项附 GitHub 文件和行号；
3. **数学正确性审计**：recurrence、VJP、A=0、active/empty beta；
4. **所有权与 live-set 表**：每类 tensor 的保存/释放/bytes；
5. **设计漏洞**：blocker/major/minor/info，区分事实与 inference；
6. **修订后的最小 API/IR**：给 dataclass/protocol 级伪代码；
7. **M0 backward 算法**：复杂度、scratch、launch 和 rematerialization 计划；
8. **阶段门禁修订表**：保留/修改/删除及理由；
9. **最短证伪实验**：一个提交可完成，列 raw 字段与预期结果；
10. **两周条件式计划**：上游不过不得提前开放下游；
11. **claim 边界**：现在、R3-1、R3-2A、R3-2B、R3-6、R3-7 分别能说什么；
12. **唯一下一动作**。

凡是没有 raw/代码直接支持的结论必须标记 `inference`。不要因为局部 TIR 有 `4.898x` 就默认 region
会快，也不要因为 B4-C2 有 `0.34x` 就默认所有 structured/custom-VJP 路线都失败。

优先参考一手资料：

- <https://docs.pytorch.org/docs/stable/notes/extending.html>
- <https://docs.pytorch.org/docs/stable/notes/autograd.html>
- <https://docs.pytorch.org/docs/stable/autograd.html>
- <https://docs.pytorch.org/docs/stable/torch_cuda_memory.html>
- <https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html>

---
