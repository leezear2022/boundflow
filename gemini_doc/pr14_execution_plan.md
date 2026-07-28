# PR-14 执行计划：Verification-Aware Execution on Real Verification Workloads

> 状态：PR-14B 已完成并判定 **VALIDATED-NO-GO**；PR-14C 被门禁阻断
> 起点：`57a854b` / `pr13-validated-reduced`
> 分支：`feat/pr14-real-verification`
> 核心目标：量化并验证 BoundFlow 的 IR、Planner、backend 与 runtime 如何覆盖和执行真实
> complete-verification workload，而不只是执行 reduced bound-query benchmark。
> 当前证据与窄化判定：`gemini_doc/pr14a_real_query_coverage_2026_07_19.md`
> 最终 replay/closure 判定：`gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`

## 1. 研究问题

### RQ1：真实 verifier 产生什么 query？

统计真实搜索流中的 method/stage、spec/domain batch、split depth/lineage、operator shape、状态版本、
memory budget、reuse opportunity 与 arrival pattern。不得用 synthetic fixed batch 替代该分布。

第一版 coverage matrix 固定包含一个 MLP、一个 CNN 和一个 ResNet block，并逐 query 输出：

```text
query_id / solver_phase / bound_method / requires_grad
alpha_enabled / beta_enabled / has_split
spec_size / domain_size / layer_pattern
backend_eligible / eligibility_reason
```

### RQ2：现有 Planner/backend 覆盖多少真实 query？

对每个 query 记录 compatibility/capability 判定和候选集合，区分：

- plain CROWN，可评估 PR-12 eager/chunked/structured/fused 候选；
- α/αβ optimize 或 split query，当前必须走 `alpha_beta_dense_split`；
- unsupported graph/op/property，fail closed 并结构化记录原因。

PR-14 不预设覆盖率一定高。低覆盖率本身是论文定位与止损决策的证据。

### RQ3：相对公平 batched verifier 是否有系统价值？

固定 host solver、模型/property、branch/split、优化步数、seed、timeout 和数值策略，仅替换
bound-call execution。主 baseline 是 original batched executor；逐节点结果只用于机制诊断。

指标至少包含 time-to-verify、solved/timeout、peak memory、node/query throughput、p50/p90/p99、
batch fill、queue wait、backend selection、fallback/OOM 和 correctness mismatch。

## 2. 冻结边界

保持不变：

- branch heuristic、priority queue、node ordering；
- α/β 优化、split/cuts、termination 与 timeout；
- property 语义和数值策略；
- PR-11/12/13 已冻结的 artifact 与 held-out split。

允许替换：

- bound evaluation adapter；
- query packing/scheduling；
- capability-safe Planner/backend dispatch；
- cache/reuse，但必须遵守 `EXACT_REUSE`、`CONDITIONAL_REUSE`、`WARM_START_ONLY`、
  `INVALIDATE` 四级规则。

## 3. 执行切片

### PR-14A：真实 verifier/workload adapter

目标是接线，不做优化。

1. 审计现有 ONNX frontend、property/spec 表达和支持算子；
2. 冻结一个 MLP、一个 CNN、一个 ResNet block 的真实模型/property 及其来源/hash；
3. 建立 ONNX/VNNLIB（或等价真实 verifier 输入）到现有 `BFTaskModule`、`InputSpec`、
   `BoundQuery` 的适配边界；
4. 复用 `FixedBabQueryRecorder`，不得再造第二套 query schema；
5. 记录 unsupported/fallback，不为了跑通而静默改写模型或 property；
6. 新增稳定的 `VerificationQueryProfile`，从现有 `BoundQuery` 派生 coverage 字段，不复制状态；
7. 对 original solver 与 adapter-off 路径做语义基线冻结。

PR-14A 完成门禁：

- MLP/CNN/ResNet-block 三类真实模型/property 可确定性导入，或对不支持项给出结构化 fail-closed
  记录；
- query ID、parent/split lineage、输入/spec 版本稳定；
- recorder on/off 不改变 solver status、node count 和 bounds；
- 0 query loss/duplicate/invalid；
- 生成真实 query trace、schema validation 与 manifest；
- 输出 method/stage 和 backend eligibility coverage 表；
- 不产生性能 claim。

### PR-14B：固定流 replay 与 backend eligibility

1. 固定 PR-14A 的真实 query stream 和 order；
2. 对 original batched、BoundFlow dense/runtime、合法的 PR-12 backend candidate 做同 query replay；
3. 报告每类 query 的 candidate coverage、拒绝原因、fallback、compile/cache applicability；
4. 对所有合法路径做 bounds、branch choice、α/β state 与 query accounting 对齐；
5. 分开 cold、warm、compile/load/cache 与 steady-state，不隐藏 N/A、OOM 或 timeout。

PR-14B 完成门禁：

- replay deterministic，所有 manifest/hash 闭合；
- 0 correctness failure、0 query loss；
- eligibility/coverage 表完整；
- original batched 为正式 baseline；
- 输出是否值得进入 full E2E 的 Go/No-Go，不预设 PR-12 fused 一定进入 αβ/split 查询。

实际结果：payload/replay/manifest 已闭合，但 VNN-COMP ResNet 的 whole-query lower bound 未保持
external computation（max diff `796.765`，符号 3/9），且唯一等价 MLP 存在 external
lower-only 与 BoundFlow lower+upper 的 requested-output 不一致。结论为 **NO-GO**。

### PR-14C：完整 verification evaluation

仅在 PR-14B Go 后启动：

- workload 梯度至少包含 CIFAR CNN、multi-block ResNet 和 VNN-COMP 代表实例中的可支持子集；
- 同一 timeout 下报告 solved/unsafe/unknown/timeout，而不是只报告成功样本；
- 至少 5 次独立重复，报告 median 与 tail；
- 记录 peak memory、真实 GPU OOM、batch fill、queue/compute/branch/prune 分解；
- 表图只能从原始 JSONL 生成，失败记录不得删除。

当前状态：**BLOCKED BY PR-14B GATE；不启动。**

## 4. 预期代码与工件边界

优先复用：

- `boundflow/frontends/onnx/`；
- `boundflow/runtime/bab_query.py`；
- `boundflow/runtime/bab_query_runtime.py`；
- `boundflow/runtime/query_batcher.py`；
- `boundflow/runtime/query_executor.py`；
- `boundflow/planner/fused_crown_backend.py`；
- `boundflow/planner/execution_candidate.py`。

新增代码应收敛在真实输入/property adapter、benchmark runner 和 contract tests，不重写 solver。

建议工件目录：

```text
artifacts/phase7a-pr14/
  pr14a-real-query-trace-<run-id>/
  pr14b-real-query-replay-<run-id>/
  pr14c-complete-verification-<run-id>/
```

每组工件必须包含 command/config、raw JSONL、normalized CSV、summary、manifest、环境与 commit/
submodule hash，以及所有 fail/OOM/timeout 记录。

## 5. 公平性与正确性门禁

- same solver、same property、same branch/split、same seed、same timeout；
- adapter-off 与 original path 必须保持独立可运行；
- capability 不兼容时 fail closed；
- 父 α/intermediate 只能 warm-start，父 β/final bounds 不得 exact reuse；
- 没有 outward rounding/proof checker 时，只声称保持相同浮点语义下的 reference computation；
- 不以逐节点 baseline 作为最终 headline；
- 不把普通 batching 收益包装成 query-runtime 独立创新。

## 6. 止损与论文定位

PR-14B 后必须做一次显式决策：

- 若真实 query 中 Planner/backend 覆盖充分，且相对 original batched 有稳定、可归因的 latency、
  memory 或 solved-count 收益，则 C3 可表述为 **Verification Query Runtime and Heterogeneous
  Execution**；
- 若覆盖率低或 runtime 相对 batched original 无稳定净收益，则 C3 降级为支撑 C1/C2 的执行
  基础设施，论文主线集中到 structured representation 与 multi-backend Planner；
- 不为保留三项贡献而新增验证算法、TIR family 或不公平 baseline。

PR-14B 已触发第二条：C3 正式降级为支撑 C1/C2 的执行基础设施。

PR-14 的最低成功标准是：BoundFlow 能正确执行并审计真实 verifier query；较强标准是至少一个
真实 phase/regime 获得可归因收益；最强标准才是 same-solver、same-property、same-timeout 下改善
time-to-verify、peak memory 或 solved instances。

## 7. Closure 后动作（历史计划）

> **2026-07-20 路线修订**：PR-14 停止结论继续有效，但下述 story-freeze 分支已被
> IR-first 代码复审取代，不再是当前执行指令。现行分支为 `feat/compiler-ir-stack-v1`，
> 详见 `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。

PR-14 不再新增实现切片。原计划是在 closure 后从最终 commit/tag 建立
`docs/asplos-c1-c2-story-freeze`，更新摘要、前两页、claims 与 artifact 阅读顺序，并以 C1+C2
重新做 paper-level Go/No-Go。未来若研究 external intermediate-bound-preserving region adapter，
必须作为新假设、新 split 和新门禁，不能继续挂在本 PR-14 上。
