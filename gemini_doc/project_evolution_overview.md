# BoundFlow 研发脉络总览

这篇文档回答五个问题：BoundFlow 想解决什么、这条线如何从 Phase 0 推到当前的 Phase 7A、各阶段主要改了哪些代码模块、现有记录分别放在哪里、以及下一步自然应该做什么。

它的定位是“入口层 + 导航层 + 演化视角层”：不替代 [gemini_doc/phase0_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase0_summary.md) 到 [gemini_doc/phase6_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase6_summary.md)，而是把这些阶段文档与总账、全流程文档串成一条可接手的主线。

---

## 1. 为什么会有 BoundFlow

BoundFlow 的目标，从一开始就不是“再写一个验证算法实现”，而是把神经网络验证工作负载当成编译器与运行时系统问题来做。这个定位在 [gemini_doc/plan.md](/home/lee/Codes/boundflow/gemini_doc/plan.md) 里写得很明确：

- 前端负责把模型和验证语义导入成统一 IR。
- 中间层负责把边界传播任务做成可规划、可复用、可批处理的执行图。
- 后端和运行时负责把这些任务稳定落到 TVM/GPU 与实验工件链。

因此，BoundFlow 的学术叙事是 ASPLOS 风格的 system contribution：强调验证感知 IR、global planner、runtime、codegen、artifact pipeline，而不是发明新的验证算法家族。Phase 6 引入 CROWN、alpha-beta-CROWN、BaB，也是在这个系统框架里落地成熟方法族，而不是重新定义它们。

---

## 2. 主线一句话概括

BoundFlow 的主线可以概括为：先把 Torch/ONNX 模型导入成统一的 Primal IR，再把验证执行抽象成 Task/Planner/Runtime，接着用 TVM 和 bench/artifact 产线把系统证据链冻结，最后把 CROWN、alpha、alpha-beta、BaB 与 E2E time-to-verify 接入这条系统主线，形成可复现、可归因、可用于论文/AE 的完整闭环。

按当前仓库状态看，可以把项目分成两层理解：

- Phase 0-5 已经在 [docs/change_log.md](/home/lee/Codes/boundflow/docs/change_log.md)、[docs/phase5_done.md](/home/lee/Codes/boundflow/docs/phase5_done.md) 与各阶段总结中形成较稳定的里程碑与证据链。
- Phase 6 已形成稳定方法族与 E2E 工件链；当前工作区与文档的真实前沿已经推进到 Phase 7A，重点体现在 [boundflow/runtime/linear_operator.py](/home/lee/Codes/boundflow/boundflow/runtime/linear_operator.py)、[boundflow/runtime/crown_ibp.py](/home/lee/Codes/boundflow/boundflow/runtime/crown_ibp.py)、[tests/test_phase7a_pr10_relu_barrier_structured.py](/home/lee/Codes/boundflow/tests/test_phase7a_pr10_relu_barrier_structured.py)、[tests/test_phase7a_pr11_shared_crown_bench.py](/home/lee/Codes/boundflow/tests/test_phase7a_pr11_shared_crown_bench.py) 与 [scripts/bench_phase7a_shared_crown_path_attribution.py](/home/lee/Codes/boundflow/scripts/bench_phase7a_shared_crown_path_attribution.py)。

---

## 3. 阶段演进总览

| 阶段 | 核心目标 | 关键能力 | 代表代码路径 | 代表测试 | 代表文档 |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | 工程止血 | 可编辑安装、包结构清理、最小 smoke | `pyproject.toml`、[tests/test_env.py](/home/lee/Codes/boundflow/tests/test_env.py) | [tests/test_env.py](/home/lee/Codes/boundflow/tests/test_env.py) | [gemini_doc/phase0_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase0_summary.md) |
| Phase 1 | IR 地基 | Node/Value 双层 Primal IR、`validate()` | [boundflow/ir/primal.py](/home/lee/Codes/boundflow/boundflow/ir/primal.py) | [tests/test_ir_primal_validate.py](/home/lee/Codes/boundflow/tests/test_ir_primal_validate.py) | [gemini_doc/phase1_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase1_summary.md) |
| Phase 2 | Torch 前端最小可用 | `torch.export -> Primal IR`、最小 normalize | [boundflow/frontends/pytorch/frontend.py](/home/lee/Codes/boundflow/boundflow/frontends/pytorch/frontend.py)、[boundflow/frontends/normalize.py](/home/lee/Codes/boundflow/boundflow/frontends/normalize.py) | [tests/test_torch_frontend_import.py](/home/lee/Codes/boundflow/tests/test_torch_frontend_import.py) | [gemini_doc/phase2_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase2_summary.md) |
| Phase 3 | 正确性基线 | Python IBP reference、auto_LiRPA 对齐 | [boundflow/domains/interval.py](/home/lee/Codes/boundflow/boundflow/domains/interval.py)、[boundflow/runtime/executor.py](/home/lee/Codes/boundflow/boundflow/runtime/executor.py) | [tests/test_phase3_ibp_against_auto_lirpa.py](/home/lee/Codes/boundflow/tests/test_phase3_ibp_against_auto_lirpa.py)、[tests/test_phase3_ibp_cnn_against_auto_lirpa.py](/home/lee/Codes/boundflow/tests/test_phase3_ibp_cnn_against_auto_lirpa.py) | [gemini_doc/phase3_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase3_summary.md) |
| Phase 4 | 系统骨架闭环 | Task/Planner/Executor、StoragePlan、Spec(C)、TVM demo、ONNX 前端 | [boundflow/ir/task.py](/home/lee/Codes/boundflow/boundflow/ir/task.py)、[boundflow/planner/interval_v0.py](/home/lee/Codes/boundflow/boundflow/planner/interval_v0.py)、[boundflow/runtime/task_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/task_executor.py)、[boundflow/runtime/tvm_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/tvm_executor.py)、[boundflow/frontends/onnx/frontend.py](/home/lee/Codes/boundflow/boundflow/frontends/onnx/frontend.py) | [tests/test_phase4_task_pipeline_against_auto_lirpa.py](/home/lee/Codes/boundflow/tests/test_phase4_task_pipeline_against_auto_lirpa.py)、[tests/test_phase4c_tvmexecutor_against_auto_lirpa.py](/home/lee/Codes/boundflow/tests/test_phase4c_tvmexecutor_against_auto_lirpa.py)、[tests/test_phase4d_onnx_frontend_matches_torch.py](/home/lee/Codes/boundflow/tests/test_phase4d_onnx_frontend_matches_torch.py) | [gemini_doc/phase4_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase4_summary.md) |
| Phase 5 | 论文/AE 工件化 | planner pipeline、TVM/Relax 可观测性、JSONL schema、artifact runner | [boundflow/runtime/tvm_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/tvm_executor.py)、[scripts/bench_ablation_matrix.py](/home/lee/Codes/boundflow/scripts/bench_ablation_matrix.py)、[scripts/postprocess_ablation_jsonl.py](/home/lee/Codes/boundflow/scripts/postprocess_ablation_jsonl.py)、[scripts/run_phase5d_artifact.py](/home/lee/Codes/boundflow/scripts/run_phase5d_artifact.py) | [tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py](/home/lee/Codes/boundflow/tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py)、[tests/test_artifact_phase5d_smoke.py](/home/lee/Codes/boundflow/tests/test_artifact_phase5d_smoke.py) | [gemini_doc/phase5_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase5_summary.md)、[docs/phase5_done.md](/home/lee/Codes/boundflow/docs/phase5_done.md) |
| Phase 6 | 方法族落地与收益归因 | InputSpec/LpBall、CROWN-IBP、alpha-CROWN、alpha-beta-CROWN、BaB、node-batch、E2E 工件 | [boundflow/runtime/perturbation.py](/home/lee/Codes/boundflow/boundflow/runtime/perturbation.py)、[boundflow/runtime/crown_ibp.py](/home/lee/Codes/boundflow/boundflow/runtime/crown_ibp.py)、[boundflow/runtime/alpha_crown.py](/home/lee/Codes/boundflow/boundflow/runtime/alpha_crown.py)、[boundflow/runtime/alpha_beta_crown.py](/home/lee/Codes/boundflow/boundflow/runtime/alpha_beta_crown.py)、[boundflow/runtime/bab.py](/home/lee/Codes/boundflow/boundflow/runtime/bab.py)、[scripts/bench_phase6h_bab_e2e_time_to_verify.py](/home/lee/Codes/boundflow/scripts/bench_phase6h_bab_e2e_time_to_verify.py) | [tests/test_phase6a_inputspec_lpball_linear.py](/home/lee/Codes/boundflow/tests/test_phase6a_inputspec_lpball_linear.py)、[tests/test_phase6b_crown_ibp_mlp.py](/home/lee/Codes/boundflow/tests/test_phase6b_crown_ibp_mlp.py)、[tests/test_phase6e_bab_mlp.py](/home/lee/Codes/boundflow/tests/test_phase6e_bab_mlp.py)、[tests/test_phase6h_bench_e2e_schema.py](/home/lee/Codes/boundflow/tests/test_phase6h_bench_e2e_schema.py) | [gemini_doc/phase6_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase6_summary.md) |

---

## 4. 按阶段展开：做了什么

### 4.1 Phase 0-2：工程地基与前端导入

这三步解决的是“能不能稳定开发、稳定导入模型”的问题。

- Phase 0 先止血，解决 editable install、重复包路径、最小环境 smoke。
- Phase 1 再把 Primal IR 加固成可校验结构，防止后续 planner/runtime 在不一致图上返工。
- Phase 2 才把 Torch 模型真正导入进来，使后续所有验证和编译工作有统一输入语义。

关键改动主题：

- 工程化基线：`pyproject.toml`、环境 smoke、pytest 收敛到仓库自己的 `tests/`
- Primal IR 结构化：Node/Value/type/meta/validate
- Torch 前端与最小 normalize：让导入图可被后续阶段消费

主要代码落点：

- [boundflow/ir/primal.py](/home/lee/Codes/boundflow/boundflow/ir/primal.py)
- [boundflow/frontends/pytorch/frontend.py](/home/lee/Codes/boundflow/boundflow/frontends/pytorch/frontend.py)
- [boundflow/frontends/normalize.py](/home/lee/Codes/boundflow/boundflow/frontends/normalize.py)
- [tests/test_env.py](/home/lee/Codes/boundflow/tests/test_env.py)

为什么这是下一阶段前提：

- 没有可校验的 IR，Phase 3 的 correctness gate 没有稳定输入。
- 没有 Torch 前端，Phase 3-6 都只能在手写 toy graph 上空转，无法形成真实模型证据链。

### 4.2 Phase 3-4：语义闭环与系统骨架

这两步解决的是“BoundFlow 先正确，再系统化”的问题。

- Phase 3 先用 Python IBP reference 与 auto_LiRPA 对齐，建立 correctness baseline。
- Phase 4 再把逐节点解释执行抽象成 Task/Planner/Executor，并打通 TVM demo、Spec(C)、ONNX 前端和 layout-only 语义。

关键改动主题：

- Interval IBP reference 与 MLP/CNN 对齐
- Task IR、StoragePlan、Planner、Scheduler 形态成型
- Python executor 与 TVM executor 两条执行路径并存
- Torch/ONNX 前端统一进入同一套 planner/executor

主要代码落点：

- [boundflow/domains/interval.py](/home/lee/Codes/boundflow/boundflow/domains/interval.py)
- [boundflow/ir/task.py](/home/lee/Codes/boundflow/boundflow/ir/task.py)
- [boundflow/planner/interval_v0.py](/home/lee/Codes/boundflow/boundflow/planner/interval_v0.py)
- [boundflow/planner/interval_v1.py](/home/lee/Codes/boundflow/boundflow/planner/interval_v1.py)
- [boundflow/runtime/task_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/task_executor.py)
- [boundflow/runtime/tvm_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/tvm_executor.py)
- [boundflow/frontends/onnx/frontend.py](/home/lee/Codes/boundflow/boundflow/frontends/onnx/frontend.py)
- [boundflow/planner/passes/layout_only.py](/home/lee/Codes/boundflow/boundflow/planner/passes/layout_only.py)

为什么这是下一阶段前提：

- Phase 5 的 bench、artifact、compile cache、planner pipeline，都必须建立在“Task/Planner/Executor 已经闭环”的前提上。
- Phase 6 要接 CROWN、alpha-beta、BaB，也需要先有统一的运行时与任务表示，而不是继续堆在 Phase 3 的 interpreter 上。

### 4.3 Phase 5：论文/AE 工件与可复现产线

Phase 5 解决的是“怎么把系统收益说清楚、跑出来、冻结成证据链”的问题，而不是新增验证算法。

关键改动主题：

- TaskGraph、PlanBundle、planner pipeline、reuse/memory instrumentation
- TVM/Relax 可观测性、编译缓存、compile/run 拆分口径
- bench JSONL schema、contract tests、postprocess、artifact runner
- baseline 与 fairness 口径进入同一套产线

主要代码落点：

- [boundflow/planner/core.py](/home/lee/Codes/boundflow/boundflow/planner/core.py)
- [boundflow/planner/pipeline.py](/home/lee/Codes/boundflow/boundflow/planner/pipeline.py)
- [boundflow/planner/storage_reuse.py](/home/lee/Codes/boundflow/boundflow/planner/storage_reuse.py)
- [boundflow/runtime/scheduler.py](/home/lee/Codes/boundflow/boundflow/runtime/scheduler.py)
- [boundflow/runtime/tvm_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/tvm_executor.py)
- [scripts/bench_ablation_matrix.py](/home/lee/Codes/boundflow/scripts/bench_ablation_matrix.py)
- [scripts/postprocess_ablation_jsonl.py](/home/lee/Codes/boundflow/scripts/postprocess_ablation_jsonl.py)
- [scripts/run_phase5d_artifact.py](/home/lee/Codes/boundflow/scripts/run_phase5d_artifact.py)

为什么这是下一阶段前提：

- 没有 schema、artifact、claims、MANIFEST，Phase 6 就算把 alpha-beta/BaB 跑出来，也很难转成 reviewer-proof 的工件链。
- Phase 5 冻结出的“如何做对照、如何做产线、如何做可复现实验”实际上成为 Phase 6 E2E 工具链的模板。

### 4.4 Phase 6：方法族落地与系统收益归因

Phase 6 解决的是“如何在同一个系统框架里接入更强方法族，并把收益拆成可归因证据”的问题。

关键改动主题：

- `InputSpec + LpBallPerturbation` 替代早期只服务 Linf 的输入扰动表达
- CROWN-IBP 的 forward/backward 最小闭环
- alpha-CROWN 的优化参数与 warm-start
- alpha-beta-CROWN 的 beta 编码与可行性/不可行性接口
- BaB driver、node-batch、NodeEvalCache、branch hint、partial prune
- Phase 6H 的 E2E time-to-verify bench、sweep、report、plot 与 AE README

主要代码落点：

- [boundflow/runtime/perturbation.py](/home/lee/Codes/boundflow/boundflow/runtime/perturbation.py)
- [boundflow/runtime/crown_ibp.py](/home/lee/Codes/boundflow/boundflow/runtime/crown_ibp.py)
- [boundflow/runtime/alpha_crown.py](/home/lee/Codes/boundflow/boundflow/runtime/alpha_crown.py)
- [boundflow/runtime/alpha_beta_crown.py](/home/lee/Codes/boundflow/boundflow/runtime/alpha_beta_crown.py)
- [boundflow/runtime/bab.py](/home/lee/Codes/boundflow/boundflow/runtime/bab.py)
- [scripts/bench_phase6c_crown_ibp_multispec_throughput.py](/home/lee/Codes/boundflow/scripts/bench_phase6c_crown_ibp_multispec_throughput.py)
- [scripts/bench_phase6g_bab_node_batch_throughput.py](/home/lee/Codes/boundflow/scripts/bench_phase6g_bab_node_batch_throughput.py)
- [scripts/bench_phase6h_bab_e2e_time_to_verify.py](/home/lee/Codes/boundflow/scripts/bench_phase6h_bab_e2e_time_to_verify.py)

为什么这是下一阶段前提：

- Phase 7 若要扩 skip/branch/general graph、Conv、甚至更大规模后端下沉，必须建立在 Phase 6 已经把方法族接口与收益归因口径钉住的前提上。

---

## 5. 代码改动脉络

这里不按单文件穷举，而按模块看项目演化的重心迁移。

### 5.1 `boundflow/frontends`

角色：把 Torch/ONNX 模型统一导入为 BoundFlow 可消费的 Primal IR。

主要演进阶段：

- Phase 2：Torch 前端最小闭环，重点在 [boundflow/frontends/pytorch/frontend.py](/home/lee/Codes/boundflow/boundflow/frontends/pytorch/frontend.py)
- Phase 4：ONNX 子集闭环与 normalize/permute/reshape 语义补齐，重点在 [boundflow/frontends/onnx/frontend.py](/home/lee/Codes/boundflow/boundflow/frontends/onnx/frontend.py)

### 5.2 `boundflow/ir`

角色：定义系统真正稳定的中间表示，包括 Primal IR、Task IR、Spec、TaskGraph。

主要演进阶段：

- Phase 1：Primal IR 加固，重点在 [boundflow/ir/primal.py](/home/lee/Codes/boundflow/boundflow/ir/primal.py)
- Phase 4：Task/StoragePlan/Spec 成型，重点在 [boundflow/ir/task.py](/home/lee/Codes/boundflow/boundflow/ir/task.py) 和 [boundflow/ir/spec.py](/home/lee/Codes/boundflow/boundflow/ir/spec.py)
- Phase 5：TaskGraph 与调度/规划载体继续长出来，重点在 [boundflow/ir/task_graph.py](/home/lee/Codes/boundflow/boundflow/ir/task_graph.py)

### 5.3 `boundflow/planner`

角色：把“做什么”转成“怎么执行”，包括 partition、passes、reuse、pipeline、verification。

主要演进阶段：

- Phase 4：最小 planner 与 Spec(C) 融合
- Phase 5：planner pipeline、liveness、buffer reuse、determinism、instrument、verify 成为主线

代表路径：

- [boundflow/planner/interval_v0.py](/home/lee/Codes/boundflow/boundflow/planner/interval_v0.py)
- [boundflow/planner/interval_v1.py](/home/lee/Codes/boundflow/boundflow/planner/interval_v1.py)
- [boundflow/planner/core.py](/home/lee/Codes/boundflow/boundflow/planner/core.py)
- [boundflow/planner/pipeline.py](/home/lee/Codes/boundflow/boundflow/planner/pipeline.py)
- [boundflow/planner/passes/buffer_reuse_pass.py](/home/lee/Codes/boundflow/boundflow/planner/passes/buffer_reuse_pass.py)

### 5.4 `boundflow/runtime`

角色：承接 reference execution、TVM execution、以及后期方法族与 BaB 运行时。

主要演进阶段：

- Phase 3：以 [boundflow/runtime/executor.py](/home/lee/Codes/boundflow/boundflow/runtime/executor.py) 为主的 Python reference
- Phase 4：Task executor 与 TVM executor 双路径
- Phase 5：重心集中到 [boundflow/runtime/tvm_executor.py](/home/lee/Codes/boundflow/boundflow/runtime/tvm_executor.py) 的 compile cache、stats、instrument、scheduler 对齐
- Phase 6：重心明显转向 [boundflow/runtime/perturbation.py](/home/lee/Codes/boundflow/boundflow/runtime/perturbation.py)、[boundflow/runtime/crown_ibp.py](/home/lee/Codes/boundflow/boundflow/runtime/crown_ibp.py)、[boundflow/runtime/alpha_crown.py](/home/lee/Codes/boundflow/boundflow/runtime/alpha_crown.py)、[boundflow/runtime/alpha_beta_crown.py](/home/lee/Codes/boundflow/boundflow/runtime/alpha_beta_crown.py)、[boundflow/runtime/bab.py](/home/lee/Codes/boundflow/boundflow/runtime/bab.py)

### 5.5 `boundflow/backends/tvm`

角色：把 Task lower 到 Relax/TIR/TVM 运行时，承担系统论文里“后端可编译性”的那部分证据。

主要演进阶段：

- Phase 4：线性与卷积的 TVM demo
- Phase 5：从 demo 走向 Relax task lowering、分析、fusion、compile observability

代表路径：

- [boundflow/backends/tvm/interval_linear.py](/home/lee/Codes/boundflow/boundflow/backends/tvm/interval_linear.py)
- [boundflow/backends/tvm/interval_conv2d.py](/home/lee/Codes/boundflow/boundflow/backends/tvm/interval_conv2d.py)
- [boundflow/backends/tvm/relax_task_lowering.py](/home/lee/Codes/boundflow/boundflow/backends/tvm/relax_task_lowering.py)
- [boundflow/backends/tvm/relax_interval_task_ops.py](/home/lee/Codes/boundflow/boundflow/backends/tvm/relax_interval_task_ops.py)
- [boundflow/backends/tvm/relax_analysis.py](/home/lee/Codes/boundflow/boundflow/backends/tvm/relax_analysis.py)

### 5.6 `scripts`

角色：把系统能力转成 bench、artifact、AE 工具链。

主要演进阶段：

- Phase 4：有少量 demo/bench
- Phase 5：重心是 `bench -> postprocess -> artifact`
- Phase 6：重心切到 `bench_phase6*` 的 throughput/E2E 工具链

关键重心转移：

- Phase 5 主要集中在 [scripts/bench_ablation_matrix.py](/home/lee/Codes/boundflow/scripts/bench_ablation_matrix.py)、[scripts/postprocess_ablation_jsonl.py](/home/lee/Codes/boundflow/scripts/postprocess_ablation_jsonl.py)、[scripts/run_phase5d_artifact.py](/home/lee/Codes/boundflow/scripts/run_phase5d_artifact.py)
- Phase 6 主要集中在 [scripts/bench_phase6c_crown_ibp_multispec_throughput.py](/home/lee/Codes/boundflow/scripts/bench_phase6c_crown_ibp_multispec_throughput.py)、[scripts/bench_phase6g_bab_node_batch_throughput.py](/home/lee/Codes/boundflow/scripts/bench_phase6g_bab_node_batch_throughput.py)、[scripts/bench_phase6h_bab_e2e_time_to_verify.py](/home/lee/Codes/boundflow/scripts/bench_phase6h_bab_e2e_time_to_verify.py)、[scripts/run_phase6h_artifact.sh](/home/lee/Codes/boundflow/scripts/run_phase6h_artifact.sh)

### 5.7 `tests`

角色：把每个阶段的语义 gate、系统 gate、schema gate、artifact gate 固定下来。

主要演进阶段：

- Phase 1-4：correctness 对齐测试为主
- Phase 5：schema/postprocess/artifact smoke 成为主线
- Phase 6：除了正确性之外，开始系统性测试 cache、batch、branch hint、E2E schema

这也是为什么看 `tests/` 就能大致看出项目重心转移：从 `test_phase3_*`、`test_phase4*`，到 `test_phase5d_*`，再到 `test_phase6*`，每一层都是上一层能力冻结后的下一阶段入口。

---

## 6. 现有哪些文档已经在记录这条脉络

现有文档并不缺，缺的是“把它们串起来的总入口”。可以按角色来读：

- 总账：
  - [docs/change_log.md](/home/lee/Codes/boundflow/docs/change_log.md)
  - 作用：记录每一批修改做了什么、为什么做、怎么验证，是最完整的历史流水账。

- 阶段总结：
  - [gemini_doc/phase0_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase0_summary.md) 到 [gemini_doc/phase6_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase6_summary.md)
  - 作用：按阶段总结目标、完成定义、代码落点与验证入口。

- 全流程文档：
  - [gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md](/home/lee/Codes/boundflow/gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md)
  - 作用：从 claim、命令、产物、AE 视角串全链路。

- 指挥视角：
  - [gemini_doc/boundflow_full_pipeline_director_view.md](/home/lee/Codes/boundflow/gemini_doc/boundflow_full_pipeline_director_view.md)
  - 作用：从工程组织与系统主线角度看项目，不强调逐阶段细节。

- 设计/评审：
  - [gemini_doc/phase6_review_three_axis_stage_pipeline.md](/home/lee/Codes/boundflow/gemini_doc/phase6_review_three_axis_stage_pipeline.md)
  - [gemini_doc/bound_methods_and_solvers_design.md](/home/lee/Codes/boundflow/gemini_doc/bound_methods_and_solvers_design.md)
  - 作用：解释 Phase 6 方案为什么这样分层，以及落地时哪些坑必须提前规避。

- 论文/AE 交付：
  - [docs/phase5_done.md](/home/lee/Codes/boundflow/docs/phase5_done.md)
  - [gemini_doc/ae_readme_phase6h.md](/home/lee/Codes/boundflow/gemini_doc/ae_readme_phase6h.md)
  - 作用：告诉读者怎么跑、怎么验、哪些口径冻结了。

这篇新文档补的缺口是：把“目标、阶段演进、代码重心、现有记录、下一步路线”汇成一篇面向接手者的研发主线总整理，而不是让读者自己在十几份文档之间来回跳。

---

## 7. 当前状态判断

如果按“已经在 git 历史和阶段总结里形成稳定里程碑”的标准看，Phase 0-6 是当前最清晰的一层：

- 它们在 [docs/change_log.md](/home/lee/Codes/boundflow/docs/change_log.md) 中有连续记录。
- 它们在 [gemini_doc/phase0_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase0_summary.md) 到 [gemini_doc/phase6_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase6_summary.md) 中已有较完整收官文档。
- Phase 5 还有 [docs/phase5_done.md](/home/lee/Codes/boundflow/docs/phase5_done.md) 作为冻结口径，Phase 6 则已有完整方法族与 E2E 工件链。

如果按“当前仓库工作区与文档已经推进到哪里”的标准看，Phase 7A 才是当前真实开发前沿：

- `runtime` 下的 [boundflow/runtime/linear_operator.py](/home/lee/Codes/boundflow/boundflow/runtime/linear_operator.py) 与 [boundflow/runtime/crown_ibp.py](/home/lee/Codes/boundflow/boundflow/runtime/crown_ibp.py) 已经连续完成 PR-8 到 PR-14 的 shared CROWN / DAG backward 扩展。
- `tests` 下已有 [tests/test_phase7a_pr9_dag_linear_operator.py](/home/lee/Codes/boundflow/tests/test_phase7a_pr9_dag_linear_operator.py)、[tests/test_phase7a_pr10_relu_barrier_structured.py](/home/lee/Codes/boundflow/tests/test_phase7a_pr10_relu_barrier_structured.py)、[tests/test_phase7a_pr11_shared_crown_bench.py](/home/lee/Codes/boundflow/tests/test_phase7a_pr11_shared_crown_bench.py) 三组专项回归。
- `scripts` 下新增了 [scripts/bench_phase7a_shared_crown_path_attribution.py](/home/lee/Codes/boundflow/scripts/bench_phase7a_shared_crown_path_attribution.py)，把“layout-only 是否已有收益、ReLU 路径热点在哪、PR-14 后 hotspot 是否归零”这些问题都变成了可复现口径。
- 文档侧已经有 [gemini_doc/phase7a_pr11_shared_crown_benchmark_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase7a_pr11_shared_crown_benchmark_summary.md) 与 PR-11 到 PR-14 的 `change_*.md` 记录。

当前更准确的判断不是“Phase 7A 还在做 correctness 收尾”，而是：

- correctness 与 observability 已经闭合；
- `split_pos_neg_dense` 旧热点已从 ReLU workload 主路径清掉；
- 剩余性能问题集中到 `relu_relax_pullback()` 内部仍存在的精确 dense materialization 成本。

---

## 8. 下一步自然路线

基于 [gemini_doc/phase7a_pr11_shared_crown_benchmark_summary.md](/home/lee/Codes/boundflow/gemini_doc/phase7a_pr11_shared_crown_benchmark_summary.md) 与 [gemini_doc/next_plan_after_phase7a_pr14.md](/home/lee/Codes/boundflow/gemini_doc/next_plan_after_phase7a_pr14.md)，Phase 7A 在 PR-14 之后最自然的主线有四条：

### 8.1 继续压缩 ReLU pullback 的 dense materialization 成本

PR-14 清掉的是旧的 `split_pos_neg_dense` hotspot，但 `RightMatmul` / `SliceInput` 的 `relu_relax_pullback()` 仍会内部精确 materialize dense 系数。下一步最值钱的工作，是把这部分成本拆出来并继续下降，而不是再回头重写 `split_pos_neg()` contract。

### 8.2 继续强化 shared CROWN 的 benchmark / observability

现在已经有同进程 baseline、固定 workload、dense fallback 计数。下一步应继续把 ReLU pullback 内部的物化次数、张量尺寸或 timing breakdown 变成可记录字段，让“为什么还慢”不再停留在推测层。

### 8.3 继续做 operator-specific 的 sound 优化，而不是泛化成错误 algebra

PR-12 已经证明 `RightMatmul` 的常见四项 sign split 不能满足当前逐元素 exact contract。后续更自然的方向是继续做 ReLU 专用、operator-specific 的 sound 实现，或引入更合适的中间表示，而不是把一个不 exact 的分解硬塞回 `split_pos_neg()`。

### 8.4 后端方向仍然保持不变

后端的自然原则仍然不变：保持 BaB 控制流在 Python runtime，把真正重的张量计算继续往后端推进。下一阶段不是把整个 BaB 逻辑强行 lower 到 TVM，而是继续扩大“哪些重计算块值得编译/融合/缓存/批处理”的覆盖面。
