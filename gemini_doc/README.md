# gemini_doc 导引（BoundFlow 工程文档索引）

本目录用于存放“由大模型协助生成/维护”的工程文档与变更记录（changelog-style notes），目标是：

- 让每次 PR/阶段推进都有可审计的文字记录；
- 让别人（或未来的你）能快速定位：某个决策/某个口径/某个脚本是“为什么这样做”；
- 让论文/AE 的证据链（claim → 命令 → 产物 → 字段）能闭环。

> 约定：每次工程改动都应在 `gemini_doc/` 新增一份 `change_YYYY-MM-DD_*.md` 记录，并在 `docs/change_log.md` 追加一条总账。

---

## 1) “我应该从哪读起？”

按目的给四条阅读路径：

### A. 论文/AE 视角（最推荐）

1. `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`（当前最新：
   production Schedule ownership/memory 门禁、NO-GO 与下一 native real-network 分支）
2. `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`（真实 verifier
   correctness/integration 关闭审计与不可升级边界）
3. `gemini_doc/rvir_external_audit_handoff_2026_08_03.md`（可直接交给其他模型的自包含
   RVIR 审计请求、证据和复核顺序）
4. `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`（RVIR 所有权与门禁）
5. `gemini_doc/pr14_external_model_audit_handoff_2026_07_19.md`（从项目起点到 PR-14
   No-Go 的自包含外部审计入口）
6. `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`（PR-14
   后 IR-first 架构重置、对象协议与实施门禁）
7. `gemini_doc/current_status_after_pr13.md`（全项目当前状态）
8. `gemini_doc/asplos_claims_map.md`（论文主张→代码→实验→工件证据）
9. `gemini_doc/asplos_execution_memo_v1_0.md`（唯一历史顺序与门禁）
10. `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`（ASPLOS 顶层计划）

### B. 全流程总览（从 claims 到工程到 AE）

- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
- `gemini_doc/boundflow_full_pipeline_director_view.md`（指挥视角的系统路线）

### C. 研发协作流程（人与大模型怎么配合）

- `gemini_doc/boundflow_build_and_run_workflow.md`（按源码类型编译、运行和测试）
- `gemini_doc/llm_collaboration_workflow.md`（输入计划→修正测试→总结→下一步计划）
- `/home/lee/.codex/skills/boundflow-workflow/SKILL.md`（上述工作流的本机 skill 入口）

### D. 研发演化/接手视角

1. `gemini_doc/current_status_after_pr13.md`（当前冻结状态与下一门禁）
2. `gemini_doc/project_evolution_overview.md`（项目目标、阶段推进、代码落点、未来路线）
3. `docs/change_log.md`（按时间看每一批修改做了什么）
4. `gemini_doc/phase6_summary.md`（当前方法族与 E2E 工件链的阶段总览）

---

## 2) 本目录文件分类

### 2.1 关键交付文档（“长期有效”）

- `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_PLAN_2026_08_04.md`：RVIR 后
  production Schedule ownership、materialization/storage 与 multi-budget memory 门禁；
  `NO_GO` 后转向 native real-network Bound IR
- `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_CHANGELOG_2026_08_04.md`：上述
  audit module、runner、artifact、验证与权威文档同步记录
- `gemini_doc/pr14_external_model_audit_handoff_2026_07_19.md`：项目起点、ASPLOS 路线、
  PR-14A/B 证据/限制、外部复核命令与下一步的自包含审计交接
- `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`：复审当前
  Bound/Plan/Task/Schedule/runtime 实现后冻结的一等 IR 分层、动态规划边界与逐阶段门禁
- `gemini_doc/change_2026-07-20_ir_planner_schedule_runtime_contract.md`：上述架构重置、
  Claims Map 降级和权威文档修订记录
- `gemini_doc/change_2026-07-28_bound_ir_v1_schema_foundation.md`：IR-1A typed Bound IR
  schema、verifier、deterministic dump/hash、兼容性与未完成 builder/interpreter 边界
- `gemini_doc/change_2026-07-28_bound_ir_v1_plain_crown_lowering.md`：IR-1B plain-CROWN
  Task/trace lowering、显式 affine/fanout 语义、独立 dense interpreter 与 final-bound 对齐边界
- `gemini_doc/change_2026-07-28_bound_ir_v1_representation_rewrite.md`：IR-1C 显式
  dense/structured cast、materialization rewrite、structured reference 执行与 IR-1 closure
- `gemini_doc/change_2026-07-28_plan_ir_v1_schema_and_legacy_migration.md`：IR-2A typed
  PlanTemplate/PlanInstance、跨决策 verifier、instance replay 与 PR-11/12 迁移边界
- `gemini_doc/change_2026-07-28_plan_ir_v1_reference_builder_selector.md`：IR-2B typed
  evidence→PlanTemplate builder、预算/deadline selector 与不可变 selection artifact
- `gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`：IR-2C query-time state
  validity、legacy atomic assembly、raw-record absence audit 与 IR-2 validated-reduced closure
- `gemini_doc/change_2026-07-28_schedule_ir_v1_schema_lowering.md`：IR-3A typed
  ScheduleModule/action schema、PlanInstance lowering、memory/use-def/query verifier foundation
- `gemini_doc/change_2026-07-28_schedule_ir_v1_control_executor.md`：IR-3B batch/event/state/
  retry/replan actions、reference executor、canonical trace 与 fresh-process artifact
- `gemini_doc/change_2026-07-28_task_ir_v1_foundation.md`：IR-3C typed TaskIRModule/Unit、
  Plan region lowering、Task↔Schedule launch linkage 与 dispatch trace
- `gemini_doc/change_2026-07-28_task_schedule_ir_v1_semantic_closure.md`：IR-3D stateful
  Bound stepping、逐 Task 数值执行、shape/transfer 契约补项、artifact v2 与 closure audit
- `gemini_doc/change_2026-07-28_ir4a_typed_backend_dispatch.md`：IR-4A typed backend
  dispatch key、PyTorch reference adapter、prepared-task cache 与 stale/capability rejection
- `gemini_doc/change_2026-07-28_ir4b_pytorch_backend_registry.md`：IR-4B backend-specific
  Task identity、fused ReLU→Affine stepping 与真实 dense/structured/chunked registry
- `gemini_doc/change_2026-07-28_ir4c_tvm_backend_cache_fallback.md`：IR-4C typed TVM
  fused/unfused、dispatch-namespaced disk cache、fresh-process replay 与 semantic OOM fallback
- `gemini_doc/change_2026-07-28_ir4d_compiler_query_state_runtime.md`：IR-4D
  capability-gated typed query、Plan/Task cache、exact-version state load/store/task skip、
  fresh-process artifact 与 PR-14 No-Go 保留
- `gemini_doc/change_2026-07-28_ir4e_pr13_query_migration_closure.md`：PR-13
  DynamicBatchManager→typed compiler adapter、legacy α/β historical opt-in、artifact v2 与
  IR-4 validated-reduced closure
- `gemini_doc/change_2026-07-28_ir5a_adaptive_plan_context.md`：IR-5A query-time
  memory/deadline/cache/distribution context、compile amortization 与 per-query plan cache
- `gemini_doc/change_2026-07-28_ir5b_fair_policy_evaluator.md`：IR-5B
  fixed/local/global/oracle 统一 observation evaluator、tail/TTV/peak/regret 与 synthetic
  contract artifact
- `gemini_doc/change_2026-07-28_ir5c0_typed_measured_workload_foundation.md`：IR-5C0
  正式 typed MLP benchmark workload、候选 Plan/Schedule 构造与 predicted/measured compile
  防泄漏
- `gemini_doc/change_2026-07-28_ir5c1_leakage_free_measurement_runner.md`：IR-5C1
  calibration-only 预测、CUDA cold/warm/peak/TVM phase 测量、冻结 split/resource context
  与目录级 manifest/semantic replay
- `gemini_doc/change_2026-07-28_ir5c2_cuda_heldout_partial.md`：IR-5C2 fresh CUDA
  typed MLP artifact、四策略 regret/TTV/tail/peak、低内存切换及 workload-family/batching
  未闭环的 PARTIAL 判定
- `gemini_doc/change_2026-07-28_ir5c3a_independent_cnn_family.md`：IR-5C3A
  deterministic chain-CNN typed workload、跨 architecture calibration feature 与
  PyTorch/TVM fused CUDA semantic probe
- `gemini_doc/change_2026-07-28_ir5c3b_fair_batching_contract.md`：IR-5C3B
  fixed-single/ordinary-batching/batched-original 公平 evaluator、physical-batch 归一化、
  MLP→CNN runner 与 batch 语义门禁
- `gemini_doc/change_2026-07-28_ir5c3c_family_fair_nogo.md`：IR-5C3C
  architecture-held-out CUDA fair artifact、Global p90 regret 70.263×、host
  validate/hash hot-path 归因与 IR-5 v1 VALIDATED-NO-GO
- `gemini_doc/change_2026-07-28_ir5d_prepared_execution_capsule.md`：IR-5D
  prepared Bound/Task execution、production/audit trace 分离、from-forward-trace 公平基线
  与 calibration-only CUDA remediation 诊断；该文档当时尚未执行的 residual final
  已由 IR-5E—H 完成并判定最终 No-Go
- `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`：IR-5 No-Go 后独立的
  真实 verifier correctness/integration 路线、所有权与 RVIR-1—4 门禁
- `gemini_doc/change_2026-08-03_rvir1_external_intermediate_semantics.md`：external
  intermediate bounds + adaptive slope 的 ResNet initial-CROWN 语义修复
- `gemini_doc/change_2026-08-03_rvir2_typed_external_calls.md`：activation-BaB external exact
  call 的 Bound/Plan/Task/Schedule 类型化、真实调度与 lineage closure
- `gemini_doc/change_2026-08-03_rvir3_cpu_correctness_artifact.md`：394 个历史 activation
  typed admission、377 次真实 CPU exact dispatch、ResNet 等价与自包含 replay artifact
- `gemini_doc/change_2026-08-03_rvir_online_raw_replay_v2.md`：外部审计 M4 后续，冻结
  377 条在线 query/typed-record 原文并在 fresh replay 中重算 lineage、accounting 与 IR hash
- `gemini_doc/change_2026-08-03_rvir_resnet_raw_rerun.md`：外部审计 F5/M5 后续，在固定
  αβ-CROWN 与 VNN-COMP commit 上连续两次重跑 ResNet 原始数值，核对冻结摘要与 tensor digest
- `gemini_doc/change_2026-08-03_rvir_post_hardening_audit_handoff.md`：PR #5—#8
  审计后加固的 AC1—AC6、独立复核命令、claim boundary 与新 DocOps exchange 交接
- `gemini_doc/change_2026-08-04_rvir_post_hardening_audit_closure.md`：外部复审 approve、
  AC1—AC6/F1—F5 关闭、正式 exchange closure 与完整审计附件的 Git 固定记录
- `gemini_doc/change_2026-07-28_ir5e_residual_final_protocol_freeze.md`：IR-5E
  residual-CNN typed workload、chain-CNN calibration→residual final v2 冻结 split、
  from-trace 公平协议及 p90/Pareto 一次性门禁
- `gemini_doc/change_2026-07-28_ir5f_residual_final_v2_protocol_invalid.md`：IR-5F
  v2 首次运行在 fixed-single 输入身份门禁失败、shape-dependent RNG 根因与
  `7401/7402` 永久退役记录
- `gemini_doc/change_2026-07-28_ir5g_exact_input_slice_v3_freeze.md`：IR-5G
  fixed-single 显式切片 batched input 的方法学修复、v3 schema 与未执行
  `7501/7502` final freeze
- `gemini_doc/change_2026-07-28_ir5h_residual_final_v3_nogo.md`：IR-5H
  fresh residual final v3 完整 artifact、Global p90 1.26160×、gray Pareto 缺失与
  当前 ASPLOS system-performance 路线最终 No-Go
- `gemini_doc/change_2026-08-03_ir5_route_closure_and_publish.md`：IR-5 路线封存、
  权威状态去过期、外部 replay 命令与真实 Verifier IR 新路线准入条件
- `gemini_doc/real_verifier_ir_integration_contract_v1_2026_08_03.md`：真实 verifier
  intermediate-bound semantics、relaxation policy、activation external-call IR 与门禁
- `gemini_doc/change_2026-08-03_start_real_verifier_ir_integration.md`：新 correctness/
  integration 路线启动与 ResNet 根因复核记录
- `gemini_doc/change_2026-08-03_rvir1_external_intermediate_semantics.md`：external ReLU
  intermediate bounds/adaptive policy 入 IR 与 ResNet CPU correctness closure
- `gemini_doc/asplos_execution_memo_v1_0.md`：ASPLOS 研发的短执行入口与门禁
- `gemini_doc/current_status_after_pr13.md`：PR-13 closure 后的真实状态、证据边界与当前缺口
- `gemini_doc/pr14_execution_plan.md`：真实 verifier workload coverage/execution 的切片、门禁与止损
- `gemini_doc/change_2026-07-19_pr14a_abcrown_query_profile_adapter.md`：外部 αβ-CROWN
  `compute_bounds` → PR-13 `BoundQuery` → coverage profile 的可撤销接入边界
- `gemini_doc/pr14a_real_query_coverage_2026_07_19.md`：MLP/CNN/VNN-COMP ResNet-2B 的真实
  method/phase/backend coverage、observer baseline 与 PR-14B 窄化判定
- `gemini_doc/change_2026-07-19_pr14a_real_query_traces.md`：真实 trace 生成与 fail-closed
  frontend 审计的变更记录
- `gemini_doc/change_2026-07-19_tvm_ffi_library_search_path.md`：新环境中新版 tvm-ffi
  动态库发现与 Conda hooks 的修复记录
- `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`：exact-box MLP/ResNet fixed
  replay、requested-output/bound-equivalence 门禁与 PR-14/C3 最终 No-Go
- `gemini_doc/change_2026-07-19_pr14b_initial_crown_fixed_replay.md`：PR-14B 代码、contract、
  ignored artifacts 与验证记录
- `gemini_doc/change_2026-07-19_fresh_clone_test_split_fixtures.md`：完整测试从代码冻结 split
  重建临时 fixture，不再依赖新环境中不存在的 ignored PR-12 artifacts
- `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`：ASPLOS 总体研发、论文与 artifact 执行计划
- `gemini_doc/asplos_claims_map.md`：ASPLOS 三项贡献的动态证据映射
- `gemini_doc/materialization_trace_schema_v1.md`：PR-10 trace JSONL 与内存口径
- `gemini_doc/pr10_materialization_profile_summary_2026_07_12.md`：PR-10 第一版 clean GPU profile 与 claim 边界
- `gemini_doc/pr10_dense_structured_comparison_2026_07_12.md`：PR-10 双模式 guardrail 与最终判定
- `gemini_doc/pr10_review_integration_2026_07_12.md`：PR-10 外部评审意见、PR-11 收敛与本次修改记录
- `gemini_doc/pr11_materialization_planner_start_2026_07_12.md`：PR-11 第一实现切片、测试证据与剩余 blocker
- `gemini_doc/materialization_plan_schema_v1.md`：PR-11 plan/context/candidate JSON schema 与 capability 语义
- `gemini_doc/pr11_heldout_eval_2026_07_12.md`：PR-11 cost model、final held-out 结果与未通过项
- `gemini_doc/pr11_multi_barrier_placement_start_2026_07_12.md`：非退化 multi-barrier Global placement 与 mixed runtime foundation
- `gemini_doc/pr11_barrier_evaluator_and_retry_2026_07_12.md`：measured Oracle、Global Retry held-out 结果与真实 OOM handling 边界
- `gemini_doc/change_2026-07-12_pr11_bounded_stratified_retry.md`：有界分层 retry、双规模 held-out 与真实 CUDA OOM 证据
- `gemini_doc/change_2026-07-12_pr11_independent_topology_nogo.md`：并行残差 held-out 失败、部署特征审计与 PR-11 No-Go
- `gemini_doc/change_2026-07-12_pr11_static_topology_cost.md`：静态 topology/liveness feature、LOO retry calibration 与三组 final held-out
- `gemini_doc/pr11_closure_audit_2026_07_12.md`：PR-11 逐项 closure、replicated final evidence 与 PR-12/PR-13 边界
- `gemini_doc/pr11_regret_attribution_2026_07_13.md`：高 regret case 的候选覆盖/后端假设归因
- `gemini_doc/pr12_fused_crown_task_plan_2026_07_13.md`：PR-12 收敛范围、接口、门禁与证据版本
- `gemini_doc/backend_candidate_schema_v1.md`：PR-12 placement/backend 二维候选与 capability 合同
- `gemini_doc/change_2026-07-13_pr12_start_and_fused_linear.md`：PR-12 起点、held-out 与 Linear TIR 第一切片
- `gemini_doc/change_2026-07-13_pr12_fused_conv2d.md`：Conv stride-1/2、codegen 与 latency sanity
- `gemini_doc/change_2026-07-13_pr12_e2e_crown_integration.md`：显式 fused region schedule、TVM/Torch executor、网络级 final-bound 与 zero-copy 门禁
- `gemini_doc/change_2026-07-13_pr12d_correctness_closure.md`：fanout soundness、完整 step contract、pre-materialization fallback 与 TVM-FFI custom-stream closure
- `gemini_doc/change_2026-07-13_pr12ef_runtime_pareto_heldout.md`：正式 runtime/memory Pareto、calibration-only Planner、frozen held-out 与性能 No-Go
- `gemini_doc/change_2026-07-13_pr12g_multibackend_planner.md`：chunked-r512 候选、全新 held-out-v2、多后端 Planner 与 canonical 工件
- `gemini_doc/pr12_mid_long_term_completion_plan.md`：PR-12H–N baseline、摊销、profile、Planner 与 closure 执行路线
- `gemini_doc/pr12_execution_status.md`：PR-12 跨会话唯一恢复入口与当前门禁
- `gemini_doc/change_2026-07-14_pr12h_benchmark_contract.md`：三层 benchmark contract、历史证据披露与 PR-12G freeze
- `gemini_doc/change_2026-07-14_pr12i_fair_baselines.md`：structured/TVM-unfused 公平 baseline、条件 torch.compile probe 与正式 Pareto
- `gemini_doc/change_2026-07-14_pr12j_compile_amortization.md`：compile/load/cache 阶段拆分、跨进程 disk hit 与 Q-sweep 摊销
- `gemini_doc/change_2026-07-14_pr12k_cupti_profile.md`：CUPTI activity profile、硬件 counter 权限边界与停止孤立 TIR 调优判定
- `gemini_doc/change_2026-07-14_pr12l_stop_tir_optimization.md`：冻结停止孤立 TIR 调优、未选分支与 PR-12M 接口约束
- `gemini_doc/change_2026-07-14_pr12m_compile_aware_planner.md`：compile/cache/reuse Planner、v3 split、多预算 held-out 与 regret
- `gemini_doc/pr12_closure_audit_2026_07_14.md`：PR-12N 最终判定、H–M 证据/限制与 PR-13 Go/No-Go
- `gemini_doc/pr12_artifact_appendix_2026_07_14.md`：PR-12 reduced artifact 依赖、工作流、expected outputs 与 claims
- `gemini_doc/pr13_execution_status.md`：PR-13 五切片跨会话状态、冻结边界与恢复命令
- `gemini_doc/change_2026-07-14_pr13a_query_contract_fixed_replay.md`：state-versioned query contract、真实 BaB 固定流 replay 与 PR-13B 门禁
- `gemini_doc/change_2026-07-14_pr13b_dynamic_batch_manager.md`：兼容分桶、预算/deadline、OOM 拆批、physical αβ batching 与 PR-13C 门禁
- `gemini_doc/change_2026-07-14_pr13c_same_solver_adapter.md`：原 host solver 仅替换 bound-call path 的对照与 PR-13D 门禁
- `gemini_doc/change_2026-07-14_pr13d_fixed_e2e_gpu.md`：RTX 4060 fixed/E2E reduced 评估、batched-original 归因与负收益
- `gemini_doc/pr13_closure_audit_2026_07_14.md`：PR-13 `VALIDATED-REDUCED` 逐项 closure 与未成立主张
- `gemini_doc/pr13_artifact_appendix_2026_07_14.md`：PR-13 reduced artifact 命令、expected outputs 与证据链
- `gemini_doc/artifact_claims_phase5d.md`：Phase 5D artifact claims（证据链/口径映射）
- `gemini_doc/artifact_appendix_phase5d.md`：Phase 5D artifact appendix（复现说明）
- `gemini_doc/project_evolution_overview.md`：研发脉络总览（目标、阶段推进、代码落点、未来路线）
- `gemini_doc/codex_superpowers_global_install.md`：Codex Superpowers 全局安装说明（主机级安装、跨主机复用、自动检测 skills 目录）
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`：全流程总览（从研究主张到工程到 AE）
- `gemini_doc/boundflow_full_pipeline_director_view.md`：指挥视角的工程主线
- `gemini_doc/why_boundflow_not_auto_lirpa_or_tvm.md`：论文辩护要点（为何不端到端用 auto_LiRPA / 为何不直接用 TVM）
- `gemini_doc/phase0_summary.md`：Phase 0 总结（工程止血：可编辑安装 + 包结构清理 + 最小 smoke）
- `gemini_doc/phase1_summary.md`：Phase 1 总结（工程止血 + Primal IR 加固）
- `gemini_doc/phase2_summary.md`：Phase 2 总结（TorchFrontend：torch.export → Primal IR + 最小 normalize）
- `gemini_doc/phase3_summary.md`：Phase 3 总结（IBP reference + auto_LiRPA 对齐：MLP/CNN）
- `gemini_doc/perturbation_support_design.md`：设计文档（支持 L∞/L2/L1/L0 输入扰动与线性算子统一公式）
- `gemini_doc/bound_methods_and_solvers_design.md`：设计文档（IBP/CROWN/IBP-CROWN/αβ-CROWN/BaB 的三轴解耦与落地路线）
- `gemini_doc/phase4_summary.md`：Phase 4 总结（Task/Planner/Executor/Spec/TVM/ONNX 的闭环与对齐口径）
- `gemini_doc/phase5_summary.md`：Phase 5 总结（bench→JSONL→postprocess→artifact 产线 + schema_version=1.0 冻结）
- `gemini_doc/phase6_summary.md`：Phase 6 总结（语义闭环 + 系统收益归因 + AE/论文工件链）
- `gemini_doc/quick_restart_ibp.md`：Quick Restart（像 auto_LiRPA 一样跑 IBP 边界）
- `gemini_doc/tvm_backend_optimization_memo.md`：TVM/Relax 后端优化备忘
- `gemini_doc/llm_collaboration_workflow.md`：与大模型协作工作流模板

### 2.2 变更记录（`change_YYYY-MM-DD_*.md`）

这些文件按时间记录“当时做了什么/为什么做/怎么验证”，适合：

- 回溯某个接口/口径的由来
- 追踪阶段推进（Phase 4 → Phase 5）

常见命名模式：

- `change_YYYY-MM-DD_phase5*_pr*_*.md`：按阶段/PR 编号
- `change_YYYY-MM-DD_*_memo.md`：备忘或总结

---

## 3) Phase 5（现状）的一句话索引

如果你只想知道“Phase 5 到底做完了什么”：

- 完成声明：`docs/phase5_done.md`
- 口径冻结：`docs/bench_jsonl_schema.md`（`schema_version=1.0`）
- 一键产线：`scripts/run_phase5d_artifact.py`（产 `results.jsonl/table_main.csv/figures/MANIFEST/CLAIMS/APPENDIX`）
- 证据链：`gemini_doc/artifact_claims_phase5d.md`

---

## 4) 维护规则（防止目录继续膨胀失控）

1. **不要移动/改名历史 `change_*.md`**（避免破坏已有引用）。
2. 新增文档时优先选择：
   - `docs/`：面向用户/读者的稳定说明（安装、schema、完成声明）
   - `gemini_doc/`：面向研发/演进的过程记录（变更记录、备忘、决策）
3. 任何影响口径的变更都要同时更新：
   - `docs/bench_jsonl_schema.md`
   - 对应的 contract tests / postprocess tests
4. 运行产物目录 `artifacts/`、`out/` 不进入 git（已在 `.gitignore` 忽略）。
