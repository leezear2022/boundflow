# BoundFlow 技术参考手册

本文档是 BoundFlow 项目的统一技术参考，合并了 CLAUDE.md、AGENTS.md、README.md 中的共性内容。各工具专属指令保留在对应根文件中。

---

## 1. 项目概述

**BoundFlow** 是一个验证感知的编译器和运行时系统，将 **LiRPA/CROWN 边界传播** 和 **Certified Training** 作为一等公民工作负载。通过专用 IR、全局规划器和 TVM 后端，系统化地消除基于 Python 的验证器中的开销（碎片化、同步、冗余）。

- **核心目标**: 在相同验证强度下，减少端到端验证时间，增加可验证规模（更大模型、更多规格、更深 BaB）。
- **学术定位**: ASPLOS 风格的 system contribution——不发明新验证算法，而是把已有方法族（IBP/CROWN/Alpha-Beta-CROWN/BaB）系统化落地。

---

## 2. 架构总览

```
PyTorch/ONNX → BoundFlow IR → Planner → BoundTasks → TVM Backend → GPU Kernels
```

分层思想：

- **前端**（Frontends）：把模型和验证语义导入成统一 IR
- **中间层**（Planner）：把边界传播任务做成可规划、可复用、可批处理的执行图
- **后端**（TVM Backend）：把任务稳定落到 TVM/GPU
- **运行时**（Runtime）：承接 reference execution、TVM execution、方法族与 BaB

---

## 3. 模块组织

核心 Python 包在 `boundflow/` 下。

### 3.1 `boundflow/ir/` — 验证感知中间表示

| 文件 | 说明 |
|------|------|
| `primal.py` | Primal IR（Node/Value/TensorType），`BFPrimalGraph.validate()` |
| `bound.py` | Bound Graph 抽象（BFBoundGraph, ApplyTransformer, DomainState） |
| `task.py` | Task IR（TaskKind/TaskOp/BoundTask/BFTaskModule/StoragePlan） |
| `task_graph.py` | TaskGraph DAG 调度（topo_sort, reachability, cycle detection） |
| `spec.py` | LinearSpec 验证性质表示 |
| `liveness.py` | Liveness 分析（BufferLifetime, buffer reuse key） |

### 3.2 `boundflow/planner/` — 全局规划与调度

| 文件 | 说明 |
|------|------|
| `core.py` | PlannerConfig, PlanBundle, PlannerPass protocol |
| `pipeline.py` | 主入口 `plan()`，协调 v0/v2 版本 |
| `interval_v0.py` | 单任务 lowering（整图一个 INTERVAL_IBP task） |
| `interval_v1.py` | LinearSpec 变体 |
| `interval_v2.py` | 多任务分区；构建 TaskGraph |
| `options.py` | 配置枚举：PartitionPolicy, LayoutOptions, LifetimeOptions |
| `storage_reuse.py` | Buffer 复用算法（LIFO/FIFO, liveness-aware） |
| `verify.py` | 后置验证（task 一致性、buffer 引用、storage plan） |
| `instrument.py` | 可观测性（TimingInstrument, VerifyInstrument, DumpPlanInstrument） |
| `passes/` | buffer_reuse_pass, layout_only, liveness_pass |

### 3.3 `boundflow/backends/tvm/` — TVM 代码生成

| 文件 | 说明 |
|------|------|
| `interval_linear.py` | Interval linear TVM module builder |
| `interval_conv2d.py` | Interval conv2d TVM module builder |
| `relax_task_lowering.py` | 统一 Relax IR lowering（RelaxLoweringMode: RELAX_OPS / CALL_TIR） |
| `relax_interval_task_ops.py` | Task 级 Relax 算子实现 |
| `relax_analysis.py` | Relax IR 分析与优化 |
| `relax_interval_linear/conv2d.py` | Relax VM 执行器 |

### 3.4 `boundflow/frontends/` — 模型导入

| 文件 | 说明 |
|------|------|
| `pytorch/frontend.py` | `import_torch()`：torch.export → Primal IR |
| `onnx/frontend.py` | ONNX → Primal IR |
| `normalize.py` | 图规范化（view→reshape, transpose→permute） |

### 3.5 `boundflow/domains/` — 抽象域

| 文件 | 说明 |
|------|------|
| `base.py` | AbstractDomain protocol（affine/relu/elementwise transformer） |
| `interval.py` | IntervalDomain 实现（IBP: linear/conv2d/relu/add/mul） |

### 3.6 `boundflow/runtime/` — 执行器与方法族

| 文件 | 说明 |
|------|------|
| `executor.py` | Executor protocol, PythonInterpreter 兼容层 |
| `task_executor.py` | PythonTaskExecutor（reference IBP 执行器） |
| `tvm_executor.py` | TVMExecutor（TVM 编译执行，compile cache） |
| `scheduler.py` | Task 调度器 |
| `perturbation.py` | 输入扰动（LpBallPerturbation, L-inf/L2/L1, concretize） |
| `crown_ibp.py` | CROWN-IBP（forward IBP + backward linear bounds；支持 structured ReLU/DAG backward 与 layout-only op） |
| `alpha_crown.py` | Alpha-CROWN（可优化 dual variables + warm-start） |
| `alpha_beta_crown.py` | Alpha-Beta-CROWN（beta 编码 split 约束） |
| `bab.py` | Branch-and-Bound（ReluSplitState, branching, node-batch, prune） |
| `linear_operator.py` | LinearOperator 抽象（Dense/Batched/Conv/Reindex/Scaled, 避免显式大张量） |
| `dag_utils.py` | DAG 工具（shape 验证、规范化） |
| `relu_shape_utils.py` | ReLU split broadcasting 与 shape 工具 |

### 3.7 `boundflow/3rdparty/` — 外部依赖（vendored submodule）

| 子模块 | 说明 |
|--------|------|
| `tvm` | TVM 编译器框架 |
| `tvm-ffi` | TVM FFI 库 |
| `auto_LiRPA` | 自动 LiRPA 库（ground truth 对齐） |

**重要**: 这些是 vendored 子模块，除非升级上游版本否则避免编辑。

---

## 4. 环境设置

### 4.1 首次安装

```bash
bash scripts/install_dev.sh
```

6 阶段：Git submodule → Conda env → TVM-FFI → TVM → auto_LiRPA → Hooks。

### 4.2 日常激活

```bash
conda activate boundflow
source env.sh
```

### 4.3 关键环境变量（由 `env.sh` 自动设置）

| 变量 | 说明 |
|------|------|
| `BOUNDFLOW_ROOT` | 仓库根目录 |
| `TVM_HOME` | TVM 子模块路径 |
| `PYTHONPATH` | 包含 TVM, TVM-FFI, auto_LiRPA |
| `TVM_FFI_CACHE_DIR` | 缓存目录（默认 `.cache/tvm-ffi`） |
| `BOUNDFLOW_QUIET=1` | 可抑制 env.sh 诊断输出 |

### 4.4 TVM 增量重编译

```bash
bash scripts/rebuild_tvm.sh
# 注意：重编译后需要重启 Python 进程
```

---

## 5. 常用命令

### 5.1 测试

```bash
pytest tests/test_env.py          # 验证环境和导入
pytest tests                       # 运行所有测试（68 个文件）
pytest -k "phase4"                 # 模式匹配
pytest -n auto                     # 并行执行
```

### 5.2 基准测试

```bash
# Phase 5 artifact pipeline
python scripts/run_phase5d_artifact.py --mode quick --workload all --run-id test

# Phase 6H E2E time-to-verify
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_run

# 消融实验
python scripts/bench_ablation_matrix.py > results.jsonl
python scripts/postprocess_ablation_jsonl.py results.jsonl
```

---

## 6. 关键概念

### 6.1 Task 抽象

- **Task** 表示原子级边界传播操作（如 Linear 层的 interval IBP）
- 由 Planner 调度为 **TaskGraph** (DAG)
- 可降低到 TVM Relax IR 或约束求解器

### 6.2 Storage Planning

- Planner 执行 buffer 复用优化以减少内存占用
- `BufferSpec` 关联 value 与 buffer ID
- `StoragePlan` 映射 buffer 生命周期
- 复用策略: LIFO/FIFO, liveness-aware, scope/dtype/layout 匹配

### 6.3 Benchmark JSONL Schema

- 性能基准输出 JSONL (JSON Lines) 到 stdout
- **stdout**: 仅 JSONL payload；**stderr**: 诊断信息
- Phase 5 schema: `schema_version=1.0`（已冻结）
- Phase 6 schema: `phase6h_e2e_v2`（含 p90/p99/timeout/speedup）
- 详见 [docs/bench_jsonl_schema.md](bench_jsonl_schema.md)

### 6.4 方法族（Verification Method Family）

```
IBP → CROWN-IBP → Alpha-CROWN → Alpha-Beta-CROWN → BaB
```

三轴解耦设计（避免组合爆炸）：
- **Axis A**: PerturbationSet（L-inf/L2/L1）
- **Axis B**: BoundMethod/DomainState（IBP/CROWN/Alpha-Beta pipeline）
- **Axis C**: Solver（BaB 控制流留在 Python, 重计算推到后端）

---

## 7. 开发规范

### 7.1 代码风格

- Python 3.10+，类型标注
- 4 空格缩进，`snake_case`（函数/模块），`PascalCase`（类），`UPPER_SNAKE`（常量）
- IR 容器优先用 dataclass（参考 `boundflow/ir/task.py`）
- 格式化: `black`；类型检查: `mypy`；lint: `pylint`
- Docstring 简洁且语义化

### 7.2 测试规范

- 测试文件: `test_*.py` in `tests/`
- 数值检查使用确定性种子
- 新增 CLI 标志或环境行为需添加 smoke test
- 覆盖 happy path 和 failure path（尤其 planner/runtime 接口）

### 7.3 提交规范

- 使用 Conventional Commits（如 `feat:`, `fix:`, `docs:`, `chore:`）
- 一次提交一个逻辑变更
- 第三方依赖更新需注明上游 commit SHA

### 7.4 PR 规范

- 包含：摘要、关联 issue、验证命令（`pytest`、`rebuild_tvm.sh`）、性能/覆盖影响
- 截图仅用于用户面向输出的变更

### 7.5 安全

- 不提交凭证或本地路径
- `env.sh` 是机器特定配置的正确位置
- 修改 `PYTHONPATH`/`TVM_HOME` 后验证 `tests/test_env.py`

---

## 8. 文档体系

### 8.1 `docs/`（稳定、面向用户）

| 文件 | 说明 |
|------|------|
| `reference.md` | 本文档（统一技术参考） |
| `bench_jsonl_schema.md` | JSONL schema 规范（schema_version=1.0） |
| `phase5_done.md` | Phase 5 完成声明（复现入口、产物结构、DoD） |
| `change_log.md` | 变更总账 |

### 8.2 `gemini_doc/`（研发过程记录）

| 文件 | 说明 |
|------|------|
| `README.md` | 目录导引（索引、阅读路径、维护规则） |
| `project_evolution_overview.md` | 研发脉络总览（7 阶段演进） |
| `llm_briefing_boundflow.md` | LLM 交接文档（问题-背景-解决方案-现状） |
| `phase0~6_summary.md` | 各阶段总结 |
| `artifact_claims_phase5d.md` | 证据链/口径映射 |
| `artifact_appendix_phase5d.md` | 复现说明 |
| `llm_collaboration_workflow.md` | LLM 协作工作流模板 |
| `bound_methods_and_solvers_design.md` | 方法族三轴解耦设计 |
| `tvm_backend_optimization_memo.md` | TVM 后端优化备忘 |
| `change_YYYY-MM-DD_*.md` | 变更记录（150+ 份） |

维护规则：不移动/改名历史 `change_*.md`。详见 `gemini_doc/README.md`。

---

## 9. 可扩展性

系统设计考虑未来扩展：

1. **可插拔抽象域**: 支持 DeepPoly、Zonotope 等（AbstractDomain protocol 已定义）
2. **完备求解器后端**: MIP/MILP、SMT 集成（Gurobi、AlphaBeta-CROWN）
3. **量化神经网络 (QNN)**: 定点和位精确语义的一等公民支持

---

## 10. 阶段演进概要

| 阶段 | 核心目标 | 状态 |
|------|---------|------|
| Phase 0 | 工程止血（editable install, 包结构清理, smoke test） | 完成 |
| Phase 1 | IR 地基（Node/Value 双层 Primal IR, validate） | 完成 |
| Phase 2 | Torch 前端（torch.export → Primal IR, normalize） | 完成 |
| Phase 3 | 正确性基线（Python IBP reference, auto_LiRPA 对齐） | 完成 |
| Phase 4 | 系统骨架闭环（Task/Planner/Executor, TVM demo, ONNX） | 完成 |
| Phase 5 | 论文/AE 工件化（JSONL schema, artifact pipeline） | 完成 |
| Phase 6 | 方法族落地 + 收益归因（CROWN/Alpha/BaB + E2E bench） | 完成 |
| Phase 7A | CNN/DAG 扩展 + LinearOperator + operator-preserving backward | **进行中** |

详见 [gemini_doc/project_evolution_overview.md](../gemini_doc/project_evolution_overview.md)。
