# BoundFlow LLM 交接文档

> 本文档用于让任何大模型或新合作者快速理解 BoundFlow 项目的全貌：问题、背景、解决方案和现状。
>
> 最后更新: 2026-03-29 | 当前前沿: Phase 7A（structured ReLU backward + layout-only shared CROWN）

---

## 1. 问题（Problem）

### 1.1 现有验证工具的系统性瓶颈

神经网络形式验证（Certified Robustness）领域目前的主流工具（auto_LiRPA、AlphaBeta-CROWN 等）本质上是 **Python 层面的库式实现**。这带来三类系统性开销：

- **Kernel 碎片化**: 每一层的边界传播是独立的 PyTorch 算子调用，GPU kernel launch 开销累积显著
- **同步开销**: Python ↔ GPU 之间频繁同步，无法流水线化
- **冗余计算**: 没有全局 buffer 复用和生命周期分析，中间结果反复分配/回收；无法在验证任务间共享计算

### 1.2 核心矛盾

验证算法研究者写的是 Python 库，但验证工作负载的本质是 **编译器 + 运行时问题**：

- 需要全局可见性（跨层优化、buffer 复用、任务调度）
- 需要后端编译能力（算子融合、代码生成、autotuning）
- 需要可审计的性能归因（而非黑盒 end-to-end timing）

二者之间缺一个 **系统层**——这就是 BoundFlow 要填补的空白。

---

## 2. 背景（Background）

### 2.1 LiRPA/CROWN 方法族

BoundFlow 处理的验证方法族来自 LiRPA 体系，按紧度递增排列：

| 方法 | 原理 | 紧度 | 计算代价 |
|------|------|------|---------|
| **IBP** | 逐层区间传播（每层独立用正负权重分解） | 最松 | O(n) 前向 |
| **CROWN** | 后向线性界传播（把非线性用线性松弛替代） | 较紧 | O(n) 前向 + O(n) 后向 |
| **CROWN-IBP** | Forward IBP + Backward CROWN 组合 | 更紧 | IBP + CROWN |
| **Alpha-CROWN** | CROWN + 可优化 dual variables（α 参数） | 再紧 | 多步梯度优化 |
| **Alpha-Beta-CROWN** | α + β 编码 ReLU split 约束 | 最紧（不完备） | 多步联合优化 |
| **BaB** | Branch-and-Bound: 分裂不稳定 ReLU + 递归验证 | 完备 | 指数但可剪枝 |

### 2.2 学术定位

BoundFlow 是 **ASPLOS 风格的 system contribution**：

- **不发明新验证算法**: IBP/CROWN/BaB 都是已有成熟方法
- **把已有方法系统化**: 通过验证感知 IR、全局 Planner、TVM 后端、可复现工件链，让这些方法在同一框架里高效、可归因地运行
- **证据链闭合**: 每个性能声明都有 JSONL schema → bench script → postprocess → artifact → MANIFEST 的可审计路径

### 2.3 技术栈选择

| 层 | 选择 | 原因 |
|----|------|------|
| 前端 | 复用 auto_LiRPA 的模型导入逻辑 | 避免重写自动推导 |
| IR | 自定义 Primal IR + Task IR | 需要验证感知的语义（bound state, spec, liveness） |
| Planner | 自研 | 核心贡献：全局可见性 + 多趟优化 |
| 后端 | TVM (Relax IR → TIR → Kernels) | 成熟编译栈，支持 autotuning |
| Runtime | Python driver + TVM compiled | BaB 控制流留 Python，重计算推后端 |
| 对齐基线 | auto_LiRPA `compute_bounds()` | 黄金参考，每个方法都与之对齐 |

---

## 3. 解决方案（Solution）

### 3.1 端到端流水线

```
PyTorch/ONNX
    ↓ import_torch() / import_onnx()
BFPrimalProgram（规范化的模型图）
    ↓ plan_interval_ibp_v0/v2()
BFTaskModule（Task IR + TaskGraph + StoragePlan）
    ↓ PythonTaskExecutor / TVMExecutor
IntervalState / CrownState（验证边界结果）
    ↓ (BaB driver)
验证判定（verified / counterexample / timeout）
```

### 3.2 IR 设计

BoundFlow 有两层 IR：

**Primal IR**（`boundflow/ir/primal.py`）:
- Node/Value 双层结构，承载模型拓扑
- TensorType 记录 shape/dtype
- `validate()` 保证图的结构一致性
- 作为前端到 Planner 的统一输入

**Task IR**（`boundflow/ir/task.py`）:
- **TaskOp**: 最小可执行验证算子（linear_ibp, conv2d_ibp, relu_ibp...）
- **BoundTask**: 一组 TaskOp 的容器，带 input/output contract
- **BFTaskModule**: 封装可执行任务 + StoragePlan + 可选 TaskGraph
- **StoragePlan**: value → buffer 映射，支持 aliasing/复用
- **TaskGraph**: 任务间 DAG 依赖，支持 topo_sort + reachability

### 3.3 Planner

Planner 是 BoundFlow 的核心系统贡献，负责从"做什么"到"怎么执行"的转换：

**多版本规划器**:
- **v0**: 整图打包为单个 INTERVAL_IBP task（最简单，正确性保证）
- **v2**: 多任务分区，尊重 layout-only 边界，构建 TaskGraph

**多趟优化 Pass**:
1. Liveness Analysis → BufferLifetime
2. Buffer Reuse（LIFO/FIFO, scope/dtype/layout 匹配）
3. Layout-only 简化
4. Verification（后置一致性检查）
5. Instrumentation（timing, dump, verify）

**配置**: `PlannerConfig` 控制优化开关、TVM target、debug 选项。

### 3.4 Runtime

**双执行器**:
- `PythonTaskExecutor`: Reference backend，保证正确性
- `TVMExecutor`: TVM 编译执行，compile cache + pass timing

**方法族实现**（`boundflow/runtime/`）:
- `perturbation.py`: 输入扰动抽象（L-inf/L2/L1, concretize）
- `crown_ibp.py`: CROWN-IBP（forward IBP + backward linear bounds）
- `alpha_crown.py`: Alpha-CROWN（K-step 优化 + warm-start）
- `alpha_beta_crown.py`: Alpha-Beta-CROWN（beta 编码 split 约束 + 可行性检测）
- `bab.py`: BaB driver（ReluSplitState, branching strategy, node-batch, prune）
- `linear_operator.py`: LinearOperator 抽象（Dense/Batched/Conv, 避免显式大张量）

**设计原则**: BaB 控制流（queue/branch/prune）留在 Python；ReLU 边界、α/β 优化、batching 推到后端。

### 3.5 TVM 后端

- **Lowering Mode**: RELAX_OPS（推荐）/ CALL_TIR
- **Compile Cache**: 跨进程共享编译产物（on-disk `.so`）
- **可观测性**: Pass timing, IR dump, compile vs run 拆分

### 3.6 工件链（Artifact Pipeline）

BoundFlow 有完整的可复现证据链：

```
Research Claim
  → Artifact Claims Doc（度量什么）
  → Bench Script（确定性执行）
  → JSONL/JSON（schema-versioned 输出）
  → Postprocess（CSV/figure 生成）
  → MANIFEST.txt（SHA256 完整性审计）
```

- Phase 5: `scripts/run_phase5d_artifact.py` → `schema_version=1.0`
- Phase 6H: `scripts/run_phase6h_artifact.sh` → `phase6h_e2e_v2`

---

## 4. 现状（Current Status）

### 4.1 阶段进度

| 阶段 | 核心目标 | 关键能力 | 状态 |
|------|---------|---------|------|
| Phase 0 | 工程止血 | Editable install, 包结构, smoke test | 完成 |
| Phase 1 | IR 地基 | Node/Value Primal IR, `validate()` | 完成 |
| Phase 2 | Torch 前端 | `torch.export` → Primal IR, normalize | 完成 |
| Phase 3 | 正确性基线 | Python IBP reference, auto_LiRPA 对齐(MLP/CNN) | 完成 |
| Phase 4 | 系统骨架 | Task/Planner/Executor, StoragePlan, TVM demo, ONNX | 完成 |
| Phase 5 | 论文工件化 | JSONL schema 1.0, artifact pipeline, baseline 对照 | 完成 |
| Phase 6 | 方法族落地 | CROWN/Alpha/BaB + node-batch + E2E 收益归因 | 完成 |
| Phase 7A | 扩模型/扩表达 | CNN CROWN, DAG backward, LinearOperator, concat/add | **进行中** |

### 4.2 代码规模

| 指标 | 数量 |
|------|------|
| 核心模块 | 7 个（ir, planner, backends, frontends, domains, runtime, 3rdparty） |
| 测试文件 | 68 个 |
| 脚本 | 21 个（bench/artifact/工具） |
| 变更记录 | 150+ 份 |

### 4.3 Phase 7A 当前前沿

已完成 PR-1 到 PR-10，并补齐 shared CROWN 的 layout-only 支持：
- PR-2: LinearOperator backward state（消除显式大张量）
- PR-3: Conv LinearOperator + CNN CROWN-IBP
- PR-5: Alpha-CROWN for CNN
- PR-6: Alpha-Beta-CROWN for CNN
- PR-7: BaB for chain CNN
- PR-8: General DAG frontend + runtime（residual/concat）
- PR-9: Operator-preserving DAG backward（adjoint merge + concat 不再 to_dense）
- PR-10: Structured ReLU backward（sign-split + `ScaledInputLinearOperator` + `RepeatedRowLinearOperator`）
- Layout-only support: shared CROWN 补齐 `reshape` 与 batch-preserving `permute/transpose`

**下一步**: 做性能记录并继续消除 `split_pos_neg()` 在复合 operator 上的剩余 dense 点。
详见 `gemini_doc/next_plan_after_phase7a_pr10.md`。

### 4.4 关键 API 入口

```python
# 1. 导入模型
from boundflow.frontends.pytorch.frontend import import_torch
program = import_torch(model, (example_input,))

# 2. 规划
from boundflow.planner import plan_interval_ibp_v0
task_module = plan_interval_ibp_v0(program)

# 3a. IBP 执行
from boundflow.runtime.task_executor import PythonTaskExecutor
result = PythonTaskExecutor().run_ibp(task_module, perturbation)

# 3b. CROWN-IBP 执行
from boundflow.runtime.crown_ibp import crown_ibp_bounds
result = crown_ibp_bounds(model, x0, eps)

# 3c. BaB 执行
from boundflow.runtime.bab import bab_verify
verdict = bab_verify(model, x0, eps, oracle="alpha_beta")
```

### 4.5 环境与命令速查

```bash
# 首次安装
bash scripts/install_dev.sh

# 日常激活
conda activate boundflow && source env.sh

# 测试
pytest tests/test_env.py      # 环境验证
pytest tests                   # 全部测试

# TVM 重编译
bash scripts/rebuild_tvm.sh   # 之后重启 Python

# Artifact
python scripts/run_phase5d_artifact.py --mode quick --workload all --run-id test
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_run
```

### 4.6 已知限制

1. **算子覆盖**: CROWN/Alpha-Beta/BaB 目前覆盖 Linear + ReLU + Conv2d 子集；更多算子（BatchNorm, MaxPool 等）待补
2. **Sign-split dense 点**: `split_pos_neg()` 在部分复合 operator（如 `RightMatmulLinearOperator`）上仍允许内部精确物化，是 shared CROWN 主路径的下一批性能优化点
3. **TVM lowering**: TVM 编译路径覆盖 interval IBP 子集；CROWN/BaB 尚未下沉到 TVM
4. **扰动类型**: L-inf/L2/L1 已支持；semantic perturbation 等未实现
5. **规模**: 当前以小/中型网络为主（MLP, MNIST CNN）；大规模网络的 scalability 待验证

### 4.7 文档导航

| 目的 | 文档 |
|------|------|
| 统一技术参考 | [docs/reference.md](../docs/reference.md) |
| 文档索引 | [gemini_doc/README.md](README.md) |
| 研发脉络 | [gemini_doc/project_evolution_overview.md](project_evolution_overview.md) |
| Phase 5 完成声明 | [docs/phase5_done.md](../docs/phase5_done.md) |
| Phase 6 总结 | [gemini_doc/phase6_summary.md](phase6_summary.md) |
| 方法族设计 | [gemini_doc/bound_methods_and_solvers_design.md](bound_methods_and_solvers_design.md) |
| JSONL Schema | [docs/bench_jsonl_schema.md](../docs/bench_jsonl_schema.md) |
| LLM 协作流程 | [gemini_doc/llm_collaboration_workflow.md](llm_collaboration_workflow.md) |
| 下一步计划 | [gemini_doc/next_plan_after_phase7a_pr10.md](next_plan_after_phase7a_pr10.md) |

---

## 5. 协作约定

### 5.1 工作流

采用**回合式 PR-by-PR LLM 协作**：

1. 输入目标 + DoD → 2. LLM 产出计划 + 接口 → 3. 实现 + 最小闭合 → 4. 测试修正 → 5. 总结 + 下一 PR 计划

每个 PR 产出：
- 代码实现（feature branch）
- 变更记录（`gemini_doc/change_YYYY-MM-DD_*.md`）
- 总账更新（`docs/change_log.md`）
- 通过测试

### 5.2 入口文件

| 文件 | 消费者 | 内容 |
|------|--------|------|
| `CLAUDE.md` | Claude Code | 专属约定 + 指向 reference.md |
| `AGENTS.md` | Codex/通用 Agent | 专属约定 + 指向 reference.md |
| `GEMINI.md` | Gemini CLI | 专属约定 + 指向 reference.md |
| `docs/reference.md` | 所有人 | 统一技术参考（架构/模块/环境/规范） |

### 5.3 规范

- **语言**: 全程中文交流
- **Conda 环境**: `boundflow`
- **文档**: 存放在 `gemini_doc/`，可新建子文件夹
- **提交**: Conventional Commits（`feat:` / `fix:` / `docs:` / `chore:`）
- **代码**: Python 3.10+, 类型标注, `black` 格式化, `mypy` 类型检查
