# BoundFlow

**BoundFlow** 是一个验证感知的编译器和运行时系统，将 **LiRPA/CROWN 边界传播** 和 **Certified Training** 作为一等公民工作负载。通过专用 IR、全局规划器和 TVM 后端，系统化地消除基于 Python 的验证器中的开销（碎片化、同步、冗余）。

**核心目标**: 在相同验证强度下，减少端到端验证时间，增加可验证规模（更大模型、更多规格、更深 BaB）。

---

## 架构

```
PyTorch/ONNX → BoundFlow IR → Planner → BoundTasks → TVM Backend → GPU Kernels
```

- **前端**: 把 PyTorch/ONNX 模型导入为统一的 Primal IR
- **Planner**: 全局规划（分区、buffer 复用、liveness、layout 优化）
- **后端**: TVM Relax 代码生成 → 优化 GPU Kernels
- **运行时**: 双执行器（Python reference + TVM compiled）+ 方法族实现

---

## 当前能力（Phase 7A）

### 验证方法族

| 方法 | 说明 |
|------|------|
| IBP | Interval Bound Propagation（最快，较松） |
| CROWN-IBP | Forward IBP + Backward Linear Bounds |
| Alpha-CROWN | 可优化 dual variables + warm-start |
| Alpha-Beta-CROWN | Beta 编码 split 约束 |
| BaB | Branch-and-Bound（node-batch, cache, prune, 完备验证） |

### 模型支持

- **MLP**: Linear + ReLU 链式网络
- **CNN**: Conv2d + ReLU 链式网络
- **General DAG**: 残差连接、concat、多分支结构

### 系统特性

- **LinearOperator 抽象**: 避免显式大张量构造，支持 Dense/Batched/Conv 变体
- **Structured shared CROWN**: DAG backward、ReLU backward 与 layout-only `reshape/permute/transpose` 保持结构化路径
- **Task IR + TaskGraph**: DAG 调度与依赖管理
- **Planner 多趟优化**: Liveness → Buffer Reuse → Layout → Fusion
- **TVM/Relax 编译后端**: Compile cache, compile/run 拆分口径
- **JSONL + Artifact Pipeline**: 可复现的基准测试与证据链

### 输入扰动

- L-inf, L2, L1 球扰动（统一 concretize 接口）

---

## 快速开始

### 环境安装

```bash
# 首次安装（含子模块、conda env、TVM 编译）
bash scripts/install_dev.sh

# 日常激活
conda activate boundflow
source env.sh
```

### 运行测试

```bash
pytest tests/test_env.py    # 环境验证
pytest tests                 # 全部测试（68 个文件）
pytest -k "phase6"           # 按阶段筛选
```

### 最小使用示例

```python
import torch
from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.task_executor import PythonTaskExecutor
from boundflow.runtime.perturbation import LpBallPerturbation

# 导入模型
model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2))
program = import_torch(model, (torch.randn(1, 2),))

# 规划与执行
task_module = plan_interval_ibp_v0(program)
x0 = torch.randn(1, 2)
spec = LpBallPerturbation(center=x0, eps=0.1, norm=float("inf"))
result = PythonTaskExecutor().run_ibp(task_module, spec)
print(f"Lower: {result.lower}, Upper: {result.upper}")
```

---

## 目录结构

```
boundflow/
  ir/              验证感知中间表示（Primal/Task/TaskGraph/Spec/Liveness）
  planner/         全局规划与调度（Pipeline/StorageReuse/Passes）
  backends/tvm/    代码生成后端（Relax IR → TIR → Kernels）
  frontends/       模型导入（PyTorch, ONNX）
  domains/         抽象域（Interval/IBP, AbstractDomain protocol）
  runtime/         执行器 + 方法族（CROWN/Alpha/BaB/LinearOperator）
  3rdparty/        外部依赖（TVM, TVM-FFI, auto_LiRPA）
tests/             68 个测试文件（按 Phase 组织）
scripts/           21 个基准/工具脚本
docs/              稳定技术文档
gemini_doc/        研发过程文档（150+ 变更记录）
```

---

## 阶段演进

| 阶段 | 核心目标 | 状态 |
|------|---------|------|
| Phase 0 | 工程止血（editable install, 包结构, smoke test） | 完成 |
| Phase 1 | IR 地基（Node/Value Primal IR, validate） | 完成 |
| Phase 2 | Torch 前端（torch.export → Primal IR） | 完成 |
| Phase 3 | 正确性基线（IBP reference, auto_LiRPA 对齐） | 完成 |
| Phase 4 | 系统骨架闭环（Task/Planner/Executor, TVM, ONNX） | 完成 |
| Phase 5 | 论文/AE 工件化（JSONL schema 1.0, artifact pipeline） | 完成 |
| Phase 6 | 方法族落地（CROWN/Alpha/BaB + E2E 收益归因） | 完成 |
| Phase 7A | CNN/DAG + LinearOperator + operator-preserving backward | **进行中** |

---

## 可扩展性

系统设计为未来扩展预留接口：

1. **可插拔抽象域**: DeepPoly、Zonotope 等（AbstractDomain protocol 已定义）
2. **完备求解器后端**: MIP/MILP、SMT 集成（Gurobi、AlphaBeta-CROWN）
3. **量化神经网络 (QNN)**: 定点和位精确语义的一等公民支持

---

## 文档导航

| 文档 | 说明 |
|------|------|
| [docs/reference.md](docs/reference.md) | 统一技术参考手册（架构/模块/环境/规范） |
| [gemini_doc/README.md](gemini_doc/README.md) | 研发文档索引与阅读路径 |
| [gemini_doc/llm_briefing_boundflow.md](gemini_doc/llm_briefing_boundflow.md) | LLM 交接文档（问题-背景-解决方案-现状） |
| [gemini_doc/project_evolution_overview.md](gemini_doc/project_evolution_overview.md) | 研发脉络总览（7 阶段演进详解） |
| [docs/bench_jsonl_schema.md](docs/bench_jsonl_schema.md) | JSONL schema 规范 |
| [docs/phase5_done.md](docs/phase5_done.md) | Phase 5 完成声明 |
| [gemini_doc/phase6_summary.md](gemini_doc/phase6_summary.md) | Phase 6 总结 |
