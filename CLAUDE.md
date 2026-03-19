# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 约定

- **全程用中文交流**（除非用户明确要求英文）
- **Conda 环境**: `boundflow`
- **文档存放**: 所有生成的文档、变更记录、备忘录都存放在 `gemini_doc/` 目录中
- **文档组织**: 可以在 `gemini_doc/` 下新建子文件夹进行分类管理
- **每次修改都要记录**: 写一个 change log 文档记录修改内容

## 项目概述

**BoundFlow** 是一个验证感知的编译器和运行时系统，将 **LiRPA/CROWN 边界传播** 和 **Certified Training** 作为一等公民。通过专用 IR、全局规划器和 TVM 后端，系统化地消除基于 Python 的验证器中的开销（碎片化、同步、冗余）。

**核心目标**: 在相同验证强度下，减少端到端验证时间，增加可验证规模（更大模型、更多规格、更深 BaB）。

## 环境设置

```bash
# 完整安装（首次或更新子模块）
bash scripts/install_dev.sh

# 激活环境
conda activate boundflow
source env.sh

# 修改 TVM C++ 代码后增量重新编译
bash scripts/rebuild_tvm.sh
# 注意：重新编译后需要重启 Python 进程
```

关键环境变量（由 `env.sh` 自动设置）:
- `BOUNDFLOW_ROOT`: 仓库根目录
- `TVM_HOME`: TVM 子模块路径
- `PYTHONPATH`: 包含 TVM, TVM-FFI, auto_LiRPA
- `TVM_FFI_CACHE_DIR`: 缓存目录（默认 `.cache/tvm-ffi`）

## 常用命令

### 测试
```bash
pytest tests/test_env.py          # 验证环境和导入
pytest tests                       # 运行所有测试
pytest -k "phase4"                 # 模式匹配
pytest -n auto                     # 并行执行
```

### 基准测试
```bash
python scripts/run_phase5d_artifact.py                   # 完整 artifact 流程
python scripts/bench_ablation_matrix.py > results.jsonl  # 消融实验（输出 JSONL）
python scripts/postprocess_ablation_jsonl.py results.jsonl  # 后处理结果
```

## 架构概览

```
PyTorch/ONNX → BoundFlow IR → Planner → BoundTasks → TVM Backend → GPU Kernels
```

### 核心模块

- **`boundflow/ir/`**: Primal 图、Bound 图、Task 抽象、TaskGraph、Spec
- **`boundflow/planner/`**: PlannerConfig、Pipeline、Storage Reuse、多趟优化 Pass
- **`boundflow/backends/tvm/`**: TVM 代码生成（Relax IR → TIR → Kernels）
- **`boundflow/frontends/`**: PyTorch/ONNX 导入、模型标准化
- **`boundflow/domains/`**: Interval Domain (IBP/CROWN)、抽象域接口
- **`boundflow/runtime/`**: Executor、Scheduler、Python/TVM 执行器

### 第三方依赖

位于 `boundflow/3rdparty/`（作为子模块）:
- `tvm`: TVM 编译器框架
- `tvm-ffi`: TVM FFI 库
- `auto_LiRPA`: 自动 LiRPA 库

**重要**: 这些是 vendored 子模块，除非升级上游版本否则避免编辑。

## 关键概念

### Task 抽象
- Task 表示原子级边界传播操作（如 Linear 层的 interval IBP）
- 由 Planner 调度为 TaskGraph (DAG)
- 可降低到 TVM Relax IR 或约束求解器

### Storage Planning
- Planner 执行 buffer 复用优化以减少内存占用
- `BufferSpec` 关联 value 与 buffer ID
- `StoragePlan` 映射 buffer 生命周期

### Benchmark JSONL Schema
- 性能基准输出 JSONL (JSON Lines) 到 stdout
- **stdout**: 仅 JSONL payload；**stderr**: 诊断信息
- 详见 [docs/bench_jsonl_schema.md](docs/bench_jsonl_schema.md)
- 环境变量 `BOUNDFLOW_QUIET=1` 可抑制 env.sh 输出

## 开发规范

### 代码风格
- Python 3.10+，类型标注
- 4 空格缩进，`snake_case` (函数/模块)，`PascalCase` (类)
- IR 容器优先用 dataclass（参考 [boundflow/ir/task.py](boundflow/ir/task.py)）
- 格式化: `black`；类型检查: `mypy`

### 测试规范
- 测试文件: `test_*.py` in [tests/](tests/)
- 数值检查使用确定性种子
- 新增 CLI 标志或环境行为需添加 smoke test

### 提交规范
- 使用 Conventional Commits (如 `feat:`, `fix:`)
- 一次提交一个逻辑变更
- 第三方依赖更新需注明上游 commit SHA

## 重要文档

### 入口索引
- **gemini_doc/README.md**: gemini_doc 目录导引（文档索引、阅读路径、维护规则）
- **AGENTS.md**: 仓库指南和约定（包含中文工作流备注）
- **docs/phase5_done.md**: Phase 5 完成声明（复现入口、产物结构、DoD）

### 核心文档
- **docs/bench_jsonl_schema.md**: JSONL schema 规范（schema_version=1.0）
- **gemini_doc/artifact_claims_phase5d.md**: Artifact claims（证据链/口径映射）
- **gemini_doc/artifact_appendix_phase5d.md**: Artifact appendix（复现说明）
- **gemini_doc/llm_collaboration_workflow.md**: LLM 协作工作流模板
- **gemini_doc/tvm_backend_optimization_memo.md**: TVM 后端优化备忘

## 可扩展性

系统设计考虑未来扩展:
1. **可插拔抽象域**: 支持 DeepPoly、Zonotope 等
2. **完备求解器后端**: MIP/MILP、SMT 集成（Gurobi、AlphaBeta-CROWN）
3. **量化神经网络 (QNN)**: 定点和位精确语义的一等公民支持
