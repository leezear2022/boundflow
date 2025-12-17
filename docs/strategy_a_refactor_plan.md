# BoundFlow 策略 A：在现有仓库基础上重构与补齐（计划文档）

## 1. 背景与动机

BoundFlow 的目标不是“再写一个 verifier 算法库”，而是把 **LiRPA/CROWN 边界传播**（以及可选的 **BaB complete verification**、**certified training**）当成一等公民工作负载，通过 **验证感知 IR + 全局 Planner + TVM(TIR) 后端 + 运行时**，系统性解决现有 Python/算子库式实现的性能问题：

- kernel 颗粒度碎片（launch 多、带宽利用差）
- CPU↔GPU 同步频繁（BaB/多 spec 调度拖吞吐）
- 多 spec / 多 BaB 节点的重复前缀计算难复用

当前仓库是“按计划自动生成的骨架”，目录边界基本合理，但缺少端到端可运行链路，且存在会阻碍后续迭代的结构性问题（包结构重复、IR 语义不清、安装假设过强、测试不足）。因此采纳 **策略 A：在现有仓库基础上改** —— 保留可取的目录/脚本/依赖组织，集中重构关键 P0，并尽快跑通最小闭环，用可运行结果验证路线。

## 2. 指导思想（我们要做什么、不做什么）

### 2.1 核心想法（不变）

- **三层语义分离**：Primal（模型语义）→ Bound（抽象域语义）→ Task（可执行任务，TVM Relax/TIR 承载）
- **Node/Value 分离的 IR 形态**：IR 显式区分 Operation（Node）与 Data（Value），Value 承载 shape/dtype/布局/生命周期等元信息，便于后续内存规划与融合（参考 `torch.fx` / StableHLO 的风格）
- **少量 primitive + transformer 插件化**：前端尽量把复杂算子分解到少数 primitive；域只需实现这些 primitive 的 transformer
- **Planner/后端以任务为中心**：先跑通“正确性与闭环”，再逐步引入 fusion/batching/reuse 的优化策略
- **统一 Executor 接口**：Runtime 以统一 `Executor` 抽象承载执行；`PythonInterpreter` 作为可调试的 fallback backend，`TVMExecutor` 作为优化后端
- **TVM 集成优先拥抱 Relax**：Task 的高层编排（分配/调用/串接）统一用 Relax 表达，优先复用 TVM Relax VM 运行时，而不是手写胶水去拼接 TIR 调用

### 2.2 v0.1 版本范围（明确收敛）

v0.1 的目标是：**从 Torch/ONNX 导入一个小模型，做 Interval bound propagation，生成一个可执行 Task，并能跑通测试**。

非目标（v0.1 不做）：
- 覆盖所有 PyTorch/ONNX 算子（优先 VNN-COMP 风格 CNN/MLP 子集）
- 动态 shape（默认静态 shape）
- complete BaB runtime（只预留接口与数据结构，不追求可用）
- TVM autotune / 高性能调度（先用“可编译、可运行”的最小 lowering）

## 3. 现状评估（为什么需要重构）

### 3.1 正向：骨架方向基本对
- 目录划分与计划一致：`boundflow/ir`, `boundflow/frontends`, `boundflow/domains`, `boundflow/planner`, `boundflow/backends`, `boundflow/runtime`
- 第三方依赖在 `boundflow/3rdparty/`，脚本在 `scripts/`，测试在 `tests/`

### 3.2 关键问题（阻碍后续实现）

1) **包结构重复/歧义**
- 存在 `boundflow/boundflow/domains/` 的空文件，与 `boundflow/domains/` 重叠，容易造成 import 与重构混乱。

2) **IR 语义不足以支撑实现**
- Primal IR 当前用 `inputs/outputs` 表示“node 名”，但更合理的是表示 **value/tensor 名**（SSA 风格、常量/参数、多输出等都需要 value 级别标识）。
- 进一步地，建议在 IR 层显式分离 `Node`（算子）与 `Value`（边/张量），把 shape/dtype/布局等元数据挂到 Value 上，为 planner 的 liveness/memory planning 预留位置。
- 缺少一致性校验与构建器：`node_map` 可能与 `nodes` 不一致，BoundGraph/Task 也缺少约束。

3) **端到端链路缺失**
- Torch/ONNX 前端与 normalize 目前为占位符；planner/backend/runtime 基本为空。

4) **安装假设过强**
- 安装脚本强制启用 CUDA/LLVM，Conda 环境固定 `pytorch-cuda=12.1`，对不同机器（CPU-only/不同 CUDA）不友好。

5) **测试不足**
- 只有 import smoke test，缺少对 IR、前端导入、域 transformer 的单元测试，无法约束迭代质量。

## 4. v0.1 目标与验收标准（可执行的“完成定义”）

### 4.1 功能目标
- Torch 前端：`torch.export` 导入一个小 MLP/CNN，生成非空 `BFPrimalProgram`
- ONNX 前端：读取 `.onnx` + shape inference（能导入至少 1 个 toy 模型）
- Normalizer：将一小组算子规范化到 primitive op-set（至少覆盖 Linear/Conv/ReLU/Add/Reshape）
- Interval 域：实现 `affine/relu/elementwise` 的 transformer（正确性优先）
- Spec 定义清晰：至少区分 **输入约束（Input Constraints）** 与 **待验证性质（Property / Output Constraints）**
- 执行闭环：给定 spec，能输出最终输出层的 (l,u) 或 margin bound（先以 `PythonInterpreter` 跑通，TVM 作为可选加速路径）

### 4.2 工程目标
- BoundFlow 可 `pip install -e .`（减少依赖 PYTHONPATH）
- `pytest tests` 至少包含：前端导入 + interval transformer 正确性（数值可重复）
- 安装脚本支持 CPU-only 与 CUDA 可选配置（不强制写死）

## 5. 修改计划（策略 A 的具体执行路径）

下面按“先止血、再闭环、再扩展”的顺序列计划。每个阶段都给出产出物与验收点，避免做成“无限期大重构”。

### Phase 0：止血与工程化整理（P0）

**目标**：消除包结构歧义，建立可依赖的开发/测试入口。

- [ ] 清理重复包路径：移除/合并 `boundflow/boundflow/` 之类的重复目录，仅保留一个 `boundflow/` 包根
- [ ] 增加标准打包：补 `pyproject.toml`（或 `setup.cfg`）使 `pip install -e .` 成为主路径
- [ ] 明确 `env.sh` 定位：保留为“开发便利”，但不再作为唯一运行方式；hooks 仍可保留

**验收**：新建 conda env 后，执行 `pip install -e .` + `pytest -q tests/test_env.py` 通过。

### Phase 1：IR 语义加固（P0）

**目标**：把 IR 从“dataclass 草图”升级为可实现/可校验的数据结构，避免后续 planner/runtime 反复返工。

- [ ] 将 Primal IR 的边从“node 名”统一为“value 名”（tensor/SSA value）
- [ ] 引入 `Node`/`Value` 双层结构：
  - `Node` 表示计算（op_type + attrs + 输入/输出 Value 引用）
  - `Value` 表示数据（shape/dtype/可选 layout、是否为 param/const、debug 信息）
- [ ] 添加构建与校验工具：
  - 唯一命名、拓扑顺序（或显式依赖）
  - `node_map`/`value_meta` 一致性
  - 输入输出 value 的存在性检查
- [ ] 明确 `Spec` schema（至少：Input Constraints + Property；v0.1 可先覆盖鲁棒性分类的常见形式）
- [ ] 明确 `DomainState` 的最小接口与可序列化字段（v0.1 只实现 IntervalState）
- [ ] 定义统一 `Executor` 抽象（接口先定、实现后补），为 `PythonInterpreter/TVMExecutor` 预留同一运行入口

**验收**：对一个手写 toy graph，IR 校验能捕获常见错误（重复名/悬空输入/缺 meta）。

### Phase 2：前端导入 + normalize（P0→P1）

**目标**：跑通“模型 → Primal IR”，并把算子集收敛到可支持的 primitives。

- [ ] TorchFrontend：
  - 使用 `torch.export.export()` 获取图与参数
  - 转换到 Primal IR（value 名、shape/dtype meta、params/const）
- [ ] ONNXFrontend：
  - `onnx.load` + `onnx.shape_inference.infer_shapes`
  - 初步映射到 Primal IR（覆盖 toy 模型）
- [ ] Normalizer：
  - Linear → MatMul + Add（或统一成 Affine）
  - 规范 attrs（stride/pad/dilation/groups 等）
  - 明确不支持算子的报错与 fallback 规则

**验收**：导入一个小 MLP，Primal IR 非空且 meta 完整；normalize 后 op_type 全在允许集合内。

### Phase 3：Interval 域 + 最小 bound propagation（P0→P1）

**目标**：实现可验证的“正确性闭环”，先不追求性能。

- [ ] 实现 `IntervalState(l, u)` 与 `AbstractDomain` 的 interval 版本
- [ ] 覆盖 primitives：
  - affine（Linear/Conv）
  - relu
  - elementwise add/mul（先做 add，mul 可选）
- [ ] 绑定 `Spec`：把输入 perturbation 初始化成 IntervalState
- [ ] `PythonInterpreter`（reference executor）：作为 Runtime 的一种 fallback backend，不依赖 TVM，直接用 torch/numpy 执行 bound graph，既用于 ground truth/回归，也用于调试新算子

**验收**：对小网络（随机权重、固定 seed），interval 输出满足：对采样扰动点的真实输出均落在 (l,u) 内。

### Phase 4：Task 生成与 TVM 后端“最小可编译”（P1）

**目标**：把 Phase 3 的 reference executor 逐步替换为 Task 承载，打通“IR → Task → 执行”的系统路径。

- [ ] 定义最小 `BoundTask` schema（输入/输出 state、batch 轴占位、memory_plan 占位）
- [ ] 顶层结构从一开始就支持 Multi-Task：
  - `BFTaskModule` 包含多个 `BoundTask`
  - 提供一个明确的 entry/main（即使 v0 只是顺序执行），避免未来引入 BaB/分支后重构顶层架构
- [ ] Planner v0：先做“整图一个任务”（不做优化），保证接口正确
- [ ] Backend v0：
  - 优先选择 1–2 个 kernel 模板（例如 affine interval 的核心公式）作为 TVM TIR lowering demo
  - Relax 作为 orchestration（v0.1 即采用），用 Relax 表达执行流并复用 TVM Relax VM
  - 避免手工拼接 TIR 调用：优先走 Relax → `call_tir` 的标准路径，降低 Runtime 复杂度
- [ ] `TVMExecutor`：提供 `run(task_module, inputs, spec)` 的统一入口（与 `PythonInterpreter` 同接口）

**验收**：相同输入/模型下，TVM 路径输出与 reference executor 一致（允许小数误差）。

### Phase 5：测试与脚本增强（P1）

**目标**：用测试与安装选项把工程“钉住”，后续才能安全迭代 planner/优化。

- [ ] 增加 pytest：
  - `test_torch_frontend_import`
  - `test_normalize_primitives`
  - `test_interval_transformers`
  - `test_end_to_end_interval`（可选：CPU-only 版本）
- [ ] 安装脚本参数化：
  - CPU-only / CUDA 可选
  - 不强制修改 `config.cmake`（用模板/显式参数覆盖）
  - 失败时打印更明确的诊断信息（TVM_HOME、lib 路径等）

**验收**：在 CPU-only 环境下至少能跑通 reference executor 的端到端测试；GPU/TVM 路径在具备环境时可选通过。

## 6. 风险与对策

- **IR 语义返工风险**：最先做 Phase 1，并用测试锁住 value 命名与 meta schema，避免 planner/runtime 大面积重写。
- **TVM 接入复杂度**：先有 `PythonInterpreter` 保底；TVM 后端以“最小可编译 demo”逐步替换，不阻塞整体闭环；优先复用 Relax VM 而不是自研运行时胶水。
- **环境不一致**：脚本参数化 + 文档明确 CPU-only 路径，避免安装被 CUDA 版本绑死。

## 7. 里程碑摘要（建议作为每周验收）

- M0：包结构清理 + 可 `pip install -e .` + smoke test 通过
- M1：IR 语义加固 + toy graph 校验测试
- M2：Torch 导入 + normalize 覆盖小 MLP
- M3：Interval reference executor 端到端跑通 + 基础正确性测试
- M4：Planner/Task v0 + TVM lowering demo（可选）与对齐测试

---

如你确认这个计划没问题，下一步我建议先做 **Phase 0 + Phase 1**（两步完成后，后面的实现会顺很多）。
