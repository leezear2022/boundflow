下面我按**工程交付**来把 Phase 4 的“尾巴”补全成一个可执行计划（你让模型照着生成代码就行），同时每一项都会明确：**它如何为 Phase 5 的 Planner（切分/融合/缓存/批处理）铺路**。

我会引用你们现有方向中最关键的外部接口约束：

* PyTorch 前端基于 `torch.export.export()` 产出 `ExportedProgram`（带 `graph_module/state_dict/range_constraints` 等信息）
* ONNX 前端依赖 shape inference 将推断结果写入 `value_info`
* Spec/Property（C/margin）要对齐 auto_LiRPA 的 `compute_bounds(C=..., method=...)` 语义
* TVM demo 建议走 Relax→`LegalizeOps`→混合 IRModule（Relax+TIR，op 变 `call_tir`），并用 VM 的 `save_function` 降低调用开销做对比/benchmark

---

## Phase 4 总目标（工程视角）

把你现在“IBP correctness 闭环 + Task pipeline + StoragePlan 钉子”推进到：

1. **Task pipeline 支持 Spec/Property（C/margin）并对齐 auto_LiRPA**（Phase 4B.2）
2. **Layout/Transpose 变成可分析、可消除的对象（至少能做保守消除）**（Phase 4B.3）
3. **同一个 BFTaskModule：PythonTaskExecutor 与 TVMExecutor 输出一致**（Phase 4C v0）
4. 额外补齐：**ONNX import 最小闭环**（可选但强烈建议放在 Phase4 末尾，避免 Phase5 前端分叉）

> Phase 5 的 Planner（切分/融合/缓存/批处理）依赖 Phase4 的“语义完整 + lowering 通路完整”。所以 Phase4 的每一项都要“钉住可优化接口”，但不强行实现优化。

---

# Phase 4B：接口加固（在不做“大优化”的前提下，把语义补齐）

## 4B.1（补一刀）：把 “Spec/Property/Layout 元数据”放进 IR 的稳定位置

**目的**：避免 4B.2/4B.3 做完后，信息散落在 executor/planner 临时字段里，Phase5 再返工。

**交付内容**

* `Value/TensorType`（你们 Primal IR 已有）增加最小元信息槽位：

  * `layout: Optional[str]`（如 `"NCHW"|"NHWC"|None"`）
  * `storage_kind: Optional[str]`（cpu/gpu/shared/…先占位）
* `BoundTask` / `TaskOp` 增加 `meta: Dict[str, Any]`（强烈建议统一口子，后续 phase5 直接塞 cost model / tiling hint / cache hint）

**验收**

* 不改语义，仅通过单测验证这些字段序列化/validate 不破坏现有 MLP/CNN IBP 测试。

**对 Phase 5 的连接**

* Phase 5 要做“按 layout 消除 transpose / 选择 fusion group / 选择缓存点”，必须有**稳定挂载点**（meta + layout）。

---

## 4B.2：Spec/Property（C 矩阵 / margin）闭环（对齐 auto_LiRPA）

auto_LiRPA 明确把 `C` 定义为“把模型输出再接一层线性映射到 margins 的 specification matrix”——你们要对齐的就是这个语义。

### 设计建议（工程落地版）

**新增 `SpecIR`（结构化，而不是直接塞 Tensor）**

* `LinfInputSpec(center, eps)`（你已有）
* `LinearSpec(C, rhs=None, sense=">=")`：先只做分类 margin，rhs 可缺省
* 以后扩展：sparse-C / onehot-C / VNNLIB AND-OR（Phase5+）

**Property lowering（关键）**

* 把 `C` 当成 “PropertyLinearLayer” 插到图末端（概念上相当于 `y = C @ logits`），这样：

  * PythonTaskExecutor：最后多执行一次线性 bound
  * TVMExecutor：后续可把它 fuse 到最后一层（Phase5 的优化点）

### 交付内容

* `boundflow/ir/spec.py`（新）：定义 SpecIR
* `planner/interval_v1.py`（或在 v0 基础上扩展）：TaskModule 里携带 SpecIR
* `runtime/task_executor.py`：支持 margin bound 输出（输出 shape 和 auto_LiRPA 对齐）
* 新增测试：

  * `test_phase4b2_margin_against_auto_lirpa.py`：对齐 `compute_bounds(C=..., method="IBP")`

### 验收标准

* 同一模型、同一输入扰动：

  * `BoundFlow(IBP + C)` 输出 lb/ub 与 auto_LiRPA 一致（允许浮点误差）
* 输出接口稳定：返回的不是“裸 tensor”，而是带 `spec_id / property_shape` 元信息（为 Phase5 批处理铺路）

**对 Phase 5 的连接**

* Phase 5 的“multi-spec batching”“C-aware fusion”“property cache key”都建立在 **SpecIR 一等公民** 的前提上。

---

## 4B.3：Layout/Transpose 作为可优化对象（先做“保守消除”）

TVM 的 `LegalizeOps` 会把 Relax op 变 `call_tir`，并进入混合 IRModule。如果你在 Phase4 不把 layout/transpose 在 IR 里规范化，后面很容易把“transpose”固化成不可消除的 kernel 链。

### 交付内容（最小但有实际收益）

1. **LayoutHint 语义**

   * `TensorType.layout` 仅记录“当前张量解释方式”
2. **Normalize Pass（primal 级）**

   * 规则1：相邻 `permute` 互逆 → 消除
   * 规则2：`permute → reshape/flatten → permute` 的可交换情况先不做，但把模式识别打 log（为 Phase5）
3. **TaskOp attrs**

   * 对每个 op 标记输入/输出 layout（只要能解释清楚即可）

### 验收

* 新增一个专门的 layout 单测：构造 `permute -> permute(inv)` 的小图，planner 输出不包含多余 transpose task。
* 现有 MLP/CNN 测试不退化。

**对 Phase 5 的连接**

* Phase 5 做“layout propagation / transpose sinking / fusion”时，直接复用 Phase4 已有 layout pass scaffold，不返工 IR。

---

# Phase 4C：TVM lowering demo（先打通后端，不追求最强性能）

你们的工程判断“不要急着把 orchestration 全塞进 Relax VM”是对的。工程上推荐：

* 先做 **TVMExecutor**：Python 侧驱动、编译并调用 TVM 产物
* 逐步增加覆盖面：先 interval-Linear / interval-Conv / interval-ReLU

TVM 侧关键点：

* Relax 提供 `call_tir` 用来调用 TIR PrimFunc
* `LegalizeOps` 能把 Relax 降成 mixed module（Relax+TIR），op 转 `call_tir`
* VM 的 `save_function` 可减少运行时字典查找开销，利于做准确 benchmark

## 4C v0：TVMExecutor（linear + relu + add 的 IBP）

**交付内容**

* `boundflow/backends/tvm/`（新）

  * `lower_interval_linear.py`：生成 TIR（或 TE）实现 interval linear
  * `lower_interval_relu.py`
  * `compile_task_module.py`：把 BFTaskModule → TVM IRModule → build → runnable
* `runtime/tvm_executor.py`：API 对齐 PythonTaskExecutor
* 新增测试：

  * `test_phase4c_tvmexecutor_matches_python.py`：同 task module，两种 executor 输出一致

**验收**

* 至少 1 个 MLP、1 个 MNIST CNN：

  * PythonTaskExecutor == TVMExecutor（容忍误差）
* 加一个 micro-bench 脚本：用 VM `save_function` 对同一函数做 timing（减少调用开销干扰）

**对 Phase 5 的连接**

* Phase 5 的 Planner 要输出“多个任务 + fusion group + schedule meta”。Phase4C 你先把 **“一个任务如何 lower”**跑通，Phase5 才能把 task 切分/融合接上去。

---

# Phase 4D（建议作为 Phase4 收口）：ONNX import 最小闭环（避免 Phase5 前端分裂）

你之前明确要 PyTorch + ONNX。工程上如果 Phase4 不做 ONNX，你 Phase5 会遇到“两个前端两套 normalize / 两套 op 语义”的雪崩返工。

## 交付内容

* `frontends/onnx/frontend.py`：

  * 读取 ModelProto
  * 调 `onnx.shape_inference.infer_shapes` 把推断 shape 写进 `value_info`
  * 映射到 Primal IR（先支持你当前闭环算子子集）
* 新增测试：

  * 同一小模型（MLP 或小 CNN）：

    * Torch export → Primal IR
    * ONNX → Primal IR
    * 两者在 normalize 后（允许 node 名不同）拓扑等价、shape/dtype 一致

**对 Phase 5 的连接**

* Phase 5 的 Planner 需要假设“输入 Primal IR 语义统一”。ONNX 最小闭环能逼你把 normalize/primitive set 做成真正的单一真源。

---

# 建议你 Phase 4 的执行顺序（务实版）

如果你想尽快进入 Phase5，同时保证不返工，我建议顺序是：

1. **4B.2（Spec/Property）**：先把 verifier 任务表达补齐（对齐 auto_LiRPA 的 C 语义）
2. **4C v0（TVMExecutor）**：先把后端通路打通（call_tir / mixed module / build / run）
3. **4B.3（Layout/Transpose）**：在 TVM 通路存在后再做 layout（否则你不知道哪些 transpose 最致命）
4. **4D（ONNX import）**：作为 Phase4 收口项（不然后面更痛）

---

# 你可以直接复制给“生成工程的模型”的 Phase4 任务清单

* [ ] 4B.1：IR 增加 `layout` 与 `meta` 挂载点（不改语义，单测保障）
* [ ] 4B.2：新增 `SpecIR` + property lowering（C/margin），对齐 auto_LiRPA `compute_bounds(C=...)`
* [ ] 4B.3：layout normalize pass（至少支持互逆 permute 消除）
* [ ] 4C：TVMExecutor v0（interval linear/relu/add），同 task module 输出对齐；加 VM `save_function` 基准脚本
* [ ] 4D：ONNX import 最小闭环 + shape_inference（infer_shapes 写 value_info）

等 Phase4 全部完成后，你的 Phase5 Planner 才是真正“可落地的优化规划器”，不会变成只写文档的空中楼阁。
