下面是一份可以直接丢给其它 LLM/合作者的 **BoundFlow 前端设计文档（Best Frontend Plan）**。目标是：**最少自研 IR、最快跑通端到端、同时为 DeepPoly/Zonotope/MIP/SMT/QNN 等扩展预留接口**。

---

# BoundFlow 前端设计文档（Best Plan）

## 1. 设计目标与边界

### 目标

1. **双入口导入**：支持 `PyTorch nn.Module` 与 `ONNX ModelProto`。
2. **三层语义分离**：把“原模型语义 / 边界语义 / 可执行任务”拆开，保证可维护与可优化。
3. **最少自研**：

* 第一层尽量复用 PyTorch Export/FX、ONNX 图；
* 第三层尽量复用 TVM Relax/TIR IRModule（直接交给 TVM 编译/调优）。TVM IRModule 可同时包含计算图与张量程序，这是我们要的承载体。

4. **预留扩展**：抽象域（DeepPoly/Zono/LiRPA/…）、完备求解后端（MIP/SMT/BaB）、量化语义（QNN）都是“一等公民”。

### v1 非目标（先写清楚，避免工程爆炸）

* 不追求支持 PyTorch 所有算子；优先覆盖 VNN-COMP 风格 CNN/ResNet/MLP 子集。
* 不追求完整动态 shape；默认 **静态 shape**（Torch export 默认也会按静态 shape specialize）。
* MIP/SMT/QNN v1 只留接口，不强行实现完整求解。

---

## 2. 前端总体结构：三层 IR（推荐方案）

> 关键点：**Layer3 不是“再来一套 primal op”**，而是 **Executable Task IR（可编译任务）**。

### Layer 1：Primal IR（原模型语义图）

**载体不自研**：

* PyTorch：用 `torch.export.export()` 得到 `ExportedProgram`。它包含一个 `torch.fx.Graph` 表示纯张量计算，并且图里仅包含 `torch.ops.aten`/自定义算子，且可检查元信息。
* ONNX：用 `onnx.shape_inference.infer_shapes`，推导出的 shape 会写入 `graph.value_info`。

**Layer1 的职责**：导入 + 规范化（normalize）

* 统一算子集到少量 primitive（Conv/Linear/Add/Mul/Relu/Pool/Reshape/Transpose/Concat/MatMul 等）；
* 补齐/验证 shape、dtype、常量与参数绑定；
* 记录源映射（便于 debug & 与 Bound/BaB 对齐）。

> ONNX shape inference 并非总是成功（例如 Reshape 的目标 shape 非常量时），需要在设计里写明 fallback 规则。

---

### Layer 2：Bound IR（你自定义的 Bound 语义层）

**自研的核心层**：用于表达“抽象域状态如何随算子传播/收紧”。

我推荐把 Layer2 设计成 **Transformer Graph**（而不是为每个域生成一套完全不同 BoundOp）：

* `BoundOp = ApplyTransformer(primitive_op, attrs, domain_id, state_in -> state_out)`
* 具体公式由 **Domain 插件**实现（LiRPA/DeepPoly/Zono/…）。

这样你能天然满足“预留 DeepPoly/Zonotope/MIP/SMT/QNN 接口”：只要为 primitive op 实现对应 transformer，就能扩域；不用改 IR 核心。

---

### Layer 3：Compute/Task IR（可执行任务层，TVM Relax/TIR）

**不自研 IR，直接用 TVM IRModule 承载**：

* Planner 把 Bound IR 切分/融合/批处理后，输出 `BoundTask`。
* Backend 把每个 `BoundTask` lower 成：

  * 一个或多个 **TensorIR (TIR) PrimFunc**（真正的 fused kernel）
  * 一个 **Relax function** 负责 orchestration（分配、调用、串接）。TVM Relax 支持在同一 IRModule 内同时包含 Relax 与 TensorIR。

Relax/TIR 的关键机制是把高层算子 legalize 成 `call_tir` 调用对应的 PrimFunc：TVM Unity 的 `LegalizeOps` 就是做这个的，legalize 后 IRModule 会同时包含 Relax 函数和 TensorIR 函数。

> 对你来说：Layer3 的“op”不是原模型的 Conv/ReLU，而是 **“边界计算任务”**（例如 `L_out = conv(W+, L_in) + conv(W-, U_in)` 这种张量程序）——语义已经是 bound-workload 了。

---

## 3. 为什么这三层是“最优工程解”

### 3.1 最少自研、最快见效

* Layer1 直接复用 PyTorch ExportedProgram（FX Graph）与 ONNX；ExportedProgram 自带 `state_dict`/元信息与（可选）range_constraints 等，便于后续扩动态 shape。
* Layer3 直接复用 TVM Relax/TIR IRModule（不用造执行 IR），且 TVM pass 生态天然支持模块级变换。
* 你真正要攻克的是 Layer2 + Planner（这正是 ASPLOS 的贡献点）。

### 3.2 可扩展性最好

* Domain 插件化：DeepPoly/Zono/LiRPA 只是“同一套 primitive 上的不同 transformer”。
* Complete backend 插件化：当需要 MIP/SMT/QNN 时，Layer3 可以并列一种 `ConstraintTask`（v1 只留接口）。

---

## 4. 前端导入与规范化流程（Torch + ONNX）

### 4.1 TorchFrontend（首选 torch.export，必要时 FX fallback）

**入口 API**：

```python
def import_torch(module, example_inputs, *,
                 strict=False,
                 normalize=True,
                 export_mode="export") -> BFPrimalProgram:
    ...
```

**实现要点**：

1. `torch.export.export(module, example_inputs, strict=...) -> ExportedProgram`

   * `ExportedProgram` 含 `torch.fx.Graph` 的纯张量计算图 + `state_dict` 等元信息。
2. normalize：

   * 只保留/映射到 primitive op-set（你支持的子集）
   * 明确禁止 in-place（export IR 通常是 functional 风格，文档也强调图可用且不含如 `torch.add_` 的 in-place）。
3. 动态 shape（v1 可禁用）：

   * export 默认按静态 shape specialize；若未来支持动态，读取 `range_constraints`。

> 备注：PyTorch 的 ONNX 新导出器本身也基于 `torch.export.ExportedProgram`，并推荐 `dynamo=True` 路径；这意味着 Torch/ONNX 两条入口在语义上可以更一致。

---

### 4.2 ONNXFrontend（infer_shapes + 映射到同一 primitive）

**入口 API**：

```python
def import_onnx(model_or_path, *,
                do_shape_infer=True,
                input_shapes=None,
                normalize=True) -> BFPrimalProgram:
    ...
```

**实现要点**：

1. `onnx.load` / 读取 ModelProto
2. `onnx.shape_inference.infer_shapes(model)`

   * shape 会写入 `graph.value_info`。
   * 注意：shape inference 不总成功，尤其 Reshape 的 shape 非常量时可能无法推导。
3. normalize：ONNX op 映射到同一 primitive op-set（与 TorchFrontend 对齐）。

---

## 5. 三层 IR 的数据结构规格（给 LLM 直接建工程）

### 5.1 Layer1：`BFPrimalProgram`

**核心字段**（建议 dataclass）：

* `source`: `"torch_export" | "onnx"`
* `graph`: 统一后的 primal graph（自定义轻量结构，或保留 FX/ONNX 并做 wrapper）
* `params`: 权重/常量张量（Torch 来自 ExportedProgram.state_dict；ONNX 来自 initializer）
* `tensor_meta`: 每个 value 的 shape/dtype（ONNX 来自 value_info；Torch 来自 export meta）
* `debug_map`: node ↔ source span/name（便于报错定位）

**PrimalNode** 建议最小集合：

* `op_type`（primitive 名）
* `inputs`/`outputs`
* `attrs`（stride/pad/dilation/groups 等）
* `dtype`/`shape`

### 5.2 Layer2：`BFBoundProgram`

* `primal`: 指向 `BFPrimalProgram`
* `domain_id`: `"interval" | "lirpa_linear" | "deepoly" | "zonotope" | ...`
* `spec`: 扰动集合/验证目标（ε、norm、target class/margin 等）
* `bound_graph`: Transformer DAG（节点是 `ApplyTransformer`）
* `state_table`: value -> DomainState

**DomainState**（先定义抽象基类）：

* interval: `(l, u)`
* linear(LiRPA): `(A, b)` 或其它你选定的紧凑表示
* 预留：zonotope/affine forms、deepoly 线性上下界、αβ 参数等

### 5.3 Layer3：`BFTaskModule`（TVM IRModule + task metadata）

* `tvm_mod`: TVM `IRModule`（包含 Relax + TIR）
* `tasks`: List[`BoundTask`]
* `bindings`: 参数/常量如何绑定到 Relax entry function
* `profiling_hooks`: 可选（性能测量入口）

**BoundTask** schema：

* `task_id`
* `input_states` / `output_states`
* `batch_axes`: spec_batch / bab_batch / class_batch
* `memory_plan`: 要缓存哪些中间 state
* `lowering`: `"tvm_tir"`（v1） / `"constraint"`（预留）

---

## 6. Domain 插件接口（预留 DeepPoly/Zono/MIP/SMT/QNN）

### 6.1 抽象域插件：`AbstractDomain`

每个域实现“primitive transformer 集合”（只覆盖你支持的 primitive op）：

* `affine`（Linear/Conv 的统一抽象）
* `relu`
* `pool`
* `reshape/transpose/concat`（多为形状/重排语义，域状态搬运为主）

### 6.2 量化语义（QNN）预留

在 Layer1 就要能标记数值语义（否则后面补很难）：

* `NumericSemantics`: `FP32 | FP16 | INT8 | FIXED_POINT(qm, qn, round, saturate) | BV(width)`
* ONNX 量化模型常以 Q/DQ 形式表达；v1 先做到“识别并贴标签”，后续再接约束求解/位精确。

### 6.3 完备求解后端（MIP/SMT）预留

Layer3 的 `lowering="constraint"`：输出 `ConstraintTask`（约束 IR + solver 配置）。v1 只定义 schema，不实现。

---

## 7. Planner 与 Layer3（TVM）对接要点

BoundFlow 的 planner 输出 fused/batched `BoundTask`，TVM 侧建议用：

* Relax function 做任务 orchestration
* 每个 fused task 编译为一个 TIR PrimFunc，并通过 `call_tir` 绑定到 Relax。TVM 的 legalize 流程与 API 明确支持把 Relax 算子变成 call_tir，并在同一 IRModule 内共存。

（你也可以直接用 Relax frontend helper 创建 call_tir binding。）

---

## 8. 工程目录建议（LLM 直接照着生成）

```text
boundflow/
  boundflow/
    frontends/
      torch_frontend.py      # torch.export -> BFPrimalProgram
      onnx_frontend.py       # onnx + infer_shapes -> BFPrimalProgram
      normalize.py           # primitive 化、shape/dtype 校验、常量折叠

    ir/
      primal.py              # BFPrimalProgram, BFPrimalGraph, BFPrimalNode
      bound.py               # BFBoundProgram, ApplyTransformer, DomainState
      task.py                # BFTaskModule, BoundTask, ConstraintTask(schema)

    domains/
      base.py                # AbstractDomain 接口
      interval.py            # v1 实现
      lirpa_linear.py        # v1/ v1.5 实现（可只覆盖 affine+relu）

    planner/
      planner.py             # bound -> tasks
      passes/
        fuse_blocks.py
        batch_specs.py
        reuse_cache.py

    backends/
      tvm_backend/
        lower_to_tvm.py       # BoundTask -> TVM IRModule (Relax+TIR)
        build.py              # relax.build / runtime wrapper

  tests/
    test_import_torch.py
    test_import_onnx.py
    test_primal_equivalence.py
    test_interval_bound_correctness.py
```

---

## 9. 验收测试（必须写进计划）

1. **导入一致性测试**：同一个模型

* TorchFrontend 导入 → primal graph（normalized）
* ONNXFrontend 导入 → primal graph（normalized）
  两者 primitive 序列、shape/dtype、权重绑定一致（允许 node 名不同）。

2. **正确性测试**：interval bound

* Layer2 interval domain 输出与参考实现对齐（可先用简单手算或 auto_LiRPA baseline）。

3. **TVM 可执行性**：Layer3 IRModule

* IRModule 同时含 Relax + TIR（legalize/call_tir 路径跑通）
* 能编译并运行一次 `compute_bounds(entry)`。

---

## 10. v1 具体落地顺序（最省时间）

1. TorchFrontend（export）+ Normalize（只支持 Conv/Linear/ReLU/Add/Reshape）
2. ONNXFrontend（infer_shapes）+ 同一套 Normalize
3. Interval domain（只做 forward bound）
4. Planner v0：一层一 task（不融合）
5. TVM backend v0：把一个 affine bound task lower 成 TIR PrimFunc + Relax 调用
6. 加 fusion/batching/reuse（作为你系统论文的核心优化点）

---

如果你希望我再“更硬一点”，我可以把上面这份文档补上两部分，方便 LLM 直接生成可运行骨架代码：

* **primitive op-set 的精确定义表**（每个 op 的 attrs 与 shape 规则）；
* **`AbstractDomain` 的必选方法签名 + interval 参考实现伪代码**。
