# 变更记录：Bound IR v1 typed schema foundation

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`8daa363`（IR-first 文档冻结）
> 状态：IR-1A schema/verifier foundation；IR-1 尚未完成

## 1. 目标

把原 `boundflow/ir/bound.py` 的空 `DomainState`、`objectives: Any` 和通用 attrs dict 占位结构，
替换为第一套独立于 runtime、PyTorch 和 TVM 的一等 Bound IR schema。

本切片只关闭：

- typed value/type/spec/domain/op/graph/module schema；
- SSA/use-def 和语义合同 verifier；
- deterministic JSON dump 与 content hash；
- 旧 runtime `DomainState` 继承兼容。

本切片不关闭：

- Primal/Task trace → Bound IR builder；
- dense reference interpreter；
- CROWN runtime → Bound IR lowering/migration；
- Plan IR、Schedule IR 或自适应 runtime；
- α/β/split 优化执行。

## 2. 实现

### 2.1 类型与 verification axes

`boundflow/ir/bound.py` 新增：

- `BoundTensorType`、`BoundBatchAxis`；
- `BoundValue`、`BoundValueRole`、`BoundPolarity`；
- `BoundRepresentation`；
- 分离的 sample/spec/domain batch-axis identity。

已验证的不变量包括 shape/dtype/layout/device、batch dimension 范围、axis kind/dimension
唯一性，以及 batch axes 的确定性顺序。

### 2.2 强类型 Spec 与 method state

新增：

- `PerturbationSpec`：L∞/L2/L1/box 及 radius/payload identity；
- `ObjectiveSpec`：identity/linear/margin；
- `VerificationSpec`：requested bounds 与 numeric policy；
- `BoundDomainConfig`：interval/CROWN/α-CROWN/αβ-CROWN 状态组合。

非法 beta-without-alpha、plain CROWN 携带 α/β/split、非有限 radius、含糊 requested bounds
均 fail closed。

### 2.3 typed BoundOp

`BoundOpKind` v1 覆盖 input/spec bind、Linear/Conv backward、ReLU relaxation、
coefficient/bias compose、add、reshape、materialize、representation cast、concretize 和
objective reduce。

关键 op 使用独立 frozen attrs dataclass，不再接受无约束 `Dict[str, Any]`。Verifier 检查：

- op/attrs 类型匹配；
- arity、unknown reference、duplicate output；
- polarity 和 tensor-type 保持；
- reshape static numel；
- materialization 只改变 representation 且目标为 dense；
- concretize/objective reduce 的角色边界。

### 2.4 SSA graph 与模块 identity

`BFBoundGraph` 现在检查：

- value/op/IO ID 唯一；
- topological use-before-def；
- SSA redefinition；
- 每个非输入 value 都有 producer；
- graph IO 均存在。

`BFBoundModule` 提供 schema version、Primal graph hash、typed spec/domain/graph、
canonical JSON 和 SHA-256 stable hash，并将 input/spec binding、concretization reference
交叉解析到 typed `VerificationSpec`，未知或不一致的 spec ID fail closed。

旧 `DomainState` 保留为 runtime compatibility base；未使用的旧名字 `Spec`、
`ApplyTransformer`、`BFBoundProgram` 仅作为指向新 schema 的迁移 alias，不保留旧 Any/dict
容器。

## 3. 测试

新增 `tests/test_bound_ir_v1.py`，覆盖：

- deterministic dump/hash；
- duplicate value/op；
- unknown reference/use-before-def；
- materialization semantic preservation；
- fanout/residual merge；
- polarity/dtype mismatch；
- reshape/batch-axis failure；
- op attrs/method state failure；
- spec failure；
- `IntervalState(DomainState)` 兼容；
- Bound IR 源文件不存在 runtime/backend/torch/TVM import。

正式环境：

```text
Conda env: boundflow
Python: 3.12.12
```

已执行：

```bash
python -m black --target-version py310 \
  boundflow/ir/bound.py tests/test_bound_ir_v1.py
mypy boundflow/ir/bound.py tests/test_bound_ir_v1.py
pylint boundflow/ir/bound.py tests/test_bound_ir_v1.py
pytest -q tests/test_bound_ir_v1.py tests/test_env.py
```

结果：

- Black：2 files unchanged；
- Mypy：0 issues；
- Pylint：10.00/10；
- Pytest：15 passed，1 个 PyTorch deprecation warning。

随后执行关键兼容回归：

```bash
pytest -q \
  tests/test_bound_ir_v1.py \
  tests/test_env.py \
  tests/test_phase3_ibp_against_auto_lirpa.py \
  tests/test_phase3_ibp_cnn_against_auto_lirpa.py \
  tests/test_phase6b_crown_ibp_mlp.py \
  tests/test_phase7a_pr8_general_dag_runtime.py \
  tests/test_phase7a_pr10_relu_dense_reference.py \
  tests/test_phase7a_pr14b_box_perturbation.py
```

结果：42 passed，1 warning。

全量：

```bash
pytest -q tests
```

最终结果：384 passed、1 skipped、6 warnings，用时 55.16 s。skip 为 TVM 已可用时跳过重复的
allow-no-tvm 编译 smoke；warnings 均来自 PyTorch/ONNX/profiler 依赖的既有
deprecation/future/user warning。

## 4. 下一门禁

IR-1B 必须实现：

```text
Primal/Task trace
  -> Bound IR builder
  -> dense reference interpreter
  -> 与现有 run_crown_ibp_mlp final bounds 对齐
```

在 builder/interpreter/lowering 和非 toy residual/fanout 对齐之前，C1 只能称
`schema/verifier foundation validated`，不能称完整 Structured Bound-Operator IR。
