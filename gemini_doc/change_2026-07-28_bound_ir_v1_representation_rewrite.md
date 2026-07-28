# 变更记录：Bound IR v1 显式 representation/materialization rewrite

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`659c4aa`（IR-1B plain-CROWN dense semantic closure）
> 状态：IR-1 最小语义闭环门禁通过；下一阶段为 IR-2 Plan IR

## 1. 目标

IR-1B 已证明 Task/trace 能 lower 为 typed Bound IR，且独立 dense interpreter 与旧 CROWN
oracle final bounds 对齐。但如果 structured 表示和 materialization 仍只存在于旧 runtime 的 Python
类型分支中，C1 的“显式物化语义”仍未成立。

本轮关闭 IR-1 最后一组门禁：

```text
dense Bound IR
  -> verified structured-region rewrite
  -> explicit RepresentationCast / Materialize
  -> dense/structured reference execution
  -> final-bound equivalence
```

## 2. Verifier 收紧

`boundflow/ir/bound.py` 现在要求：

- Linear/Conv/ReLU/Reshape affine-state transform 不得隐式改变 coefficient representation；
- coefficient compose 的所有输入和输出 representation 必须一致；
- Add/Concat backward route 必须保持 coefficient representation；
- 任何 dense/structured 变化必须经过一等 `REPRESENTATION_CAST` 或 `MATERIALIZE` op。

直接把某个 affine op 的输出 metadata 从 dense 改成 structured 会 fail closed，而不是被 verifier
当作合法 backend 选择。

## 3. Structured-region rewrite

新增 `boundflow/ir/bound_rewrite.py`：

- 只接受已验证、全 dense 的 IR-1 plain-CROWN source module；
- 在 maximal affine/routing region 入口插入
  `REPRESENTATION_CAST(Dense -> Structured)`；
- Linear、Conv、Reshape、Add/Concat route 和 CoefficientCompose 均保持 structured；
- ReLU 的 sign-dependent relaxation 是 v1 显式 dense boundary，因此在其前插入
  `MATERIALIZE(Structured -> Dense)`；
- concretization 前同样显式 materialize；
- rewrite 结果再次执行完整 `BFBoundModule.validate()`；
- 相同输入产生相同 canonical JSON 和 stable hash；
- 对已经 structured 的输入重复 rewrite 会 fail closed。

这里没有做 cost-based 选择。rewrite 只是 IR-1 reference transformation，用于确定表示改变的语义
和合法性；候选、成本和选择属于下一阶段 Plan IR。

## 4. Reference interpreter 扩展

`boundflow/runtime/bound_ir_interpreter.py` 现在同时执行：

- dense coefficient tensor；
- explicit dense→structured cast；
- `LinearOperator` structured Linear/Conv/Reshape/Add/Concat/Compose；
- explicit structured→dense materialization；
- dense ReLU boundary；
- final perturbation concretization。

解释器仍不 import 或调用 `runtime/crown_ibp.py`。旧 `LinearOperator` 在这里是 structured
representation 的 reference data type，不拥有 graph 顺序、rewrite 决策或 Bound IR 语义。

## 5. 测试与结果

新增/扩展测试覆盖：

- multi-spec MLP；
- Conv→ReLU→Flatten→Linear chain CNN；
- residual add + input fanout；
- concat + input fanout；
- dense IR 与 rewritten structured IR final lower/upper 对齐；
- rewrite dump/hash 确定性；
- IR 中真实存在 structured coefficient value、cast 和 materialize；
- affine op 隐式 representation change 被拒绝；
- 非 dense source 的重复 rewrite 被拒绝。

专属：

```bash
pytest -q tests/test_bound_ir_v1.py tests/test_bound_ir_v1_plain_crown.py
```

结果：`25 passed`。

相邻：

```bash
pytest -q \
  tests/test_bound_ir_v1.py \
  tests/test_bound_ir_v1_plain_crown.py \
  tests/test_phase7a_pr8_general_dag_runtime.py \
  tests/test_phase7a_pr7_bab_chain_cnn.py \
  tests/test_phase7a_linear_operator_concretize.py \
  tests/test_env.py
```

结果：`47 passed`，1 个既有 PyTorch deprecation warning。

全量：

```bash
pytest -q tests
```

结果：`397 passed, 1 skipped, 6 warnings`，用时 45.58 s。

静态门禁：

- Black：clean；
- Mypy：0 issues；
- Pylint：10.00/10。

## 6. IR-1 closure 与下一步

对照架构契约 §8 的 IR-1 门禁：

| 门禁 | 结果 |
|---|---|
| typed schema、builder/lowering、verifier、dump/hash、interpreter | 通过 |
| MLP、chain CNN、fanout/residual final bounds | 通过 |
| materialize/structured/dense rewrite 数值一致 | 通过 |
| malformed polarity/shape/state/fanout/representation fail closed | 通过 |
| 不以 `runtime/crown_ibp.py` 隐藏分支定义 IR 语义 | 通过 |

因此可以关闭 **IR-1 最小 reference semantic closure**。

这仍不等于完整 ASPLOS C1 已成立：

- 尚无统一 Plan IR；
- 尚无 Task/Schedule IR；
- backend/runtime 尚未迁移为消费 Plan/Task/Schedule IR；
- 尚无 IR-driven E2E artifact 和性能证据。

下一阶段必须进入 **IR-2 Plan IR v1**，而不是继续在 Bound IR 上增加 α/β/BaB 泛化：

1. `PlanTemplate` / `PlanInstance` typed schema；
2. representation、materialization、backend、batch、storage/lifetime 的统一 decision；
3. 跨 decision verifier；
4. PR-11/12 旧计划对象的显式 adapter 或 unsupported 记录；
5. deterministic plan dump/hash/replay。
