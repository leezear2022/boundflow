# 变更记录：Bound IR v1 plain-CROWN lowering 与独立解释器

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`192d37f`（IR-1A typed schema/verifier foundation）
> 状态：IR-1B dense semantic closure validated；IR-1 尚未全部完成

## 1. 为什么继续修 schema

IR-1A 把原占位 `Bound IR` 升级为 typed schema，但开始对照
`runtime/crown_ibp.py` 的真实 backward trace 后发现，plain CROWN 的最小状态不能诚实地表示为
单个 coefficient value。真实状态是：

```text
(A_u, b_u, A_l, b_l)
```

而 residual/fanout 还需要同时表达：

- 一个 backward state 向多个动态父节点路由；
- 常量 add 的 affine bias 吸收；
- 路由时原 bias 只交给第一个 child，避免重复计数；
- 同一 primal value 收到多路贡献时，四个分量分别合并；
- concat 按 primal axis 切 coefficient，bias 同样只分配一次。

因此本轮先补齐 `BoundAffineStateRef`、`ADD_BACKWARD`、`CONCAT_BACKWARD` 和四元 state arity/
type verifier，再实现 lowering；没有把上述语义继续藏在 builder 的临时 Python 字典里。

## 2. Task/trace → Bound IR lowering

新增 `boundflow/frontends/plain_crown_bound_ir.py`：

- 输入现有单任务 `BFTaskModule`、`InputSpec`、IBP trace、ReLU pre-activation trace 和可选
  `linear_spec_C`；
- 输出经过完整验证的 `BFBoundModule`；
- 显式生成 `SPEC_BIND`、Linear/Conv backward、ReLU relaxation、reshape、add/concat route、
  coefficient compose 和 concretize；
- coefficient type 显式保存 domain/spec 两个 batch axis；
- identity objective 与 rank-2/rank-3 linear objective 均形成稳定 typed spec；
- L∞/L2/L1/box perturbation 均映射到 typed perturbation spec；
- TaskOp、attrs 和参数 tensor 内容共同形成 `primal_graph_hash`；
- objective tensor 内容形成 payload hash；
- canonical dump/hash 不依赖 Python object identity。

当前 lowering 的有意边界与原 plain-CROWN reference 子集一致：

- 单 `INTERVAL_IBP` task；
- 最后一个 TaskOp 的 rank-2 输出；
- Linear、Conv2d、ReLU、Flatten/Reshape、exact-shape Add；
- feature/channel-first dynamic Concat；
- 不包含 α、β、split、cuts 或 fused backend 决策。

## 3. 独立 dense reference interpreter

新增 `boundflow/runtime/bound_ir_interpreter.py`。解释器只按 `BFBoundGraph.ops` 顺序执行：

- 不 import 或调用 `runtime/crown_ibp.py`；
- 从 typed attrs 解析参数、ReLU pre-bound、concat slice 和 perturbation；
- 独立实现 Linear/Conv transpose、plain-CROWN ReLU relaxation、fanout compose 和 concretize
  公式；
- 执行前重新核对 Task/参数 fingerprint、objective payload hash 和 perturbation identity；
- stale model weights、stale objective 或缺失 ReLU trace 均 fail closed。

这条路径的用途是给后续 rewrite/Plan/Task/Schedule lowering 提供语义 oracle，不是替换当前优化
runtime，也不构成性能声明。

## 4. 测试与结果

新增 `tests/test_bound_ir_v1_plain_crown.py`，并扩展 `tests/test_bound_ir_v1.py`。

专属覆盖包括：

- 两批输入的 identity-objective MLP；
- rank-2 multi-spec linear objective；
- residual add + input fanout，检查显式 `ADD_BACKWARD` 与 `COEFFICIENT_COMPOSE`；
- 双分支 concat + input fanout，检查显式 `CONCAT_BACKWARD`；
- Conv2d → ReLU → Flatten → Linear chain CNN；
- L∞、L2 和 exact box concretization；
- deterministic lowering/dump/hash；
- missing ReLU trace、stale objective、stale parameter fingerprint；
- 解释器源码不得 import CROWN oracle。

执行环境：

```text
Conda env: boundflow
Python: 3.12.12
```

专属 schema/lowering/interpreter：

```bash
pytest -q tests/test_bound_ir_v1.py tests/test_bound_ir_v1_plain_crown.py
```

结果：`20 passed`。

相邻旧路径：

```bash
pytest -q \
  tests/test_bound_ir_v1.py \
  tests/test_bound_ir_v1_plain_crown.py \
  tests/test_phase7a_pr8_general_dag_runtime.py \
  tests/test_phase7a_pr7_bab_chain_cnn.py \
  tests/test_env.py
```

结果：`32 passed`，1 个既有 PyTorch deprecation warning。

全量：

```bash
pytest -q tests
```

结果：`392 passed, 1 skipped, 6 warnings`，用时 45.96 s。skip 仍为 TVM 可用时跳过重复
allow-no-tvm 编译 smoke；warnings 均为既有依赖 warning。

静态门禁：

```bash
python -m black --target-version py312 --check \
  boundflow/ir/bound.py \
  boundflow/frontends/plain_crown_bound_ir.py \
  boundflow/runtime/bound_ir_interpreter.py \
  tests/test_bound_ir_v1.py \
  tests/test_bound_ir_v1_plain_crown.py
mypy \
  boundflow/ir/bound.py \
  boundflow/frontends/plain_crown_bound_ir.py \
  boundflow/runtime/bound_ir_interpreter.py \
  tests/test_bound_ir_v1_plain_crown.py
pylint \
  boundflow/ir/bound.py \
  boundflow/frontends/plain_crown_bound_ir.py \
  boundflow/runtime/bound_ir_interpreter.py \
  tests/test_bound_ir_v1_plain_crown.py
```

结果：Black clean、Mypy 0 issues、Pylint 10.00/10。

## 5. Claim 边界与下一门禁

本轮可升级的表述仅是：

> Bound IR v1 已拥有 plain-CROWN dense reference 子集的 typed lowering、显式
> residual/fanout semantics 和独立 final-bound interpreter。

仍不能宣称完整 C1 或 IR-1 closure，因为以下门禁尚未关闭：

1. `MATERIALIZE` / `REPRESENTATION_CAST` rewrite 尚未作用于真实 lowering graph；
2. structured/dense rewrite 前后 final bounds 尚未通过同一 IR interpreter/runtime 对齐；
3. 当前生产 `run_crown_ibp_mlp` 尚未改为消费 Bound IR；
4. 尚无 IR-driven artifact/replay；
5. Plan IR、Task IR 和 Schedule IR 仍未实现。

下一切片应为 IR-1C：

```text
dense Bound IR
  -> explicit representation/materialization rewrite
  -> structured/dense reference execution
  -> MLP/CNN/residual final-bound equivalence
  -> malformed rewrite fail closed
```

完成 IR-1C 后才能审计 IR-1 是否达到进入 Plan IR 的门禁。
