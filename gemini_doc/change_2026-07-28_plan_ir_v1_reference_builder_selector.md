# 变更记录：Plan IR v1 reference builder、selector 与 selection artifact

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`57c64e8`（IR-2A Plan IR schema/verifier/replay/migration foundation）
> 状态：IR-2B reference planning path validated；IR-2 尚未完成

## 1. 目标

IR-2A 已定义 `PlanTemplate`/`PlanInstance`，但完整实例只由测试手工拼装，尚不能回答：

1. Bound IR、hardware/workload、backend capability 和 cost evidence 如何生成候选空间；
2. memory budget、available memory 和 deadline 改变时如何选择不同实例；
3. 每个未选候选如何留下可审计拒绝原因；
4. 一次选择如何落成拒绝覆盖和篡改的 artifact。

本轮建立一条不依赖旧 planner `Any/meta` 的 reference planning path。它是 correctness/reference
实现，不是生产级高性能搜索器。

## 2. 类型化 reference builder

新增 `boundflow/planner/plan_ir_builder.py`。输入由 `ReferencePlanEvidence` 及其类型化子项组成：

- `RegionEvidence`：连续 Bound IR op span 和 cost；
- `TransitionEvidence`：source value、before-op、cast/materialize 和 representation；
- `RepresentationEvidence`：region、representation、所需 transition；
- `BackendEvidence`：region、representation、capability、artifact key；
- `BatchEvidence`：domain/spec/sample 三轴及 payload；
- `StorageEvidence`/`ValueLayoutEvidence`：arena、兼容关系、非 dense layout；
- `StateEvidence`：state/value/version 对应的 cache/recompute/evict 候选。

builder 不接受 `dict[str, Any]` 作为计划语义。它负责：

- 从 Bound IR use-def 自动推导 region input/output boundary；
- 拒绝 unknown/non-contiguous region；
- 根据 capability 与 workload/op/representation 计算静态 rejection reasons；
- 从 tensor type、producer/user 自动推导 logical bytes 和 lifetime；
- 按 hardware alignment 构建 storage binding，以实际 arena size 覆盖 evidence 中的 peak；
- 从 Bound IR value 读取 state version，拒绝无版本的 state evidence；
- 对 evidence/config 生成稳定 SHA-256 和稳定 template ID；
- 构建结束后执行完整 `PlanTemplate.validate()`。

当前 reference storage builder 只支持 static shape，并使用保守的无 alias 单 arena 分配；这保证
正确性和可审计性，不宣称 memory-optimal allocation。

## 3. 有界 reference selector

新增 `boundflow/planner/plan_ir_selector.py`。selector 对候选轴进行有上限的 deterministic
exhaustive search：

- 枚举恰好覆盖 Bound IR 的 region partition；
- 为每个 region 选择 legal representation 和其全部 required transitions；
- 选择 capability-compatible backend；
- 独立选择 domain/spec/sample batch；
- 选择每个 state group 和 compatible storage；
- 同时应用 hardware、available memory、configured budget 与 deadline；
- 按 latency、peak bytes、compile/setup cost、candidate IDs 稳定排序。

输出实例把 template 中每个 candidate 恰好记为 selected 或 rejected。无可行实例时返回聚合
`PlanSelectionFailure(reason, count)`；超过 `max_evaluated_combinations` 时 fail closed。

测试中的同一 Bound IR/evidence：

- 高预算选择全 dense、零 materialization 的较快计划；
- 低预算选择 structured storage 和 4 个显式 cast/materialize transition；
- 不可能的 memory/deadline 和过小搜索上限均拒绝，而不是静默 fallback。

## 4. 不可变 artifact 与 replay

新增 `boundflow/planner/plan_ir_artifact.py`：

- 写入 canonical `bound_module.json`、`plan_template.json`、`plan_instance.json` 和 manifest；
- 非空目标目录拒绝覆盖；
- manifest 固定 Bound/Template/Instance hash 和每个文件的 SHA-256；
- verifier 要求精确 typed Bound IR/PlanTemplate，逐文件核对 canonical bytes/hash，再调用
  `PlanInstance.from_canonical_json()` 完整重放；
- 任一文件篡改、template/module 不匹配或 instance 非 canonical 均 fail closed。

这关闭了 selection artifact 的代码/API contract，但还不是自包含 CLI：独立进程仍需从同一
Bound IR + typed evidence 确定性重建 template。

## 5. 验证结果

专属 `tests/test_plan_ir_v1.py`：`11 passed`。新增覆盖：

- 多预算产生不同且合法的完整实例；
- deterministic selector 和 plan hash；
- memory/deadline/search-bound fail closed；
- builder 自动推导 boundary/lifetime/capability/稳定 hash；
- artifact 拒绝覆盖、精确 replay 和 tamper rejection。

相邻回归：

```bash
pytest -q \
  tests/test_plan_ir_v1.py \
  tests/test_plan_ir_v1_legacy_adapter.py \
  tests/test_bound_ir_v1.py \
  tests/test_bound_ir_v1_plain_crown.py \
  tests/test_phase7a_pr11_materialization_planner.py \
  tests/test_phase7a_pr11_materialization_placement.py \
  tests/test_phase7a_pr12_execution_candidate.py \
  tests/test_phase5b_pr3_buffer_reuse.py \
  tests/test_env.py
```

结果：`92 passed`，1 个既有 PyTorch deprecation warning。

全量 `pytest -q tests`：`413 passed, 1 skipped, 6 warnings`，用时 46.08 s。

静态门禁：

- Black：clean；
- Mypy：0 issues；
- Pylint：10.00/10；
- `git diff --check`：通过。

## 6. 边界与下一门禁

本轮关闭 **IR-2B reference builder/selector/artifact API**，但不宣称 IR-2 完成。尚缺：

1. 将 PR-11/12 legacy adapter 批量输出组装到同一 template，并检测重复/孤立 candidate；
2. 对真实历史记录生成逐项 migration report，而非只测合成 fixture；
3. 把 cache/state validity 作为 query-time selector 输入并形成选择/拒绝证据；
4. 提供可独立进程执行的 template reconstruction/artifact replay CLI；
5. 做 IR-2 closure audit，逐条核对旧决策映射、`meta/Any`、capability/memory/state 门禁。

下一切片为 **IR-2C legacy artifact assembly + state-validity + closure audit**。IR-2 关闭后才进入
IR-3 Schedule IR v1，不能把当前 `runtime/scheduler.py` 的 topo loop 当作 Schedule IR。
