# PR-14B Initial Plain-CROWN Fixed Replay 结论

> 状态：**VALIDATED-NO-GO**
> 决策：不进入 PR-14C；不把 current whole-query executor 接入真实 complete verifier；C3
> 降级为支撑 C1/C2 的 query/state/capability 基础设施。
> 原始工件：`artifacts/phase7a-pr14/pr14b-initial-replay-20260719-v4/`（本地 ignored，按
> manifest 重生成）。

> **2026-08-03 后续修订**：本报告的 PR-14B No-Go 是当时 whole-query executor 的真实
> 结论。新分支 `feat/real-verifier-ir-integration-v1` 已定位其根因是丢失 external
> intermediate bounds 与 adaptive relaxation policy，并在 CPU fresh replay 上将 ResNet
> lower max diff 修复到 `3.10e-6`、sign `9/9`。新证据见
> `gemini_doc/change_2026-08-03_rvir1_external_intermediate_semantics.md`；这不追溯改写
> PR-14B artifact，也不产生性能 claim。

## 1. 本次回答的问题

PR-14A 已证明当前 backend 在 activation-BaB phase 为 0/394 eligible，只留下 initial
plain-CROWN 这一条窄化路线。PR-14B 因此只回答：在不引入 α/β/split kernel 的前提下，现有
BoundFlow whole-query executor 能否用真实 `x_L/x_U/C` 保持 external αβ-CROWN 的 initial
bound computation，并形成公平性能对照。

Runner 强制上游使用：

```text
complete_verifier=skip
bound_prop_method=crown
init_bound_prop_method=same
pgd_order=skip
```

observer 在真实 `BoundedModule.compute_bounds` 返回后冻结逐元素 input box、linear spec C、
external lower/upper、method、phase 和 requested outputs；同一进程内可再次调用原方法，避免
把文件反序列化差异混入 fixed replay。

## 2. 为 VNNLIB 补齐的 exact-box 语义

VNN-COMP ResNet property 经过像素 clipping/normalization 后不是统一 ε 的 L∞ 球。新增
`BoxPerturbation(lower, upper)`：

- 保留每个元素的精确上下界；
- affine concretization 使用 midpoint/radius 的精确 box 公式；
- lazy `LinearOperator` 只在最终输入行 materialize；
- query identity 显式包含 lower/upper content hash，不能把同中心、不同 width distribution
  的 box 错误复用；
- exhaustive vertices、batched boxes、plain-CROWN 和 query identity contract 均已覆盖。

正式 ResNet payload 为 `[1,3,32,32]`，box width 最小 `0.05996275`、最大 `0.06442014`，
共有 28 个 FP32 unique width；不能用单一 ε 替代。

## 3. Fixed replay 结果

| Workload | nominal BF vs ONNX | external replay | BoundFlow lower | requested outputs | 判定 |
|---|---:|---:|---|---|---|
| official simple MLP | max diff `0` | max diff `0` | eager/chunked/TVM 均 max diff `0` | external lower-only；BF lower+upper | 等价通过；性能 N/A |
| VNN-COMP ResNet-2B prop0 | max diff `1.67e-6` | max diff `1.07e-6` | eager/chunked max diff `796.765` | external lower-only；BF lower+upper | **bound equivalence FAIL** |

ResNet 的失败不是 ONNX 导入错误：同一中心点的 10 维输出与 ONNX Runtime 对齐到
`1.67e-6`。BoundFlow eager 与 chunked lower 彼此对齐（max diff `6.10e-5`），说明 backend
变体没有互相分叉；但两者都没有保持 external CROWN 的 bound computation。

对 9 个 CIFAR robustness specs：

- external lower 有 6 个非负项；
- BoundFlow lower 有 0 个非负项；
- 符号只对齐 3/9。

因此 direct whole-query replacement 会改变 incomplete verifier 的 prune/verified decision，不能
进入 same-solver adapter。这里的 `bound_equivalence_failure` 不等于证明 BoundFlow bounds
数学 unsound；它表示当前路径没有满足“相同浮点语义下保持 reference bound computation”的
编译替换门禁。

## 4. 为什么没有性能数字

正式 v4 runner 在两个门禁之后才计时：

1. BoundFlow bounds 必须与 external bounds 等价；
2. requested outputs 必须相同。

ResNet 在第 1 项失败。MLP 虽通过第 1 项，但真实 external call 是 `bound_upper=False`，而当前
BoundFlow executor 总是同时计算 lower+upper，因此第 2 项失败。两类 workload 都把 BoundFlow
timing/peak-memory 写为 N/A；更早 debug artifact 中的时间不用于 claim。

这也修正了 PR-14A 的 143/146 eligibility 解读：它只表示 query 内有 capability-legal 的
Affine→ReLU region，不表示 whole-query relaxation、intermediate bounds 和 requested-output
contract 已经可以替换 external executor。

## 5. Go/No-Go

1. **Activation-BaB：NO-GO。** 真实 coverage 为 0/394，不新增 α/β/split kernel；
2. **Initial whole-query replacement：NO-GO。** 非平凡 ResNet 不能保持 external bounds；
3. **公平性能 claim：NO-GO。** 唯一等价的 MLP 又存在 lower-only vs lower+upper contract
   不一致；
4. **PR-14C：BLOCKED BY GATE。** 不运行 complete-verification E2E 来掩盖 bound mismatch；
5. **C3：DOWNGRADED。** 保留 `BoundQuery`、state validity、batching、capability routing 和
   observer 作为 C1/C2 的基础设施，不再作为“真实 verifier runtime acceleration”核心贡献。

## 6. 下一步（2026-07-19 历史决定）

> **2026-07-20 路线修订**：下述 `docs/asplos-c1-c2-story-freeze` 已被 IR-first 代码复审
> 取代，不再是当前执行指令。现行分支为 `feat/compiler-ir-stack-v1`，顺序与门禁见
> `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`。

当时判定不应继续造 kernel 或扩 verifier 算法。PR-14 closure 后原建议的下一条分支是
`docs/asplos-c1-c2-story-freeze`，任务是：

1. 把摘要、前两页和 claims map 收敛到 C1 structured representation + C2 multi-backend
   Planner；
2. 明确写出 C3 的 reduced positive evidence、真实 coverage 与 No-Go 限制；
3. 以现有 C1/C2 证据重新做一次 ASPLOS 2027 paper-level Go/No-Go；
4. 只有未来提出“复用 external intermediate-bound semantics、仅替换合法 region”的新假设时，
   才能另开研究线；它不属于当前 PR-14。
