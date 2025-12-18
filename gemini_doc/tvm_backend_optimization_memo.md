# TVM 后端优化备忘（当前 baseline 与后续改进点）

## 背景与目标

BoundFlow 目前的 TVM 后端目标是：在不牺牲正确性的前提下，把 verifier 的热点算子（尤其是 **interval IBP** 与后续 **CROWN/backward LiRPA** 的核心线性传播）逐步迁移到 TVM 上，并形成可持续迭代的性能闭环（benchmark + ablation）。

本备忘用于明确：

- **当前阶段的 baseline**：`Relax op + VM`（工程侧不手写 TE/TIR）
- **后续的改进点**：`call_tir + PrimFunc`（更高可控性与性能上限）
- 如何在 Phase 5 的 Planner/PlanBundle 里“挂载”这些 lowering 策略，避免返工

---

## 1) 当前 baseline：Relax op + VM（推荐保留为默认路径）

### 现状（仓库实现）

- `TVMTaskExecutor` 默认用 **Relax VM** 执行，并以 **Relax op** 表达 interval kernel：
  - interval linear：`boundflow/backends/tvm/relax_interval_linear.py`
  - interval conv2d：`boundflow/backends/tvm/relax_interval_conv2d.py`
- legacy demo（手写 TE → build）仍保留，可通过 `TVMExecutorOptions(kernel_style="te")` 回退。

### 为什么把它作为 baseline

- 开发速度快：更像“写算子图”，少写底层细节。
- 对齐/回归稳：更容易保持与 Python reference、auto_LiRPA 的一致性。
- 适合 Phase 5 的系统骨架：TaskGraph / StoragePlan / CachePlan / batching / ablation 先跑起来。

### 已知约束（本仓库 TVM fork）

- `tvm.nd` 不可用（`hasattr(tvm, "nd")==False`），运行时使用 `tvm.runtime._tensor`（`tvm.runtime.Tensor`）。
- 因此 executor 侧的 “numpy ↔ tvm tensor ↔ torch tensor” 往返会带来开销；对性能评测需要特别注意：
  - 减少 host/device 拷贝
  - 尽快引入 buffer 复用（StoragePlan）与编译缓存

---

## 2) 后续改进点：call_tir + PrimFunc（作为 Phase 5/6 的性能主干）

### 何时需要上 call_tir

当满足以下任一条件时，建议把热点 task 从 Relax-op baseline 升级为 `call_tir + PrimFunc`：

- Relax op 生成的 kernel 结构不可控（难以 fuse、难以保证“一次 kernel 同时算上下界”等）
- VM/算子粒度导致 launch 过碎，吞吐受限
- 需要明确的 schedule/tiling 接口（为论文/benchmark 提供可控变量）

### 两条落地路线

**路线 A（推荐）**：BoundFlow 自己生成 mixed IRModule

- 为每个 Task（或 fused task）生成：
  - 一个或多个 `tir.PrimFunc`（用 TVMScript/IRBuilder）
  - 一个 Relax wrapper，调用 `R.call_tir(primfunc, args, out_sinfo)`
- 优点：完全绕开 TE；对 kernel 结构可控；更利于融合与后续 meta-schedule。

**路线 B（可选）**：映射到 Relax op + 自定义 legalization

- 把 bound op 映射到 Relax op，再通过自定义 legalization 接管 lowering。
- 优点：更容易吃到 TVM 的 FuseOps 等 pipeline；缺点：工程复杂度更高。

### 与 Planner 的接口对齐建议

在 Phase 5 的 PlanBundle / lowering plan 里建议显式表示：

- `LoweringStrategy`：`python` / `relax_vm` / `call_tir` / `tilelang_jit`（可选）
- `KernelKey`：包含 shape/dtype/layout/target（用于编译缓存与复用）
- `ScheduleHint/LayoutHint`：先占位，后续用于 schedule 或布局传播/约束

这样后续从 `relax_vm` 迁移到 `call_tir` 不会引入上层接口返工。

---

## 3) TileLang JIT（可选路线，建议定位为“热点 kernel 加速器”）

如果后续考虑 TileLang，建议策略是：

- 不作为 baseline（避免系统工程分叉）
- 作为 `LoweringStrategy=tilelang_jit` 的可选分支，专攻少数热点 kernel（例如 backward 的 GEMM/conv）
- 强制要求：
  - 编译缓存（进程内 + 可选磁盘）
  - 明确 fallback：tilelang 不可用/shape 不匹配 → 回退 `call_tir` 或 `relax_vm`

---

## 4) 推荐的阶段性演进顺序（避免返工）

1. Phase 5A/5B/5E：先用 `relax_vm` 路径把 **TaskGraph + cache/reuse/batching + benchmark/ablation** 骨架做全。
2. Phase 5E（进阶）：挑 1–2 个最重的 task（CNN 的 interval conv2d/linear），落 `call_tir + PrimFunc`。
3. Phase 6：BaB 引入后，如果 profiling 显示 backward GEMM/conv 成为新热点，再评估 TileLang JIT。

