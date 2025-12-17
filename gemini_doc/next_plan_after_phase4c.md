# 下一步计划：继续 Phase 4 收尾，随后进入 Phase 5

## 当前进度（你现在已经“有资格进入 Phase 5”，但建议先把 Phase 4 的关键尾巴收掉）

已完成并有单测对齐：

- **Phase 4A/4B.0**：Task pipeline + StoragePlan schema 钉子。
- **Phase 4B.2**：Spec/Property（`LinearSpec(C)`）对齐 auto_LiRPA（IBP + C）。
- **Phase 4C v0**：`TVMTaskExecutor` 跑通（目前仅加速 `linear(w:2D)`），并通过：
  - `PythonTaskExecutor == auto_LiRPA`
  - `TVMTaskExecutor == PythonTaskExecutor`

这意味着：**语义闭环与后端 demo 闭环都成立**。接下来要做的是“让 Phase 5 不返工”的工程收尾。

---

## 我建议：先做 Phase 4D/4C.1/4B.3（工程化收口），再开 Phase 5

原因很简单：

- Phase 5（CROWN/BaB/多任务 planner）会显著放大运行次数与图规模；如果 **layout-only/缓存/前端统一** 没钉好，后面每一步都会被迫回头修 Phase 4。
- Phase 4 的收尾项是“不可逆接口”：一旦 Phase 5 基于现有接口长出来，再改会很痛。

---

## Phase 4（收尾）执行计划（推荐顺序）

### 4B.3：Layout-only（permute/reshape 等）最小可优化语义

**目标**
- 把 `permute` 明确为 **layout-only**：允许 planner 做“保守消除/合并”，避免未来被固化成 kernel 调用。

**交付**
- IR/TaskOp：为 layout-only op 增加统一标记（例如 `attrs["layout_only"]=True` 或 `TaskOp.meta`）。
- Planner pass（最小版）：
  - 合并连续 `permute`（compose dims）
  - 删除 identity `permute`
- 测试：
  - 新增一个只含 `permute` 的小图：优化前后语义一致，且 TaskOps 数量减少。

**验收**
- 现有 auto_LiRPA 对齐测试不退化。

---

### 4C.1：TVMExecutor 工程化（缓存 + 覆盖面）

**目标**
- 让 “TVM 后端”从 demo 变成可持续扩展的执行器：可观测、可缓存、可增量覆盖更多算子。

**交付（优先级从高到低）**
1. **编译缓存（进程内 + 可选磁盘）**
   - 进程内已有 `lru_cache`；补一个可选 `cache_dir`（key→artifact）避免每次 pytest 重新编译。
2. **显式统计/日志**
   - 运行时记录：哪些 op 走了 TVM，哪些 fallback（方便你判断“到底加速了没有”）。
3. **补齐 batched linear（可选，但很值）**
   - 让 `w` 为 rank-3 `[B,O,I]` 也能走 TVM（这能覆盖 `C` 融合后的 property 场景）。
4. （后续）conv2d TVM kernel：先不追求最优 schedule，只追求 correctness + 对齐 Python。

**验收**
- `tests/test_phase4c_tvmexecutor_against_auto_lirpa.py` 继续通过。
- 新增一个 “batched linear/spec” 的 tvm 对齐测试（如果实现了 batched linear）。

---

### 4D：ONNX 前端最小闭环（强烈建议）

**目标**
- 避免 Phase 5 时出现“两套前端/两套 normalize”的分叉。

**交付**
- `frontends/onnx/frontend.py`：`onnx.shape_inference.infer_shapes` + 映射到 Primal IR（先支持已闭环算子子集：linear/conv2d/relu/add/reshape/flatten/permute）。
- 测试：
  - 同一 MLP（或小 CNN）分别用 Torch/ONNX 导入，normalize 后的拓扑/shape/dtype 一致（允许 node 名不同）。

**验收**
- Torch/ONNX 走同一 planner+executor，输出一致（至少 forward shape 一致；IBP 对齐先用 MLP）。

---

## Phase 5（真正的 Planner + 更复杂域 + BaB）计划（在 Phase 4 收口后启动）

### 5.1：Domain/Rule 插件化（防止 `IntervalDomain` 变巨无霸）

**目标**
- 把“算子语义/边界传播规则”从 domain 类中拆到 registry/rules，domain 只负责状态表示与通用算子组合。

**交付**
- `OpRule`/`RuleRegistry`（按 op_type 分发）：`interval_linear_rule`、`interval_conv2d_rule`、`relu_rule` 等。
- Executor/Planner 只调用 registry，不直接写 if-else 巨分支。

---

### 5.2：CROWN（反向线性界）reference 版（先正确，再优化）

**目标**
- 在 Python reference 路径上实现：
  - backward bound（A,b 形式）
  - relu relaxation（CROWN）
  - 输出支持 `LinearSpec(C)`（C 直接作为 backward 起点）

**验收**
- 对齐 auto_LiRPA：`method="CROWN"`（先 MLP，再 CNN）。

---

### 5.3：BaB（Branch-and-Bound）骨架

**目标**
- 跑通最小 BaB：split、branching heuristic、缓存、终止条件。

**验收**
- 在一个小网络/小 batch 上能得到比纯 IBP 更紧的 bounds（并能复现 auto_LiRPA 的趋势）。

---

## 你现在可以直接做的下一步（建议从这里开始）

1. 先做 **4B.3（permute 的 layout-only pass）**：改动小、收益大、最不容易被依赖卡住。
2. 再做 **4C.1（TVM 编译缓存 + “走 TVM/fallback” 统计）**：马上提升可用性与可调试性。
3. 然后上 **4D（ONNX 最小闭环）**：为 Phase 5 清障。

