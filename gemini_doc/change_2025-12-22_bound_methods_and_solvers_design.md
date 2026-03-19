# 变更记录：新增 IBP/CROWN/αβ-CROWN/BaB 的统一设计文档

## 动机

Phase 5 已冻结为 interval IBP 主线；Phase 6 需要引入更强的 LiRPA 方法族（CROWN/IBP-CROWN/αβ-CROWN）以及可选的 complete verification（BaB）。如果直接“按算法名字堆接口”，容易导致 TaskKind/缓存/后端路径爆炸，新增扰动或新方法时牵一发动全身。

因此新增一份系统设计文档，明确三轴解耦与落地路线，并回答“引入新扰动/新边界方法是否麻烦”。

## 主要改动

- 新增：`gemini_doc/bound_methods_and_solvers_design.md`
  - 提出三轴解耦：`PerturbationSet × BoundMethod/DomainState × Solver(BaB)`。
  - 将 IBP/CROWN/IBP-CROWN/αβ-CROWN 统一为可组合 stages（forward/relax/backward/optimize），避免 TaskKind 组合爆炸。
  - 给出与现有 BoundFlow 分层（Primal IR/Planner/TaskGraph/StoragePlan/Runtime/TVM）的映射与推荐实现顺序。
  - 讨论 cache key 设计、可复用状态、batching 优先级，以及控制流留在 Python 的工程原则。

## 验证

- 文档变更（无额外运行时验证）。

