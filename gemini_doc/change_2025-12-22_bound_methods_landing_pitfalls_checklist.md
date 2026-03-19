# 变更记录：补强 bound_methods 设计的“落地避坑清单”

## 动机

在 IBP/CROWN/αβ-CROWN/BaB 的统一设计里，容易出现“文档方向对，但落地时撞墙”的问题，例如：

- backward LiRPA 的 `A` 若被默认成显式矩阵会导致内存爆炸；
- TaskKind 虽然不爆炸，但缺少 task contract 会让 executor/scheduler 难以泛化；
- αβ 的 warm-start/reuse 若不进入 cache key 会产生 silent bug；
- BaB 的 subproblem 约束若只是松散 dict，后续 LINEAR_BWD/RELAX_OPT 很难利用；
- cache key 有了但没有 cache 粒度（缓存什么 stage/state），容易缓存错对象；
- correctness 对齐没有写成 DoD，reviewer 会质疑语义正确性。

因此把这些点补充进设计文档，作为 Phase 6 的接口约束/DoD。

## 主要改动

- 更新：`gemini_doc/bound_methods_and_solvers_design.md`
  - 新增：`§7 落地避坑清单（建议作为 Phase 6 的接口约束/DoD）`
    - `A` 的表示建议支持 `LinearOperator`（不强迫显式矩阵）
    - task contract：produces/consumes/batch_axes
    - αβ 优化阶段：warm-start state machine + cache key 约束
    - subproblem 约束结构：β-CROWN 风格留槽
    - CachePlan 粒度：value/layer/stage 级别与 cache vs batch 的取舍
    - correctness DoD：逐阶段对齐 auto_LiRPA/参考实现

## 验证

- 文档变更（无额外运行时验证）。

