# BoundFlow 工程概览与进度报告

## 1. 工程简介

**BoundFlow** 是一个面向神经网络验证（Neural Network Verification）和认证训练（Certified Training）的**验证感知编译器与运行时系统**。

*   **核心目标**：解决现有验证工具（如 `auto_LiRPA`）在“库式执行”模式下存在的系统效率问题（碎片化 Kernel、重复计算、CPU-GPU 同步开销等）。
*   **方法论**：将“边界传播”（Bound Propagation）视为一种可编译、可规划的工作负载。
    *   **前端**：复用 `auto_LiRPA` 的算法逻辑进行推导。
    *   **中间层**：引入 **BoundFlow IR** 和 **Global Bound Planner**，进行全局图优化、算子融合（Fusion）、批处理（Batching）和存储复用（Memory Reuse）。
    *   **后端**：利用 **TVM (TIR)** 生成高效的 GPU Kernel。
*   **学术定位**：目标是发表在 ASPLOS 等体系结构/系统顶级会议上，强调“系统级”贡献而非“算法级”创新。

## 2. 核心架构与代码对应

工程代码结构清晰，与架构设计文档高度一致：

| 模块 | 路径 | 功能描述 | 实现状态 |
| :--- | :--- | :--- | :--- |
| **Frontend** | `boundflow/frontends` | 负责导入模型和 Spec，将 `auto_LiRPA` 的计算图转换为 BoundFlow IR。 | ✅ 已实现 |
| **IR** | `boundflow/ir` | 定义了 `PrimalGraph`（原始计算）、`TaskGraph`（任务依赖）、`StoragePlan`（存储规划）等核心数据结构。 | ✅ 已实现 |
| **Planner** | `boundflow/planner` | 全局规划器，包含 `Pipeline` 和各种 Pass（如 `buffer_reuse_pass`），负责生成优化的执行计划。 | ✅ 已实现 |
| **Backend** | `boundflow/backends/tvm` | TVM 后端实现，负责将 BoundFlow 任务 Lowing 到 TVM Relax/TIR 并生成代码。 | ✅ 已实现 |
| **Runtime** | `boundflow/runtime` | 包含 `PythonExecutor`（Reference）和 `TVMExecutor`（高性能执行），以及 BaB 相关的调度逻辑。 | ✅ 已实现 |
| **Docs** | `gemini_doc` | 包含详尽的开发文档、计划（`plan.md`）、变更记录（`change_...md`）和设计原理。 | ✅ 非常详尽 |

## 3. 当前实现进度

根据最新的变更日志（`change_2025-12-24_phase6g_pr3c_per_node_infeasible_and_partial_prune.md`），项目目前处于 **Phase 6: Hardening & Extensions** 阶段，具体为 **Phase 6G**。

*   **已完成阶段**：
    *   Phase 1-4：基础 IR、Planner、TVM 后端打通。
    *   Phase 5：工程化完善（Engineering Plan），包括 Jsonl Schema、Artifact Runner 等复现基础设施。
*   **当前焦点（Phase 6G - 2025.12.24）**：
    *   **Branch-and-Bound (BaB) 优化**：正在深入优化完备验证（Complete Verification）的效率。
    *   **Infeasible Pruning**：实现了 Per-node 的不可行分支剪枝（Infeasible Pruning）和缓存（Node Eval Cache）。
    *   **Cache & Reuse**：正在增强跨节点、跨 Batch 的计算复用能力。

## 4. 计划与一致性检查

*   **一致性**：**极高**。代码库的演进严格遵循 `gemini_doc/plan.md` 中规划的路线图。
*   **文档驱动开发**：每一个 PR 或功能变更都有对应的 Markdown 文档（`change_...md`）记录动机、改动和测试方法，这是一种非常高质量的开发模式。
*   **目标对齐**：
    *   **Performance**：通过 TVM 后端和 Planner 优化 Launch overhead 和 Memory footprint 的目标在代码中得到了体现（如 `StoragePlan`）。
    *   **Correctness**：保留了与 `auto_LiRPA` 的全流程对齐测试（`test_phase3_ibp_against_auto_lirpa.py` 等），确保了数值正确性。
    *   **Extensibility**：代码中预留了对不同 Domain 和 Solver 的扩展接口，符合设计文档中的“Extensibility Principles”。

## 5. 总结

BoundFlow 是一个成熟度较高、设计严谨的科研工程项目。它不仅完成了从前端到编译后端的全链路打通，目前正处于通过高级特性（BaB 优化、剪枝、缓存）进一步提升性能和鲁棒性的深水区。工程实现与设计文档高度自洽，展现了良好的软件工程实践。
