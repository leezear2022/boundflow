---
name: "BoundFlow Documentation Index"
description: "Index of key architectural and historical documents for the BoundFlow project to provide context on its evolution, implementation phases, and current state."
---

# BoundFlow Documentation Index

This skill provides a comprehensive index of the documentation for the BoundFlow project. When you need to understand the project's architecture, historical context, or specific implementation phases, refer to the documents listed below. 

All documents are located in the `gemini_doc/` directory unless otherwise specified.

## Core Directives & Main Indexes
- **`README.md`**: The master index of all documents in the `gemini_doc/` folder. It provides the highest-level overview of available records and AE (Artifact Evaluation) steps.
- **`project_evolution_overview.md`**: The definitive overview of the project's evolution from Phase 0 to Phase 6. It maps out the core goals, key capabilities, codebase locations, and test paths for each phase. Start here to trace the development timeline.

## End-to-End & Architectural Views
- **`boundflow_full_pipeline_director_view.md`**: A high-level director's view of the BoundFlow system's engineering and architectural progression.
- **`boundflow_full_pipeline_from_claims_to_ae.md`**: Maps the paper claims down to the engineering implementation and the final AE artifacts.
- **`bound_methods_and_solvers_design.md`**: Details the design rationale behind decoupling the Abstract Domains, Solvers (BaB), and TVM Code Generation.
- **`why_boundflow_not_auto_lirpa_or_tvm.md`**: Explains the core system contribution of BoundFlow compared to standalone algorithms (like `auto_LiRPA`) and generic compilers (like `TVM`).

## Phase Summaries
Each phase of development has a dedicated summary detailing the completed tasks, code changes, and verification commands:
- **`phase0_summary.md`**: Phase 0 (Engineering baselines, editable installs, package cleanup).
- **`phase1_summary.md`**: Phase 1 (Engineering fixes & Primal IR solidification).
- **`phase2_summary.md`**: Phase 2 (PyTorch frontend and model importation to Primal IR).
- **`phase3_summary.md`**: Phase 3 (Correctness baseline, IBP reference, and alignment with `auto_LiRPA`).
- **`phase4_summary.md`**: Phase 4 (System core loop completion: Task/Planner/Executor mapping).
- **`phase5_summary.md`** & **`../docs/phase5_done.md`**: Phase 5 (Paper AE artifacts, reproducible pipelines, and JSONL schemas).
- **`phase6_summary.md`**: Phase 6 (Landing domain methods, complete verification features like BaB, alpha-beta-CROWN, and performance attribution).

## Additional Key Docs
- **`perturbation_support_design.md`**: Documents the design for supporting L-infinity, L2, L1, L0 perturbations uniformly.
- **`tvm_backend_optimization_memo.md`**: Memo on TVM/Relax backend optimizations.
- **`llm_collaboration_workflow.md`**: Standard template for collaborating with LLMs during development.

**Usage:** When tasked with modifying a specific module (e.g., BaB, TVM backend, Planner), always consult the corresponding Phase Summary or Architecture Document to understand the original design decisions and "done definitions".
