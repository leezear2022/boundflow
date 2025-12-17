# BoundFlow

**BoundFlow** is a verification-aware compiler and runtime system that treats **LiRPA/CROWN bound propagation** and **Certified Training** as first-class workloads. It leverages a dedicated IR, a global planner, and a TVM-based backend to systematically eliminate the overheads (fragmentation, synchronization, redundancy) inherent in existing Python-based verifiers.

---

## Project Manifesto

*   **North Star Metric**: Decrease end-to-end verification time and increase verifiable scale (larger models, more specs, deeper BaB) under the same verification strength.
*   **Engineering Constraints**:
    *   **Frontend**: Reuse `auto_LiRPA`'s automatic derivation logic where possible.
    *   **Backend**: Built upon **TVM** for code generation and autotuning.
    *   **System Core**: Centered around the **Global Bound Planner** and **Runtime**.

---

## Extensibility Principles

**BoundFlow** is designed to be future-proof. The IR and Planner are built with the following extensibility in mind:

1.  **Pluggable Abstract Domains**: Interfaces are reserved for various robust verification frameworks (e.g., DeepPoly, Zonotope) beyond LiRPA.
2.  **Complete Solver Backends**: Support for offloading to complete solvers like MIP/MILP and SMT (e.g., Gurobi, AlphaBeta-CROWN's BaB).
3.  **Quantized & Bit-Precise Semantics**: **Quantized Neural Networks (QNN)** are treated as first-class citizens. The architecture reserves abstractions for fixed-point and bit-precise semantics, ensuring correctness for QNN verification without rewriting the core compiler.

---

## Architecture

```
PyTorch Model / ONNX
        │
        ▼
   BoundFlow IR  (Primal + Bound Graph + Domain States)
        │
    Global Planner (Fusion, Batching, Reuse)
        │
    BoundTasks
        │
   TVM Backend  --->  Optimized GPU Kernels
```

## Directory Structure

*   `boundflow/ir`: Verification-aware Intermediate Representation.
*   `boundflow/planner`: Global optimization and scheduling.
*   `boundflow/backends`: Code generation backends (TVM).
*   `boundflow/frontends`: Model importers (PyTorch, ONNX).
*   `boundflow/3rdparty`: External dependencies (TVM, auto_LiRPA).
