# 2026-07-14：PR-13C Same-Solver Adapter

## 阶段判定

现有 host-side BaB solver 已增加可选 query-runtime adapter。默认 `query_runtime=None` 路径不变；
启用后只替换 bound-call execution，搜索、branch、heap、node ordering、α/β/split、termination 和
seed 仍由原 `solve_bab_mlp` 控制。

```text
same host solver:                   PASS
query IDs/order/parent:             PASS
per-query bounds/branch/αβ state:   PASS
solver status/node counters:        PASS
invalid capability non-invocation:  PASS
PR-13C overall:                     VALIDATED FOUNDATION
PR-13D evaluation:                  NEXT
```

## 实现

新增 `boundflow/runtime/bab_query_runtime.py`：

- `SameSolverRuntimeConfig` 只配置 batch/memory/wait，不拥有 solver 算法；
- `SameSolverQueryRuntime.execute()` 接收由 solver 已选出的 request group，经 PR-13B
  BatchManager 执行并按原 query order 返回；
- `alpha_beta_dense_split` 只调用 physical αβ dense batch executor；`alpha_dense` 走逐 query
  reference；其他 capability 在调用 executor 前拒绝；
- runtime audit 复用 submitted/emitted/completed/loss/queue/batch/OOM counters。

`solve_bab_mlp` 新增可选 `query_runtime`：single-node 和 existing node-batch 两条路径均可委托；
runtime 返回实际 bounds、α/β tensors 和 branch hint，solver 继续使用这些状态做 warm start、
branch 和 cache。`BoundQueryResult` 因此补充 owned α/β state payload；离线 recorder 可计算
content hash，runtime 热路径不逐节点 hash GPU state。replay gate 以 state tensor 数值对齐为
correctness，精确 hash 只作浮点归约差异诊断。

## Same-solver 对照

专项测试使用 αβ steps=3、node batch=4：

- original executor 与 query runtime 的 status、visited/evaluated/expanded、max queue、batch
  rounds、best lower/upper 一致；
- query ID/order 完全相同；每 query bounds、branch、α version、β version 一致；
- alpha-only serial + 1D restriction path 也保持 solver status/counters；
- forged `plain_crown_fused` capability 在 physical αβ executor 调用计数仍为 0 时被拒绝；
- dispatch-plan cache 的 hit/miss 可观测；αβ/split 的 compiled-plan cache 明确 N/A；
- 非默认 CUDA stream 使用 event-only 同步通过，executor stream ID 与 Torch custom stream 一致。

Authoritative smoke：`artifacts/phase7a-pr13/pr13c-same-solver-v5-20260714/`：

```text
original/runtime query count: 7 / 7
query replay:                 7/7
max abs diff:                0
solver status:               unsafe / unsafe
nodes visited/evaluated:     7/7 for both
nodes expanded:              3 for both
runtime query loss/invalid:  0 / 0
```

工件中的单次 wall time 明确标为 `non_authoritative`：baseline 首次运行含初始化/cache effects，
不能解释为 runtime speedup。

## 后续判定

PR-13D 必须建立正式 fixed-tree 与 true E2E 双评估：

- 同一冻结 query stream 的 per-node original、batched original、runtime eager-only、runtime
  planner/cache ablation；
- 至少一个 non-toy CNN/mini-ResNet/VNN-COMP 风格 workload；
- compile/queue/bound/branch/prune/memory 分解，p50/p90/p99 和 node count；
- true E2E 报 safe/unsafe/timeout 与 time-to-verify，不用搜索树差异隐藏成本；
- 真实 GPU OOM、default/custom stream 和 plan/cache hit 仍需证据。

PR-13D 随后完成 reduced GPU 双层评估，最终 closure 为 `VALIDATED-REDUCED`。上述 non-toy、
真实 GPU OOM、PR-12 Planner coverage 仍未补齐，详见
`gemini_doc/pr13_closure_audit_2026_07_14.md`。
