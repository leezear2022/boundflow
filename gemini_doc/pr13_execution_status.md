# PR-13 执行状态（跨会话恢复入口）

> 当前总状态：**CLOSED / VALIDATED-REDUCED**
> 起点：PR-12 closure `3492d79` / tag `pr12-validated-reduced`
> 代码：`fda5b82`；closure tag：`pr13-validated-reduced`；不自动启动下一研究 PR。

## 冻结目标

PR-13 正式名称为 **Same-Solver BaB Runtime with Query-Aware Multi-Backend Execution**。保持
branch heuristic、priority queue、node ordering、α/β 配置、split/cuts、termination、timeout 和
seed 不变，只替换 bound-call execution path。

## 五切片状态

- [x] PR-13A：`BoundQuery`、compatibility key、state validity、真实 BaB fixed replay；
- [x] PR-13B：dynamic BatchManager、budget/deadline、partial flush、OOM split/retry、metrics；
- [x] PR-13C：same-solver adapter 与 capability-safe α/αβ execution；plain-CROWN multi-backend 保留到评估接线；
- [x] PR-13D：frozen replay + true E2E BaB 分层评估（reduced GPU）；
- [x] PR-13E：closure audit、claims/artifact、tag。

## PR-13A 权威证据

- 代码：`boundflow/runtime/bab_query.py`、`boundflow/runtime/bab.py`；
- 测试：`tests/test_phase7a_pr13a_query_contract.py`、
  `tests/test_phase7a_pr13a_fixed_replay.py`；
- 工件：`artifacts/phase7a-pr13/pr13a-fixed-replay-v6-20260714/`；
- 结果：8/8 replay、max abs diff 0、0 loss、0 duplicate；仅 smoke，不是性能结论；
- 记录：`gemini_doc/change_2026-07-14_pr13a_query_contract_fixed_replay.md`。

## PR-13B 权威证据

- 代码：`boundflow/runtime/query_batcher.py`、`query_executor.py`；
- 测试：`tests/test_phase7a_pr13b_dynamic_batch_manager.py`，以及 PR-13A physical batch replay；
- 工件：`artifacts/phase7a-pr13/pr13b-dynamic-batch-v7-20260714/`；
- 结果：dynamic 8/8、3 batches、0 loss/invalid；fault OOM 8→4+4→2+2+2+2、8/8；
- 边界：CPU smoke、逻辑 clock、fault injection，不是性能/真实 OOM；
- 记录：`gemini_doc/change_2026-07-14_pr13b_dynamic_batch_manager.md`。

## PR-13C 权威证据

- 代码：`boundflow/runtime/bab_query_runtime.py` 与 `bab.py` 可选 adapter；
- 测试：`tests/test_phase7a_pr13c_same_solver_adapter.py`；
- 工件：`artifacts/phase7a-pr13/pr13c-same-solver-v5-20260714/`；
- 结果：original/runtime 7/7 query IDs，7/7 bounds/branch/αβ state，solver status/node counters
  一致，0 loss/invalid；
- forged plain-CROWN capability 在 physical αβ executor 0 调用时拒绝；dispatch-plan cache 可观测，
  compiled-plan cache 明确不适用；
- CUDA custom stream event-only 回归通过；
- 边界：toy CPU correctness smoke；单次 wall time non-authoritative；
- 记录：`gemini_doc/change_2026-07-14_pr13c_same_solver_adapter.md`。

## PR-13D/E 权威证据

- 代码基线：`fda5b82`；工件：`artifacts/phase7a-pr13/pr13d-bab-runtime-v5-20260714/`；
- fixed 16-query：runtime / per-node 96.52×；runtime / batched original 1.024×；
- hard E2E 16-node：runtime / per-node 9.93×；runtime / batched original 0.980×；
- safe/unsafe/unknown status 与 node count 一致，0 correctness failure/loss；
- 判定：`VALIDATED-REDUCED`；non-toy、真实 OOM、PR-12 Planner dispatch 均未完成；
- 记录：`gemini_doc/change_2026-07-14_pr13d_fixed_e2e_gpu.md`、
  `gemini_doc/pr13_closure_audit_2026_07_14.md`。

## 不得越过的边界

- α/β/split query 必须进入 `alpha_beta_dense_split` capability，不得选 plain-CROWN fused TIR；
- 父 α/intermediate 只能 warm-start，父 β/final bounds 不得当 child exact result；
- 不重写 host solver，不实现 persistent GPU queue，不扩 split-aware TIR；
- 96×/9.93× 必须归因于物理 batching；不得声称 runtime abstraction 超越 batched original；
- 不得把 chain-CNN 16-node 结果泛化为 VNN-COMP/non-toy time-to-verify；
- 不得声称 compiled Planner/cache 已进入 αβ/split 查询；
- 下一阶段必须先由人工/多模型重审主张，不自动扩 scope。

## 恢复命令

```bash
conda activate boundflow
source env.sh
git status --short --branch
pytest -q tests/test_phase7a_pr13a_query_contract.py \
  tests/test_phase7a_pr13a_fixed_replay.py \
  tests/test_phase7a_pr13b_dynamic_batch_manager.py \
  tests/test_phase7a_pr13c_same_solver_adapter.py
```
