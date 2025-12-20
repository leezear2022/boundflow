# 变更记录：Phase 5D PR#13A（bench_ablation_matrix：统一 JSONL schema + 最小矩阵 + smoke 回归）

## 动机

Phase 5D 目前已经具备：

- planner 侧：`config_dump/timings_ms/verify/dump_plan` 的可复现与可观测（Phase 5C）
- TVM 侧：compile/run 拆分、fusion/call_tir 统计、memory planning baseline（PR#12 系列）

下一步进入论文/系统化消融阶段，最缺的是：**统一的实验矩阵与输出 schema**（保证每次跑出来的数据能直接进论文表格/画图，而不用反复改脚本/字段）。

因此本 PR#13A 先落地：

- 一个可复用的 `bench_ablation_matrix.py`（JSONL：一行一个 run）
- 默认矩阵覆盖关键控制变量（partition/reuse/static_plan/fusion）
- 一个 smoke 测试，防止脚本不可运行/字段漂移

## 本次改动

### 1) 新增 ablation matrix bench（JSONL）

- 新增：`scripts/bench_ablation_matrix.py`
  - 默认矩阵（2×2×2×2=16）：
    - `partition_policy`: `v0_single_task` / `v2_baseline`
    - `reuse_on`: off/on（BoundFlow logical→physical）
    - `memory_plan_mode`: `default` / `disable_static_plan`（TVM StaticPlanBlockMemory on/off）
    - `fusion_on`: off/on（task-level fusion pipeline）
  - 输出：JSONL（每行一个 run），字段包含：
    - `planner`: `num_tasks/num_edges/plan_ms_total/timings_ms/storage(logical/physical bytes est)/reuse_stats`
    - `tvm`: `compile_ms_total/call_tir(after_legalize/after_fuse_tir)/memory_by_scan bytes + estimator stage`
    - `runtime`: `run_ms_avg/p50/p95`
    - `baseline.auto_lirpa`（可选）：`compute_bounds_ms_avg/p50/p95` + correctness gate
  - 提供 `--matrix small`：只跑 1 个配置点，供 CI/快速排查。

### 2) 新增 smoke 回归测试

- 新增：`tests/test_phase5d_pr13_ablation_matrix_smoke.py`
  - 调用 `bench_ablation_matrix.main([...])` 跑 `--matrix small --iters 1 --warmup 1 --no-auto-lirpa`
  - 断言输出为 JSONL 且包含核心顶层字段（meta/planner/tvm/runtime）。

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py

# 真实跑一组配置点（MLP）
conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 3 --iters 20

# 全量矩阵（会编译多次，较慢）
conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix default --warmup 3 --iters 20
```

## 已知限制

- 当前 workload 仅包含 `mlp`。后续扩到 `mnist_cnn` 需要先补齐 TVMTaskExecutor 在 Phase 5 multi-task/scheduler 路径下对 `conv2d/flatten/...` 的覆盖（计划作为后续 PR#14）。
- 若将输出管道到 `head`（例如 `... | head -n 1`），Python/conda 可能仍会在进程退出时打印 stdout flush 的 BrokenPipe 警告；建议改用 `--output <file>` 或直接不截断输出。
