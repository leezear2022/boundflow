# 变更记录：Phase 5D PR#13B（bench 计时公平性/可解释性增强 + env.sh stdout 清洁）

## 动机

Phase 5D 进入系统化消融阶段后，bench 输出需要满足两个“论文级硬约束”：

1) **输出可被机器稳定解析**：JSONL/CSV 的 stdout 不应被环境脚本或提示信息污染；  
2) **计时与对照可解释**：需要区分 compile vs steady-state run，记录编译缓存命中情况，并补充 correctness 的差异幅度（不仅仅是 allclose）。

本 PR 做两类加固：bench 输出与计时字段完善、以及 `env.sh` 的 stdout 污染修复（默认改为 stderr 并支持静默）。

## 本次改动

### 1) `env.sh`：避免污染 stdout（默认改为 stderr，可静默）

- 修改：`env.sh`
  - 将提示信息从 stdout 改为 stderr，避免重定向输出（例如 `> result.jsonl`）被污染。
  - 新增静默开关：`BOUNDFLOW_QUIET=1` 时不输出任何提示。

### 2) TVMTaskExecutor：增加 task-level compile cache 统计（用于 bench 公平性）

- 修改：`boundflow/runtime/tvm_executor.py`
  - 新增 `get_task_compile_cache_stats()`：返回 `task_compile_cache_hit/miss/fail`（json-able）。
  - 在 task-level RELAX_OPS 编译缓存路径中记录 hit/miss/fail（用于矩阵实验解释“为什么某些点 compile_ms_total 不可比/是否发生重复编译”）。

### 3) ablation bench：补齐 compile/run 拆分、baseline setup 拆分、diff 幅度指标

- 修改：`scripts/bench_ablation_matrix.py`
  - 增加 `runtime.compile_first_run_ms`：把“首次运行（包含编译触发）”单独计时。
  - 增加 `tvm.compile_cache_stats`：记录 task-level compile cache 的 hit/miss/fail。
  - auto_LiRPA baseline 增加 `setup_ms`：将 `BoundedModule` 构建开销与 `compute_bounds` 运行开销分开记录。
  - correctness 输出增加 `python_vs_tvm_max_abs_diff_*`、`python_vs_auto_lirpa_max_abs_diff_*`（可解释差异幅度）。
  - 增加 `SIGPIPE` 处理：便于 `... | head` 等用法不产生 BrokenPipe 警告。

### 4) 测试：回归 env.sh 不污染 stdout

- 新增：`tests/test_env_sh_quiet_stdout.py`
  - 断言默认 `source env.sh` 不输出 stdout（只输出 stderr）。
  - 断言 `BOUNDFLOW_QUIET=1` 时 stdout/stderr 都为空。

## 如何验证

```bash
# 单测
conda run -n boundflow python -m pytest -q tests/test_env_sh_quiet_stdout.py
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py

# bench：输出到文件（stdout 应为纯 JSONL）
conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py \
  --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check \
  --output /tmp/boundflow_ablation.jsonl

# 检查首行是 JSON
head -n 1 /tmp/boundflow_ablation.jsonl
python -c 'import json; import pathlib; p=pathlib.Path(\"/tmp/boundflow_ablation.jsonl\"); json.loads(p.read_text().splitlines()[0]); print(\"ok\")'
```

## 备注

- `env.sh` 的提示现在默认写到 stderr：交互使用仍可见；做 artifact/bench 重定向时不会破坏 stdout 的机器可解析性。
- `task_compile_cache_hit` 目前统计的是“编译缓存查询命中次数”（run 期间可能会增长）；做论文统计时更建议重点关注 `task_compile_cache_miss`（实际发生编译的次数）。

