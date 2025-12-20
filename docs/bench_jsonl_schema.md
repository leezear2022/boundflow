# Bench JSONL Schema（Phase 5D）

本项目的系统化消融/性能评测统一使用 **JSON Lines（JSONL）**：每行一个独立的 JSON 对象（一次配置点/一次 run），便于增量追加、流式处理与后处理画图。

本文档固定 `scripts/bench_ablation_matrix.py` 的输出 schema，避免后续新增旋钮时反复改字段/改后处理脚本。

## 输出规范（强约束）

- **stdout 仅输出 JSONL payload**；提示/诊断信息必须走 stderr（见 `env.sh` 的约定）。
- **每行一个完整 JSON 值**，不允许夹杂进度条/日志文本。
- 所有字段必须是 JSON-able（标量/列表/字典），避免写入 Python 对象 repr（除非明确说明）。

## 顶层字段

- `schema_version`：字符串；schema 版本号（当前 `0.1`）。
- `meta`：运行环境与可复现信息。
- `workload`：工作负载定义（模型、输入形状、eps、domain/spec）。
- `config`：planner config dump + tvm options（用于复现）。
- `planner`：planner 产物与统计（task DAG、storage reuse）。
- `tvm`：TVM 编译侧统计（compile、call_tir、memory estimate、cache）。
- `runtime`：运行时统计（compile-first-run 与 steady-state run 分离）。
- `baseline`：可选 baseline（auto_LiRPA）。
- `correctness`：对齐/误差统计。

## `meta`

- `git_commit`：字符串；短 commit hash（用于定位代码版本）。
- `timestamp`：整数；Unix epoch（秒）。
- `time_utc`：字符串；ISO-8601 UTC 时间（便于日志/多机对齐）。
- `host`：字符串；主机名。
- `platform`：字符串；`platform.platform()`。
- `python`：字符串；Python 版本号。
- `torch`：字符串；PyTorch 版本字符串。
- `torch_num_threads`：整数；`torch.get_num_threads()`。
- `tvm`：字符串或 null；`tvm.__version__`（best-effort）。
- `tvm_home`：字符串；`$TVM_HOME`。
- `seed`：整数；当前固定为 0（便于可复现）。

## `runtime`（计时口径）

- `compile_first_run_ms`：浮点数；第一次 `run_ibp_scheduled` 的 wall time（通常包含编译触发 + 执行）。
  - 用途：区分 cold-start / first-run。
- `run_ms_avg/p50/p95`：浮点数；steady-state 运行时间（warmup 后多次 iters 统计）。
- `warmup/iters`：整数；计时参数。

## `tvm.compile_cache_stats`（公平性）

该字段来自 `TVMTaskExecutor.get_task_compile_cache_stats()`，用于解释矩阵实验中 compile 开销不可比的问题。

- `task_compile_cache_miss`：整数；发生编译（cache miss）的次数。
- `task_compile_cache_hit`：整数；查询命中次数（可能在 warmup/iters 期间持续增长）。
- `task_compile_fail`：整数；编译失败次数。

## `tvm.compile_cache_stats_delta_compile_first_run`

围绕 `compile_first_run_ms` 的增量统计（`after_first_run - before_first_run`），用于避免累计值歧义：

- 同 `tvm.compile_cache_stats` 的三项 key，但语义是 **首次运行阶段的增量**。

## `correctness`

- `bounds_allclose_to_python`：bool 或 null；TVM vs PythonTaskExecutor 是否 allclose。
- `bounds_allclose_to_auto_lirpa`：bool 或 null；PythonTaskExecutor vs auto_LiRPA 是否 allclose（若 baseline 可用）。
- `python_vs_tvm_max_abs_diff_lb/ub`：浮点数；max absolute diff（lb/ub）。
- `python_vs_tvm_max_rel_diff_lb/ub`：浮点数；max relative diff（lb/ub，分母为 `abs(ref).clamp_min(1e-12)`）。
- `python_vs_auto_lirpa_*`：同上（若 baseline 可用）。

## 版本演进

- `schema_version=0.1`：引入 `time_utc`、compile cache delta、max_rel_diff 等字段，用于论文级消融的“口径可解释性”。

## 后处理（JSONL → CSV/表格/图）

- 脚本：`scripts/postprocess_ablation_jsonl.py`
  - 读入一个或多个 JSONL 文件，输出到 `out/phase5d/`：
    - `ablation.csv`：扁平化后的逐点记录（适合画图/透视）
    - `tables/ablation_summary.csv`：按关键旋钮分组的汇总表（当前为最小版本）
    - `figures/cache_miss_vs_compile_first_run.png`：示例散点图（若安装了 matplotlib）

### 缺失值约定（非常重要）

- 如果 bench 使用了 `--no-check` 导致 `correctness` 缺失/为 null，后处理不会把缺失当成 0；
  - `ablation_summary.csv` 中对应 `python_vs_tvm_max_rel_diff_max` 会为空；
  - 并额外输出 `python_vs_tvm_rel_diff_missing` 计数，避免误读为“误差为 0”。
