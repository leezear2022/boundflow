# Bench JSONL Schema（Phase 5D）

本项目的系统化消融/性能评测统一使用 **JSON Lines（JSONL）**：每行一个独立的 JSON 对象（一次配置点/一次 run），便于增量追加、流式处理与后处理画图。

本文档固定 `scripts/bench_ablation_matrix.py` 的输出 schema，避免后续新增旋钮时反复改字段/改后处理脚本。

## 输出规范（强约束）

- **stdout 仅输出 JSONL payload**；提示/诊断信息必须走 stderr（见 `env.sh` 的约定）。
- **每行一个完整 JSON 值**，不允许夹杂进度条/日志文本。
- 所有字段必须是 JSON-able（标量/列表/字典），避免写入 Python 对象 repr（除非明确说明）。

## 顶层字段

- `schema_version`：字符串；schema 版本号（当前 `1.0`）。
- `status`：字符串；`"ok"` 或 `"fail"`。**即使失败也必须写一行**，避免矩阵实验“静默丢点”。
- `error`：对象或 null；当 `status="fail"` 时包含错误信息（见下）。
- `meta`：运行环境与可复现信息。
- `workload`：工作负载定义（模型、输入形状、eps、domain/spec）。
- `config`：planner config dump + tvm options（用于复现）。
- `planner`：planner 产物与统计（task DAG、storage reuse）。
- `tvm`：TVM 编译侧统计（compile、call_tir、memory estimate、cache）。
- `runtime`：运行时统计（compile-first-run 与 steady-state run 分离）。
- `baseline`：可选 baseline（auto_LiRPA）。
- `correctness`：对齐/误差统计。

## `error`（失败记录）

当 `status="fail"` 时：

- `error.error_type`：字符串；异常类型名（例如 `ModuleNotFoundError`）。
- `error.error_msg`：字符串；异常消息（简短）。
- `error.traceback_hash`：字符串；traceback 的 sha256，用于去重与聚合。
- `error.traceback`：字符串；完整 traceback（JSON 字符串转义，不会污染 JSONL 的单行格式）。

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
- `device`：对象；设备指纹（最小版本）：
  - `cuda_available`：bool
  - `cuda_device`：int 或 null
  - `cuda_name`：string 或 null
- `env_flags`：对象；关键环境变量快照（用于 AE/复现排障）。

## `runtime`（计时口径）

- `compile_first_run_ms`：浮点数；第一次 `run_ibp_scheduled` 的 wall time（通常包含编译触发 + 执行）。
  - 用途：区分 cold-start / first-run。
- `run_ms_cold`：浮点数；**编译触发后**的一次“冷但不含编译”的运行时间（用于解释 VM init / cache miss 抖动）。
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

## `tvm.compile_keyset_*`（公平性：任务集合指纹）

为了避免“看起来变快其实只是复用了不同的编译结果/任务划分变了”的口径争议，每行还会写入本次 run 内实际编译过的 **task cache key 集合摘要**：

- `tvm.compile_keyset_size`：整数；本次 run 编译过的 unique task key 数量（来自 `TVMTaskExecutor.get_compile_stats()` 的 key 集合）。
- `tvm.compile_keyset_digest`：字符串；对排序后的 key 集合做 sha256 的短摘要（用于跨 run 对比是否是“同一批任务”）。

并额外镜像：

- `tvm.compile_cache_tag`：字符串；compile cache tag（用于人为区分不同实验批次/配置）。


## `correctness`

- `bounds_allclose_to_python`：bool 或 null；TVM vs PythonTaskExecutor 是否 allclose。
- `bounds_allclose_to_auto_lirpa`：bool 或 null；PythonTaskExecutor vs auto_LiRPA 是否 allclose（若 baseline 可用）。
- `python_vs_tvm_max_abs_diff_lb/ub`：浮点数；max absolute diff（lb/ub）。
- `python_vs_tvm_max_rel_diff_lb/ub`：浮点数；max relative diff（lb/ub，分母为 `abs(ref).clamp_min(1e-12)`）。
- `python_vs_auto_lirpa_*`：同上（若 baseline 可用）。

## `baseline.auto_lirpa`（第三方对照：auto_LiRPA）

`baseline.auto_lirpa` 用于把第三方实现（auto_LiRPA）的 **计时口径** 与 **correctness gate** 固化到同一条 JSONL 证据链中。

重要约定：

- baseline **不依赖** partition/reuse/static_plan/fusion 等矩阵旋钮；因此 `scripts/bench_ablation_matrix.py` 会在进入矩阵循环前先计算一次 baseline，并按 `(workload,input_shape,eps,method,warmup,iters,device,dtype,spec)` 做进程内缓存，避免 16 点矩阵重复跑 16 次 baseline。

字段（最小集合）：

- `baseline.auto_lirpa.available`：bool；是否可用（依赖缺失/不支持 op 时为 false）。
- `baseline.auto_lirpa.reason`：string；不可用原因（best-effort）。
- `baseline.auto_lirpa.version`：string；auto_LiRPA 版本（best-effort）。
- `baseline.auto_lirpa.method`：string；当前固定为 `"IBP"`（后续可扩展）。
- `baseline.auto_lirpa.init_ms`：float；构建 `BoundedModule`/绑定扰动输入的初始化耗时（类比编译/构图开销）。
- `baseline.auto_lirpa.run_ms_cold`：float；第一次 `compute_bounds` 的 wall time。
- `baseline.auto_lirpa.run_ms_p50/p95`：float；warmup 后的稳态 `compute_bounds` 计时分位数。
- `baseline.auto_lirpa.cache_hit`：bool；该行是否复用同一进程内的 baseline 缓存。
- `baseline.auto_lirpa.baseline_key`：string；baseline 缓存 key 的短摘要（用于后处理去重/避免矩阵点重复计数）。
- `baseline.auto_lirpa.spec_hash`：string；spec/C 的短摘要（避免未来扩 spec 时缓存误命中）。

兼容字段（历史遗留，等价映射）：

- `setup_ms ~= init_ms`
- `compute_bounds_ms_* ~= run_ms_*`

## `config.tvm_options.compile_cache_dir`（可选：跨进程 TVM 编译缓存）

当 bench 使用 `--tvm-cache-dir <dir>` 时，`TVMTaskExecutor` 会尝试将 task-level RELAX_OPS 的编译产物落盘（共享库 + json spec），并在下次运行时直接加载，从而减少重复编译。

- `config.tvm_options.compile_cache_dir`：string；落盘缓存根目录（空字符串表示关闭）。
- `config.tvm_options.compile_cache_refresh`：bool；是否强制刷新缓存条目。

## 版本演进

- `schema_version=0.1`：引入 `time_utc`、compile cache delta、max_rel_diff 等字段，用于论文级消融的“口径可解释性”。
- `schema_version=1.0`：冻结 Phase 5D 的字段集合与计时/分组口径；后续若扩展字段应 bump 版本并保持后处理兼容。

## 后处理（JSONL → CSV/表格/图）

- 脚本：`scripts/postprocess_ablation_jsonl.py`
  - 读入一个或多个 JSONL 文件，输出到 `out/phase5d/`：
    - `ablation.csv`：扁平化后的逐点记录（适合画图/透视）
    - `tables/ablation_summary.csv`：按关键旋钮分组的汇总表（当前为最小版本）
    - `tables/table_main.csv`：论文主表最小版本（核心分组键 + plan/cold/hot + bytes_est + call_tir）
    - `figures/cache_miss_vs_compile_first_run.png`：示例散点图（若安装了 matplotlib）

### 缺失值约定（非常重要）

- 如果 bench 使用了 `--no-check`，`correctness` 中的 diff 字段会输出为 `null`（结构稳定），后处理不会把缺失当成 0；
  - `ablation_summary.csv` 中对应 `python_vs_tvm_max_rel_diff_max` 会为空；
  - 并额外输出 `python_vs_tvm_rel_diff_missing` 计数，避免误读为“误差为 0”。

## Workload 参数化（用于分组验证）

- `scripts/bench_ablation_matrix.py` 当前支持 `workload`：
  - `mlp`
  - `mnist_cnn`

- `scripts/bench_ablation_matrix.py` 支持额外参数（对所有 workload 生效）：
  - `--batch <int>`：覆盖输入 batch size（影响 `workload.input_shape`）
  - `--eps <float>`：覆盖 `workload.eps`
  - 用途：生成真实 JSONL 中 eps/input_shape 不同的记录，验证 postprocess 的 group key 不混组。
