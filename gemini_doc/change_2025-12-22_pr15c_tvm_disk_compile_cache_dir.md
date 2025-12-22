# 变更记录：PR#15C（TVM task-level compile cache 落盘：跨进程复用）

## 动机

当前 TVMTaskExecutor 的编译缓存主要是**进程内**（dict），在 AE/大矩阵多次运行或多进程跑实验时，会重复触发 Relax build → 造成：

- cold-start 时间显著增加；
- 算力浪费（重复编译同一任务）。

因此补齐一个可选的“落盘编译缓存”：

- 编译时将 task-level RELAX_OPS 的产物导出为共享库（`.so`）并同时写入 json 版 lowering spec；
- 下次运行若命中同一 cache_key_hash，则直接 `tvm.runtime.load_module` 加载并跳过编译。

## 本次改动

### 1) TVMExecutorOptions 新增落盘缓存选项

- 修改：`boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions.compile_cache_dir: str = ""`
  - `TVMExecutorOptions.compile_cache_refresh: bool = False`

### 2) task-level RELAX_OPS 编译：disk hit/miss

- 修改：`boundflow/runtime/tvm_executor.py`
  - `_compile_interval_task_relax_ops()`：
    - 先查进程内 cache
    - 再查 disk cache（若启用且存在 `task_<hash>.so` + `task_<hash>.spec.json`）
    - disk hit：加载 module + spec，记录 `compile_cache_event="disk_hit"` 并计入 hit 统计
    - disk miss：正常编译后 best-effort 写入 disk cache（并支持 refresh 删除旧条目）

### 3) bench CLI：透传 cache_dir 选项

- 修改：`scripts/bench_ablation_matrix.py`
  - 新增：`--tvm-cache-dir`、`--tvm-cache-refresh`
  - 透传到 `TVMExecutorOptions`
  - `compile_cache_tag` 默认使用当前 git commit（降低跨版本误命中风险）

### 4) schema 文档补充

- 修改：`docs/bench_jsonl_schema.md`
  - 说明 `config.tvm_options.compile_cache_dir/compile_cache_refresh`

### 5) 测试

- 新增：`tests/test_phase5d_pr15c_tvm_disk_cache.py`
  - 同一 `--tvm-cache-dir` 下二次运行应避免 compile miss（delta miss=0）

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr15c_tvm_disk_cache.py
```

## 备注

- 该落盘缓存是 best-effort：若缓存条目损坏或与当前 TVM/ABI 不兼容，会自动回退到重新编译，不阻塞运行。
- 缓存 key 已包含 task signature + target + fusion/memory 相关旋钮；bench 侧默认附加 git commit 作为 `compile_cache_tag`，用于隔离不同代码版本的缓存空间。

