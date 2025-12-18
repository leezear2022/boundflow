# 变更记录：Phase 5D PR#10（TVM 编译侧可观测性：PassTimingInstrument + DumpIR）

## 动机

Phase 5C 已经具备 planner 侧的 `config_dump/timings_ms/verify/dump_plan`，Phase 5D PR#9 已经跑通 TVM executor 的 compile+execute 闭环。

下一步要做系统消融/写论文，必须把 **TVM compile 开销**从黑箱变成可观测数据（compile vs run 拆清楚，并能解释“为什么慢/快”）。

本 PR 增加 TVM 编译侧的：

- per-pass timing（`tvm.ir.instrument.PassTimingInstrument`）
- 可选 per-pass IR dump（`tvm.ir.instrument.DumpIR`）

并把结果作为 json-able 的 `compile_stats` 暴露出来（默认关闭，不影响正常运行）。

## 本次改动

### 1) TVMExecutorOptions：新增 compile-side observability 开关

- 修改：`boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions` 新增：
    - `enable_pass_timing: bool = False`
    - `enable_dump_ir: bool = False`
    - `dump_ir_dir: str = ".benchmarks/tvm_ir"`
    - `dump_ir_refresh: bool = False`
    - `compile_cache_tag: str = ""`（可选 tag，避免不同配置误命中 cache）

### 2) 编译 wrapper：PassContext + instruments（并保证 render 时机正确）

- 修改：`boundflow/runtime/tvm_executor.py`
  - `_compile_interval_linear_executable()` 在 `relax.build(...)` 外包：
    - `tvm.transform.PassContext(instruments=[PassTimingInstrument(), DumpIR(...)])`
  - 注意：`PassTimingInstrument.render()` 必须在 **退出 PassContext 之前**调用，否则 profile 会被 `exit_pass_ctx` 清空；因此在 `with PassContext(...):` 内部完成 render。
  - `compile_stats`（json-able）包含：
    - `compile_ms`
    - `pass_timing_render`（原始字符串）
    - `pass_timing`（best-effort 解析出的 `[{pass,time_ms}, ...]`）
    - `dump_ir_dir`（若开启 dump）

### 3) 测试：确保 timing/dump 可用

- 新增：`tests/test_phase5d_pr10_tvm_compile_instruments.py`
  - 开启 `enable_pass_timing/enable_dump_ir`
  - 触发一次 compile
  - 断言 `compile_stats` 含 timing 与 dump 目录落盘

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5d_pr10_tvm_compile_instruments.py
conda run -n boundflow python -m pytest -q
```

