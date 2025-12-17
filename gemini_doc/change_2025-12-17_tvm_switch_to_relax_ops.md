# 变更记录：TVM 后端从 TE/TIR demo 切换到 Relax 算子实现（默认路径）

## 背景

此前的 TVM 后端 kernel（interval linear/conv2d）是用 TE → `te.create_prim_func` → `tvm.build` 生成的。

考虑到 TE 作为入口逐渐不再是 TVM 推荐方向，这次把“默认路径”切换为：

- 使用 **Relax op** 表达 kernel 逻辑
- 用 **Relax VM** 执行
- 避免我们在仓库内继续手写 TE/TIR（TIR 仍会在 TVM 内部 lowering 产生，但不需要工程侧维护）

## 本次改动

### 1) 新增 Relax kernel builder

- `boundflow/backends/tvm/relax_interval_linear.py`
  - 用 Relax op 实现 IBP 的 interval linear
  - 输出 `(y_l, y_u)` 二元组（VM 返回 `tvm_ffi.container.Array`）
- `boundflow/backends/tvm/relax_interval_conv2d.py`
  - 用 Relax op.nn.conv2d 实现 IBP 的 interval conv2d（NCHW）
  - v0 限制：`groups==1`

### 2) TVMTaskExecutor 默认走 Relax

- `boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions.kernel_style` 默认 `"relax"`
  - 仍保留 `"te"` 作为 legacy demo（方便对照/回退）

## 验证

已通过 TVM 相关对齐测试：

```bash
conda run -n boundflow python -m pytest -q \
  tests/test_phase4c_tvmexecutor_matches_python.py \
  tests/test_phase4c_tvmexecutor_matches_python_cnn.py \
  tests/test_phase4c_tvmexecutor_against_auto_lirpa.py
```

## 备注（关于“先不用 TIR”）

这里的含义是：**工程侧不再手写 TE/TIR**；Relax op 在 TVM 内部仍会 legalize/lower 到 TIR 并编译成可执行代码，这是 TVM 的正常工作流。

