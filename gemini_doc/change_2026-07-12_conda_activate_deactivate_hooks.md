# 变更记录：BoundFlow Conda 激活与反激活钩子

## 修改

- 重写 `scripts/setup_hooks.sh`，成对安装 `activate.d` 与 `deactivate.d` 钩子。
- 激活时自动 source 仓库 `env.sh`。
- 反激活时恢复激活前的变量值；变量原先不存在时保持 unset。
- 覆盖 `BOUNDFLOW_ROOT`、`PYTHONPATH`、`TVM_HOME`、`TVM_LIBRARY_PATH`、
  tvm-ffi 配置和 `TMPDIR`，避免退出环境后残留 BoundFlow 路径。
- 安装过程幂等，重复运行会覆盖同名生成文件。

## 验证

- hook 的状态保存/恢复不再使用 shell-specific 间接参数展开，覆盖仓库默认 zsh 和 bash。
- 在干净 Bash 子进程中设置哨兵 `PYTHONPATH`，激活 `boundflow` 后验证 TVM 与
  BoundFlow 路径存在。
- 执行 `conda deactivate` 后验证哨兵值恢复，BoundFlow 专属变量不再残留。
