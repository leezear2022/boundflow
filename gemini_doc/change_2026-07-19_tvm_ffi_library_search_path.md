# 变更记录：修复新环境的 tvm-ffi 动态库搜索路径

## 问题

新环境中 TVM 与 tvm-ffi 已经完成构建，但激活脚本只设置
`TVM_LIBRARY_PATH=<TVM_HOME>/build-boundflow`。当前 vendored tvm-ffi 的 loader 不读取该变量，
只从 `LD_LIBRARY_PATH`/`PATH` 搜索；其动态库实际位于 `build-boundflow/lib/libtvm_ffi.so`，
因此 `import tvm` 在加载阶段失败。

## 修改

- `env.sh` 同时把 `build-boundflow/lib` 与 `build-boundflow` 加入 `LD_LIBRARY_PATH`；
- `scripts/install_dev.sh` 的显式 staged 命令使用相同搜索路径；
- Conda activation/deactivation hooks 成对保存并恢复原 `LD_LIBRARY_PATH`；
- `tests/test_env.py` 增加两个构建目录均已暴露的回归门禁。

## 验证

- 重新安装 hooks 并激活 `boundflow` 后，`import tvm, tvm_ffi` 通过；
- `bash scripts/install_dev.sh audit` 应从 TVM unavailable 恢复为 available；
- 不重编 TVM，不改 vendored third-party 源码。
