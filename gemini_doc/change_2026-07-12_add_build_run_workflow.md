# 变更记录：新增日常构建与运行工作流

- 按 BoundFlow Python、TVM Python、TVM C++/TIR、内嵌 tvm-ffi ABI 和 LLVM/MLIR 边界给出
  对应编译与验证命令。
- 固定 TVM `build-boundflow` 和单一内嵌 tvm-ffi 来源。
- 明确禁止普通 `pip install -e` 重新驱动 TVM 根目录 scikit-build。
- 增加分层测试与 ASPLOS 证据要求。
