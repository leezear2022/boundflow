# S4-1C 外审修正记录

date: 2026-08-31
performance-claimed: false

## 修改

- 将 `indices[-1]` 的负向篡改移到自定义 CUDA stream 之外，并在进入该 stream 前同步，消除默认流初始化/写入与自定义流读取之间的测试竞态；
- 为测试文件补充惰性 TVM 导入对应的 Pylint `import-error` 例外。

生产 TIR、runtime 和数值合同均未修改。

## 验证

- 冻结 S4/R3 联合测试连续三次：`200 passed`、`200 passed`、`200 passed`；
- `mypy tests/test_asplos27_s4_compressed_gradient.py`：clean；
- `pylint tests/test_asplos27_s4_compressed_gradient.py`：`10.00/10`；
- `git diff --check`：PASS。

本修正只关闭测试确定性与静态检查问题，不形成性能结论；后续不中断性能主线，继续实现 S4-1D/S4-2。
