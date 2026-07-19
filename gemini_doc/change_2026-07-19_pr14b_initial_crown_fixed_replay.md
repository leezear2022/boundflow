# 变更记录：PR-14B Initial Plain-CROWN Fixed Replay

## 修改

- 新增 exact `BoxPerturbation`，支持 VNNLIB per-element clipped input bounds；
- `InputSpec.box` 与 `BoundQuery` identity 显式携带 box content identity；
- 新增可撤销的 `ABCrownInitialCrownCapture`，冻结真实 `x_L/x_U/C`、bounds、phase、method 和
  requested outputs，并保留 process-local external replay closure；
- 新增 PR-14B runner，对 external replay、ONNX nominal forward、BoundFlow eager/chunked/TVM
  做分层门禁；bound 或 requested-output 不等价时不生成性能数据；
- 原始 tensor payload 与 manifest 写入 ignored artifact 目录，Git 只提交代码、contract 和结论。

## 结果

- official MLP：nominal、external replay、三条 BoundFlow lower 全部 0 diff；因 external
  lower-only、BoundFlow lower+upper，公平性能门禁 N/A；
- VNN-COMP ResNet-2B prop0：nominal forward max diff `1.67e-6`，但 BoundFlow lower 对
  external max diff `796.765`，符号只对齐 3/9；
- PR-14B 判定 `VALIDATED-NO-GO`，PR-14C 不启动，C3 降级为 C1/C2 基础设施。

## 验证

- PR-14B Box/capture contracts：10 passed；
- PR-14A/PR-13/query/general-DAG/environment 聚焦回归合计 37 passed；
- 完整 `pytest -q tests`：`372 passed, 1 skipped, 6 warnings`；
- Mypy：5 个 runtime/runner 文件通过；
- Pylint：adapter、runner、PR-14B tests 为 10.00/10；
- raw artifacts 保持 `.gitignore`。
