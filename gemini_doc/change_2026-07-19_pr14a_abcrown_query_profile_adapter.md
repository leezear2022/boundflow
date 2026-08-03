# 变更记录：PR-14A αβ-CROWN Query Profile Adapter

## 背景

PR-13 已有稳定的 `BoundQuery`、compatibility key、state validity 与 fixed replay，但这些合同只在
仓库内 reduced solver 中验证过。PR-14A 的第一门禁是先观察真实 complete verifier 的
`compute_bounds` 调用分布，并量化当前 backend capability 的覆盖率，不提前替换执行器。

## 修改

- 新增 `VerificationQueryProfile`：只从既有 `BoundQuery` 投影 solver phase、method、α/β/split、
  spec/domain size、layer pattern 与 backend eligibility，不建立第二套 query identity；
- 新增 `VerificationCoverageReport`，保留所有不支持调用及其明确拒绝原因；
- 将 `IBP`、`forward`、`alpha-forward` 纳入统一 `BoundMethod` 枚举，使外部 verifier 的真实方法
  不会被错误记作 CROWN；
- 新增可撤销的 `ABCrownBoundQueryProfiler`，进程内包装外部 `BoundedModule.compute_bounds`，调用
  前生成 PR-13 `BoundQuery` 与 profile，调用后保持上游返回值原样，并在 context 退出时恢复方法；
- 新增 ONNX+VNNLIB runner，可直接调用官方 `ABCrownSolver`，输出 `queries.jsonl`、
  `profiles.jsonl`、`coverage.json` 与来源/hash/config/result manifest；
- adapter 不修改 αβ-CROWN、auto_LiRPA 或仓库 vendored third-party，也不接管 branch/split/cuts/
  termination。

## 当前边界

- 该切片只做 identity/profile capture，不保存 tensor payload，因此还不是 PR-14B fixed replay；
- `compute_bounds` hook 无法恢复 host solver 的 parent query lineage，当前显式保留
  `parent_query_id=None`；β state 无法解析时会写 `split_state_values_unresolved`，不伪造 lineage；
- profile 会做 tensor content hash 和 Python stack phase classification，有观测开销，不能用于性能
  claim；性能测试必须先冻结 trace，再在 recorder-off 路径执行；
- 上游 αβ-CROWN `e5c7e17` 声明 Python 3.11/Torch 2.11，而当前 BoundFlow 环境为 Python 3.12/
  Torch 2.12.1+cu132。本次只补齐缺失 Python 依赖，不降级 Torch；该组合只能作为 integration
  compatibility evidence，正式 artifact 仍需冻结受支持环境或补充兼容性门禁。

## 验证

- PR-14A contract：4 passed；
- PR-13 focused + PR-14A：19 passed；
- Mypy：新增三个 source 文件 success；
- Pylint：新增/修改 source 与 contract test 为 10.00/10；
- Black：Python 3.12 target 无改动；
- 未修改 `boundflow/3rdparty/**`。

