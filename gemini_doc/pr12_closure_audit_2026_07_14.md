# PR-12 closure audit（2026-07-14）

## 最终判定

```text
PR-12 closure: VALIDATED-REDUCED
closure tag:   pr12-validated-reduced
PR-13 gate:    GO / READY，尚未启动
```

PR-12 不能标记为 `VALIDATED`：fused compile 在 PR-12J 的 Q≤1024 目标区间 0/3 可摊销，硬件
performance counter 因权限不可用，收益只存在于部分 shape/budget/reuse regime，也尚未接真实
BaB/VNN-COMP query stream。但它也不是 `MECHANISM-ONLY`：完整 final-bound correctness、非 toy
mini-ResNet/Conv memory Pareto、预算可行性、compile-aware 自动选择与独立 held-out 已形成系统
价值，且限制和负结果完整保留。

## H–M 门禁审计

| 阶段 | 门禁 | 结论 |
|---|---|---|
| H | 三层机器可读 benchmark contract；历史不公平证据标记 | PASS |
| I | eager/structured/chunked/TVM-unfused/fused 公平 E2E baseline | PASS；fused latency headline FAIL |
| J | compile/load/cache/restart 分解，Q=1..1024 | PASS；0/3 在 Q≤1024 可摊销 |
| K | profiler 与机制归因 | CUPTI activity PASS；硬件 counter UNAVAILABLE |
| L | 只冻结一个优化分支 | PASS：`E_STOP_OPTIMIZING_TIR` |
| M | compile-aware、多预算、全新 held-out | PASS / validated-reduced |

## 关键证据

- PR-12I：72 rows，54 ok、18 structured N/A、0 correctness failure；mini-ResNet fused E2E
  7.009 ms vs eager 7.234 ms，peak 18.27 vs 19.74 MiB，属于小幅 non-toy Pareto；
- PR-12J：3/3 correct；Linear/Conv not amortizable；mini-ResNet fresh/disk/process break-even
  4668/1062/4450 queries，均超出 Q≤1024，且不优于 chunked；
- PR-12K：30/30 correct；相对 TVM-unfused 最大 launch 降幅 1.96%；5% 阈值下 3/6 退化、
  1/6 改善、2/6 中性；
- PR-12M：calibration/final 各 25/25 correct；75 decisions；72/72 feasible opportunities
  选择可行 backend；0 unsafe；feasible median/p90/max regret 1.000/1.000/1.016×；
- PR-12M 自动选择 eager/chunked/structured/fused 为 47/12/3/13，fused 从 cold/mixed 各
  1 次变为 warm-Q1024 的 11 次；
- 3 个 16 MiB capacity failure、torch.compile N/A、ncu 权限失败、v1/v2 测量实现 bug 均未删除。

## 正确性与安全边界

- general-DAG fanout 只允许 single-consumer Affine→ReLU fusion，额外 consumer 确定性 fallback；
- runtime 重验 graph/version/step contract；α、β、grad、split、不支持 Conv 在 executor 前拒绝；
- TVM-FFI 显式桥接当前 PyTorch CUDA stream，测试不依赖 global synchronize；
- complete final-bound 与 eager reference 对齐；PR-12M 0 unsafe backend；
- capability 仍限 static FP32 CUDA plain CROWN、Linear 与有限 groups=1/dilation=1 Conv。

## 工件完整性

closure 重新计算 I/J/K/M 十一个 primary hash。唯一初始异常是人工审计命令误填 PR-12I
expected SHA；PR-12I raw、runner manifest 与 report manifest 三者实际一致为
`fe65faf578a6597aeac85c4435348ff1bd9fac85d06f353114406bf9573ebc31`，不是工件漂移。
J/K/M 记录值全部匹配。PR-12M fit/replay model SHA 均为
`dc56c58b83ea355097ff14fe42e48599d16b3ed3e391c7c30f3febf7b2dcfa59`。

第三方子模块仍为：

```text
auto_LiRPA  9d100ec070868440b48d34e2f1dd21b97aab9172
tvm         6248b5db43505fbcfb13cc289d11877d5d2649e8
tvm-ffi     438f6439148b059d424ce2cc2a348736923f6948
```

仓库源码、脚本、测试和正式文档未发现来自无关项目的标识或实现；第三方 SHA 未变化。

## PR-13 Go/No-Go

PR-13 硬门禁逐项满足：

- closure tag：本 closure 提交创建；
- E2E correctness：满足；
- structured/TVM-unfused baseline：满足；
- compile amortization：已测且保留负结果；
- 独立 final held-out：满足；
- unsafe：0；
- non-toy repeated-query value：mini-ResNet Pareto + compile-aware 避免错误 fused 选择；
- closure 不是 `MECHANISM-ONLY`。

因此 PR-13 可启动，但不在本提交中启动。PR-13 只允许推进真实 `QueryState/QueryBatch`、BaB
adapter、scheduler/cache observability；不得回到 TIR 试参，也不得把 PR-12 reduced 证据描述成
论文级完整验证。

## 最终验证

```text
PR-12M focused/integration: 9 passed
全量：                      340 passed、1 skipped
mypy：                      success
pylint：                    10.00/10
Black / git diff --check：  通过
```
