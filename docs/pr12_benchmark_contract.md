# PR-12 Benchmark Contract v1

> Schema：`boundflow.pr12-benchmark-contract/v1`
> 状态：PR-12H 冻结；后续结果必须声明 contract id，历史结果不得冒充新合同。

PR-12 使用三个不可互换的测量层级。任何表格或 headline 必须说明使用哪一层；不同层的数据
不能直接相除生成 speedup。

## 1. Kernel contract：`pr12-kernel-preallocated-v1`

回答“数值 kernel 本身是否更快”。

- 输入已在同一 CUDA device/stream；
- 所有候选输出均预分配；
- 不包含 compile、allocator、backend dispatch、interop、Planner 或 concretization；
- 使用被测 stream 上的 CUDA Events；
- steady state 与 compile 分开报告；
- 不用未同步的 host wall time作为 kernel latency。

## 2. Region-runtime contract：`pr12-region-runtime-v1`

回答“从统一 fused-region backend API 进入后是否仍有收益”。

- 输入已在 CUDA；
- 包含 backend dispatch、必要输出分配、Torch/TVM interop 和 stream management；
- 不包含首次 compile、region matching、Planner 与最终 concretization；
- CUDA Event 与 host wall time都在同一被测 stream 边界闭合；
- 记录 allocator peak allocated/reserved delta、output bytes 与 temporary workspace upper bound。

## 3. End-to-end contract：`pr12-end-to-end-final-bound-v1`

回答“完整 plain-CROWN final-bound 查询是否有效”。

- 包含 region matching、Planner、backend、interop、必要分配、concretization 与最终 bounds；
- compile/load 单列，steady-state 查询不包含首次 compile；
- 对每个 query count 报总时间、warm latency、peak memory、最终 bound correctness；
- default/custom stream 使用相同依赖和同步口径；
- fallback、OOM、timeout 和 compile failure 作为结构化结果保留。

## 4. 共同正确性与计时规则

- 同一 model/input/spec/domain/dtype/device/method；
- FP32 plain CROWN 默认 `rtol=atol=2e-4`，同时记录 max abs/rel diff；
- finite、`lower <= upper` 和最终 decision 必须一致；
- warmup 与 measured groups 分开，默认至少 5 个独立组；
- CUDA event 在被测 stream 上；timed region 不调用全局 `torch.cuda.synchronize()`；
- host wall time允许在边界同步被测 stream；
- compile、IR construction、schedule、serialization、load、first run、warm run 分字段；
- `not amortizable` 必须显式记录，禁止以负 break-even 表示。

## 5. 历史证据边界

- `scripts/benchmark_phase7a_pr12_fused_sanity.py` 的 PyTorch/TVM allocation contract 不一致，
  只能作为 calibration，声明 `compliant=false`；
- `scripts/benchmark_phase7a_pr12_runtime_pareto.py` 的候选执行包含 final-bound computation，
  但 region matching 和 Planner 在 timed call 外，也声明 `compliant=false`；
- PR-12E/G 的历史数值保留，不因合同升级被删除或重写；PR-12I 起生成新 schema 工件。

合同的机器可读定义位于 `boundflow/benchmarks/contracts.py`，每个正式 manifest 必须保存
contract payload 与 SHA-256。
