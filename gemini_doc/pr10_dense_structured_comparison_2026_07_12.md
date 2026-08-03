# PR-10 Dense vs Structured ReLU 对照结论

> Artifact：`pr10-dense-structured-frozen-weights-clean-20260712`  
> 代码：`dfcc185`，`git_dirty=false`  
> 矩阵：5 workload × 3 method × 2 mode × 4 spec × 3 domain = 360 rows  
> 结果：354 ok、6 structured OOM、0 correctness failure

## 1. 公平性

- dense/structured 使用同一模型、输入、spec、synthetic domain batch、seed 与 solver 参数；
- 权重明确 `requires_grad=false`；α/β 是 verifier 中唯一优化状态；
- latency 为 warm trace-off 3 次 median/p90；
- peak 为清空 allocator cache 后的独立 trace-off 单次执行；
- trace-on 只分析 logical materialization/lifetime，不进入 latency。

原始证据：

```text
artifacts/phase7a-pr10/pr10-dense-structured-frozen-weights-clean-20260712/profile/raw.jsonl
artifacts/phase7a-pr10/pr10-dense-structured-frozen-weights-clean-20260712/profile/normalized.csv
artifacts/phase7a-pr10/pr10-dense-structured-frozen-weights-clean-20260712/profile/manifest.json
```

## 2. 结构门禁

- dense 的全部 180 个成功 query 均存在 persistent `relu_sign_split`；
- structured 的 174 个成功 query 中 persistent materialization 全部为 0；
- structured materialization 均有 `relu_bias_sign_reduce` 或 `sign_split_*` reason/site，并标记
  ephemeral；
- local A/b、α gradient、plain CROWN、multi-step α、fixed-split αβ 和真实
  `solve_bab_mlp` 搜索与 dense reference 对齐；
- 全量 179 passed、1 个预期 skip。

因此 C1 的“消除 ReLU 后永久 dense coefficient state”成立，但不等于总 materialization 或
真实 peak 必然下降。

## 3. 代表点

### mini-ResNet CROWN，spec=128/domain=32

| mode | events | total logical | persistent | ephemeral | peak allocated | median latency |
|---|---:|---:|---:|---:|---:|---:|
| dense | 14 | 469,762,048 | 469,762,048 | 0 | 440,064,000 | 77.49 ms |
| structured | 30 | 1,006,632,960 | 0 | 1,006,632,960 | 308,959,232 | 710.56 ms |

structured 将 peak 降低约 29.8%，但累计 logical materialization 增加 2.14×，latency 增加
9.17×。原因是 Python eager 下对 nested operator 做多次 ephemeral 重算。

### mini-ResNet α-CROWN，spec=32/domain=8

| mode | total logical | peak allocated | median latency |
|---|---:|---:|---:|
| dense | 58,720,256 | 210,854,912 | 27.39 ms |
| structured | 125,829,120 | 2,152,948,736 | 121.09 ms |

structured autograd 必须保存/重算 nested SignSplit 图，peak 约为 dense 的 10.21×。在
spec=32/domain=32、spec=128/domain=8/32 上，α 和 αβ structured 均 OOM，而 dense 成功。

## 4. 全矩阵聚合

仅比较双方均成功的 paired queries：

| method | pairs | structured peak wins | latency wins | mean peak ratio | mean latency ratio |
|---|---:|---:|---:|---:|---:|
| CROWN | 60 | 50 | 4 | 0.873× | 3.079× |
| α-CROWN | 57 | 0 | 1 | 2.302× | 2.721× |
| αβ-CROWN | 57 | 0 | 2 | 2.223× | 2.236× |

`mean ratio` 只用于机制决策，不作为论文 headline 统计；正式论文仍需独立重复与置信区间。

## 5. Go/No-Go

### PR-10

- Correctness Gate：PASS；
- Structural Gate：PASS；
- Opportunity Gate：PASS（存在 plain CROWN memory-budget regime 与 α/β dense regime）；
- 默认主路径 guardrail：FAIL。

最终决策：**PR-10 以 guarded research path 完成**。默认保持 dense；通过
`BOUNDFLOW_RELU_BACKWARD_MODE=structured` 开启 structured。

正式研究解释为：**PR-10 通过了表示、正确性与研究机会门禁，但否定了“structured 应成为
统一默认表示”的假设。** plain CROWN、无梯度且显存紧张时，structured 是可能牺牲速度换取
可运行性的 memory escape path；α/αβ 或其他需要保留梯度图的 optimized-bound regime 中，
dense 是当前唯一允许自动选择的候选。避免 persistent materialization 不等于更快或更省真实
峰值显存，这一冲突是 PR-11 的直接动机。

### 后续

证据已经证明“一个固定策略始终最好”不成立，因此 Planner 有研究动机。但 Planner 必须把
`requires_grad/bound_method`、spec/domain、memory budget 和 recompute/autograd cost 纳入输入：

- plain CROWN、严格显存预算：structured 可能值得；
- α/β：当前必须 dense，除非先有 fused/custom-autograd/checkpoint lowering；
- 不得因 persistent bytes 为 0 就选择 structured。

在 saved-tensor、allocator 与 optimization-loop 生命周期完成分解前，文档只能表述为“测量
结果提示 autograd-retained relaxation state 可能主导 optimized-bound regime”，不得断言
autograd 是 6 个 OOM 的唯一原因。

真实 BaB domain profile 仍未完成；当前 profile 的 domain 是 synthetic fixed batch，只有
correctness oracle 使用了真实 `solve_bab_mlp` 搜索。
