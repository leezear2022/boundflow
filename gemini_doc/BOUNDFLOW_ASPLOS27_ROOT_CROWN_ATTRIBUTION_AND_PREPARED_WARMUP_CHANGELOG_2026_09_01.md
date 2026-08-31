# BoundFlow root α-CROWN 归因与 Prepared Warmup 修改记录

status: internally-validated-five-fresh-diagnostic
date: 2026-09-01
external-audit: deferred-by-user
performance-claimed: false

## 1. 本轮结论

本轮没有启动外审。工作从完整 same-solver query 的最大剩余热点出发，先建立 root incomplete
α-CROWN 的闭合嵌套时间账，再实现一个不会改变求解语义的 prepared-runtime 优化：在正式 warm query
之前，对同一 model/property 执行一次 root-only incomplete verifier，丢弃其结果，只保留进程级 CUDA、
autograd、library kernel 与 allocator warm state。

在 RTX 4060 Laptop GPU、ResNet2B/property 0、同一 αβ-CROWN host solver、5 对交错 fresh Python
进程上，`S4-ROOT-WARM` 相对 `S4-PREP` 的诊断结果为：

- complete-query speedup geomean：`1.594232243136117x`；
- complete-query worst pair：`1.5619469814489118x`；
- root incomplete speedup geomean：`2.785741712871747x`；
- root incomplete worst pair：`2.729070694071354x`；
- lower 最大绝对误差：`2.0265579223632812e-06`；
- lower sign 与离散 solver semantics：全部一致；
- 平均 warm query 节省：`430.0315198 ms`；
- 平均额外 root warmup：`717.3746414 ms`；
- 仅按本项增量计算的 break-even：`1.6682` 个后续同构 query，即第 2 个 query 后开始净节省。

这是 cached/prepared warm-query 结果，不是 TVM 编译收益，也不是 10x 或 ASPLOS-ready claim。当前只形成
内部诊断证据，`performance_claimed=false` 保持不变。

## 2. Root α-CROWN 的实测结构

代表性 `S4-PREP` 深度归因样本中：

| 事务 | 时间 |
|---|---:|
| complete query | `1314.418 ms` |
| root incomplete | `834.928 ms` |
| root optimized-bounds transaction | `473.129 ms` |
| 5 次 nested compute-bounds | `207.090 ms` |
| 4 次 autograd backward | `243.472 ms` |
| best-return 更新 | `7.169 ms` |
| Adam step | `1.786 ms` |
| clear intermediate | `2.967 ms` |
| 其余 optimized-bounds 自身开销 | `8.379 ms` |

四次 backward 分别约为 `201.531/14.679/13.595/13.667 ms`。第一次占绝对多数，说明查询内混入了
显著的一次性 CUDA/autograd 准备成本；它不是 Adam 标量更新、clip 或 best-state 保存造成的。

嵌套 observer 只有显式 `attribute_root_incomplete=true` 时才安装。正式性能配置默认关闭，避免计时
instrumentation 自身污染 headline。

## 3. 实现边界

新增 `boundflow/runtime/prepared_root_optimizer_warmup.py`：

1. 临时把 complete verifier 设置为 `skip`，只执行真实 incomplete/root α-CROWN；
2. 使用同一 solver 的 model/property/config 形状和算子路径触发 CUDA/autograd 首次准备；
3. 不保留 warmup 产生的 lower、alpha、model 或 solver reference；
4. 在 `finally` 中恢复 complete-verifier policy 与 solver-visible mutable fields；
5. receipt 记录 root-only wall time、status、lower tensor/element 数量；
6. receipt 明确 `exact_model_property_warmup=true`、`query_timing_excluded=true` 和
   `performance_claimed=false`。

新增诊断配置 `S4-ROOT-WARM`。它继承 `S4-PREP` 的 Prepared Verification Request 与 S4 exact-call
executor，但在 query 计时前额外执行上述 root warmup。每个 fresh candidate 进程独立支付 warmup，
没有跨 control/candidate 共享缓存。

## 4. 五对原始结果

| pair | S4-PREP query ms | S4-ROOT-WARM query ms | query speedup | root speedup |
|---:|---:|---:|---:|---:|
| 0 | `1143.132` | `719.700` | `1.588344x` | `2.770621x` |
| 1 | `1148.548` | `722.903` | `1.588799x` | `2.729071x` |
| 2 | `1160.215` | `716.940` | `1.618288x` | `2.823175x` |
| 3 | `1151.322` | `737.107` | `1.561947x` | `2.837775x` |
| 4 | `1165.534` | `721.942` | `1.614442x` | `2.769469x` |

原始诊断目录：`/tmp/boundflow-root-warm-pairs.tGCbCg`。该目录不是版本化 artifact，机器清理或重启后
可能消失。

## 5. 内存与冷启动披露

- candidate warm-query peak allocated 与 control 基本相同，最大增加 `695808 B`；
- peak reserved 从 `390070272 B` 增到 `411041792 B`，增加 `20971520 B`；
- 平均 root warmup 为 `717.375 ms`；
- S4 TIR/static prepare 与 Prepared Verification Request 的既有冷成本仍另行存在；
- 单次一次性 query 把 warmup 算入后不占优；该优化只面向同一编译/验证上下文的重复请求。

## 6. 被否决的 CIBC 直接接法

本轮还实测了把 CIBC 纯 IBP intermediate bounds 直接注入 root α-CROWN：lower 从正常约 `-0.5`
恶化到约 `-566`，unstable neuron 从约 `701` 增到 `3015`，并改变后续 sparse topology。S4 runtime
因 dynamic sparse layout 不一致而 fail closed。

因此该接法已经从生产配置撤销，相关临时 runtime module/test 也未保留。结论是：CIBC 图级 IBP 的
`2.45631x` 不能仅靠把较松 bounds 注入 root 来传播；root 需要 tight CROWN 语义下的编译。

## 7. 下一性能动作

Prepared warmup 解决的是一次性初始化，不是论文所需的 TVM 核心创新。下一动作保持为：

1. capture root 的 5 次 CROWN evaluation、4 次 backward、alpha state 与 start-node topology；
2. 把现有 B4-B2 differentiable Linear/Conv TIR 从 BaB anchor 泛化到 root shape/signature；
3. 用 verification-aware fused forward/custom backward 替换 `207.090 + 243.472 ms` 主体；
4. dense A 只允许 tile-local scratch/recompute，不跨层保存；
5. 先做单 root correctness，再比较 `S4-PREP / S4-ROOT-WARM / compiled-root` 完整 query。

只有 compiled-root 在 wrapper-inclusive query 中继续带来收益，才进入多层 fusion、arena 与 CUDA Graph。
