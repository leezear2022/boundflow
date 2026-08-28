---
status: closed-by-s3-v2-internal-validation
updated: 2026-08-28T13:20:00+08:00
type: plan
topic: boundflow
slug: asplos27-s3-optimizer-runtime
stage: s03
performance-claimed: false
external-audit: deferred-to-next-round-by-user
---

# ASPLOS'27 S3：Optimizer / Runtime 累计候选预注册

## 1. 目标与边界

S2 已证明固定 ResNet2B P-anchor 的一次 CROWN lower + compressed dα VJP 经 canonical
Relax/cuDNN/TIR prepared path 相对 native eager 达到 `4.24538196457207x`。S3 回答下一层问题：

> 这条 prepared CROWN/VJP 路径能否在冻结的 10 evaluation / 9 Adam mutation 本地 wrapper 中保住
> 至少 `3x` 的累计收益，同时保持每一步 optimizer state transition 等价？

本阶段不是通用 optimizer IR、same-solver exact-call、complete-query、BaB queue 或 10x 系统 claim。冻结的
10/9 只代表当前 P-anchor artifact 的受限轨迹；production 的 stop、patience、timeout、prune、keep-best、
restore、active mask 与 scheduler policy 仍由 host 拥有。

S1+S2 combined external exchange 继续保持 `ready_for_audit`，本阶段不覆盖、不关闭它。用户已授权先连续
执行 S3，外审留到下一轮合并进行。

## 2. 唯一候选与所有权

三方固定为：

- `N`：独立 PyTorch eager/autograd full-region oracle；
- `D`：R3-D2B 旧 prepared staged backward，用作历史 bridge 对照；
- `P`：S2 canonical prepared CROWN program 驱动的 S3 direct-VJP wrapper。

`P` 必须：

1. 每个 sample 只 `begin_sample()` 一次；
2. evaluation ordinal 严格为 `0..9`；
3. 每步直接调用 `forward()` 与 `backward(upstream)`，禁止经过
   `_R31B2CompiledFunction`、全局 executor registry 或 PyTorch autograd graph；
4. ordinal `0..8` 使用与 production oracle 相同的 Adam(`lr=0.01`)、
   `clamp_(0,1)`、ExponentialLR(`gamma=0.98`)；ordinal 9 只读取 terminal lower；
5. Relax/TIR module、persistent arena/views、CUDA Graph 与 DLPack binding 在 prepare 阶段建立；warm path
   不得 fallback、eager candidate、native shadow 或动态 DLPack view；
6. 每一步执行后回到 host policy cut；不得把一次无 early-stop witness 伪装为一般 10/9 device loop。

## 3. 正确性门禁

五个 fresh correctness triplet 从同一个冻结 pre-state 独立创建 N/D/P owner。对每个 ordinal 冻结并比较：

- `alpha_before`；
- six lower；
- compressed dα（`[2,1,6,86]`）；
- `alpha_after`；
- Adam `step`、`exp_avg`、`exp_avg_sq`；
- learning rate、`update_after`、evaluation/mutation ordinal；
- terminal lower、terminal α、10/9 evaluation/mutation/scheduler counters。

容差固定：lower `atol=rtol=2e-4`；gradient/α/Adam moments `atol=rtol=2e-5`；lower 与 gradient
sign 必须 exact。P 必须报告 custom forward/backward=`10/10`、CUDA graph replay=`10/10`、fallback/eager/
native-shadow/autograd-registry=`0`、saved dense A=`0`、saved autograd history=`false`。

任一 ordinal 不等价即 correctness NO-GO，不进入 formal performance headline。

## 4. Wrapper-inclusive 性能协议

- 6 个 fresh subprocess，覆盖 N/D/P 六个全排列：`NDP/NPD/DNP/DPN/PND/PDN`；
- 每个 worker：5 个完整 10/9 warmup group + 30 个完整 measured group；
- 每个 mode 在计时外恢复同一 initial α；每个 mode 内新建 Adam 与 scheduler；
- host `perf_counter_ns`，sample 前后 device synchronize；覆盖 Python loop、host policy cut、Adam、clamp、
  scheduler、CUDA submission 与 GPU completion；
- prepare/compile/capture 冷成本单独披露，不进入 warm headline；
- 不删 outlier、不续跑 partial raw、不根据结果改样本数或顺序。

冻结判定：

| 判定 | P/N geomean | P/N worst | P/D geomean | 其他 |
|---|---:|---:|---:|---|
| `VALIDATED-S3-3X-LOCAL-OPTIMIZER` | `>=3.00x` | `>=2.50x` | `>=1.50x` | 全部语义/结构门禁通过 |
| `VALIDATED-REDUCED-S3` | `>=1.20x` | `>=0.98x` | 无硬门槛但必须披露 | 全部语义/结构门禁通过 |
| `VALIDATED-NO-GO-S3` | 其余 | 其余 | 其余 | 或任一语义/结构失败 |

即便通过 3x，`performance_claimed` 仍保持 `false`，正式 claim 只由 closure 文档在 replay、tamper、测试与
DocOps validation 全部完成后授予；它仍只覆盖本地 P-anchor 10/9 wrapper。

## 5. Memory、replay 与篡改门禁

- 在 prepared owner 已建立后，独立 untimed warm run reset peak；报告 PyTorch allocated/reserved delta；
- P warm dynamic allocated/reserved 目标为 `0/0`，若非零必须披露并解释；
- raw-first 空目录生成，manifest 绑定 source commit、代码 blob、model/capture digest、protocol 与全部 raw；
- replay 必须从 raw 重新计算逐步语义、速度、计数与 verdict，逐字节匹配 summary；
- 至少 10 类 fully outer-resigned tamper，覆盖 latency、step tensor、optimizer state、ordinal、execution
  counters、receipt/module identity、gate、claim flag 与 manifest，必须全部拒绝。

## 6. 实现与提交顺序

1. `docs: preregister ASPLOS27 S3 optimizer runtime`；
2. `feat(runtime): add direct-VJP S3 optimizer wrapper`；
3. `test(runtime): close S3 trajectory and fail-closed gates`；
4. `bench: add S3 six-order worker replay and tamper protocol`；
5. source-exact commit 后生成 formal artifact；
6. 同步主计划、execution memo、claims map、README 索引与 closure/change log；
7. targeted/full regression、black/mypy/pylint、DocOps validation，提交并推送；
8. 生成下一轮 combined external-audit handoff，但不在本轮自行批准。

## 7. 后继

只有 `VALIDATED-S3-3X-LOCAL-OPTIMIZER` 才开放 S4/RVIR same-solver exact-call cumulative integration。
`VALIDATED-REDUCED-S3` 只允许做 residual attribution，不自动开放 same-solver 性能；NO-GO 则关闭当前
host-Adam/S2组合，依据 raw 判断是否需要 functional optimizer epilogue 或更粗 submission，禁止改门槛。

## 8. Source-exact safety addendum 与最终状态

v2最终formal之前的连续fresh尝试暴露了selected-value CUDA Graph捕获临时cuDNN workspace、以及TVM
cuDNN TLS workspace析构顺序未定义的问题。为满足fail-closed生命周期要求，source=`1766cbc`在formal前
冻结了更严格的receipt：selected graph replay=`0`、VM invocation=`10`、persistent output-copy TIR=`10`、
warm DLPack=`0`。这取代§3中原定graph replay=`10`的结构预期，但性能门槛与数值协议没有降低；失败证据、
修复顺序和最终结果见S3 change log与formal closure。

最终v2通过3x门槛并关闭为`VALIDATED-S3-3X-LOCAL-OPTIMIZER-V2-PENDING-EXTERNAL-AUDIT`。whole-wrapper
warm dynamic allocated/reserved实测为`13,824/0 B`，非零项来自host-owned Adam首次状态建立，故不主张
whole-wrapper `0/0`。只开放S4 same-solver implementation/correctness，不自动形成same-solver性能claim。
