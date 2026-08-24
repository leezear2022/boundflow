---
status: validated-no-go-r3-2b-dispatch-granularity
updated: 2026-08-25T06:15:00+08:00
type: changelog
topic: boundflow
slug: r3-2b-wrapper-timing-formal-no-go
stage: s01
---

# R3-2B Wrapper-Inclusive Timing 正式 NO-GO 关闭

## 1. Verdict

R3-2B 以 `VALIDATED-NO-GO-R3-2B-DISPATCH-GRANULARITY` 关闭。formal source=
`f43eb7629d4e3006c9f6d9acd1567e92bc49457b`；5对fresh、每worker 3个warmup+30个完整10/9
wrapper样本、semantic/statistical replay与10/10 fully re-signed tamper全部通过。

candidate correctness和memory继续成立，但host-wall wrapper geomean=`0.1339893740788718x`、worst=
`0.13037077164706176x`，远低于冻结GO门槛`1.20x/0.98x`。因此当前single-site compiled recurrence
正式kill，R3-3 active-beta、multi-site、R3-4+和same-solver全部继续关闭。

`performance_claimed=false`。不能把memory收益、R3-1b局部correctness或旧B4-B2 microkernel数字写成
R3-2B speedup。

## 2. 正式性能

artifact：`artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1/`

5个pair的native/candidate median与speedup：

| pair | native (ms) | candidate (ms) | speedup |
|---:|---:|---:|---:|
| 0 | 96.3100 | 738.1405 | 0.1304765x |
| 1 | 101.2534 | 739.1607 | 0.1369843x |
| 2 | 101.0694 | 739.3380 | 0.1367026x |
| 3 | 103.5039 | 793.9197 | 0.1303708x |
| 4 | 100.2321 | 739.2913 | 0.1355787x |

- geomean=`0.1339893740788718x`，即candidate约慢`7.4633x`；
- worst pair=`0.13037077164706176x`，即约慢`7.6704x`；
- 每个median来自30个未裁剪host `perf_counter_ns`样本，sample边界device synchronize；
- Adam创建、10次forward/backward、9次Adam/scheduler/clamp、Python/autograd/FFI dispatch全部计入；
- ONNX/plan/TVM compile/persistent owner preparation、warmup、initial α reset和raw capture均在计时外；
- timed path源码无SHA、`.cpu()`、profiler、memory reset或显式synchronize。

## 3. Correctness与memory保持

- maximum terminal lower/α max abs diff=
  `8.58306884765625e-06 / 2.384185791015625e-07`；
- terminal lower allclose/sign exact，terminal α在`2e-5`内；
- 两侧evaluation/mutation/scheduler=`10/9/9`；candidate custom forward/backward=`10/10`；
- candidate fallback/eager/native-shadow/timing-capture=`0/0/0/0`；
- native absolute peak allocated/reserved=
  `20,687,872 / 27,262,976 B`（5/5一致）；
- candidate absolute peak allocated/reserved=
  `1,209,344 / 4,194,304 B`（5/5一致）；
- worst allocated/reserved ratio=
  `0.05845666485175469 / 0.15384615384615385`。

因此R3的structured owner/custom VJP对memory的假设被支持；失败集中在当前compiled execution的物理
实现，而不是语义漂移或dense-A重新存活。

## 4. 失败机理边界

当前candidate每个evaluation执行约15个compiled forward launch、15个reverse recurrence launch和约
10个P-α VJP/clear launch；完整wrapper约400个细粒度TVM-FFI kernel dispatch。timed path已经移除formal
SHA、CPU raw copy、逐步receipt和每步sync，仍慢7.46x，因此不能把失败归因于审计instrumentation。

但本轮没有CUPTI/Nsight kernel-sum与host-gap归因，尚不能断言主因究竟是：

1. TVM-FFI/launch host dispatch；
2. 未调优的Conv/right-contraction kernel本体；
3. kernel间全局memory traffic与缺少纵向fusion；
4. 三者组合。

所以允许的下一研究动作只能是新的只读R3-D0 microphysics attribution预注册；不得直接声称CUDA Graph、
monolithic TIR或schedule tuning必然可解，也不得继续扩当前variant到active β/multi-site。

## 5. Replay / tamper / regression

- manifest=`36ae84d6b875f8cc036e7dc21b8bf3dec8b8c3dd1329513f5dd19a390b71022c`；
- protocol=`21cc9bc9b3492458691b00db79b0da1af169e592ae871071bd9cfc23ffc5c563`；
- summary=`27a704d20da095edde3ebcd985cc920b6cb01c75f9617f03935991533422510b`；
- tamper report=`3f72ea40a617e522d57373ef1455fa8a0daeeaff1a10c1d9f987c69b1228ffa3`；
- 10类全重签攻击全部拒绝：latency、terminal lower/α、counter、memory、sample count、clock、protocol
  threshold、summary verdict/order；
- R3 targeted=`54 passed`；
- mypy=`clean`，pylint=`10.00/10`；
- full regression=`1606 passed, 3 skipped, 6 warnings in 663.19s`。

## 6. 下一动作

当前工程路线在R3-2B kill gate停止。若继续，必须先单独预注册R3-D0只读归因，测量native/candidate的：

- host wall、CUDA kernel sum、launch count与host gap；
- 每个TIR symbol的count/sum/p50/p95；
- Conv/right-contraction、ReLU coefficient、VJP、Adam的region share；
- CUDA Graph空收益上限与每类kernel无限加速Amdahl上限。

只有归因能证明新variant数学上可达`1.20x`，才能预注册“纵向kernel fusion/调优”或“graph replay”；否则
R3-SO-CVJP性能路线整体NO-GO，但R3-2A correctness/memory证据保留。

