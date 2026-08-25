---
status: preregistered-active
updated: 2026-08-26T18:58:00+08:00
type: plan
topic: boundflow
slug: mr5-multi-conv-production-bridge-correctness
stage: s01
---

# MR5 Multi-Conv Production Bridge Correctness 预注册

## 1. 前置与唯一问题

MR4 census在5 fresh/150 rows上确认三条生产direct ReLU→Conv edge均具备稳定10/9、absent β与
consumer handoff，并给出`4.5P`静态MAC机会。MR3又证明单C2/P bridge correctness成立但physical
timing NO-GO。

MR5只回答：

> 一个first-class generalized bridge能否在同一个真实outer exact call中，按C2→C1→C0顺序接管
> 三条lower ReLU+Conv路径，并保持逐site lower、compressed dα、10/9 optimizer mutation、完整
> owner state与termination-visible result等价？

本阶段不计时。MR4 full regression已以`1764 passed, 3 skipped`通过，本文件现为active，允许按
第8节顺序实现candidate。

## 2. 三个typed site instance

| Site | ReLU←Conv | Conv input→output | Weight | stride/padding |
|---|---|---|---|---|
| C0 | `/input-4←/input` | `3×32×32 → 8×16×16` | `[8,3,3,3]` | `2/1` |
| C1 | `/input-12←/input-8` | `8×16×16 → 16×8×8` | `[16,8,3,3]` | `2/1` |
| C2 | `/input-24←/input-20` | `16×8×8 → 16×8×8` | `[16,16,3,3]` | `1/1` |

Template可以共享语义，但Instance必须分别绑定channels/spatial/stride/padding/α feature count、buffer
size、symbols与module receipt。任何shape自动广播或fallback到P硬编码实现均拒绝。

## 3. Generalized TIR语义

每site forward仍融合：

1. lower-CROWN ReLU slope/intercept选择；
2. Conv-transpose A传播；
3. Conv bias与ReLU intercept归约；
4. lower A与bias联合输出。

stride=`s`、padding=`p`的forward A映射必须显式满足
`source_h = input_h*s + kernel_h - p`与同构width映射；backward adjoint必须是该映射的精确转置，
不能沿用C2的stride-1索引。custom backward返回incoming-A VJP、compressed α VJP和uniform bias VJP；
不保存跨evaluation dense A，不执行higher-order grad。

## 4. Production ownership

- provider继续拥有split/history、全部α tensors、Adam、clamp、termination和非三site路径；
- bridge只在一个evaluation内持有三site pending handoff和各site persistent plan buffers；
- evaluation顺序必须恰为`C2→C1→C0`，每site begin/consume恰一次；
- bridge不得跨evaluation保留pending dense A；
- three-site forward/backward总计=`30/27`，fallback/eager/native shadow=`0/0/0`；
- C0/C1/C2 β都必须是一个`[6,0]` tensor；任一site active β立即fail closed；
- site module/cache/buffer receipt必须独立，不能把一个site receipt冒充三site稳定。

## 5. Correctness协议

- 5 pair/10独立process，顺序=`PB/BP/PB/BP/PB`；
- baseline是完全原生provider，candidate只替换三site lower path；
- 每evaluation、每site在Conv返回点记录provider/candidate lower A+bias；
- 每evaluation记录outer inner result与full α snapshot；
- 前9次逐步记录compressed α gradient、Adam `exp_avg/exp_avg_sq`、pre/post clamp、step/lr；
- final no-grad evaluation记录termination-visible outer result、target α与完整module owner state；
- general tolerance=`atol=rtol=2e-4`且sign exact；optimizer trajectory=
  `atol=rtol=2e-5`且clamp mask/step/lr exact；
- candidate必须额外用独立PyTorch closed expression检查C0/C1 stride-2 forward/VJP，不得只做TIR自比。

## 6. Atomic failure

独立candidate failure worker在evaluation 5、site C1 forward之后注入异常。outer atomic wrapper必须：

- staged emit=`0`、commit=`0`、rollback=`1`；
- 所有α/optimizer/split/history owner tensor content与storage pointer恢复；
- device/stream不漂移；
- C2已执行产生的临时buffer不得泄露为provider-visible state。

## 7. 机械门禁

只有全部满足才输出`VALIDATED-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS`：

1. source/protocol/三site template-instance-module receipt稳定；
2. 5 pair的逐site、evaluation、optimizer与final semantics全部通过；
3. candidate launches=`150 forward/135 backward`累计，fallback/eager/native shadow=`0`；
4. 三site β absent、order/consumer closure/pending lifecycle exact；
5. stride-2独立oracle通过；
6. atomic failure全部owner恢复；
7. replay及至少18类fully re-signed tamper拒绝；
8. targeted/full regression、Black/mypy/pylint/DocOps通过。

通过只开放另行预注册的MR5 multi-site outer exact-call timing；不自动claim speedup、complete query、
queue、B0/B3 parity或ASPLOS-ready。

任一项失败则状态=`VALIDATED-NO-GO-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS`，保留MR4 census
与MR3 single-site correctness，不计时。

## 8. 提交顺序

1. 本预注册（MR4 full=`1764 passed, 3 skipped`后已激活）；
2. generalized TIR + runtime + isolated C0/C1 math tests；
3. production bridge worker + synthetic/negative tests；
4. clean-source five-pair formal raw/replay/tamper；
5. closure/claims传播；若通过才可另写timing预注册。

## 9. Claim边界

当前文件只冻结未来correctness实验。没有multi-site implementation、没有candidate raw，也没有新增
performance/memory claim；MR3 NO-GO不变，MR4的`4.5P`仍只是一项静态结构账本。
