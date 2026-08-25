# BoundFlow MR3 P-anchor Production Exact-Call Bridge Correctness 预注册

> 日期：2026-08-26  
> 性质：production integration correctness；无timing、无speedup claim  
> 前置：MR2 selected=`P:25/Conv_8`，bridge未实现  
> 性能声明：`performance_claimed=false`

## 1. 唯一问题

在冻结ResNet2B property-0的真实αβ-CROWN exact call中，能否只把P-anchor `25/Conv_8`的
coefficient-sign region替换为已验证的typed structured owner/custom backward，同时让provider继续
拥有其余start nodes、split/history、10/9 optimizer mutation、scheduler/clamp、termination与最终提交。

MR3不优化、不调schedule、不扩S-anchor或multi-site，也不记录任何latency。它只补MR2唯一missing
gate：`production_exact_call_connection`。

## 2. 冻结输入与身份

- model/property SHA256=
  `791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d` /
  `89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff`；
- source capture SHA256=
  `f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc`；
- start node=`25/Conv_8`，P-anchor ordinal与topology receipt必须来自冻结capture，不能按shape猜；
- α=`[2,1,6,86]`、β=`[6,0]`且absent、bounds=`[6,16,8,8]`、weight=`[16,16,3,3]`；
- evaluation/mutation=`10/9`，provider Adam、ExponentialLR、clamp顺序不得改变；
- base evidence=MR2 formal manifest及R3-D2B correctness manifest，均须先replay。

若真实provider版本、model/property、start-node identity或state shape任一不符，worker必须在candidate
launch前fail closed。

## 3. Bridge ownership

### BoundFlow拥有

- P-anchor单site typed admission、Plan/Task/Schedule identity；
- compressed α读取、absent-β验证、bounds/weight/bias读取；
- 该site forward lower/A与custom backward compressed dα；
- 每evaluation一次candidate region dispatch/receipt，最后一次evaluation不执行backward；
- call结束前的candidate result staging，不直接写provider最终query结果。

### αβ-CROWN拥有

- 其余start nodes与所有active-β site；
- split/history/cuts/intermediate bounds及parent lineage；
- loss aggregation、optimizer step、moment、scheduler、clamp；
- termination、domain/result提交与exception语义。

bridge不得保存跨evaluation dense A，不得构造伪zero β，不得绕过provider optimizer，也不得在错误后
部分提交candidate state。

## 4. Paired five-fresh protocol

固定5个pair、10个独立进程，顺序=`PB/BP/PB/BP/PB`：

- `P`=原provider exact call；
- `B`=同一provider exact call，仅P-anchor由bridge替换；
- 每一侧从同一冻结pre-state独立加载，禁止复用前一侧mutation；
- 每进程只运行一次exact call；compile/cache receipt在raw披露但不计时；
- 所有raw先落盘，任一缺失不得resume或生成partial summary。

## 5. Correctness gates

五个pair全部满足：

1. exact call entry identity、phase、start-node顺序、split/history hash相同；
2. candidate P-anchor region dispatch=`10`、forward/backward=`10/9`；外层typed exact-call
   launch/emit/atomic commit=`1/1/1`；
3. provider fallback/eager/native shadow=`0/0/0`，其他site仍走provider且call count不变；
4. 10个evaluation逐步比较P lower、compressed dα、aggregate loss；
5. 9个mutation逐步比较α、Adam `exp_avg/exp_avg_sq`、lr、clamp mask；
6. final exact-call lower/A、termination-visible result、module state与provider baseline等价；
7. discrete identity/count/hash/sign exact；finite float使用`atol=2e-4,rtol=2e-4`，optimizer state
   使用`atol=2e-5,rtol=2e-5`；
8. bridge receipt必须`performance_claimed=false`、`timing_recorded=false`；
9. exception/tamper注入时candidate staging丢弃，provider-visible state保持pre-call；
10. current device/stream、tensor pointer/version ownership在call前后符合合同。

任一pair或任一步失败，结论=`VALIDATED-NO-GO-MR3-P-BRIDGE-CORRECTNESS`，当前bridge关闭。

## 6. Negative gates

专项拒绝至少覆盖：wrong model/property/start node/ordinal、active/nonempty β、α shape/layout、bounds/
weight shape/dtype/device/nonfinite、split/history hash、optimizer step count、duplicate/missing dispatch、
fallback/eager/native shadow、partial commit、current stream/device drift、higher-order grad、receipt/state
hash tamper。

formal artifact tamper需至少14类fully re-signed攻击，并由replay重新执行ledger/trajectory/atomicity
语义检查，不能只验manifest digest。

## 7. Mechanical verdict

- 全部门禁通过：`VALIDATED-MR3-P-PRODUCTION-BRIDGE-CORRECTNESS`，只开放另行预注册的
  single-site bridge timing；multi-site/S-anchor/same-solver complete-query仍不开；
- 任一失败：NO-GO，保留MR2 inventory与历史local correctness，不扩site、不换容差；
- 无法构造真实exact-call hook：`BLOCKED-PRODUCTION-HOOK-MISSING`，先补capture/hook contract，
  不能退化成local wrapper自比。

## 8. 提交顺序

1. `docs(research): preregister MR3 P-anchor production bridge correctness`；
2. `feat(runtime): add fail-closed P-anchor exact-call bridge`；
3. `test(runtime): freeze MR3 bridge negative and atomicity gates`；
4. `bench(research): freeze MR3 five-pair production bridge artifact`；
5. `docs(research): close MR3 bridge correctness route`。

每一步修改独立记录。implementation commit前不得生成formal raw；formal source冻结后从pair 0完整生成。

## 9. Claim boundary

MR3通过最多证明单个P-anchor在真实exact call中完成等价typed bridge。不得claim speedup、B0/B3
parity、multi-site coverage、S-anchor、query/queue收益或ASPLOS-ready。
