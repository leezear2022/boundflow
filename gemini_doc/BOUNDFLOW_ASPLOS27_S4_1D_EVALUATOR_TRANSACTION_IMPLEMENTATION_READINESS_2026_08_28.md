---
status: implementation-readiness-frozen-pending-s3-external-audit
date: 2026-08-28
type: implementation-readiness
topic: boundflow
slug: asplos27-s4-1d-evaluator-transaction-readiness
stage: s04
depends-on: s4-1bc-selector-gradient-tir-readiness
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1D evaluator事务实施就绪冻结

## 0. 结论

S4-1D现在可以被实现为一个明确的prepared-runtime事务，但S3独立外审尚未返回，因此本文只冻结合同，
不开放production代码。

本轮只读审计纠正了旧蓝图的四个实质问题：

1. S4-1D logical correctness ledger不是`386,712 B`，而是`438,726 B`；旧账漏掉
   `49,152 B` residual scratch和`2,862 B` compressed metadata，共漏`52,014 B`；
2. evaluation一旦进入GPU写阶段，失败后不能在同一prepared object上reset/retry；正确状态是
   `POISONED_NO_RETRY`，generation永久烧毁；
3. lower、六dα、六dβ和可选terminal lA不是若干可独立释放的松散view，而是一个composite result lease；
4. 单次输出很小，5个nonterminal加5个terminal worker的完整IEEE tensor raw合计仅`919,680 B`
   （`0.877075 MiB`），所以S4-1D禁止只保存hash+bounded projection。

以上都是设计修正，不是S4实现、GPU formal或性能结果。

## 1. owner与非owner

唯一owner仍是：

```text
PreparedS4AllStateCrownEvaluatorV1
```

它负责：

- request admission和evaluation generation；
- 六α、一active β和五empty β token的ordered ABI；
- pass A/B/C module与persistent arena；
- 48个prepare-time view的lifetime；
- final finite gate；
- composite result lease与terminal child transfer；
- component receipt、live counter和最终execution receipt。

它不负责：

- Adam、scheduler、keep-best、stop/prune；
- live provider state mutation或atomic commit；
- KFSB三次child CROWN；
- official post、host packet、queue/branch/termination；
- performance timing。

这些边界分别属于S4-2、S4-3、host solver与后续S4-P，不能塞进evaluator回调。

## 2. request admission必须完全只读

`evaluate(request)`先执行read-only admission，不得在此阶段reset counter、清空buffer、增加generation或launch：

```text
1. evaluator state == READY
2. no live result/terminal child lease
3. request schema/version exact
4. ordinal/version/terminal-mode tuple exact
5. schedule action hash exact
6. admission/plan/trace/ordered-buffer hashes exact
7. live source object/storage/version/content guards exact
8. module/cache/hardware/dtype/device/stream identities exact
9. all 48 prepared views pointer/shape/stride/dtype exact
10. component receipts independently validate
```

任一项拒绝：

- state保持`READY`；
- generation不增加；
- buffer/counter/phase不变；
- launch/copy=`0`；
- 没有lease；
- 调用方可以修正request后再次调用同一prepared object。

这里的“可以再次调用”只适用于尚未开始事务的request rejection，不适用于任何GPU写入后的失败。

## 3. 状态机

### 3.1 顶层状态

```text
PREPARED_READY
  -- read-only admission pass --> EVALUATING(generation=1)
  -- read-only admission reject --> PREPARED_READY

EVALUATING
  -- success + receipt validate --> RESULT_LEASED
  -- any failure -------------> POISONED_NO_RETRY

RESULT_LEASED
  -- nonterminal composite close --> CLOSED
  -- terminal child transferred --> PARENT_CLOSED_CHILD_LIVE

PARENT_CLOSED_CHILD_LIVE
  -- terminal child close -------> CLOSED

POISONED_NO_RETRY
  -- close/release --------------> CLOSED
```

S4-1D formal中每个fresh process只执行一次evaluation，所以success后不会隐式返回`READY`。S4-2若需要10次
evaluation，必须由sealed driver引入显式parameter mutation/state-version transition和下一generation合同，不能让
S4-1D先偷偷支持不受控复用。

### 3.2 generation规则

- read-only rejection不消耗generation；
-进入`EVALUATING`时原子分配唯一generation；
- generation不得回退、重置或复用；
- post-begin failure烧毁generation并毒化owner；
- receipt中的pass A/B/C、finite gate、lease和result必须绑定同一generation；
- partial component receipt不能被下一evaluation继承。

独立状态机枚举覆盖14个case，canonical model hash为：

```text
8942bb5970f268f47314265e0a1683947e7d5cddf6d421d3fd80cd778a9627eb
```

枚举结果：5类pre-begin rejection均不改状态；7类post-begin failure均进入`POISONED_NO_RETRY`；
nonterminal与terminal success各一类，均不隐式返回`READY`。

## 4. exact transaction sequence

只读admission全部通过后，执行：

```text
0. atomically enter EVALUATING and allocate generation
1. reset device counters/phase tags
2. pass A: lower + coefficient propagation + six selector packs
3. pass B: six selected-primal V values
4. pass C: recompute coefficient + six dα + one physical dβ
5. terminal only: after each gradient read, copy pre-transform incoming A to lA slot
6. final finite/discrete gate
7. seal one composite result lease
8. build execution receipt from component receipts + live counters
9. independently validate execution receipt
10. publish result and enter RESULT_LEASED
```

`counter reset`已经属于post-begin事务。步骤1—9任一异常都执行同一个慢路径：

- 不发布result或terminal child；
- 不修改live solver和parameter state version；
- 释放本次临时host owner，但保留prepared allocation直到close；
- state=`POISONED_NO_RETRY`；
- retry/fallback/native shadow/queue continue=`0`；
- 保留failure receipt所需的阶段、generation和稳定reason；
- 调用方必须close并重新prepare。

不承诺把半写GPU arena恢复到调用前内容，因为恢复本身会引入额外写入和新的失败面；禁止消费即可形成更强的
fail-closed边界。

## 5. composite result lease

成功结果是一个不可复制、不可序列化、single-close的composite lease：

```text
S4AllStateResultLeaseV1
  lower_view
  six_alpha_gradient_views
  six_beta_gradient_slots       # 1 physical + 5 typed token
  terminal_child_transfer       # terminal only, one-shot
  execution_receipt
```

规则：

- lower与所有gradient view共同持有arena generation；不得单独release后重写其中一部分；
- lease存活时拒绝第二次evaluate；
- nonterminal close直接关闭prepared evaluator；
- terminal child只能transfer一次，重复transfer拒绝；
- parent result可以在child仍存活时close，但arena直到child close才最终释放；
- terminal child只包含六lA的不可变view和lineage/phase receipt，不得携带optimizer/provider callback；
- child close之后任何view访问都拒绝；
- close不等于CUDA allocator立即归还reserved memory，artifact不得混淆logical release和physical free。

## 6. corrected logical memory ledger

formal fixture、排除模型parameters、fixed bounds和compiled-module内部workspace：

| 类别 | logical bytes | physical owner说明 |
|---|---:|---|
| active α/β parameters | 17,016 | 7 physical parameter buffers |
| dα/dβ outputs | 17,016 | 7 physical gradient buffers |
| six selectors | 55,296 | 6 int8 buffers，`-128`为invalid |
| V/terminal-lA arena | 149,856 | 1 storage，6 non-overlap slots |
| two coefficient arenas | 147,456 | 2 persistent storages |
| residual scratch | 49,152 | 2 existing staged storages |
| lower + upstream + bias | 72 | 3 scalar/small outputs |
| compressed static metadata | 2,862 | α indices + β location/sign |
| **合计** | **438,726** | **36个logical physical buffers** |

旧账`386,712 B`漏项为：

```text
49,152 residual scratch + 2,862 static metadata = 52,014 B
386,712 + 52,014 = 438,726 B
```

CUDA allocation探针得到：

```text
logical total                 438,726 B
torch allocated delta         448,000 B
torch reserved delta        2,097,152 B
allocator minus logical         9,274 B
existing source lease bytes    34,008 B  # 只延长lifetime，不是新增allocation
```

该探针只说明ledger在本设计fixture中的物理可实例化性。`allocated/reserved`含allocator行为，不能与logical sum互换；
fixed bounds和compiled workspace仍需implementation receipt单独披露。

修正后的下游已知小计：

```text
S4-1D                                  438,726 B
+ S4-2 Adam m/v                         34,032 B
= S4-2 known subtotal                  472,758 B
+ S4-3 candidate + rollback             68,016 B
= S4-3 known subtotal                  540,774 B
```

这些仍不是peak-memory claim。

## 7. 48-view ABI与component receipt

prepare-time view固定为：

- S4-1A base views=`16`；
- S4-1B/1C emitter所需unique views=`46`；
- emitter与base重叠=`14`；
- additional TIR views=`32`；
- prepared total=`48`。

`46`是七个gradient emitter signature中的unique view scope；`48`是整个prepared evaluator scope，不能互换。
全部view必须在prepare建立且pointer exact，warm invocation的DLPack view creation=`0`。

每个component receipt至少绑定：

- canonical TIR/template/schedule/module/device-source hashes；
- view ordinal、storage token、pointer、offset、shape、stride、dtype；
- selector legality与IEEE exponent nonfinite policy；
- safe-index、clamp endpoint、bound polarity和operation order；
- workspace/alloc-buffer声明；
- launch/copy/cache/fallback counters；
- generation和phase。

最终receipt必须从这些component receipt和live device/host counters重算，不能由Python wrapper填固定常量。

## 8. final finite/discrete gate

在lease seal前必须核验：

- lower、六dα、active dβ和terminal lA（如有）全部finite；
- 六selector只含各自合法值：Ainput为`{-1,0,+1}`，其余为`{0,1}`；
- selector `-128`、gradient qNaN或任意nonfinite一律失败；
- α index、β normalized location/sign、slot ordinal、shape和pointer exact；
- 五empty β仍为metadata token；
- output emitter count无缺失、重复或乱序；
- terminal/nonterminal lA inventory与phase exact。

kernel用canonical qNaN传播错误，final gate把它转为稳定fail-closed reason；不得把NaN替换为0继续。

## 9. formal worker与完整raw

S4-1D冻结为至少10个fresh subprocess：

```text
5 × ordinal0/version0/nonterminal
5 × ordinal9/version9/terminal
```

每个process：

- 重载并核验同一冻结source/model/property；
- 独立prepare A/B/C state；
- exact one evaluation；
- raw先落盘，summary后生成；
- 不resume、不复用candidate process、不从expected trace构造candidate output。

每类5个worker内部预注册A/B/C执行顺序或pair顺序，必须同时保存A production capture、B independent native oracle、
C compiled candidate的输入identity与输出。

### 9.1 full IEEE payload budget

每个worker的candidate numeric payload：

| tensor | bytes |
|---|---:|
| lower `[6,1]` | 24 |
| six dα + active dβ | 17,016 |
| terminal six lA | 149,856 |

因此：

```text
nonterminal worker = 17,040 B
terminal worker    = 166,896 B
5 + 5 total        = 919,680 B = 0.877075 MiB
```

预算canonical hash：

```text
1e2aab39a7f7049a09371fef6ec1e0a01dc1e2ec6b25ed7c4060b2cf78e2f0d6
```

raw必须以stdlib可解码的base64 IEEE bytes保存全部lower/gradient/lA，并绑定dtype、shape、endianness、signed-zero/
NaN policy和content hash。projection只可作为便于阅读的附加摘要，不能替代full payload。

## 10. independent replay与tamper

replayer必须不import BoundFlow、PyTorch、TVM、NumPy或αβ-CROWN，并从protocol/source/raw独立重建：

- worker/fixture/order inventory；
- tensor shape/element/byte count；
- IEEE finite/sign/max-abs/max-rel；
- empty β token和terminal inventory；
- component hash chain、generation、counter和memory ledger；
- summary及其canonical hash。

除既有source/module/plan/state/slot/lower/gradient/lA/counter/claim tamper外，S4-1D新增至少验证：

1. pre-begin rejection伪造为consumed generation；
2. post-begin failure伪造为retry success；
3. poisoned generation被下一raw复用；
4. partial result在receipt validate前发布；
5. composite lease拆成可独立重写的view；
6. terminal child transfer两次；
7. parent/child close顺序伪造；
8. 48-view receipt改为46或反之；
9. 漏掉49,152 B scratch或2,862 B metadata；
10. 以projection替换full IEEE payload；
11. qNaN被改为0并重签；
12. terminal/nonterminal worker配比或fixture串换。

攻击必须同步重签payload/file/summary/manifest外层digest，仍由semantic invariant拒绝。

## 11. implementation顺序与kill gate

S3外审批准后，S4-1D只能在S4-0、1A、1B0、1B、1C逐级关闭后按以下顺序实现：

1. `feat(runtime): add S4-1D read-only request admission and state machine`；
2. `feat(runtime): assemble pass A/B/C with 48 prepared views`；
3. `feat(runtime): add final gate and composite result lease`；
4. `test(runtime): close pre-begin/post-begin/lease negative matrix`；
5. `artifact: close 5+5 full-IEEE replay and fully re-signed tamper`；
6. `docs: close S4-1D and only then open S4-2`。

STOP条件：

- 任一component仍由per-site Python wrapper动态创建view/allocate output；
- failure后需要同一owner reset/retry；
- result不能由一个composite lease覆盖；
- memory receipt不能解释logical与allocator差异；
- formal只保存projection或依赖`.pt`/production validator；
- 为接S4-2提前在evaluator内部修改parameter/moment/scheduler。

## 12. 当前门禁

```text
S3 external audit                         pending
S4-0/1A/1B0/1B/1C/1D implementation      closed
S4-1D GPU correctness/formal/timing       closed
S4-2/S4-3/S4-4/S4-P                       closed
performance/same-solver/query claims       false
```

本文只使S4-1D实现合同达到readiness，不改变任何代码门禁或claim。
