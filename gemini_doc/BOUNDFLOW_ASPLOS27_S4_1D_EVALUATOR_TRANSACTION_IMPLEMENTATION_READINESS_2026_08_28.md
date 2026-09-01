---
status: implementation-readiness-corrected-by-s4-1d-construction-v2
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

此前只读审计纠正了旧蓝图的四个实质问题；2026-08-29逐文件施工又纠正三项组合边界：

1. 2026-08-29进一步源码复核把S4-1D logical correctness ledger从`438,726 B`纠正为`389,574 B`：旧审计把
   coefficient arena内部两个residual scratch slice误作独立storage，重复加了`49,152 B`；真正相对
   `386,712 B`旧账新增的只有`2,862 B` compressed metadata；
2. evaluation一旦进入GPU写阶段，失败后不能在同一prepared object上reset/retry；正确状态是
   `POISONED_NO_RETRY`，generation永久烧毁；
3. lower、六dα、六dβ和可选terminal lA不是若干可独立释放的松散view，而是一个composite result lease；
4. 旧`919,680 B`只是5+5 candidate output，不是A/B/C三方raw；formal改为两fixture各六全排列、共12 worker，
   三方output+terminal V sidecar最低numeric raw=`4,209,984 B`；
5. terminal child transfer与parent close是正交动作；状态机修正为9 states/14 legal transitions/67 invalid；
6. raw Tensor getter无法提供可撤销capability；result/child必须opaque，只能被exact sealed consumer消费。

权威逐文件施工包为`BOUNDFLOW_ASPLOS27_S4_1D_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`，canonical
hash=`76da1864...3cd1`。以上都是设计修正，不是S4实现、GPU formal或性能结果。

## 1. owner与非owner

唯一owner仍是：

```text
PreparedS4AllStateCrownEvaluatorV1
```

它负责：

- request admission和evaluation generation；
- 六α、一active β和五empty β token的ordered ABI；
- pass A/B/C module与persistent arena；
- S4-1B 90个、完整S4-1A/B/C 110个prepare-time argument descriptor的lifetime；
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
9. all 110 prepared argument descriptors pointer/shape/stride/dtype/offset exact
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

### 3.1 顶层与parent/child正交状态

```text
PREPARED_READY
  -- read-only admission pass --> EVALUATING(generation=1)
  -- read-only admission reject --> PREPARED_READY

EVALUATING
  -- nonterminal success --> NT_PARENT_OPEN
  -- terminal success ----> T_PARENT_OPEN_CHILD_EMBEDDED
  -- any failure -------------> POISONED_NO_RETRY

NT_PARENT_OPEN
  -- parent close --> CLOSED

T_PARENT_OPEN_CHILD_EMBEDDED
  -- transfer child --> T_PARENT_OPEN_CHILD_LIVE
  -- parent close ---> CLOSED  # embedded child一起撤销

T_PARENT_OPEN_CHILD_LIVE
  -- child close  --> T_PARENT_OPEN_CHILD_CLOSED
  -- parent close --> PARENT_CLOSED_CHILD_LIVE

T_PARENT_OPEN_CHILD_CLOSED
  -- parent close --> CLOSED

PARENT_CLOSED_CHILD_LIVE
  -- terminal child close -------> CLOSED

POISONED_NO_RETRY
  -- close/release --------------> CLOSED
```

完整模型为9 states、9 events、14 legal transitions；其余67种state/event组合稳定拒绝，canonical hash=
`963e723f...599d`。transfer child绝不隐式close parent；child-first与parent-first都必须合法。

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

旧独立failure/success枚举仍覆盖14个case，canonical model hash为：

```text
8942bb5970f268f47314265e0a1683947e7d5cddf6d421d3fd80cd778a9627eb
```

枚举结果：5类pre-begin rejection均不改状态；7类post-begin failure均进入`POISONED_NO_RETRY`；
nonterminal与terminal success各一类，均不隐式返回`READY`。该14-case failure模型与上方14-transition
capability模型是两个口径，不得混淆。

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
10. publish result and enter NT_PARENT_OPEN or T_PARENT_OPEN_CHILD_EMBEDDED
```

S4-1C construction进一步冻结步骤4—5的内部计数：nonterminal=`10 coefficient+7 emitter=17` actions；
terminal=`17+6 copy=23` actions。site31必须在dα和dβ两个V reader都入队后才copy；完整module最多为7个
gradient symbol+6个copy symbol，但copy新增argument descriptor/storage均为0。

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

成功结果是一个不可复制、不可序列化、single-close的opaque composite capability：

```text
S4AllStateResultLeaseV1
  tensor-free execution_receipt
  consume_into_exact_sealed_policy(driver)
  transfer_terminal_child()       # terminal only, one-shot
  serialize_into_exact_formal_sink(sink)
  close()
```

不得公开`.lower/.gradients/.lA`、Tensor tuple/dict、`__iter__`、DLPack或generic callback。现场反例证明：即使lease
close后property拒绝，先前取得的raw Tensor引用仍可读取；因此“close后任何已逃逸view访问都拒绝”在Python/PyTorch中
不可实现。诚实合同是API禁止raw Tensor escape，close后拒绝新的consume/transfer/serialize。

规则：

- lower与所有gradient private view共同持有arena generation；不得单独release后重写其中一部分；
- lease存活时拒绝第二次evaluate；
- nonterminal close直接关闭prepared evaluator；
- terminal child只能transfer一次，重复transfer拒绝；
- parent result可以在child仍存活时close，但arena直到child close才最终释放；
- terminal child为opaque capability，只允许exact sealed KFSB/formal consumer，不公开六lA raw Tensor；
- child close之后拒绝新的consume/serialize；此前raw Tensor逃逸由API禁止与consumer retention audit防止，不能声称可撤销；
- close不等于CUDA allocator立即归还reserved memory，artifact不得混淆logical release和physical free。

sealed consumer必须是repo内exact class并绑定implementation hash，拒绝subclass/duck typing/arbitrary callable；消费后检查
consumer字段与return没有保留source Tensor/storage。

## 6. corrected logical memory ledger

formal fixture、排除模型parameters、fixed bounds和compiled-module内部workspace：

| 类别 | logical bytes | physical owner说明 |
|---|---:|---|
| active α/β parameters | 17,016 | 7 physical parameter buffers |
| dα/dβ outputs | 17,016 | 7 physical gradient buffers |
| six selectors | 55,296 | 6 int8 buffers，`-128`为invalid |
| V/terminal-lA arena | 149,856 | 1 storage，6 non-overlap slots |
| two coefficient arenas | 147,456 | 2 persistent storages |
| residual scratch | 0 additional | 2 offset views inside coefficient arenas |
| lower + upstream + bias | 72 | 3 scalar/small outputs |
| compressed static metadata | 2,862 | α indices + β location/sign |
| **合计** | **389,574** | **34个logical physical buffers** |

该合计隐含一个尚待S4-1B implementation关闭的phase alias：ternary input select的`73,728 B selected_endpoint`
复用existing coefficient arena，而不是独立分配。S4-1B0隔离module实测仍需要distinct output；若production
live-reader/generation/stream证明失败，本表必须增加`73,728 B`至`463,302 B`，不能继续沿用`389,574 B`。

修正关系为：

```text
residual scratch additional physical bytes = 0
386,712 + 2,862 static metadata = 389,574 B
```

旧CUDA allocation探针曾得到：

```text
old over-allocated logical total 438,726 B
torch allocated delta         448,000 B
torch reserved delta        2,097,152 B
allocator minus old logical     9,274 B
existing source lease bytes    34,008 B  # 只延长lifetime，不是新增allocation
```

该probe按36个独立buffer手工实例化，真的给两个scratch分配了额外storage，所以只能说明旧过度分配设计可实例化，
不能验证production reuse。新probe必须从真实prepared owner按storage `_cdata`去重，并核对scratch offset。
`allocated/reserved`含allocator行为，不能与logical sum互换；fixed bounds和compiled workspace仍需implementation
receipt单独披露。

修正后的S4-1D本阶段小计不变；下游S4-2实施就绪审计随后补齐了optimizer step、compressed best、`ret_0`和
validate-before-commit shadow，故下游数字以2026-08-29修订为准：

```text
S4-1D                                  389,574 B
+ S4-2 policy/optimizer additions      102,200 B
= S4-2 known subtotal                  491,774 B
+ S4-3 candidate + rollback             68,016 B
+ S4-3 persistent upper/depths              48 B
= S4-3 known subtotal                  559,838 B
```

这些仍不是peak-memory claim。

## 7. 90/110-view ABI与component receipt

prepare-time argument descriptor固定为：

- S4-1A base=`16`；
- S4-1B selected graph=`49`，与base active α重叠`5`；
- S4-1B pass A额外=`30`；
- S4-1B union=`16+49-5+30=90`；
- S4-1C emitter isolated=`46`，与base重叠`14`、与S4-1B flattened bounds再重叠`12`；
- S4-1C新增=`20`，完整S4-1A/B/C union=`110`。

`46`是七个gradient emitter signature中的unique scope；`48`只是base+emitter局部并集，不是整个prepared evaluator。
全部110个argument descriptor必须在prepare建立且pointer exact，warm invocation的DLPack view creation=`0`。

composite result另使用6个普通Torch view（五个Conv-shaped terminal reshape+lower `[D,1]`）；site31
terminal shape与现有emitter `[D,1,100]` view一致。它们不属于110个argument DLPack，也不新增storage。

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

S4-1D修正为12个fresh subprocess，每类fixture覆盖A/B/C六全排列：

```text
6 × ordinal0/version0/nonterminal   # ABC/ACB/BAC/BCA/CAB/CBA
6 × ordinal9/version9/terminal      # ABC/ACB/BAC/BCA/CAB/CBA
```

每个process：

- 重载并核验同一冻结source/model/property；
- 独立prepare A/B/C state；
- exact one evaluation；
- raw先落盘，summary后生成；
- 不resume、不复用candidate process、不从expected trace构造candidate output。

每类六个worker的A/B/C顺序逐项冻结；三实现使用source-equivalent但mutable storage-independent输入，必须同时保存A
production、B independent native oracle、C compiled candidate的输入identity与完整输出。

### 9.1 full IEEE payload budget修正

每个实现每worker numeric payload：

| tensor | bytes |
|---|---:|
| lower `[6,1]` | 24 |
| six dα + active dβ | 17,016 |
| terminal six lA | 149,856 |

因此：

```text
candidate 6+6 outputs        = 1,103,616 B
A/B/C three-way outputs      = 3,310,848 B
terminal candidate V sidecar =   899,136 B
minimum numeric raw          = 4,209,984 B = 4.01495361328125 MiB
```

旧`919,680 B/1e2aab...`仅为5+5 candidate-only历史估算，不能再称formal完整raw。terminal V sidecar必须在lA
覆盖前抓取；三方output和sidecar外的JSON/receipt/environment bytes另计。

raw必须以stdlib可解码的content-addressed IEEE bytes保存三方全部lower/gradient/lA及terminal candidate V sidecar，并绑定dtype、shape、endianness、signed-zero/
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
8. 90/110-view receipt改成46/48局部口径；
9. 把49,152 B scratch重复算成独立physical storage，或漏掉2,862 B metadata；
10. 以projection替换full IEEE payload；
11. qNaN被改为0并重签；
12. terminal/nonterminal worker配比或fixture串换。

攻击必须同步重签payload/file/summary/manifest外层digest，仍由semantic invariant拒绝。

## 11. implementation顺序与kill gate

S3外审批准后，S4-1D只能在S4-0、1A、1B0、1B、1C逐级关闭后按以下顺序实现：

1. `feat(runtime): add S4-1D read-only request admission and state machine`；
2. `feat(runtime): assemble pass A/B/C with 110 prepared argument descriptors`；
3. `feat(runtime): add final gate and composite result lease`；
4. `test(runtime): close pre-begin/post-begin/opaque-capability negative matrix`；
5. `artifact: close 12-worker six-permutation full-IEEE replay and fully re-signed tamper`；
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
