---
status: draft-corrected-by-s4-1d-construction-v3
date: 2026-08-28
type: implementation-plan
topic: boundflow
slug: asplos27-s4-1d-all-state-evaluator-closure
stage: s04
depends-on: validated-s4-1c-compressed-gradient-emitters
execution-authority: false-pending-s3-external-audit-s4-0-s4-1a-s4-1b-s4-1c
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1D：all-state single-evaluation evaluator closure蓝图

## 0. 直接结论

S4-1D不再增加新算法，而是把此前1A/1B0/1B/1C收束为一个唯一production-shaped prepared evaluator：

```text
S4-0  tensor-free mutable-state admission + live source lease projection
  ↓
S4-1A ordered parameter/gradient buffers + lease/version
  ↓
S4-1B pass A ternary/binary selectors + pass B six selected-primal V values
  ↓
S4-1C pass C six dα + active dβ + terminal lA phase
  ↓
PreparedS4AllStateCrownEvaluatorV1.evaluate(request)
```

一次调用必须产生一个lower、六条ordered dα、六条ordered dβ（1 physical+5 empty token）和一个可重放receipt。
terminal mode只在ordinal 9额外产生六lA lease；非terminal不能出现handoff。

S4-1D通过只证明single-evaluation correctness/ownership，不接Adam、不证明10/9 trajectory、不计时。

本蓝图的事务、内存和artifact细节已由
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_1D_EVALUATOR_TRANSACTION_IMPLEMENTATION_READINESS_2026_08_28.md`
收紧：旧`386,712 B`账和“失败后reset/retry”语义已被取代。

2026-08-29逐文件施工又修正：result必须为opaque sealed-consumer capability而非raw Tensor getter；parent/child
采用9-state正交lifecycle；formal由5+5改为两fixture各六全排列、12 fresh；minimum three-way output+terminal V
sidecar numeric raw=`4,209,984 B`。权威施工包canonical hash=`76da1864...3cd1`。

## 1. 唯一runtime owner

建议新增：

```text
boundflow/runtime/asplos27_s4_all_state_evaluator.py
```

`PreparedS4AllStateCrownEvaluatorV1`唯一拥有：

- S4-0 admission；
- S4-1A parameter/gradient/lower/upstream buffers与version state machine；
- tensor-free compiled module cache handles；
- 两个coefficient arena与bias accumulator；
- 六sign bitmap；
- 一个six-slot effective/terminal-lA arena；
- prepare-time DLPack views；
- evaluation counters、phase tags、leases与receipt builder。

禁止再由S2 wrapper、R31B2 executor、per-site B4-B2 wrapper或global registry共同拥有同一execution。
这些历史对象只作oracle/代码生成资产。

## 2. prepare合同

构造入口：

```text
prepare_s4_all_state_evaluator_v1(
    admission,
    production_plan,
    bounded_arena_trace,
    ordered_buffers,
    fixed_runtime_inputs,
    hardware_identity,
) -> PreparedS4AllStateCrownEvaluatorV1
```

prepare阶段允许：compile/cache lookup、GPU allocation、source→active parameter pack、metadata normalization和DLPack
view建立。禁止provider compute、optimizer mutation、timing headline或live solver mutation。

module cache只含compiled code/receipts，key至少绑定：graph/plan/trace、slot layouts、dtype、compute capability、TVM/
CUDA identities、bound polarity、endpoint/clamp policy。instance tensor/pointer不能进入global cache。

## 3. exact evaluate序列

```text
0. read-only validate request ordinal/version/lease/stream/module/view identities
1. atomically enter EVALUATING, allocate generation, then reset scalar counters and phase tags
2. pass A: compute lower + coefficient propagation + six sign bitmaps
3. pass B: ternary-endpoint selected-primal lowering writes six V slots; coefficient-action VJP remains the oracle
4. pass C: recompute coefficient; at each site emit compressed gradients
5. if terminal: after each gradient read, copy incoming A to same value slot as lA
6. seal lower/gradient/(optional lA) leases
7. build and independently validate execution receipt
8. return result lease
```

2026-08-29 S4-1C施工已把步骤4—5机械展开：coefficient动作10项，gradient emitter 7项，因此nonterminal
Pass C固定17-action；terminal插入六个copy后固定23-action。site31必须
`dα31→dβ31→copy lA31→ReLU31`，其他site才是单reader的`dα→copy→transform/reuse`。S4-1D receipt
必须验证这两个完整action inventory，不能只检查“7 emitter已执行”。

任何步骤失败：

- parameter state version不变；
- result lease不发布；
- live solver不变；
- 若在read-only admission前拒绝，state/generation/buffer/counter全不变，可修正request后重试；
- 若已进入`EVALUATING`，owner立即进入`POISONED_NO_RETRY`，generation永久烧毁；
- 半写arena不恢复、不消费、不跨generation复用；调用方必须close并重新prepare。

## 4. request/result ABI

### 4.1 request

复用S4-1A：

```text
evaluation_ordinal
expected_state_version
require_terminal_handoff
schedule_action_hash
```

S4-1D correctness只准两种fixture：ordinal0/version0/nonterminal与ordinal9/version9/terminal。后者的version9 state来自
冻结production/native trace或独立prepared fixture，不在S4-1D内部运行九次Adam。

### 4.2 result

```text
S4AllStateResultCapabilityV1:
    tensor-free execution_receipt
    consume_into_exact_sealed_policy(driver)
    transfer_terminal_child()       # terminal only, opaque/one-shot
    serialize_into_exact_formal_sink(sink)
    close()
```

它是一个不可复制、不可序列化的opaque composite capability，不公开raw Tensor/view/dict。raw Tensor一旦逃逸就无法由
lease close撤销，因此只能由exact sealed policy/KFSB/formal consumer消费，并在返回后检查consumer不保留source storage。
capability存活时拒绝下一次evaluate。terminal child只可transfer一次；parent/child可任一先close，最后一个close才结束
capability lifecycle。结果metadata按admission slot顺序；S4-1D success不隐式回到`READY`。

## 5. logical memory ledger

当前formal、排除模型parameters/fixed bounds/compiled module内部workspace：

| 类别 | logical bytes | owner |
|---|---:|---|
| active α/β parameters | 17,016 | S4-1A |
| dα/dβ outputs | 17,016 | S4-1A |
| six sign bitmap | 55,296 | S4-1B |
| coefficient-adjoint/terminal-lA shared arena | 149,856 | S4-1B/1C |
| two coefficient arenas | 147,456 | existing R31B1 |
| residual scratch | 0 additional | two offset views inside coefficient arenas |
| lower + upstream + bias accumulator | 72 | evaluator |
| compressed indices + β metadata | 2,862 | S4-1C static metadata |
| 合计 | **389,574** | corrected correctness design ledger |

terminal lA复用V arena、不增加physical bytes或argument DLPack。result-facing普通Torch view相对110个argument
descriptor只额外6个：五个Conv-shaped terminal reshape和一个lower `[D,1]`；site31 `[D,1,100]`复用emitter view。

`389,574 B`以S4-1B证明`selected_endpoint[18,432]`在pass B安全复用一块coefficient arena为前提；S4-1B0
isolated pack/select的caller-owned selector+selected output为`92,160 B`，并不自动满足该alias。若phase proof失败，
本ledger增加`73,728 B`至`463,302 B`。

2026-08-29源码复核推翻了“residual scratch是独立storage”的旧假设：`residual11`与`residual6`分别是两个
coefficient arena的offset slice，因此只新增descriptor、不新增physical bytes。相对最早`386,712 B`账，只应增加
`2,862 B` compressed metadata。VM/cuDNN workspace、allocator metadata与module storage仍须分项披露。
S4-2 Adam m+v另加34,032 logical bytes，不属于S4-1D。

该表不是peak allocated/reserved claim。implementation必须用CUDA allocator计数独立测量，不得用logical sum替代。

## 6. execution receipt

`S4AllStateEvaluationReceiptV1`至少绑定：

```text
admission / plan / trace / ordered-buffer hashes
forward / effective / gradient module hashes
hardware / stream / dtype identities
evaluation ordinal / input state version / generation

logical_evaluation_count=1
lower_result_count=1
coefficient_pass_count=2
effective_graph_count=1
effective_slot_count=6
alpha_gradient_emitter_count=6
physical_beta_gradient_emitter_count=1
empty_beta_token_count=5
terminal_lA_slot_count=(6 if terminal else 0)
terminal_duplicate_crown_count=0

coefficient_arena_count=2
saved_dense_coefficient_count=0
effective_arena_count=1
gradient_output_allocation_count=0
warm_dlpack_view_count=0
warm_python_dispatch_count=0
full_alpha_repack_count=0
autograd_function/registry/history_count=0
provider/fallback/eager/native_shadow_count=0

actual kernel/VM/copy counts
all logical bytes from component receipts
base/S4-1B/full-S4-1ABC prepared argument descriptor count=16/90/110
timing_recorded=false
performance_claimed=false
receipt_hash
```

receipt必须从component receipts与live counters重算，不能由wrapper填常量蒙混。

## 7. correctness protocol

### 7.1 three-way comparison

每个fresh run：

```text
A: production captured/provider result
B: provider-independent native full PyTorch/autograd oracle
C: S4 compiled evaluator
```

另以coefficient-action VJP和float64 no-autograd局部公式复核六V values与compressed emitters。site19必须
显式证明旧二元规则复现`0.0011564247542992234/9`、三元规则关闭至`4.2375177145e-08/0`；existing S2/R31B2、B4-B2
site31/25交集继续作为局部oracle，不能替代A/B full comparison。

### 7.2 6+6 fresh six-permutation order

固定12个fresh subprocess：6个ordinal0/version0/nonterminal，6个ordinal9/version9/terminal；每类逐一覆盖
`ABC/ACB/BAC/BCA/CAB/CBA`。每个process的A/B/C使用source-equivalent、storage-independent state，并各执行一次。
每个process重新：

- load frozen source；
- verify source/model/property/module hashes；
- prepare独立instance buffers；
- run exactly one admitted evaluation；
- serializeraw first，再生成summary。

不得resume partial raw、复用candidate process或从expected trace构造candidate输出。

### 7.3 numeric/discrete gates

- lower max abs/rel `<=2e-4`，sign exact；
- 六dα与active dβ max abs/rel `<=2e-5`，sign exact；
- five empty β token exact；
- ordinal9 terminal six lA max abs/rel `<=2e-4`，shape/order exact；
- all module/state/path/shape/dtype/device/pointer/version discrete fields exact；
- unowned projected dense α/β gradient为0；
- no fallback/shadow/duplicate CROWN。

## 8. artifact/replay

S4-1D closure artifact至少：

```text
manifest.json
protocol.json
source_identity.json
component_receipts.jsonl
raw.jsonl
summary.json
replay.py
```

raw逐run保存A/B/C三方lower、六dα、active dβ、empty token metadata、terminal lA（如适用）、candidate terminal
V-pre-overwrite sidecar、component receipts和environment。全部numeric tensor保存stdlib可解码IEEE raw并绑定dtype/shape/
endianness/content hash。最低numeric raw=`4,209,984 B`；旧`919,680 B`只是5+5 candidate-only历史估算。
projection只能作附加摘要，replay必须从冻结payload重算summary。

tamper至少：source/module/plan/state version、slot order、lower、任一dα、dβ location/sign、empty β、lA phase、counter、
logical bytes、kernel/copy count、claim flag；全重签外层digest后仍应被semantic replay拒绝。

## 9. negative gates

除S4-0—1C已有reason外，S4-1D至少覆盖：

1. component receipt hash链断裂；
2. module cache混入instance tensor；
3. request ordinal/version/terminal mode非法组合；
4. pass A/B/C generation不一致；
5. lower来自pass C或额外CROWN；
6. 任一site gradient缺失/重复/乱序；
7. empty β被physicalized；
8. result发布前component未完成；
9. result lease未释放即下一evaluate；
10. post-begin异常后owner未poison、generation被retry/reuse或半写result被消费；
11. terminal lA在nonterminal出现或terminal缺失；
12. third coefficient pass/11th CROWN；
13. warm allocation/view/dispatch；
14. autograd registry/history出现；
15. provider/native shadow/fallback；
16. logical memory ledger漏掉scratch/metadata或把effective伪写dense-A=0；
17. raw缺12 worker/任一六全排列、串换terminal fixture或partial resume；
18. replay只校digest不重算；
19.全重签semantic tamper未拒；
20. performance/timing/same-solver flag提前true。

## 10. S4-1D关闭与后继

全部通过才允许：

```text
VALIDATED-S4-1D-ALL-STATE-SINGLE-EVALUATION-CORRECTNESS
```

它只开放S4-2 sealed production 10/9 policy trajectory。S4-2必须比较每ordinal lower、六α/六β before/gradient/after、
moments、LR、clamp、keep-best/stop及terminal bridge；S4-P timing继续关闭。

S4-2不得复用S3简化循环；精确driver/receipt/negative gate见
`gemini_doc/BOUNDFLOW_ASPLOS27_S4_2_SEALED_PRODUCTION_POLICY_DRIVER_BLUEPRINT_2026_08_28.md`。其中
production计数口径冻结为evaluation/parameter mutation/scheduler call=`10/9/10`。

建议提交：

1. `feat(runtime): assemble S4 all-state prepared evaluator`；
2. `test(runtime): close S4-1D 12-worker six-permutation correctness`；
3. `artifact: add S4-1D replay and tamper closure`；
4. `docs: close S4-1D and preregister S4-2 trajectory`。

当前状态：

```text
S3 exchange = ready_for_audit
S4-0/S4-1A/S4-1B/S4-1C/S4-1D implementation = closed
S4-2/S4-P = closed
```
