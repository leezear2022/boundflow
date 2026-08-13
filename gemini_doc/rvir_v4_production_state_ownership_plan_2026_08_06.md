# RVIR-v4 Production-State Ownership 计划与门禁

## 目标

把真实αβ-CROWN solver-core边界上的输入、start-node keyed α、SparseBeta、split/history、
intermediate/reference bounds与optimizer mutation变成BoundFlow独立拥有、可重建、可执行的typed
payload。完成后重新判断B2；在此之前不运行性能比较。

## FSG2事实修正

FSG2 inventory检查了`alpha`、`sparse_beta`、`beta`、`split_beta`，但当前auto_LiRPA实际把
BaB beta对象存放在`node.sparse_betas`（复数）中，每个`SparseBeta`包含：

- `val`：可优化beta值，shape=`[domain_batch,max_history_per_layer]`；
- `loc`：split neuron ordinal；
- `sign`：split方向；
- `bias`：一般branch point时的bias，可为空；
- history owner：`UpdateBoundPreReturn.d_dict["history"]`；
- optimizer copy-out：`UpdateBoundCoreReturn.working_beta`与postprocess后的domain beta values。

因此FSG2“显式beta tensor为0”的结论只能解释为旧探针漏字段，不能作为RVIR-v4的事实基础。
FSG2 initial-CROWN正向结论与“当时B2未准入”仍成立；RVIR-v4必须用修正后的capture重新审计。

## 执行边界

RVIR-v4同时记录两层边界：

1. `BoundedModule.compute_bounds` call tree：保持原24-call parent/depth/phase/result账本；
2. `update_bounds_core(pre_result -> core_result)`：这是production beta/split GPU-heavy ownership边界，
   可在provider attach state之前捕获完整pre-state，在extract之后捕获mutation/result。

只hook第1层会错过`BetaFullData.attach_to_net()`之前的domain/history owner；只hook第2层又会漏掉
initial/alpha初始化。两层必须用共同run/call lineage绑定。

## Typed payload

每个tensor必须包含：

- stable semantic path，不使用Python object address；
- role：input/spec/intermediate/reference/alpha/beta-value/beta-location/beta-sign/beta-bias/
  decision-threshold/result/state-output；
- axes：domain/spec/alpha-polarity/start-node/relu-feature/history-slot；
- shape/dtype/device/content digest；
- ownership：read-only、copy-in、mutable-copy-out；
- alias group与pre/post digest；
- activation layer、start node、split layer等semantic keys。

history必须保存domain ordinal、layer、locations、coefficients、optional bias和depth；optimizer合同必须
保存iteration、learning rates、stop criterion identity、bound polarity、intermediate-bound policy和
determinism设置。

## 阶段与kill gates

### V4-0 — Corrected Capture

- 正式ResNet fixed-one-iteration运行重新捕获24-call tree；
- `sparse_betas.val/loc/sign[/bias]`在beta/split边界均非空且与history一致；
- pre/core/post tensor、alias与mutation closure=100%；
- 旧字段遗漏加入负向回归，禁止再次把plural field漏记为0。

未过：停止，不实现backend。

### V4-1 — Frozen-State Evaluation

- 将provider已优化的post α/β/split state映射到BoundFlow native evaluator；
- 对每个beta/split outer call复算lower，`atol=rtol=2e-4`、soundness方向单独通过；
- state key、domain/spec axes和result shape exact。

未过：记录representation mismatch，不进入optimizer replacement。

### V4-2 — Optimizer Mutation

- 独立BoundFlow backend从pre-state执行相同steps/policy；
- result、post α、post beta逐tensor复核；
- original callback/fallback=`0/0`，mutation copy-out原子提交；
- nested-call count不要求复刻provider内部实现，但outer solver-core work/result/state必须exact。

未过：B2继续NO-GO。

### V4-3 — Whole Core Replacement

- 替换`update_bounds_core`而不调用provider core/compute_bounds；
- branch decision、accepted/pruned domains、parent/depth/node、termination/verdict exact；
- 至少5 fresh correctness runs全部通过，才恢复B2 AB/BA timing。

## 性能纪律

- V4-0—V4-3均`performance_claimed=false`；
- 不用capture/replay开销代表candidate性能；
- B2恢复后仍按原FSG3协议执行5 fresh counterbalanced pairs；
- B3—B7不因RVIR-v4实现而自动升级。

## 提交序列

1. `docs(runtime): preregister RVIR-v4 production state ownership`；
2. `feat(runtime): add RVIR-v4 typed state payload`；
3. `feat(bench): capture corrected αβ-CROWN core state`；
4. `feat(runtime): add frozen-state evaluator mapping`；
5. `feat(runtime): replace optimizer mutation and solver core`；
6. `bench(runtime): close RVIR-v4 and decide B2 admission`。

## V4-0A Typed State 实现记录

- 新增`boundflow/runtime/rvir_v4_production_state.py`：digest-bound tensor、semantic path/axes、
  source device、read/copy-in/mutable-copy-out与alias group；
- alpha按`activation/start-node`保存`alpha_polarity/start_spec/domain/feature`axes；
- beta只从正确的plural `sparse_betas`捕获`val/loc/sign[/bias]`，singular-only对象在beta phase
  fail closed；
- history按domain/layer保存location/coefficient/optional bias，并与SparseBeta loc/sign/bias前缀
  逐项一致性校验；
- pre/post mutation receipt要求path集合、role/axes/shape/dtype均不漂移；tensor tamper和history
  mismatch均拒绝；
- 11项定向单测通过；这只完成typed ownership合同，不代表backend已通过。

## V4-0B Corrected Capture 实现记录

- 新增`scripts/run_rvir_v4_production_state_capture.py`，同一真实GPU run同时观察
  `BoundedModule.compute_bounds` call tree与`update_bounds_core(pre_result -> core_result)`；
- 修正FSG2探针遗漏：beta phase只接受plural `node.sparse_betas`，真实run捕获6个
  `SparseBeta.val/loc/sign`组；singular或只有空容器均fail closed；
- 捕获input/spec/intermediate/decision-threshold、activation/start-node keyed alpha、domain/layer keyed
  history、optimizer policy以及pre/post alpha/beta mutation；history的location/sign/bias/score/depth均进入
  stable hash，ReLU隐式零bias仅在history bias逐项为0时允许省略tensor；
- 诊断GPU run通过冻结门禁：24 calls=`12 initial + 1 alpha + 11 beta`，真实core=`1`，
  history entries=`36`，beta value tensors=`6`，mutation receipts=`12`、changed=`7`；
- capture只允许PyTorch `weights_only=True`安全加载；NumPy scalar在worker边界归一化为Python整数；
- artifact replay会重建typed snapshot、重算tensor/snapshot/mutation/summary digest与JSON projections，
  并验证代码来源；`performance_claimed=false`；
- 本节数字来自正式生成前的诊断run。提交并冻结代码后还必须生成正式artifact和独立replay，
  V4-0才可关闭；V4-1/V4-2/B2当前仍未准入。

## V4-0 Closure

状态：`VALIDATED-CORRECTED-CAPTURE`，V4-0关闭，V4-1准入。

- source commit=`6ecab7c68b56734831a297eeef487234e622a43a`；
- artifact=`artifacts/rvir-v4-production-state/resnet2b-core-capture-v1`；
- summary hash=`86d3365c…a2ff2`，manifest hash=`d8fe50fd…2deb4`；
- 正式GPU run与诊断结构一致：24 calls、1 core、36 history entries、6 beta value tensors、
  12 mutation receipts、7 changed mutations；
- 原样semantic replay=`replay-passed`；payload tensor被修改且外层file/manifest digest同步重签后，
  replay仍因内部tensor content hash失败；
- 全量回归=`1118 passed, 3 skipped`；新增artifact replay/tamper focused tests另为`2 passed`；
- V4-0只证明修正后的production-state ownership/capture/replay，尚未证明BoundFlow evaluator或
  optimizer replacement。V4-1只允许消费冻结post α/β/split state复算lower；V4-2与B2仍关闭。

## V4-0C Alpha Layout 修正与重新开启

V4-1映射前检查发现v1 capture仍缺少sparse-feature alpha的原神经元索引。证据是生产alpha长度与
当前intermediate unstable count不一致，例如`/input-16`为`121 vs 119`。auto_LiRPA明确通过
`node.alpha_indices`解释压缩feature轴，并可能通过`alpha_lookup_idx`解释压缩spec轴；只保存alpha
数值无法无损重建relaxation。

因此上节closure降级为“value/history capture closure”，不能作为完整V4-0 ownership closure：

- v1 artifact保留且可replay，但标记superseded-for-V4-1；
- V4-1准入撤回，B2继续关闭；
- v2 capture新增每层full feature shape、每个coordinate的`alpha_indices`与非空
  `alpha_lookup_idx`，全部进入typed tensor digest和snapshot hash；
- 正式门禁固定ResNet2B应有6个feature-shape tensors和16个coordinate-index tensors；
- v2正式artifact/replay/tamper通过后才重新关闭V4-0并准入V4-1。

v2诊断run已通过：6个activation layer均有feature shape，coordinate indices合计16个
（五个CNN layer各3维、最后linear layer 1维），所有index range和compressed alpha末轴长度逐层
exact；24-call/core/beta/history/mutation计数保持不变。诊断summary hash=`9d1c71b0…d1dbdb`，
`performance_claimed=false`。该数字仍需clean committed source的正式artifact冻结。

### V4-0C 正式关闭

- source commit=`586590553c2304f23d3cb760ca327a2b03568c44`；
- artifact=`artifacts/rvir-v4-production-state/resnet2b-core-capture-v2`；
- summary/manifest hash=`9d1c71b0…d1dbdb`/`eea6547a…e3199`；
- 原样replay通过；layout index被改为越界值后，攻击方同步重算tensor content hash、snapshot hash、
  capture file digest与manifest hash，仍被alpha feature index semantic range gate拒绝；
- V4-0现在以`VALIDATED-CORRECTED-CAPTURE-V2`关闭，V4-1重新准入；v1保留为可重放历史工件，
  但不得作为frozen-state evaluator输入；V4-2/B2仍关闭。

## V4-1A Frozen-State Evaluator 实现记录

- 新增`boundflow/runtime/rvir_v4_frozen_state.py`，显式绑定provider activation/preactivation/
  start-node与BoundFlow primal value，不使用对象地址或隐式ordinal；
- 用v2 `alpha_indices`把post alpha polarity-0重建为per-domain dense lower slopes；
- 用history location/sign重建dense split，并把post SparseBeta value按location散射为dense beta；
- provider intermediate bounds直接作为external refined ReLU preactivation，input/spec/model参数由
  frozen v2 capture与锁定ONNX提供；
- frozen state经Bound/PlanTemplate/PlanInstance/Task/Schedule五层native IR编译执行，provider callback=0；
- ResNet2B真实6-child core诊断：native lower与production core lower max abs diff=
  `2.0265579223632812e-06`，sign=`6/6`，满足`2e-4`门禁；focused=`1 passed`；
- 当前是实现与诊断通过，尚未生成独立V4-1 artifact/semantic replay，因此V4-1尚不正式关闭；
  V4-2 optimizer mutation与B2 timing继续关闭。

### V4-1B Artifact 预冻结

- 新增`scripts/run_rvir_v4_frozen_state_artifact.py`；
- artifact内复制v2 capture并固定capture、source manifest、ONNX三项SHA256；
- topology作为独立canonical JSON保存，同时由runner内冻结的六层语义映射复核，禁止仅重签文件；
- replay重新导入ONNX、重建typed snapshots、重新执行五层IR并逐字段比较execution/summary；
- 正式门禁：shape=`[6,1]`、allclose=`2e-4`、sign=`6/6`、IR hashes=`10`、provider/fallback=
  `0/0`、`performance_claimed=false`；
- runner提交并获得clean source commit后才生成正式artifact；当前仍不关闭V4-1。

### V4-1C Replay 数值语义修正

首次全量回归的唯一失败暴露了replay实现错误：对含CPU浮点执行结果的整份`execution.json`使用
Python字典exact equality，强于V4-1已经预注册的`atol=rtol=2e-4`数值门禁。修正后的责任边界为：

- 字段集合、shape、production lower、IR/state hashes、dispatch/callback/fallback计数、sign与
  `performance_claimed`仍exact；
- 重执行的`native_lower`与`lower_max_abs_diff`必须finite，并按`2e-4`比较；
- 冻结summary自身的canonical hash和manifest引用仍exact；
- 容差内数值漂移必须replay通过，容差外漂移必须fail closed。

修正代码提交并重新生成正式artifact前，V4-1、V4-2与B2准入状态不变。

## V4-1 Formal Closure

状态：`VALIDATED-REDUCED-FROZEN-STATE-EVALUATION`；V4-1关闭，V4-2准入，B2不准入。

| 证据 | 正式结果 |
|---|---|
| source commit | `c74a2049d3d2484aade7fd5b3dd805df53823d78` |
| artifact | `artifacts/rvir-v4-frozen-state/resnet2b-core-v1` |
| manifest hash | `ba6ee2fc32109adc38326d58f7253a0cdeba2dd988ccb957f7a626d6544adf95` |
| summary hash | `3541318b226ffd28cad0862e1b43055cc701d0973144cb58f4e17122a49f60e9` |
| topology/state hash | `9be36162…bca35` / `8f8cd55d…793fe` |
| real core/domain | `1/6` |
| lower parity | max diff=`2.0265579223632812e-06`，sign=`6/6` |
| IR/dispatch | 10 hashes；replacement/original/fallback=`1/0/0` |
| semantic replay | original PASS；topology/state resigned tamper拒绝；`1e-6`数值漂移准入；`1e-2`拒绝 |
| validation | focused=`21 passed`；full=`1092 passed, 39 skipped`；mypy clean；Pylint=`10.00/10` |

该关闭只说明真实solver core的post α/β/split/intermediate state已能脱离provider callback，经BoundFlow
五层IR独立复算lower。它没有把pre-state推进到post-state：10-step optimizer mutation、learning-rate
policy、stop criterion、mutable copy-out和原子提交仍属于V4-2。因此B2 same-solver timing继续关闭，
不得用本artifact的capture/replay时延形成性能claim。

下一动作固定为V4-2预注册：冻结同一core的pre/post mutation逐tensor判据、10-step policy、callback/
fallback=`0/0`、state copy-out atomicity和失败回滚，然后才实现optimizer replacement。

## V4-2 Formal Closure

状态：`VALIDATED-OPTIMIZER-REPLACEMENT`；V4-2关闭，V4-3准入，B2仍不准入。

- V4-2B正式GPU trace冻结1 core/6 domains、10 evaluations/9 updates及production policy；
- V4-2C从pre-snapshot恢复6组native α/β/split与external intermediate bounds；
- V4-2D在零provider callback下独立执行完整10/9 mutation，逐step lower/α/β最大误差均低于`2e-4`；
- V4-2E私有stage并原子commit 12 paths，其中7 paths改变；post α/β/final lower最大误差=
  `1.4663e-05/3.6135e-07/2.6226e-06`且sign exact；NaN、stale target和mid-copy fault均fail closed；
- formal artifact original replay通过，六类同步完全重签攻击在outer provenance和semantic reexecution
  两层6/6拒绝；full=`1175 passed, 3 skipped`，static/DocOps通过；
- artifact manifest/tamper SHA256=`b76ee573...0136`/`621d5485...f70`，
  `performance_claimed=false`。

这关闭的是pre-state→optimizer mutation→post-state事务，不是whole `update_bounds_core`。V4-3必须把
executor接入真实host，禁止调用provider core/compute_bounds，并以至少5次fresh correctness核对branch、
accepted/pruned domains、lineage/node accounting、termination与verdict。V4-3通过前不得恢复B2计时。

## V4-3A Formal Closure

状态：`VALIDATED-WHOLE-CORE-TRUTH`；V4-3A关闭，V4-3B准入，V4-3/B2仍不准入。

- source=`bfdeefc`，artifact=`artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1`；
- 1 core/6 domains/24 calls、6 intermediate、6 pre-KFSB lA、3 KFSB candidates与最终decision完整冻结；
- fresh replay逐树覆盖451 tensors、213,060 signs，最大差`8.8215e-06 <=2e-4`；
- lA/intermediate/candidate/decision/accounting五类full resign及字段删除共6/6拒绝；
- targeted=`12 passed`，full=`1180 passed, 3 skipped`，mypy clean，Pylint=`10.00/10`；
- `whole_core_replacement_admitted=false`、`b2_same_solver_timing_admitted=false`、
  `performance_claimed=false`。

下一门禁为V4-3B native lA/intermediate export；original truth中仍有3次provider KFSB child-bound调用，
不得误写成whole-core replacement已完成。
