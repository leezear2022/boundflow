---
status: b4b2-b2-3-validated-pending-external-audit
updated: 2026-08-23T11:33:28Z
type: plan
topic: boundflow
slug: BOUNDFLOW_FSG4_B4B_DIFFERENTIABLE_CUDA_TIR_V1
stage: s01
---

# FSG4/B4-B Differentiable CUDA/TIR v1 Plan

> **2026-08-23 B2-3内部关闭**：P-anchor dense Conv correctness 5/5 raw、20/20 metrics通过；
> 当前只开放B2-3外审，B2-4/B2-5/timing/B4-B3继续关闭。详见
> `BOUNDFLOW_FSG4_B4B2_B2_3_DENSE_CONV_TIR_CHANGELOG_2026_08_23.md`。

> **2026-08-23 B2-2外审批准**：`APPROVE`，0 blocker/major/minor；独立数学、GPU、
> workspace与回归全通过。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS`。唯一下一动作=
> B2-3 P-anchor Conv dense correctness；timing/B2-4/B2-5/B4-B3关闭。

> **2026-08-23 B2-2内部关闭**：S-anchor compressed alpha/beta直接TIR与compressed
> gradient projection已通过5 raw，max diff=`8.642673492431641e-07`，禁止dense-state
> workspace count=`0`。当前=
> `VALIDATED-B4-B2-B2-2-SPARSE-SOURCE-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。下一唯一
> 动作=B2-2外审；P-anchor/timing/B2-3/B2-4/B2-5/B4-B3关闭。
> 该待审状态已由上方外审批准状态取代。

> **2026-08-23 B2-1外审批准**：`APPROVE`，0 blocker/0 major；独立float64重算与
> 现场GPU复跑全部通过。最终=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS`。唯一下一动作=
> B2-2 S-anchor sparse-source fused forward/backward；timing/P-anchor/B2-4/B2-5/B4-B3关闭。
> 该“下一动作B2-2”状态已由上方B2-2内部关闭状态取代。

> **2026-08-23 B2-1内部关闭**：S dense Linear forward/backward 5/5 raw、20/20 metrics通过，
> 状态=`VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。
> 下一唯一动作=外审；B2-2与timing关闭。
> 该待审状态已由上方外审批准状态取代。

> **2026-08-23 B2-0外审批准**：`APPROVE`，0 blocker/0 major；GPU probe三hash现场逐位复现。
> 最终=`EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-0-ABI-PROBE`，下一唯一动作=B2-1。

> **2026-08-23 B4-B2/B2-0更新**：identity CUDA/TIR与first-class receipt ABI已内部通过，
> 状态=`VALIDATED-B4-B2-B2-0-ABI-PROBE`。下一唯一动作=B2-1 S-anchor dense correctness；
> 尚无region融合、timing或performance claim。

## 0. 准入、目标与非目标

B4-0已外审关闭为`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`；B4-A已外审关闭为
`EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`。因此B4-B只使用B4-0冻结的
differentiable lower-only CROWN opportunity，直接基线仍是B3；B4-A的约1.9% core改善
不得计入候选。

B4-B v1目标是在真实αβ-CROWN optimizer evaluation中，对一个active-beta语义锚点和
一个高占比Conv性能锚点，建立从typed exact-call、pure-PyTorch reference、CUDA/TIR
forward/backward到same-solver activation receipt的完整闭环。

本阶段不宣称：

- B4 whole-core/query speedup；
- B0 parity、memory saving或ASPLOS-ready；
- 14-call或六层coverage；
- general planner/JIT/runtime效果；
- PR-12 plain-CROWN kernel已支持α/β/split/autograd。

## 1. 已冻结的代码与物理事实

### 1.1 不得放宽的旧 capability

`boundflow/runtime/fused_crown.py`对PR-12 path明确要求：

```text
plain_crown=true
requires_grad=false
alpha_enabled=false
beta_enabled=false
split_state_present=false
all input tensors requires_grad=false
```

B4-B必须创建独立schema、cache key、planner/dispatch与activation receipt。可复用TVM build/
schedule/cache的工程骨架，但禁止改宽上述旧门禁、偽造兼容性或用`detach`绕过backward。

### 1.2 真实 mutable state

B3 production contract的α shape：

- `/45`: `(2, 1, 6, 178)`；
- `/48`: `(2, 1, 6, 27)`；
- `/input-12`: `(2, 1, 6, 132)`；
- `/input-16`: `(2, 1, 6, 121)`；
- `/input-24`: `(2, 1, 6, 86)`；
- `/input-4`: `(2, 1, 6, 164)`。

β value中`/input-28`=`(6, 1)`是唯一非空层；`/39`、`/44`、`/input-20`、
`/input-8`、`/input`均为`(6, 0)`。因此只在beta-empty Conv上通过不能证明β所有权。

### 1.3 B4-0 opportunity

B4-0外审冻结CROWN14=`9196 kernels / 32618329 ns / 3291 mat ops / 57292800 B`，
换算为B3 core span share=`0.6771722591159042`。该share不是单一kernel占比，region内还有
forward/KFSB score等非CROWN工作。CUDA kernel行`input_shapes`为空时，shape必须从同一
correlation parent CPU operator恢复并绑定lineage，不得猜测。

## 2. 双锚点冻结

### 2.1 S-anchor：active-beta语义锚点

固定B4-A lineage中的`node31 / Gemm_14`：

- incoming coefficient shape=`[6, 1, 100]`；
- preactivation shape=`[6, 100]`；
- producer=`/input-28`；
- production compressed α source=`alpha/%2F48/%2F49`，shape=`(2, 1, 6, 27)`；
- mapped native dense α shape=`(6, 100)`；
- production compressed β source=`beta/%2Finput-28/0/value`，shape=`(6, 1)`；
- mapped native dense β与`relu_pre_add_coeff_l` shape=`(6, 100)`；
- dtype/device/layout=`float32/cuda/contiguous`；
- 必须观测native dense α/β gradient；production incoming A是否requires-grad按raw事实记录。
  顶层S-anchor可以是非可微objective常量；incoming-A custom-backward责任在B4-B1/B2
  micro通过显式requires-grad clone验证，不伪造production gradient。

S-anchor用于证明稀疏α选取、beta scatter/sign/loc/value、ReLU sign selection、bias reduction、
Gemm backward与optimizer mutation的端到端所有权，不用于宣称高占比性能。

### 2.2 P-anchor：Conv性能锚点

首选B4-A lineage中的`node25 / Conv_8`：

- incoming coefficient shape=`[6, 1, 16, 8, 8]`；
- preactivation shape=`[6, 16, 8, 8]`；
- producer=`/input-20`；
- dtype/device/layout=`float32/cuda/contiguous`；
- production compressed α source=`alpha/%2Finput-24/%2F49`，shape=`(2,1,6,86)`，
  mapped native dense α shape=`(6,16,8,8)`；
- production compressed β source=`beta/%2Finput-20/0/value`，shape=`(6,0)`，但mapped native dense
  β和`relu_pre_add_coeff_l` shape=`(6,16,8,8)`；因此P-anchor不替代S-anchor。

B4-0 raw还观察到高频`cudnn_convolution_transpose` production shapes，例如
`[6,16,8,8] x [16,16,3,3]`。B4-B0必须从gradient-active的exact call现场捕获
weight shape、stride/padding/dilation/groups、operator ordinal和correlation parent，再确认P-anchor的精确
signature。如`Conv_8`不对应高share signature，只能以原始证据调整P-anchor并新建
DocOps change，不得静默替换。

## 3. B4-B0：read-only production exact-call capture

本步不实现TIR，只在optimizer evaluation 0的gradient-active exact call上捕获语义。不能只捕获
terminal no-grad evaluation，否则无法冻结custom backward。

每个锚点的typed record至少包含：

- source/code/external repo/model/property/protocol identity；
- start-node key、node ordinal/name、producer ordinal/name、phase/call/evaluation ordinal；
- lower/upper bounds、incoming lower A、previous lower A、weight/bias与Conv/Gemm attributes；
- production compressed alpha value/index/lookup/start-node mapping，mapped native dense alpha，以及两者
  round-trip lineage；
- production compressed beta value/sign/loc/bias/update_mask与空/非空状态，mapped native dense beta、
  `relu_pre_add_coeff_l`以及两者round-trip lineage；
- eager output lower A、bias delta、下一producer输入；
- loss-seeded native dense α/native dense β gradients；incoming A仅在production raw本身
  `requires_grad=true`时要求其gradient；production compressed α/β是映射源状态，不伪造为
  exact region直接的autograd leaf；
- shape/dtype/device/layout/requires_grad/alias/stream；
- raw tensor payload与canonical digest，所有写盘/hash位于timed region之外。

capture必须是显式opt-in、provider/fallback-free，默认B3路径不变。不支持的Patches、residual
fanout、empty/ambiguous lineage、未知alpha lookup或beta location必须fail closed。

### B4-B0准入门禁

- S/P两锚点各至少5个fresh process capture；
- 同一锚点5/5 discrete structure exact；
- raw tensor重放出eager output/gradient，atol/rtol=`2e-4`，sign exact；
- S-anchor的beta non-empty且α/β gradient finite、shape exact；
- P-anchor的Conv属性和correlation-parent shape单义；
- root replay从raw重建全部派生字段；
- outer-resigned tamper覆盖state/start-node/topology/shape/alpha-index/beta-loc/gradient/alias/stream。

任一锚点无法稳定捕获则B4-B=`BLOCKED-SEMANTIC-CAPTURE`，不实现TIR。

## 4. B4-B1：typed IR与PyTorch reference

新建v1 contract（建议名`DifferentiableLowerRegionIR`），将以下责任纳入stable canonical
serialization/hash：

```text
beta sparse scatter
  -> alpha sparse selection/reconstruction
  -> sign-select relaxation multiply
  -> ReLU intercept / affine bias reduction
  -> Linear matmul or Conv2d transpose-contraction
  -> previous lower A + accumulated bias
```

reference必须是可微pure PyTorch实现，不调用TVM，用作唯一numerical oracle。它需要
匹配上游真实顺序：`beta_crown_backward_bound`先注入lA/bias，随后`BoundRelu.bound_backward`
完成alpha/sign/bias，最后`BoundLinear/BoundConv.bound_backward`传播。

admission必须显式检验：start-node key、lower-only、Tensor/Patches、shape/dtype/device/layout、
requires-grad、α/β/split、Conv attrs、fanout、alias、stream。对每个拒绝原因建立专用负向测试。

## 5. B4-B2：独立CUDA/TIR forward + backward

顺序冻结为：

1. S-anchor Linear/Gemm forward TIR；
2. S-anchor custom backward，覆盖incoming A/α/β gradients；
3. S-anchor micro parity与optimizer mutation parity；
4. P-anchor Conv transpose-contraction forward TIR；
5. P-anchor custom backward与production-shape micro timing；
6. 两锚点独立compile/load cache。

forward-only通过不允许进入same-solver。不得用PyTorch eager backward隐式回退来宣称
TIR custom backward。如TVM暂无法直接接入autograd，可创建显式`torch.autograd.Function`
wrapper，但forward/backward两侧的backend receipt、module hash、stream和cache key都必须可审计。

### Micro门禁

- forward lower A/bias：atol/rtol=`2e-4`，sign exact；
- micro中显式requires-grad的incoming A/native α/native β gradient：atol/rtol=`2e-4`，
  sign exact for finite nonzero entries；
- 常量/离散结构exact，NaN/Inf一律拒绝；
- 5个固定seed x 2 anchors x forward/backward全过；
- warm时延排除compile/load/cache/hash，单列cold与warm；
- P-anchor warm forward+backward region geomean至少`1.05x`才允许进入exact-call integration；
- S-anchor未过语义门禁时，P-anchor即使很快也必须NO-GO。

`1.05x`只是“继续集成”的局部kill gate，不是B4 performance claim。

## 6. B4-B3：optimizer exact-call integration

只在B4-B2通过后，为optimizer evaluation 0建立显式opt-in same-solver path：

- eligible S/P region的forward/backward receipt各且只有1次；
- eligible region provider/fallback/eager-backward=`0`；
- unsupported region仍走原B3，但必须在coverage ledger显式列出；
- optimizer lower 10/10、α/β update 9/9、terminal state与B3逐项parity；
- 5 fresh B3/B4-B semantic pairs通过后，只报锚点region与optimizer evaluation-0 attribution；
- 本步不运行B4-D 36-process whole-query performance protocol。

## 7. 量化go/no-go与claim边界

对可加速share `s`、region speedup `r`，whole-core理论上限为：

```text
S_core = 1 / ((1 - s) + s / r)
required_r(target) = s / (1 / target - (1 - s))
```

若未来B4-C真实覆盖全部`s=0.6771722591`：

- `1.03x` core需region约`1.04494x`；
- reduced-B4 `1.10x` core需region约`1.15510x`；
- full-B4 `1.20x` core需region约`1.32647x`。

上述数字是“全share都被同等加速”的数学条件，不属于B4-B v1证据。B4-B每次扩展都必须
从activation receipt重算eligible measured share和required-r；单shape micro speedup禁止乘以67.72%
外推whole-core/query。

终止条件：

- S-anchor无法拥有β/gradient语义：`VALIDATED-NO-GO-B4-B-SEMANTICS`；
- P-anchor语义通过但warm region `<1.05x`：`VALIDATED-NO-GO-B4-B-PHYSICS`；
- 两锚点通过但eligible measured share低于B4-0 core的5%：仅保留mechanism，不开B4-C；
- 两锚点过且share至少5%：只开放B4-C coverage预注册，不自动形成performance claim。

## 8. 产物与建议提交序列

1. `docs: preregister B4-B differentiable CUDA/TIR v1`；
2. `feat: capture B4-B production differentiable regions`（B4-B0）；
3. `feat: add typed differentiable lower region reference`（B4-B1）；
4. `feat: add differentiable linear TIR forward backward`（S-anchor）；
5. `feat: add differentiable conv TIR forward backward`（P-anchor）；
6. `feat: integrate B4-B exact-call activation`（B4-B3）；
7. `bench: close B4-B five-fresh semantics and attribution`。

每个工程提交各自生成change record。artifact必须raw-first、source/code/protocol-bound、无本机路径、
root replayable并含outer-resigned tamper probes。

## 9. Validation

- 新增的typed contract/reference/capture/backend/integration都要有happy与fail-closed tests；
- 固定seed，数值容差预注册，离散结构exact；
- touched Python：Black、Mypy的精确文件清单、Pylint；
- TVM/TIR变更后执行`bash scripts/rebuild_tvm.sh`并重启Python process；
- targeted、FSG4 B3/B4回归、full `pytest tests`；
- `git diff --check`对source/docs执行，immutable artifact raw单独说明并不重写；
- `dol ch add`、`dol va add`、exchange audit、`dol lint --soft`。

## 10. Rollback / fail closed

B4-B必须默认关闭，仅显式opt-in。任一schema/hash/admission/compile/load/activation/gradient门禁
失败时，待测eligible region必须拒绝，不得在同一正式worker内静默回退。不启用B4-B时
B3行为必须bit-for-bit保持原语义。无论B4-B结果为GO或NO-GO，都保留raw、replay和拒绝证据。

## 11. Links

- cumulative roadmap：`gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_PLAN_2026_08_16.md`；
- B4-0 closure：`gemini_doc/change_2026-08-16_fsg4_b4_0_external_audit_closure.md`；
- B4-A closure：`gemini_doc/change_2026-08-18_fsg4_b4a_external_audit_closure.md`；
- PR-12 plain capability：`boundflow/runtime/fused_crown.py`；
- production mutable state：`scripts/run_fsg4_b3_counter_diagnostic.py`。

## 12. 执行更新（2026-08-18）

B4-B0 typed capture substrate已实现，状态=
`IMPLEMENTED-B4-B0-CAPTURE-CONTRACT-PENDING-LIVE-HOOK`。实现明确分离production compressed
α/β映射源与native dense α/β/`relu_pre_add_coeff_l`算子输入及native gradients；10项测试、
fixed related 46项、full=`1366 passed, 3 skipped`和静态门禁通过。尚未接入live
solver，不支持correctness/performance claim。

下一唯一工程动作：在optimizer evaluation 0将typed contract接入显式opt-in live observer；未通过
B4-B0 five-fresh/replay/tamper前不实现TIR。

### B4-B0 live observer更新

显式opt-in evaluation-0 observer已实现并通过CPU production-state与real CUDA smoke，状态=
`IMPLEMENTED-B4-B0-LIVE-OBSERVER-PENDING-FIVE-FRESH`。observer仅在诊断路径将两锚点的
structured lower-A物化为参与后续计算的同一tensor并retain gradient；默认执行不安装
observer。已冻结真实物理事实：

- S-anchor `31/Gemm_14`：incoming-A requires-grad=false，active-beta pre-add/β gradient存在，
  weight=`(100,1024)`；
- P-anchor `25/Conv_8`：incoming-A requires-grad=true且gradient存在，production beta empty，
  因此pre-add/β gradient必须为absent，不得伪造零tensor；weight=`(16,16,3,3)`，
  stride/padding/dilation=`(1,1)`，groups=`1`。

observer在evaluation 0 backward后、首次optimizer step前冻结value/gradient，后续9次mutation不得
改写payload。下一唯一动作是5-fresh raw-first artifact、root replay与tamper；关闭前TIR仍关闭。

### B4-B0 five-fresh runner更新

状态=`IMPLEMENTED-B4-B0-FIVE-FRESH-RUNNER-PENDING-FORMAL-RUN`。typed capture新增α-index/lookup、
β-location/sign、round-trip receipt、CUDA default-stream/priority和alias ownership；新增独立CUDA
worker、5-fresh raw-first runner、root typed reconstruction与九类outer-resigned tamper。单fresh
CUDA及synthetic 5-run summary已通过，但正式5进程artifact尚未执行。下一唯一动作是先提交冻结
runner，再运行formal generate/replay/tamper；关闭前B4-B1/TIR仍不得启动。

### B4-B0 five-fresh内部关闭

source=`1dbb2de`的formal artifact已执行5个独立CUDA进程，S/P各5份capture；root replay从raw
重建10份typed capture，比较108组tensor/664,744元素，最大差=`1.1920928955078125e-07`且
sign exact。state/start-node/topology/shape/alpha-index/beta-location/gradient/alias/stream九类
outer-resigned攻击`9/9 rejected`；定向20、全量`1372 passed, 3 skipped, 6 warnings`。内部状态=
`VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`。外审批准后才开放B4-B1；B4-B2 TIR、
performance和system claim保持关闭。

### B4-B0 Round 1外审与identity binding修复

Round 1=`changes_requested`，F1 major证明全run同步改写topology或lineage并全重签可绕过原相对
一致性。v2 verifier已将source capture/model、source state、primal/split/topology/schedule、两锚点
anchor/lineage/source tensor/receipt hashes冻结为绝对身份，同时绑定manifest与protocol同源。
合法v1 replay保持，原9类+两类coordinated完整性负例=`11/11 rejected`。当前状态=
`IMPLEMENTED-B4-B0-R1-F1-IDENTITY-BINDING-PENDING-V2`；下一唯一动作是clean-source v2 formal
artifact与Round 2。B4-B1/B4-B2/TIR/performance仍关闭。

### B4-B0 v2内部关闭

source=`422a3ee`的v2 formal artifact已完成5 fresh/10 captures，比较108 tensors/664,744
elements，max diff=`1.1920928955078125e-07`、sign exact。protocol与manifest绑定冻结source、
topology、anchor及lineage绝对身份，正式完整性门禁（含Round 1两类coordinated rewrite）=
`11/11 rejected`；定向=`24 passed`，全量=`1376 passed, 3 skipped, 6 warnings`。状态=
`VALIDATED-B4-B0-V2-PENDING-ROUND2-EXTERNAL-AUDIT`；下一唯一动作
是回复F1并提交Round 2，获批前不得启动B4-B1/B4-B2、TIR或性能计时。

### B4-B0 Round 2外审关闭

Round 2=`approve`，0 blocker/major/minor/info；F1由绝对身份三层绑定与审计方自建all-run
topology/lineage全链重签拒绝正式关闭。exchange已由executor关闭，状态=
`VALIDATED-B4-B0-EXTERNALLY-APPROVED`。现在只允许另行预注册B4-B1 typed pure-PyTorch
reference；B4-B2 CUDA/TIR、performance、memory、ASPLOS-ready继续关闭。

### B4-B1 typed pure-PyTorch reference预注册

B4-B1预注册已冻结。B4-B0现有raw足以把两锚点output A重建至约`3e-8`，但缺少进入region前
的lower bias、operator bias与region output adjoints，不能自包含重建output bias或production
gradient。下一唯一动作是B4-B1a read-only capture sufficiency amendment；随后才允许typed IR/
reference。详细合同见
`gemini_doc/BOUNDFLOW_FSG4_B4B1_TYPED_PYTORCH_REFERENCE_PREREGISTRATION_2026_08_18.md`。

### B4-B1 Round 2外审关闭与B4-B2预注册

B4-B1已在Round 2关闭F1/F2并由executor关闭exchange，最终=
`EXTERNALLY-APPROVED-VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE`。B4-B2现已另行预注册为
`PREREGISTERED-B4-B2-TYPED-CUDA-TIR-NOT-IMPLEMENTED`：先做B2-0 first-class lowering/receipt/
identity-TIR ABI probe，再依次做S dense、S sparse fused、P dense、P bounded schedule与formal
five-fresh。详细门禁见
`gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_PLAN_2026_08_23.md`。

预注册没有实现、计时或准入TIR。下一唯一工程动作=B2-0；其通过前region TIR与B4-B3仍关闭。
