---
status: validated-b2-1-pending-external-audit
updated: 2026-08-23T03:10:59Z
type: plan
topic: boundflow
slug: fsg4-b4b2-typed-cuda-tir-preregistration
stage: s01
---

# FSG4/B4-B2 Typed CUDA/TIR Preregistration

> **2026-08-23 B2-1内部关闭**：S-anchor dense Linear TIR完成5 raw/20 metrics/36,750元素，
> max diff=`8.642673492431641e-07`且sign exact；full=`1437 passed, 3 skipped`。状态=
> `VALIDATED-B4-B2-B2-1-DENSE-LINEAR-CORRECTNESS-PENDING-EXTERNAL-AUDIT`。只开放外审，
> B2-2 sparse-source与timing继续关闭。

> **2026-08-23 B2-0外审批准**：auditor现场GPU复跑并逐位复现三项receipt hash，verdict=
> `APPROVE`（0 blocker/0 major）。最终状态=
> `EXTERNALLY-APPROVED-VALIDATED-B4-B2-B2-0-ABI-PROBE`。只开放B2-1 S-anchor dense
> correctness；2 minor随B2-1处理，不改变B2-0结论。

> **2026-08-23 B2-0内部关闭**：first-class Template/Instance/Schedule/Module/Launch IR、
> identity CUDA/TIR forward/backward、DLPack pointer/current stream、module cache与一阶custom
> autograd已在RTX 4060通过。状态=`VALIDATED-B4-B2-B2-0-ABI-PROBE`，full=
> `1426 passed, 3 skipped`。下一唯一动作=B2-1 S-anchor dense correctness；尚无region融合或性能claim。

## 0. 结论、准入与状态上限

B4-B1 已在 DocOps exchange `fsg4-b4b1-typed-reference-20260818` Round 2 独立外审批准并由
executor 关闭，最终状态=
`EXTERNALLY-APPROVED-VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE`。因此现在只预注册 B4-B2：
把已冻结的 `DifferentiableLowerRegionIRV1` 降低为独立 CUDA/TIR forward/backward module，建立
显式 custom-autograd、zero-copy、stream、alias、cache 与物理门禁。

本文档完成仅表示门禁冻结，当前状态=
`PREREGISTERED-B4-B2-TYPED-CUDA-TIR-NOT-IMPLEMENTED`。它不授权：

- optimizer exact-call 或 same-solver integration（属于 B4-B3）；
- 14-call、whole-core/query、B0 parity、memory saving 或 ASPLOS-ready claim；
- B5 JIT/CUDA Graph、B6 batching/multi-stream、B7 arena/reuse；
- 放宽 PR-12 plain-CROWN executor 的 non-grad/αβ-disabled/split-absent 门禁；
- 只实现 forward 后用 PyTorch eager backward 冒充 TIR custom backward；
- 在看到正式 timing 后继续增加 schedule 候选或修改阈值。

B4-B2 内部通过后的状态上限为
`VALIDATED-B4-B2-TYPED-CUDA-TIR-CANDIDATE-PENDING-EXTERNAL-AUDIT`。独立外审批准后也只开放
另行预注册 B4-B3 exact-call integration，不自动形成性能 claim。

## 1. 当前证据与环境冻结

### 1.1 B4-B1 语义根

- clean source=`e711e991bed54a16c881a2f2bbeb18d71de3c210`；
- v3 manifest=`2f8a1ffde0f99777e0ab6d9dddb1042c2f7f6c71e57882d141035553475e4e3f`；
- v3 protocol=`b95bc20c8dcaef8635741842b85d4d0bf9e41c9592c60896677907cd96914baf`；
- v3 summary=`753a9558a7c36cb89f02963dcd08fc8e76fdfcd415f7dc5d969eea77dffc7a0b`；
- raw=5 fresh、10 captures、60 metrics、196,380 elements、max diff=
  `6.109476089477539e-07`、allclose/sign exact；
- F1 receipt inventory negative cases=`20/20 rejected`；
- F2 execution-policy mode/exit cases=`6/6 restored`；
- all-run negative integrity cases=`2/2 rejected`。

B4-B2 必须读取同一 B4-B1a raw 和 v3 typed IR/instance，不得另建 synthetic-only oracle，也不得
修改 v1/v2/v3 artifact。

### 1.2 双锚点与 production shapes

S-anchor=`semantic-active-beta-gemm-14`：

- operator=`31/Gemm_14`，kind=`linear`；
- incoming A=`[6,1,100]`，result A=`[6,1,1024]`，weight=`[100,1024]`，bias=`[100]`；
- compressed α=`[2,1,6,27]`，native dense α=`[6,100]`；
- compressed β=`[6,1]`，native dense β/pre-add=`[6,100]`，active beta=true；
- production incoming A requires-grad=false；native α/β gradient必须存在。

P-anchor=`performance-conv-8-candidate`：

- operator=`25/Conv_8`，kind=`conv2d`；
- incoming/result A=`[6,1,16,8,8]`，weight=`[16,16,3,3]`，bias=`[16]`；
- stride/padding/dilation=`[1,1]`，groups=1，output_padding=`[0,0]`；
- compressed α=`[2,1,6,86]`，native dense α=`[6,16,8,8]`；
- compressed β=`[6,0]`，beta gradient必须 absent；
- production incoming A requires-grad=true且gradient必须存在。

S 证明 active-beta 与 gradient ownership；P 承担 production-shape 物理门禁。两者不可互相替代。

### 1.3 目标工具链

- GPU=`NVIDIA GeForce RTX 4060 Laptop GPU`，compute capability=`sm_89`；
- Torch=`2.12.1+cu132`，CUDA build=`13.2`；
- TVM=`0.23.dev0`，commit=`6248b5db43505fbcfb13cc289d11877d5d2649e8`；
- TVM-FFI commit=`438f6439148b059d424ce2cc2a348736923f6948`；
- dtype/device/layout=`float32/cuda/contiguous-strided`。

环境、TVM/FFI commit、compute capability、driver/runtime、Torch/CUDA/cuDNN 版本必须进入 protocol。
任一冻结项变化须新建 change record 与 artifact version，不得静默复用 cache。

### 1.4 可复用与禁止复用

可以复用：

- `tvm_ffi.use_torch_stream` 与 `tvm.runtime.from_dlpack` 的 zero-copy/stream 工程骨架；
- PR-12 compile/load cache、module export、CUDA event 与 profiler 基础设施；
- `boundflow/backends/tvm/fused_crown_{linear,conv2d}.py` 的代码组织和 schedule API 形式；
- B4-B1 typed IR、raw-bound instance、pure-PyTorch numerical oracle。

禁止复用为语义实现：

- PR-12 plain-CROWN PrimFunc/receipt 作为 B4-B2 correctness oracle；
- PR-12 `FusedReluAffineRequest` 的 upper/lower双向、non-grad、αβ-disabled contract；
- `crown_ibp.py` private helper 或 production target output/gradient反推；
- cuDNN/PyTorch eager backward fallback；
- 编译期或运行时 `detach()` 切断 optimizer gradient。

当前仓库不存在 B4-B2 differentiable TIR、custom-autograd wrapper 或 module receipt；不得把已有
plain-CROWN TIR 描述成“基本已实现”。

## 2. 数学所有权与梯度合同

对 lower-only incoming coefficient `A`、preactivation bounds `(l,u)`、native α=`a`：

```text
positive = l >= 0
negative = u <= 0
ambiguous = !positive && !negative
upper_slope = positive ? 1 : negative ? 0 : u / (u - l)
upper_intercept = ambiguous ? -l * upper_slope : 0
lower_slope = ambiguous ? clamp(a, 0, 1) : positive ? 1 : 0
selected_slope = A >= 0 ? lower_slope : upper_slope
selected_intercept = A >= 0 ? 0 : upper_intercept
R = A * selected_slope + beta_pre_add
relu_bias = incoming_bias + reduce(A * selected_intercept)
beta_pre_add = active_beta ? -native_beta * split_sign : 0
```

Linear forward：`Y = R @ W`；Conv forward：`Y = conv_transpose2d(R,W,attrs)`；operator bias
以 `reduce(R * bias)` 累加至 `relu_bias`。

Backward 输入只能是真实 `dL/dY` 与 `dL/d(output_bias)`。B4-B2 必须显式返回：

- eligible incoming-A gradient；
- native-dense α gradient；
- S native-dense β gradient；P beta gradient必须 absent；
- sparse-source fused ABI 中对应 compressed α/β value gradient。

Bounds、weight、operator bias、mapping indices/sign/location在v1均为常量，不返回gradient。`A==0`、
`a==0/1`等边界必须与PyTorch reference的`where`/`clamp`导数选择逐元素一致；禁止用数值近似替代
离散导数所有权。只支持一阶autograd；higher-order gradient显式fail closed。

## 3. 两级 ABI：先证明语义，再证明融合

### 3.1 B4-B2a dense semantic ABI

输入为 incoming A、bounds、native dense α/β/pre-add、incoming bias、weight/operator bias、attrs；
forward/backward 均由 TIR 执行。该 ABI 用于把数学合同、custom backward、DLPack 与 stream 调通，
不允许进入 timing gate，也不支持“消除dense materialization”或融合claim。

### 3.2 B4-B2b sparse-source fused ABI

输入从 production compressed α/β value 与冻结 mapping coordinates/sign/location 开始。mapping作为
PlanTemplate常量或只读runtime tensor，必须进入template/cache hash；TIR不得先分配完整dense α/β/
scaled-A global workspace。Backward直接产生compressed value gradient，同时用独立projection receipt
证明它与B4-B1 native gradient的gather/scatter映射一致。

只有 sparse-source fused ABI 通过 correctness、memory ledger 与 P-anchor物理门禁，才算 B4-B2
candidate；dense ABI 通过仅是 mechanism milestone。

## 4. 第一类 Plan/Schedule/Module IR

禁止直接从 dataclass 拼 TE 后丢失编译责任。新增对象建议为：

- `DifferentiableLowerTIRTemplateV1`：绑定 B4-B1 static IR hash、anchor/operator、ABI、mapping layout、
  forward/backward function inventory、gradient mask、target/compute capability；
- `DifferentiableLowerTIRInstanceV1`：绑定 raw instance、动态tensor hashes/presence与launch shape；
- `DifferentiableLowerTIRScheduleV1`：显式保存 schedule family、thread/block/tile/unroll、workspace、
  deterministic reduction与candidate ordinal；
- `DifferentiableLowerTIRModuleReceiptV1`：绑定unscheduled TIR hash、scheduled TIR hash、module/PTX/
  binary hash、TVM/FFI/toolchain、exported symbols和cache key；
- `DifferentiableLowerTIRLaunchReceiptV1`：绑定template/instance/module/schedule、input/output/gradient
  inventory、stream、alias、cache event、launch count和fallback count。

这些对象不得含`Any`语义字段；serialization必须canonical、stable hash、round-trip parse。PlanTemplate
不得携带动态tensor value；PlanInstance不得改变static semantics；Schedule IR不得改数学等式。

Cache key至少包含：schema/ABI、static IR hash、mapping hash、operator attrs、gradient mask、dtype、
sm_89、TVM/FFI commit、schedule hash、forward/backward symbol set。动态α/β/A数值不得进入compile key。

## 5. Custom autograd 与 zero-copy runtime

实现独立 `torch.autograd.Function` wrapper，但其责任仅为：

1. 按typed admission验证输入；
2. 分配PyTorch-owned outputs/grad outputs；
3. 在当前显式stream中创建non-owning DLPack views并各启动一次forward/backward module；
4. 保存最小必要tensor/context；
5. 按`ctx.needs_input_grad`返回精确gradient或`None`；
6. 生成timed region之外的launch receipt。

硬性门禁：

- tensor生命周期由PyTorch拥有，TVM view不得逃逸call；
- input/output/data_ptr与alias ledger逐项记录，output不得alias任一input；
- forward output不得detach，backward不得调用PyTorch数学算子或production helper；
- default/current stream raw id必须与TVM-FFI一致；不支持的stream显式拒绝，不得偷偷切default；
- forward/backward各恰一次module launch；provider/fallback/eager-backward均为0；
- compile/load/hash/serialization不进入warm timed region；
- module cache miss只能发生在显式cold阶段，formal warm worker必须100% hit；
-异常退出后Torch stream/device/global policy不得漂移。

## 6. 实现阶段与逐级门禁

### B2-0：环境、ABI 与 receipt probe

只实现typed lowering skeleton、round-trip IR、identity TIR forward/backward probe、DLPack data_ptr、
current-stream、module/cache/launch receipt。默认功能仍关闭。若zero-copy、stream或一阶autograd任一
无法在独立probe中成立，状态=`BLOCKED-B4-B2-ABI`，不得写region TIR。

**内部结果（2026-08-23）**：通过。GPU=RTX 4060/sm_89；cold miss→warm hit；forward/backward
launch=`1/1`；DLPack/stream exact；alias/fallback/eager backward=`0/0/0`；higher-order与receipt
篡改fail closed。该结果只开放B2-1，不进入timing。

### B2-1：S-anchor dense forward/backward

实现Linear/Gemm dense semantic ABI。5个B4-B1 raw instances逐项比较forward A/bias、incoming-A
clone、native α、active native β gradients；先用确定性correctness schedule，不计时。

**内部结果（2026-08-23）**：5/5 raw、20/20 metrics通过，36,750元素max diff=
`8.642673492431641e-07`、sign exact；incoming A hash不变，α/β gradient present，
fallback/eager=`0/0`。等待外审，尚未开放B2-2。

### B2-2：S-anchor sparse-source fused forward/backward

将27项compressed α与每domain单项β location/sign直接纳入TIR；返回compressed α/β gradient并
投影回native receipt。禁止global dense α/β/scaled-A workspace。S失败即=
`VALIDATED-NO-GO-B4-B2-SEMANTICS`，不得以P性能继续。

### B2-3：P-anchor dense forward/backward

实现Conv transpose-contraction dense correctness schedule，冻结`[6,1,16,8,8] x [16,16,3,3]`与
全部attrs。P incoming-A/native α gradient必须通过；empty β保持absent。

### B2-4：P-anchor sparse-source fused schedule search

先实现P0 correctness schedule，再只允许以下预登记变换族：

- output-gather thread extent=`{128,256}`；
- output-channel tile=`{4,8,16}`；
- spatial tile=`{1,2}`；
- reduction unroll=`{1,3}`；
- mapping inline=true；dense α/β/scaled-A global workspace=false。

不得笛卡尔积无限搜索：在写任何timing raw前，候选ledger最多冻结12个schedule hash；正式结果后
不得追加候选。P-anchor用于schedule calibration，最终只保留一个winner进入five-fresh confirmation。
候选全失败不允许临时改用cuDNN/CUTLASS extern掩盖TIR物理结论。

### B2-5：formal five-fresh artifact与外审

冻结source后，以独立process生成5份S/P correctness raw、6份AB/BA timing worker、root replay、
完整性probe与claim ledger。内部通过后提交独立外审；获批前B4-B3仍关闭。

## 7. Correctness 与 gradient acceptance criteria

每个ABI、anchor、fresh instance必须满足：

- forward lower A/bias：`atol=2e-4`、`rtol=2e-4`、finite、sign exact；
- eligible incoming A/native α/native β gradient：相同容差，finite nonzero entries sign exact；
- sparse-source compressed gradient projection与native reference exact mapping；
- shape/dtype/device/layout/stride/presence、operator attrs、mapping、launch inventory exact；
- S active beta value/pre-add/gradient存在且非空；
- P compressed beta=`[6,0]`且beta gradient absent，不得用zero tensor冒充；
- 2 anchors × 5 fresh process全过；任一失败不进入timing；
- NaN/Inf、unknown polarity、Patches、fanout、alias、未知stream、higher-order grad全部拒绝；
- dense ABI与sparse ABI都必须直接对B4-B1 pure-PyTorch oracle，不互相作为唯一oracle。

不得放宽容差、删除S-anchor、把production raw换成synthetic、只比较aggregate loss或只比较hash。

## 8. 物理测量协议与局部 kill gate

### 8.1 主比较

主指标为 production P-shape 的 wrapper-inclusive forward+backward wall time：

- baseline：与B4-B1等价的public PyTorch sparse reconstruction + lower region + autograd；
- candidate：sparse-source TIR custom Function；
- 两侧输入、output adjoints、stream、gradient inventory、pre/post allocation policy一致；
- candidate compile/load/cache/hash排除，但output/gradient allocation与wrapper开销计入；
- kernel-only/preallocated timing仅作attribution，不作为go判定。

### 8.2 Worker协议

- 6个fresh process，固定AB/BA/AB/BA/AB/BA顺序；
- 每worker先correctness，再10次warmup、30次measured pair；
- 使用同一current stream上的CUDA events，pair结束后同步；禁止CPU timer替代；
- 保存每次raw event、median、temperature/power/clock、allocated/reserved、saved-tensor bytes、launches；
- 任一worker cache miss、fallback、semantic mismatch或thermal/power扰动比>1.10时整worker拒绝；最多按
  冻结规则重跑一次，不得只删慢样本。

### 8.3 继续集成门禁

P-anchor wrapper-inclusive forward+backward同时满足：

- paired speedup geomean `>=1.05x`；
- 95% bootstrap confidence lower bound `>1.00x`；
- worst admitted worker `>=0.98x`；
- candidate peak allocated与reserved均不高于baseline `1.05x`；
- launch/fallback/cache/semantic门禁全过。

则只允许进入外审。`1.05x`是局部继续门禁，不是B4/system performance claim。若bounded schedule
ledger耗尽仍未达到门禁，结论=`VALIDATED-NO-GO-B4-B2-PHYSICS`，停止B4-B3，不通过追加schedule、
扩大容差或换baseline续命。

## 9. Provenance、replay 与完整性门禁

Formal artifact必须raw-first并绑定：

- repo/source commit、TVM/FFI/external repo/model/property、Torch/CUDA/cuDNN/driver/device；
- B4-B1 raw/v3 manifest、IR/instance/template/schedule/module/launch hash链；
- unscheduled/scheduled TIR、exported symbols、PTX/module binary digest、cache key；
- 每次input/output/gradient hash、data_ptr alias relation、stream id、launch/cache/fallback counters；
- correctness与timing raw、summary、replay stdout，无本机绝对路径。

Root replay必须从冻结raw重建IR→template→schedule→module identity→launch/result/summary，不能只校验
外层digest。至少覆盖以下fully-resigned negative cases：IR identity、mapping coordinates、β sign/location、
Conv attrs、gradient mask、schedule id/knobs、compute capability、module/PTX hash、symbol inventory、stream、
alias、cache event、launch count、latency、memory、semantic output/gradient。同步重签外层hash后仍须由
冻结identity或语义重算拒绝。

## 10. Claim ledger

| 结果 | 最多允许的状态/claim | 后续 |
|---|---|---|
| B2-0失败 | `BLOCKED-B4-B2-ABI` | 停止 |
| S dense/sparse语义失败 | `VALIDATED-NO-GO-B4-B2-SEMANTICS` | 停止 |
| P语义通过、物理门禁失败 | `VALIDATED-NO-GO-B4-B2-PHYSICS` | 停止B4-B3 |
| 内部门禁全过 | `VALIDATED-B4-B2-TYPED-CUDA-TIR-CANDIDATE-PENDING-EXTERNAL-AUDIT` | 只开放外审 |
| 外审批准 | `EXTERNALLY-APPROVED-VALIDATED-B4-B2-CANDIDATE` | 只开放另行预注册B4-B3 |

任何结果都不得自动主张whole-core/query speedup、B0 parity、memory saving或ASPLOS-ready。

## 11. 建议提交序列

1. `docs: preregister B4-B2 typed CUDA TIR`；
2. `feat: add B4-B2 lowering and receipt IR`（B2-0）；
3. `feat: add B4-B2 linear dense TIR forward backward`（B2-1）；
4. `feat: fuse B4-B2 linear sparse state`（B2-2）；
5. `feat: add B4-B2 conv dense TIR forward backward`（B2-3）；
6. `perf: add bounded B4-B2 conv schedule candidates`（B2-4，仍无performance claim）；
7. `bench: close B4-B2 five-fresh correctness and micro physics`（B2-5）；
8. 独立外审；批准后才另行预注册B4-B3。

每个提交必须新增独立change record、`dol ch add`与确定性`dol va add`；TIR变更后运行
`bash scripts/rebuild_tvm.sh`并重启Python process。

## 12. Validation

- targeted：B4-B2 IR/lowering/schedule/module/runtime/custom-autograd/negative tests；
- related：`tests/test_fsg4_b3*.py tests/test_fsg4_b4b*.py`；
- full：完整激活`boundflow`环境后`python -m pytest -q`；
- CUDA：默认stream、显式current stream probe、forward/backward launch count、CUDA event timing；
- static：Black、touched source精确Mypy清单、Pylint、`git diff --check`；
- artifact：generate、root replay、fully-resigned negative integrity suite；
- DocOps：change/validation、exchange validate、`dol lint --soft`。

## 13. Rollback / fail closed

B4-B2默认关闭且不修改B3/B4-B1行为。任一admission、compile、load、DLPack、stream、alias、cache、
forward、backward或receipt门禁失败，formal worker直接拒绝；不得在同一worker静默回退。回滚只需
移除显式opt-in B4-B2 selection，B4-B1 pure-PyTorch evidence与B3累计基线保持只读。GO或NO-GO都
保留raw、module、replay与拒绝证据。

## 14. 外审问题

1. dense→sparse-source两级ABI是否保持完整语义所有权，而未把materialization成本藏到timed region外？
2. custom backward是否对eligible input完整、对ineligible/higher-order明确拒绝且无eager fallback？
3. first-class Plan/Schedule/Module/Launch receipts是否足以重建编译与执行身份？
4. 6-worker物理门禁是否能阻止compile/cache/thermal/allocator造成的伪加速？
5. claim ledger是否严格限制B4-B2 micro结果，不外推B4-0的67.72% span share？

## 15. Links

- changelog：`gemini_doc/BOUNDFLOW_FSG4_B4B2_TYPED_CUDA_TIR_PREREGISTRATION_CHANGELOG_2026_08_23.md`；
- B4-B总计划：`gemini_doc/BOUNDFLOW_FSG4_B4B_DIFFERENTIABLE_CUDA_TIR_V1_PLAN_2026_08_18.md`；
- B4-B1 closure：`gemini_doc/change_2026-08-23_fsg4_b4b1_round2_external_closure.md`；
- B4-B1 IR/reference：`boundflow/ir/differentiable_lower_region.py`、
  `boundflow/runtime/fsg4_b4b1_pytorch_reference.py`；
- PR-12 plain executor：`boundflow/runtime/fused_crown.py`；
- current-stream probe：`scripts/run_nrir49_g0_cuda_smoke.py`；
- roadmap：`gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_PLAN_2026_08_16.md`。
