# BoundFlow GC0-1 Verification Graph Capture + Analysis 预注册

status: preregistered-not-implemented-not-run
date: 2026-08-26
stage: GC0-1
parent: `94166b6`
parent-state: `VALIDATED-GC0-0-GENERIC-VERIFICATION-GRAPH-SCHEMA`
implementation-open: false（须先完成本预注册独立外审）
lowering-open: false
timing-open: false
performance-claimed: false

## 0. 冻结声明

本文在任何 GC0-1 capture adapter、graph analysis、formal artifact 或 negative graph 实现之前冻结。
外审批准前不得写 GC0-1 代码。批准后也只允许一次 bounded implementation/formal 提交；不得顺带实现
GC0-2 lowering、GC-1 rule rewrite/VJP execution、physical arena、production replacement 或 timing。

冻结后不得根据实现难度或 observed admission 结果修改：

1. source ownership、adapter 输入与 deterministic ID；
2. topology/use-def/boundary/postdominator/effect/alias/VJP 分析；
3. shallow policy rejection 与 full analysis witness 的分层；
4. 七类 analysis rejection 的算法、witness 与最小 negative graph；
5. 三类 positive signature、十五类 direct replay、artifact/replay/tamper；
6. GO/NO-GO/INVALID 与唯一后继；
7. `execution_enabled=false/timing_recorded=false/performance_claimed=false`。

本文完成只表示协议冻结，不表示 capture/analysis 已实现或任一 production region 已 admitted。

## 1. 起点、问题和唯一目标

### 1.1 已关闭的父门禁

GC0-0 在 `07f02fe` 实现 generic `VerificationProgram/Region/Value/Op/Effect/VJP/Rule/LegalityResult/
Module` schema，并在 exchange `gc0-0-schema-20260826` Round 1 获得 approve；executor 在 `94166b6`
提交审计与 closure。已成立的是 schema、22 reasons、15-direct/7-analysis 分区、canonical identity、
三类 schema fixture 与非执行 registry；尚无 capture、analysis、lowering、runtime 或性能。

### 1.2 外审 finding 转化为合同

`DENSE_A_ESCAPE`、`EFFECT_ORDER_CONFLICT`、`REGION_EXTERNAL_USE`、
`UNSAFE_ALIAS_OR_LIFETIME`可在 schema validate 中从输入声明做保守拒绝。这是合法的 reject-side
shortcut，但不是 graph analysis。GC0-1 强制区分：

- `shallow_policy_rejection`：输入声明本身已违反合同；
- `analysis_witness_rejection`：从 schema-valid captured graph 导出冲突路径；
- 只有后者计入 GC0-1 analysis coverage。

### 1.3 当前代码事实

| 现有对象 | 可复用事实 | 不得直接继承的局限 |
|---|---|---|
| `BFBoundModule/BoundValue/BoundOp` | typed role、polarity、representation、batch axis | 不含 production α/β/effect/VJP 的完整 witness |
| `BFTaskModule/BoundTask/TaskOp/StoragePlan` | op order、IO、param、buffer/alias placeholder | `TaskKind`只含IBP，`Any` attrs和buffer placeholder不是verification proof |
| `StructuredLowerRegionTemplateV1` | DAG、consumer、residual/bias、scratch interval | 当前实例常数不得进入通用 algorithm |
| `ProductionDifferentiableRegionCaptureV1` | α/β lineage、gradient owner、operator metadata | 历史 anchor registry不得被新 adapter import或按名字分支 |
| `R31BBoundedArenaTraceV1` | recurrence、branch、logical slot continuity | 固定steps/start-node/D/S只可作instance data |
| `VerificationGraphModuleV1` | 唯一 canonical target schema | 目前没有 capture pass 与 causal witness analysis |

### 1.4 唯一目标

```text
typed source snapshot
  → provider-neutral capture adapter
  → VerificationGraphModuleV1
  → deterministic use-def/topology/effect/alias/VJP analyses
  → LegalityResultV1 + typed witness ledger
```

GC0-1 只回答：一个显式 candidate region 能否从冻结 metadata 构造成 canonical graph，并由因果
witness 证明 admitted 或稳定拒绝。它不回答如何 rewrite、lower、execute 或加速。

## 2. 实现范围与禁止项

### 2.1 外审批准后允许新增

- `boundflow/analysis/verification_capture.py`；
- `boundflow/analysis/verification_legality.py`与最小`__init__.py`；
- `tests/test_gc01_verification_capture_analysis.py`；
- `scripts/run_gc01_capture_analysis_artifact.py`与replay脚本；
- `artifacts/gc01-verification-capture-analysis-v1/` metadata-only artifact；
- 对应 changelog、closure 与 DocOps exchange。

如需要新 receipt/witness 类型，必须独立、backward-compatible；不得改变 GC0-0 已批准枚举与 stable-hash
payload。

### 2.2 明确禁止

- import/call TVM、Torch CUDA、DLPack、kernel 或 autograd backward；
- 捕获 object id、绝对路径、pointer 或临时名；
- rule replacement、Relax/TIR lowering、compile、arena、stream、launch；
- live provider exact call 或 auto_LiRPA state mutation；
- 按 P/S/C0/C1/C2、ResNet、node path、shape或op count分支；
- latency/CUDA event/memory/performance winner；
- 用 shallow rejection 关闭 full analysis；
- 开放 GC0-2、GC-1、runtime、same-solver、query、queue 或 ASPLOS claim。

## 3. Source snapshot 与 capture ABI

### 3.1 `VerificationCaptureRequestV1`

冻结字段：

```text
schema_version, capture_id, source_adapter_id, source_schema_id
source_graph_hash, parameter_schema_hash, numeric_policy_id, target_contract_id
candidate_source_op_ids[]
declared_boundary_input_ids[], declared_boundary_output_ids[]
declared_external_effect_ids[], requested_rule_registry_hash
execution_enabled=false, timing_recorded=false, performance_claimed=false
request_hash
```

- candidate op set 由 source semantic ID 指定，不由模型名、ordinal常数或filename选择；
- boundary/effect 是待验证声明，不是 witness；
- request 不得声明 `closed_world`或`admitted`；
- source hash 绑定完整 snapshot，不只绑定 candidate 子图；
- 任一 false flag 翻转必须在 capture 前拒绝。

### 3.2 Provider-neutral `VerificationSourceSnapshotV1`

adapter 输入先归一为不可变 metadata：

```text
source_values[]: semantic_id/type/shape/dtype/device/layout/stride/role/version/lineage
source_ops[]: semantic_id/kind/inputs/outputs/params/attrs/source_ordinal
source_effects[]: resource/kind/version/access/source_ordinal
source_alias_claims[]: value/alias_set/storage_owner/may_alias
source_vjp_claims[]: owner/saved/recomputed/endpoint/higher_order/mutation
source_external_consumers[]
source_entries[] / source_exits[]
snapshot_hash
```

snapshot 不含 tensor payload、pointer、callable 或 backend module。Torch来源必须由既有capture先冻结成
metadata；GC0-1 不调用 live provider。

### 3.3 三类 adapter，共享同一 analysis

1. `BoundTaskSourceAdapterV1`：读取 Bound/Task typed graph、IO、param、storage alias；缺 verification
   state/effect 时必须由 typed overlay提供，否则拒绝；
2. `StructuredRegionSourceAdapterV1`：读取 structured DAG、consumer、residual/bias、scratch/liveness；
3. `ProductionMetadataSourceAdapterV1`：读取 provider-neutral production capture metadata；禁止 import
   历史 anchor constants，只按 protocol 映射 α/β lineage、attrs 与 gradient owner。

三者输出同一 snapshot。后续 capture/analysis 不得知道 adapter 类型。

### 3.4 Deterministic ID

```text
kind-prefix + ":" + sha256(canonical(source_schema_id,
                                      source_semantic_id,
                                      semantic_role,
                                      source_graph_hash))[:24]
```

- source ordinal 只做拓扑 tie-break；
- ID 不含 `/49`、`Conv_8`、C0、路径或进程信息；
- 五个 fresh process 的 ID、排序、hash 逐字节相同；
- 同 semantic ID、不同 graph hash 必须不同 ID；
- mapping ledger 双向唯一；显式 normalization 的 many-to-one 必须记录 provenance。

### 3.5 `VerificationCaptureReceiptV1`

```text
request_hash, snapshot_hash, module_hash
adapter_id, adapter_version
source/captured value-op-effect-vjp-region counts
mapping_ledger_hash, normalization_ledger_hash
omission_ledger[], shallow_rejections[]
capture_complete
execution_enabled=false, timing_recorded=false, performance_claimed=false
receipt_hash
```

`capture_complete=true`要求 candidate ops、transitive inputs/params/effects、boundary consumers与VJP owner
全映射。omission只允许非语义debug metadata；semantic omission必须拒绝，不能产生partial graph。

## 4. Capture 转换语义

### 4.1 Values

每个 value 必须确定 role/polarity/representation、concrete type/layout/axis、state version/lineage/storage/
alias、producer、全部consumer、external-use count、present/empty β 与 finite policy。动态shape、unknown
device/layout、缺version或无法区分spec/domain轴必须用既有direct reason拒绝，不猜默认值。

### 4.2 Ops 与 effects

- source op 只能映射到 frozen vocabulary；opaque custom op禁止；
- provenance、IO/params、polarity、numeric policy、effect、VJP与canonical attrs完整；
- unknown attrs不得用`repr()`序列化；
- α/β/split/history/optimizer/domain/queue/commit成为versioned effect；
- 每个output version单writer，writer order与topology一致；
- queue/termination只允许external boundary，candidate内不得write；
- 缺effect不能靠假token补齐。

### 4.3 Residual、bias 与 VJP

- residual diamond保留两支provenance、共同join与bias-token accumulation；
- saved/recomputed集合disjoint；
- saved只允许compressed α/β、index/location/sign、small mask、parameter/identity token；
- dense coefficient/incoming adjoint可作内部scratch，不得成为saved、external consumer或region output；
- mutation固定`none-inside-vjp`，optimizer/commit mutation属于region外owner。

## 5. Deterministic analysis pipeline

所有 pass pure、read-only、无 timing，固定按 A0—A8 一次执行；后序不得修补前序失败。

### A0 — Schema gate

运行 GC0-0 validate。失败产生 direct reason 与 `shallow_policy_rejection`，不产生 full witness。

### A1 — Topology/use-def

以 source ordinal + stable ID 做 Kahn sort；每个非external value恰一producer；consumer ledger从op input
重建。cycle/missing/duplicate producer或ledger漂移以`RECEIPT_IDENTITY_MISMATCH`拒绝。

### A2 — Boundary/external use

从完整source graph重建input/output。internal value出现未声明outside consumer时产生
`REGION_EXTERNAL_USE`；witness含value、producer、inside/outside consumer、ordinal与最短use path。

### A3 — Entry/exit/postdominator

加入synthetic entry/exit，在反图fixed-point计算postdominator。每条entry→exit path必须经过declared
exit；dead end、missing branch或旁路产生`REGION_NOT_POSTDOMINATED`，witness记录逃逸path。

### A4 — Effect SSA/order

按`(ordinal,stable effect ID)`排序；version连续、单writer、read不越过writer、mutation ordinal不重复/
倒序/缺口，queue/termination/commit boundary不可写。分别产生`EFFECT_ORDER_CONFLICT`或
`QUEUE_OR_TERMINATION_EFFECT_CROSSED`，witness绑定resource/version/readers/writers/conflict edge。

### A5 — Residual/bias closure

diamond必须恰两支到同一join，shape/dtype/polarity/version一致；bias token每支产生、join accumulate并到
terminal owner。否则`RESIDUAL_BIAS_TOKEN_UNCLOSED`，witness含branch path与token断点。

### A6 — Alias/logical lifetime

first/last use从A1独立计算；borrowed/parameter/mutable state不得被scratch alias；同alias set的可写live
interval不得重叠，除非显式view且owner/access完全一致；in-place要求input last-use等于op ordinal且
type/role不变。冲突产生`UNSAFE_ALIAS_OR_LIFETIME`及value pair/interval/access/owner witness。本阶段
不claim physical offset safety。

### A7 — Dense escape/VJP owner

从coefficient/incoming-adjoint做representation lineage propagation。dense lineage到external consumer、
region output、saved state、persistent storage或autograd owner时产生`DENSE_A_ESCAPE`；witness含source→
sink最短path、sink storage/role、contract/owner。VJP owner还须与compressed index/location/sign一一匹配，
higher-order拒绝，mutation none-inside-vjp。

### A8 — Legality closure

`admitted=true`仅当A0—A7全过、ordered ops/boundary/witness完整、rejection为空，且analysis hash绑定
module/request/pass versions/witness。失败时`admitted=false`，reason按冻结enum ordinal排序，绝不进入
lowering/launch。

## 6. Shallow 与 full witness 分层

### 6.1 `VerificationAnalysisWitnessV1`

`LegalityResultV1` 中的 witness 字符串必须引用独立 typed ledger，不能写自由文本占位。每条 witness：

```text
witness_id, witness_version, pass_id
verdict = proof | conflict
subject_ids[], source_edge_ids[], path_ids[]
coverage_ids[], detail_code
module_hash, request_hash, witness_hash
```

- `proof`：列出本 pass 实际检查的全部 value/op/effect/contract coverage IDs 及 coverage hash；
- `conflict`：列出最小 causal subject/edge/path；
- 空字符串、固定 `ok`、只含 pass 名或未绑定 module/request 的 witness 一律拒绝；
- admitted 结果的 external-use/effect/alias/dense/VJP 五组必须各至少一个 `proof` witness；
- rejected 结果保留已完成 pass 的 proof，并为每个 reason 增加 conflict witness；
- postdominator/residual proof 也必须进入 ledger，即使当前 `LegalityResultV1` 没有独立字段，也要由
  analysis hash 和 ordered witness inventory 绑定。

### 6.2 Rejection evidence

每条 rejection record：

```text
reason
evidence_kind = shallow_policy_rejection | analysis_witness_rejection
pass_id, subject_ids[], source_edge_ids[], path_ids[]
detail_code, witness_hash
```

自由文本不进入verdict；`detail_code`为stable enum。同名reason可同时有两类证据，但coverage分开。

### 6.3 七类 analysis-only reason

| Analysis reason | Pass | full causal witness |
|---|---|---|
| `REGION_EXTERNAL_USE` | A2 | producer→outside consumer edge/path |
| `REGION_NOT_POSTDOMINATED` | A3 | entry→dead/escape path |
| `EFFECT_ORDER_CONFLICT` | A4 | resource/version conflict edge |
| `RESIDUAL_BIAS_TOKEN_UNCLOSED` | A5 | branch或bias lineage断点 |
| `UNSAFE_ALIAS_OR_LIFETIME` | A6 | alias pair + overlapping intervals/access |
| `DENSE_A_ESCAPE` | A7 | dense source→external/saved/persistent sink |
| `QUEUE_OR_TERMINATION_EFFECT_CROSSED` | A4 | region writer→external resource |

七类reason即使schema浅层可触发，也必须由schema-valid negative graph走到对应pass才算coverage。

## 7. Positive 与 negative workload

### 7.1 三类 positive snapshot

三类adapter分别构造至少一个positive并由同一pipeline admitted：

1. empty-β Conv：compressed α、显式`[D,0]` β、Conv propagation、minimal VJP；
2. active-β Linear：location/sign/history、nonempty compressed owner、Linear propagation；
3. multi-Conv/residual metadata：至少三个相邻affine site、residual/bias closure、10 evaluation/9 mutation
   effect versions，但不执行trajectory。

历史实例名只允许在artifact provenance映射中出现；production module/schema不得包含实例名或shape常数。
admitted只表示metadata closed，不表示provider replacement、rule match、lowering或性能。

### 7.2 七类 full analysis negative

每类至少两个case：一个最小synthetic graph、一个positive snapshot单点变异。共至少14个case，全部先过
A0，再在冻结pass以exact reason和full witness拒绝。

### 7.3 十五类 direct-through-capture

15 direct reason从source snapshot/request mutation进入pipeline，不直接调用leaf constructor。必须证明
reason exact、在analysis output/lowering/launch前拒绝、无伪full witness、source→captured mapping可复核。

### 7.4 Multi-reason

至少四个graph各含两个独立错误：reason按enum排序，witness不覆盖；删除一个错误并全重签后另一个仍拒绝。

## 8. Formal artifact 与 replay

### 8.1 Manifest

绑定implementation commit、parent=`94166b6`、approved prereg closure、所有source/runner/test blob hash、
adapter/schema/pass versions、Python/OS、input snapshot/request digest和全部false flags。不需要GPU/TVM；
不得含`/home/`、用户名、temp path、tensor payload或secret。

### 8.2 Five-fresh raw-first

五个独立Python process从position 0执行三positive、15 direct、至少14 full-analysis、四multi-reason和
canonical round-trip。不得resume。五组ID/order/reason/witness/hash逐字节相同；这是determinism，不是
性能采样。

### 8.3 文件

```text
manifest.json, protocol.json
source_snapshots.jsonl, capture_requests.jsonl
captured_modules.jsonl, capture_receipts.jsonl
legality_results.jsonl, witnesses.jsonl
negative_cases.jsonl, raw_process_runs.jsonl
summary.json, replay_stdout.txt
```

每行含position/case/payload hash；summary只从raw重算。

### 8.4 Semantic replay

replay先验digest，再从snapshot/request重跑adapter、strict module round-trip和A0—A8，重算witness/result/
hash/five-fresh summary，stdout逐字节一致。只读取冻结legality result再比较hash不合格。

## 9. Fully re-signed tamper

至少16类，均修改内层后重签全部outer digest：

1. op kind/provenance；2. value role/polarity；3. graph hash/ID mapping；4. candidate set/boundary；
5. consumer/external use；6. entry/exit/postdominator path；7. effect version/writer；
8. queue/termination改write；9. residual/bias lineage；10. alias/lifetime/access；
11. dense sink/saved state；12. VJP owner/policy；13. shallow evidence改标full；
14. reason/witness/pass ID；15. capture/analysis/pass version；16. false flag翻转。

必须`16/16 rejected`；5/6/7/9/10/11/13必须靠因果重算而非未重签digest拒绝。

## 10. 实现与验证顺序

预注册外审批准后只允许：

1. 以批准plan commit为父点；
2. 一次bounded implementation/formal提交加入adapter、analysis、tests、runner/replay、artifact；
3. 不做中途性能调优，不改gate；
4. targeted与related Bound/Task/Structured/R3/GC0 tests；
5. full `pytest tests`；
6. Black/Mypy/Pylint/diff；
7. DocOps change/validation/lint；
8. 独立GC0-1 implementation exchange；
9. approve并由executor close后，才开放GC0-2 lowering/arena-ABI预注册。

## 11. Plan acceptance criteria

### Plan-AC1 — 顺序/范围

parent精确为`94166b6`；实现前外审批准；无lowering/runtime/timing；algorithm无模型/site常数。

### Plan-AC2 — Capture

三adapter共享snapshot/analysis；mapping双向唯一、omission fail closed；five-fresh ID exact；三positive
由adapter构造，不能手写最终module绕过capture。

### Plan-AC3 — Analysis

A0—A8顺序/version冻结；所有witness可从source重算；admitted rejection为空且witness完整，rejected至少
一个stable reason。

### Plan-AC4 — Shallow/full

两类证据分开；七analysis reason各至少两个schema-valid full negative；浅层分支不计full coverage；
`DENSE_A_ESCAPE`必须有lineage path。

### Plan-AC5 — Rejection coverage

15 direct-through-capture、7 full analysis、至少4 multi-reason全过；reason exact/order stable，拒绝发生在
lowering/launch前。

### Plan-AC6 — Artifact/replay/tamper

five-fresh exact；replay重跑capture+analysis；16/16 full-resign tamper拒绝；artifact无本机路径/payload/
timing。

### Plan-AC7 — 工程/claim

targeted/related/full、Black/Mypy/Pylint/diff/DocOps全过；skip是既有环境边界；authority docs无漂移；
`execution_enabled=false/timing_open=false/performance_claimed=false/ASPLOS-ready=false`。

## 12. GO、NO-GO、INVALID 与后继

### 12.1 GO

Plan-AC1—Plan-AC7全部通过并经独立外审后，状态只能是：

```text
VALIDATED-GC0-1-VERIFICATION-CAPTURE-ANALYSIS
```

只开放GC0-2 lowering/arena-ABI**预注册**，不开放实现、GC-1、runtime或timing。

### 12.2 NO-GO

任一positive不能通用表达/admit、任一analysis reason不能给出full witness、必须依赖模型/site特判、
source metadata不足以证明ownership，或five-fresh不稳定且原因可归因时，状态为
`VALIDATED-NO-GO-GC0-1-CAPTURE-ANALYSIS`。保留GC0-0 claim，停止GC0-2。

### 12.3 INVALID

顺序错误、partial/resume、artifact/hash缺失、shallow冒充full、replay不重跑analysis、tamper只靠outer
digest、测试/lint/claim不闭合，均不得形成结论；只允许修复tooling后按原gate全量重跑。

## 13. 外审要求

外审不得采信changelog摘要，应独立核对parent/diff；检查三adapter是否共享protocol；从snapshot重建
至少一positive与七analysis negative；重算topology/postdominator/effect/lifetime/dense witness；验证
shallow/full不混用；重算five-fresh；自建full-resign shallow→full tamper；复跑测试、静态检查、exchange
validate/lint，并按blocker/major/minor/info报告。
