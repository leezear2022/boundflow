---
status: diagnostic-complete-code-closed
date: 2026-08-29
type: implementation-readiness
topic: boundflow
slug: asplos27-s4-4-formal-evidence-readiness
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
same-solver-claimed: false
complete-query-claimed: false
tenx-claimed: false
---

# ASPLOS'27 S4-4：formal evidence、stdlib replay与trust anchor实施就绪审计

> **2026-08-29 trust-owner supersession**：本稿的source closure、codec、raw budget、fault拓扑与artifact外anchor
> 诊断继续有效；§2中`ANCHORED_AUTHENTIC = self-check + Git/DocOps anchor exact`现收窄为E1
> `CHALLENGE_BOUND`上限。正式E2必须增加auditor-controlled/delegated fresh run、stdlib独立重算和execution
> witness；executor自签Git/DocOps记录不构成独立主体。详见
> `BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_AND_WITNESS_PLAN_2026_08_29.md`。

> 2026-08-29施工冻结修订：本稿的source/codec/anchor诊断事实继续有效；S4-3新状态机、15-fault registry、
> variant-specific raw floor、16-node seal DAG和96-case tamper现由
> `BOUNDFLOW_ASPLOS27_S4_4_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`拥有。实施以施工包为准；
> 本稿旧`1,341,776 B/run`、`37.8557—47.5839 MiB`和71类tamper均被取代。

## 0. 直接结论

S4-4原蓝图的方向正确：18个B0/R/C positive worker、完整IEEE raw、标准库replay和fully re-signed tamper都是
必要的。但它还不能直接实现。本轮对S3 v2、RVIR whole-core `.pt`、真实S3 worker source closure和raw编码做了
独立诊断，确认必须先修正六个结构问题：

1. **artifact自洽不等于真实性**：攻击者若能同时替换raw、summary、source identity和manifest，artifact内部没有
   信息可证明“原source/model才是真的”；必须增加artifact外部trust anchor；
2. **inline base64不是合适的whole-core载体**：现有B0/R raw存在大量重复payload；改为每worker内
   content-addressed binary payload sidecar + tensor index，可在不丢IEEE bits/alias语义的前提下显著减小artifact；
3. **手写source allowlist远远不足**：S3 v2只列12个`CODE_PATHS`，单个normal worker结束时却已加载101个
   BoundFlow core Python文件、4个repo script、559个TVM/TVM-FFI Python文件和3个repo native library；
4. **逐call tracing不可作为formal observer**：一次`sys.setprofile`诊断运行超过120秒仍未完成，而正常worker约
   15—17秒；source closure必须使用低扰动module/native snapshot + 显式关键组件receipt；
5. **fault worker拓扑少算了15个进程**：15个poison/abort注入不能在positive worker内串行复用；正式总数应为
   `18 positive + 15 fault = 33 subprocess`；
6. **seal顺序存在潜在循环且status过早**：summary不能绑定后生成的tamper report，tamper report也不能依赖最终
   manifest；artifact内部通过只能是`PENDING-EXTERNAL-AUDIT`，外审批准后才可写`VALIDATED`。

S3 exchange当前仍只有delivery、无audit result。因此本文仍是documentation-only，**不开放S4-4代码或formal run**。

## 1. 证据范围

### 1.1 亲读和实测对象

- `scripts/run_asplos27_s3_optimizer_artifact.py`与v2 runner；
- S3 v2 tamper probe和18-worker raw；
- `rvir-v4-five-fresh` original/candidate `.pt`；
- RVIR whole-core truth、live-return和native KFSB artifact；
- S4-1D full-IEEE预算、S4-2 10/9 trajectory预算、S4-3 transaction readiness；
- Python stdlib `json/gzip/base64/struct`对duplicate key、overflow float、half/bfloat16和deterministic gzip的行为。

### 1.2 diagnostic不是formal

本轮执行了两个single S3 worker source-closure probe和若干离线codec/size probe。它们只回答“现有基础设施的设计
缺口”，不形成S3/S4性能或correctness结论。call-trace probe因observer perturbation被人工中止，必须原样披露，不能
删掉后只保留normal module snapshot。

## 2. 外部trust anchor：关闭“全重签后自证真实性”悖论

### 2.1 不可能由artifact内部证明的事实

若攻击者可同时修改并重签：

```text
source_identity.json
protocol.json
all raw
summary.json
manifest.json
replay_stdout.txt
```

则以下替换可以保持内部完全自洽：

- 换成另一个合法Git commit并同步换code blobs；
- 换model/property并重新生成全部raw；
- 换replayer并让它接受新schema；
- 伪造一组彼此不同的PID/start-time字段。

哈希只能证明“这些文件彼此一致”，不能证明“它们就是审计委托中指定的那一批”。因此旧要求“71类全部必须在
outer re-sign后由artifact semantic invariant拒绝，且不能依赖digest”对source authenticity和process freshness并不
可实现。

### 2.2 双层验证

冻结两种不同结论：

```text
SELF_CONSISTENT
  artifact manifest + raw-derived semantics internally consistent

ANCHORED_AUTHENTIC
  SELF_CONSISTENT
  + externally supplied trust anchor exact
  + anchor itself由Git/DocOps exchange request固定
```

formal closure必须运行`ANCHORED_AUTHENTIC`；只跑self-check不能关闭S4-4。

### 2.3 trust anchor不放在artifact信任域内

建议schema：

```text
S4ExternalAuditAnchorV1:
    artifact_schema
    artifact_relative_path
    final_manifest_sha256
    semantic_root_sha256
    source_revision
    protocol_sha256
    model_sha256
    property_sha256
    config_sha256
    replayer_git_blob_sha256
    expected_positive_worker_count = 18
    expected_fault_worker_count = 15
    expected_claim_flags
```

anchor由formal result提交后的DocOps external-audit request/delivery持有并由Git commit绑定。artifact目录可保存一份
untrusted convenience copy，但formal replayer必须显式接收**artifact外部**的anchor路径；不能自动选择artifact内副本。

### 2.4 tamper enforcement分层

施工包96类攻击必须逐项声明enforcement layer；本节原71类只是其历史子集：

- `EXTERNAL_ANCHOR`：source revision、code blob集合、model/property/config等真实性；
- `FROZEN_PROTOCOL`：容差、claim flag、worker/variant/path/状态机常数；
- `RAW_SEMANTIC_DERIVATION`：trajectory、Adam、numeric parity、KFSB、commit/post/queue等；
- `EXECUTION_EVIDENCE_ONLY`：fresh process证据。

对raw semantic攻击，tamper probe应在更新临时artifact外层digest后运行self-consistency mode，并要求稳定semantic reason；
不能让original anchor的manifest mismatch抢先拒绝所有攻击。对authenticity攻击，anchor mismatch正是正确理由。

“fresh subprocess”只能由parent spawn纪律、OS process receipt和现场重跑形成高强度证据，不能声称offline JSON提供
不可伪造证明。

## 3. source closure：从12文件allowlist升级为分层闭包

### 3.1 current S3事实

S3 v2 `CODE_PATHS`共12项。single normal NDP worker结束时module/native snapshot为：

```text
BoundFlow core Python files       101
repo scripts                       4
TVM/TVM-FFI Python files          559
repo-local loaded native files      3
snapshot inventory SHA256
421ce0b764b92e70360933aed8f4e25aaca02a99313ceab491174a16cb674c83
```

三个native文件为：

```text
.cache/tvm-ffi/libtorch_c_dlpack_addon_torch212-cuda.so
boundflow/3rdparty/tvm/3rdparty/tvm-ffi/build/lib/libtvm_ffi.so
boundflow/3rdparty/tvm/build-boundflow/libtvm.so
```

loaded module是执行闭包的保守superset，不等于668个文件都走过热路径；但这已经证明12项手写列表不能叫complete
source closure。

### 3.2 为什么不用全局call trace

一次`sys.setprofile`逐call收集文件名的worker在120秒后仍卡在PyTorch optimizer触发的Dynamo/SymPy import，正常
无trace worker约15—17秒。该observer会显著改变初始化、import顺序和wall behavior；即使S4-4不形成性能claim，也
不应让formal correctness依赖一个会改变程序生命周期的全局trace。

### 3.3 冻结五层source identity

```text
L0 trusted revisions
   BoundFlow HEAD / submodule gitlinks / external repo commits / clean scope

L1 loaded Python inventory
   worker结束时sys.modules下实际存在且位于trusted roots的真实文件

L2 declared execution inventory
   runner/worker/replayer/tamper + S4-0—S4-3显式entrypoint/callable source

L3 loaded native inventory
   /proc/self/maps中的relevant .so + file SHA256/size/build receipt

L4 compiled artifact identity
   Plan/TIR/module/device-source/cache receipt hashes
```

parent取所有33个worker的variant-specific union。BoundFlow core文件逐文件记录Git blob/SHA256；TVM/TVM-FFI和外部
provider记录git revision、clean状态和loaded relative-path set；native `.so`逐文件绑定SHA256和build receipt。

### 3.4 clean scope要诚实

正式字段分开：

```text
full_worktree_clean
source_scope_clean
unrelated_dirty_paths
untracked_importable_code_count
```

用户保留的文档/PDF dirty不应自动污染source scope，但不能把全仓dirty伪写成`worktree_clean=true`。所有trusted
import root下的untracked `.py/.pyc/.so`一律fail closed，避免PYTHONPATH shadow。

### 3.5 pre/post race

- 第一个worker前冻结revision、declared inventory与input digests；
- 每worker结束保存loaded/native inventory及其digest；
- 最后一个worker后重新核对HEAD、gitlinks、external commits和source-scope file hashes；
- 任一worker观察到不同core/native digest，整批non-formal；
- formal parent本身不得import torch/tvm或初始化CUDA。

module snapshot不能发现任意`exec`字符串，因此S4 implementation禁止runtime-generated Python source；TIR/device code由
compiled receipt单独绑定。

## 4. Tensor raw：改为indexed binary sidecar

### 4.1 当前inline base64的实际成本

对现有RVIR five-fresh raw做等价编码：

| legacy raw | tensor records | logical payload | inline-base64 JSONL.gz |
|---|---:|---:|---:|
| B0 original | 577 | 1,869,400 B | 1,036,878 B |
| R/C candidate | 451 | 872,952 B | 583,659 B |

相同payload内有大量重复内容：

| legacy raw | unique-content payload | duplicate saving |
|---|---:|---:|
| B0 original | 770,500 B / 95 payload | 1,098,900 B |
| R/C candidate | 572,344 B / 64 payload | 300,608 B |

若改为content-addressed binary gzip + compressed index：

```text
B0: index 10,670 + binary 214,951 = 225,621 B
R/C: index 7,262 + binary 198,918 = 206,180 B
```

相对inline base64分别减少约`78.24% / 64.67%`。S3 P-anchor三模式trajectory也证明binary可行：logical
`619,920 B`、unique `458,880 B/139 payload`、binary gzip=`100,698 B`。

这些是旧raw的codec诊断，不是S4最终artifact size claim。

### 4.2 每worker自包含，不跨worker共享payload

目录修正为：

```text
raw/w00-b0/
  worker.json
  tensor_index.jsonl.gz
  payload_index.jsonl.gz
  tensor_payloads.bin.gz
  trajectory.jsonl.gz
  kfsb.jsonl.gz
  transaction.jsonl.gz
  post_queue.jsonl.gz
  source_inventory.json
  stdout.txt
  stderr.txt
```

同一worker内payload按raw-byte SHA256去重；不同worker之间禁止共享payload文件，保持fresh worker completeness与删除检测。

### 4.3 `TensorIndexRecordV2`

```text
worker_ordinal
variant
phase
semantic_path
ordinal
dtype
shape
stride
storage_offset
layout = strided
byte_order = little
payload_id
payload_nbytes
tensor_value_sha256       # dtype + shape + raw bytes
source_device
materialization_reason
object_group
storage_group
storage_nbytes
```

`object_group/storage_group`是worker内按lexicographically-first semantic path分配的稳定label；raw pointer、Python
`id()`和storage handle绝不进入canonical artifact。empty tensor payload相同不代表storage alias。

### 4.4 payload sidecar

`payload_index.jsonl.gz`按`payload_id`排序：

```text
payload_id = sha256(raw bytes)
offset
nbytes
```

解压后的`tensor_payloads.bin`恰为所有unique payload按`payload_id`排序连接。replayer验证offset连续、无gap/overlap、
无unreferenced payload、每段SHA256和总stream hash。

### 4.5 dtype与bits

正式支持：

```text
bool, uint8, int8, int16, int32, int64,
float16, bfloat16, float32, float64
```

complex、quantized、sparse/non-strided layout和未列dtype fail closed。writer把contiguous logical value规范为little-endian
bytes；replayer用`struct`解码float16/32/64，bfloat16通过`uint16 << 16`重建float32。

bool payload只允许0/1；base64不再是正式载体。signed zero、NaN class/sign/payload和Inf sign直接从bits解析。是否要求
NaN payload exact由per-path comparison policy决定：ternary poison path要求canonical qNaN bits，普通finite-required path
出现任意nonfinite即失败。

### 4.6 strict parser与decompression边界

stdlib replayer必须：

- JSON duplicate key拒绝；
- 禁止JSON float scalar，容差和derived float以decimal string或f64-bit hex保存；
- 拒绝`1e999→inf`、NaN/Inf token、NUL/control character；
- 拒绝非canonical UTF-8/JSON、duplicate/unsorted tensor key；
- gzip仅允许单member、固定header policy、无trailing bytes；
- compressed/uncompressed size、record count、line length和payload bytes都有上限；
- manifest tree拒绝symlink、absolute path、`..`、backslash、duplicate normalized path和unlisted file。

gzip byte-for-byte重建只在pinned Python/zlib环境保证；跨环境外审只要求解压后的canonical stream hash与semantic replay
一致，不能错误要求不同zlib版本重新压缩出同一file digest。

## 5. numeric comparison policy不是一个全局atol

每个path由frozen registry指定：

```text
EXACT_BITS
EXACT_DISCRETE
FINITE_ATOL_RTOL
FINITE_SIGN_EXACT
SIGNED_ZERO_DISCLOSE
CANONICAL_QNAN_BITS
NONFINITE_FORBIDDEN
```

`FINITE_ATOL_RTOL`仍使用lower/state=`2e-4`、compiled gradient=`2e-5`，但raw保留完整bits。replayer逐path输出max
abs/rel、argmax path/index、finite class、sign和signed-zero count；float64有限数相减若overflow必须显式标记并失败，
不能因rhs变Inf而自动通过。

fixture-specific path inventory可以冻结ResNet2B的六α、active β、six lA和KFSB数量；generic codec/schema不得硬编码
model/node/shape常数。

## 6. worker拓扑：18 positive + 15 fault

### 6.1 positive workers

保持六triplet：

```text
B0-R-C / B0-C-R / R-B0-C / R-C-B0 / C-B0-R / C-R-B0
```

每个成员独立`spawn/exec`，共18个process。它们在correctness artifact中不形成latency headline；顺序用于相邻时间段
pairing和状态污染检查。

### 6.2 fault workers

S4-3施工包修正后的15个注入点每个必须独立candidate process：

```text
preclaim validation 1
KFSB/scratch/provider constructor 3
device copy ordinals 1/6/12 3
host/container/core seal 3
official post entry/mid/return 3
queue add mid/check-worst 2
total 15
```

poisoned worker不得执行第二个case或返回到positive path。正式计数：

```text
positive_worker_count = 18
fault_worker_count    = 15
total_subprocess_count = 33
```

### 6.3 process receipt边界

每个worker保存parent-assigned ordinal/nonce、PID、parent PID、OS start-time tick、interpreter digest和CUDA init receipt；
hostname/user/home/raw CUDA pointer不进入artifact。重复/缺失/非法lineage可离线拒绝，但“确实是fresh OS process”最终仍需
parent执行纪律和外部现场重跑支持。

parent必须是stdlib-only、不得import torch/tvm或创建CUDA context；interpreter选择显式绑定，不允许静默PATH fallback。

## 7. raw payload规划预算（施工修正版）

旧账使用过时C trajectory、为B0伪加candidate snapshot并重复计入KFSB。施工包按semantic occurrence重算：

```text
B0/R/C per positive = 3,829,232 / 3,897,248 / 2,537,888 B
18 positive total   = 61,586,208 B

fault minimum/run   = C policy 1,511,936 + pre/candidate/fault 102,024
                    = 1,613,960 B
15 fault minimum    = 24,209,400 B

33-worker minimum   = 85,795,608 B = 81.8210678100586 MiB
```

这是tensor-occurrence floor，不是unique payload或compressed file size；fault阶段额外raw尚未包含。manifest必须分别
报告occurrence、unique-content、decompressed、compressed和tree bytes。

## 8. seal DAG：消除summary/tamper/manifest循环

冻结有向无环顺序：

```text
protocol/environment/source declarations
  → 18 positive raw + 15 fault raw
  → semantic_root.json（只绑定上游raw/declarations）
  → raw-derived summary.json
  → tamper_report.json（绑定baseline semantic_root，不绑定final manifest）
  → replay_stdout.txt（只输出semantic-root/summary结果）
  → README.md
  → final manifest.json（绑定上述所有文件，自身除外）
  → external trust anchor（artifact外部）
  → anchored final replay
```

`summary.json`不得包含后生成的tamper-report hash；tamper report不得声称final-manifest hash。两者由final manifest共同
绑定。`replay_stdout.txt`不输出final manifest hash，保证final replay可逐字重建；anchored replay的manifest/anchor结果
另输出到terminal或外部audit report。

manifest同时保存compressed file SHA/size/role；semantic root保存decompressed canonical stream SHA和schema role。

## 9. replayer三种模式

```text
derive
  raw/declarations → derived summary（generator内部使用）

self-check
  manifest完整性 + semantic root + raw-derived summary/tamper inventory

anchored-check
  self-check + external anchor + replayer blob identity
```

formal PASS必须是`anchored-check`。replayer只import标准库，不能importrepo artifact helper、production validator、Torch、
TVM、NumPy或provider。允许的stdlib扩展包括`decimal/zlib/pathlib/argparse`。

summary中的科学数值用稳定bit/decimal-string schema，避免不同Python版本的float JSON格式成为“语义差异”。

## 10. status与claim边界

artifact内部只允许：

```text
FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT
FORMAL-NO-GO-S4-SAME-SOLVER-CORRECTNESS
```

不能在external audit前写`VALIDATED-S4-*`。只有外审批准、exchange关闭和closure commit完成后，权威文档才可升级为：

```text
VALIDATED-S4-SAME-SOLVER-CORRECTNESS
```

三种状态都保持timing/performance/query/queue/tenx/ASPLOS-ready claim为false。`queue_claimed=false`不妨碍记录固定路径
queue accounting correctness。

## 11. 96类tamper施工registry

历史1—71全部保留其语义，并由施工包扩为T01—T96；每案新增：

```text
expected_enforcement_layer
anchored_mode_required
self_check_semantic_reason_or_none
fully_resigned_files
replayer_exit_code
stable_reason
```

特别修正：

- 1—6真实性攻击可由external anchor拒绝，不能伪称raw math发现了source/model替换；
- 7—8由frozen protocol/claim hard gate拒绝；
- 12只证明process receipt结构/uniqueness，不能升级为离线不可伪造freshness；
- 27—39必须覆盖S4-3 prefix rollback、post/queue独立状态和三个counter；
- 40若只改单侧tensor应由numeric parity拒绝；若攻击者重生成一个完全自洽的新artifact，则由anchor拒绝；
- 42 exact tamper inventory由replayer内置case registry和manifest共同关闭；
- 60—67 alias/live身份攻击需要protocol fixture expectation + worker identity projection，不能只看tensor payload。

每个semantic case必须先证明修改后的artifact外层file/manifest digest全部合法，再检查stable semantic reason。tamper report
本身仍不是不可伪造证明；外审应现场重跑并另造至少3案。

## 12. 安全与隐私门禁

- artifact tree和解压raw扫描HOME、用户名、绝对repo/external路径、SSH/token-like值；
- source path只保存`owner + relative_path`；
- environment只保存allowlist和value分类，禁止全量dump；
- failed worker traceback放在sibling non-formal failure目录，不能混入formal tree；
- successful stdout/stderr必须结构化且通过leak scanner；
- gzip/JSON解析有资源上限，避免审计方打开zip bomb或超长line；
- manifest禁止symlink和path traversal；
- formal tree从空目录一次生成，失败不能resume或补齐。

## 13. tests与acceptance

### 13.1 codec/parser

- 所有支持dtype round-trip；
- ±0、subnormal、max finite、NaN payload、±Inf；
- empty tensor相同payload但distinct storage group；
- noncontiguous view按logical contiguous bytes保存、identity metadata保留原stride/offset；
- duplicate JSON key、1e999、noncanonical base64 legacy input、gzip multistream/trailing bytes、zip bomb；
- payload offset gap/overlap、orphan/unreferenced payload；
- symlink/path traversal/duplicate normalized path。

### 13.2 source/anchor

- current core loaded module不在inventory；
- untracked importable Python/native file；
- submodule/external dirty；
- native binary digest/build receipt drift；
- model/property/config/replayer/source commit替换；
- artifact内伪anchor不被自动采信；
- self-check通过但anchored-check失败的反例。

### 13.3 process/fault

- 18/15/33 exact；
- parent CUDA未初始化；
- worker duplicate/missing/order/lineage；
- 每fault恰一个fresh worker；
- poisoned fault不继续post/queue/fallback；
- positive/fault raw严格分区且都被semantic root绑定。

### 13.4 replay/tamper

- summary所有字段都有raw/protocol/source来源；
- 96 case registry exact；95类拒绝，freshness attestation一类输出offline limitation；
- authenticity/semantic/freshness enforcement不混写；
- final manifest/tamper/summary无hash cycle；
- stored replay stdout可逐字重建；
- 外审另造攻击仍拒绝。

## 14. 实现短提交顺序

仅在S3外审批准且S4-0—S4-3依序关闭后开放：

1. `docs: preregister S4-4 anchor and evidence DAG`；
2. `feat(artifact): add strict stdlib tensor index and binary payload codec`；
3. `feat(artifact): add source scope native inventory and anchor schema`；
4. `feat(artifact): add S4 positive whole-core worker projection`；
5. `feat(artifact): add 18-worker stdlib parent runner`；
6. `feat(artifact): add 15 isolated fault workers`；
7. `feat(artifact): add derive self-check and anchored replay`；
8. `test(artifact): add 96 layered fully re-signed attacks`；
9. `artifact: generate S4 whole-core formal candidate`；
10. `docs: deliver external anchor and audit exchange`；
11. `docs: close S4 correctness or formal NO-GO`；
12. `docs: preregister S4-P timing`。

artifact代码、正式raw、external anchor/audit request和closure必须分提交。formal raw只能由clean tracked代码生成。

## 15. GO / STOP

### GO

- 18 positive + 15 fault=`33` independent subprocess完整；
- source scope、loaded modules、native binaries、compiled receipts和inputs全部闭合；
- indexed binary payload可由stdlib完整恢复IEEE、path、view和alias投影；
- semantic root、summary、tamper、manifest、external anchor形成无环证据DAG；
- self-check与anchored-check均PASS；
- B0/R/C whole-core到candidate queue insertion parity关闭；
- 96-case registry exact；95类按正确enforcement layer拒绝，1类freshness只形成execution evidence；
- artifact状态仍pending external，全部性能flag false；
- external audit批准后才升级validated。

### STOP

- 仍以12项手写CODE_PATHS声称complete source closure；
- 把全局call trace放入formal worker并忽略observer perturbation；
- 只用artifact内hash证明source/model真实性；
- 只保存`.pt`或inline summary，不保存stdlib binary raw；
- fault cases复用poisoned process；
- summary/tamper/manifest互相循环引用；
- freshness字段被宣称为不可伪造attestation；
- external audit前artifact写`VALIDATED`；
- 任何S4 correctness失败被放宽容差或删字段掩盖。

## 16. 当前状态

```text
S3 external audit                  = pending
S4-4 infrastructure audit         = complete
S4-4 codec/source/anchor design    = implementation-ready
S4-4 production artifact code      = closed
S4-4 formal run                    = closed
S4-P timing/performance            = closed
```

下一外部动作仍是S3审计。本文只确保S4-4未来开工时不会把artifact自洽误写成真实性、把18个positive worker误写成
全部fault evidence，或用低效/不可独立审计的raw格式制造新的证据债。
