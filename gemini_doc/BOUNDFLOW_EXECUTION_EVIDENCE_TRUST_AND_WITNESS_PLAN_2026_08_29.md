---
status: design-frozen-code-closed
date: 2026-08-29
type: evidence-trust-plan
topic: boundflow
slug: execution-evidence-trust-and-witness
stage: s03-s04
execution-authority: false
code-change-open: false
formal-run-open: false
external-audit-verdict: pending
performance-claimed: false
---

# BoundFlow 执行证据可信度、challenge 与外部见证计划

## 0. 一句话结论

BoundFlow 的 artifact 必须明确区分四件事：

```text
数值/状态语义正确
  ≠ artifact内部自洽
  ≠ 某个新鲜进程真的执行过
  ≠ 某个独立可信主体见证了该执行
```

S3影子预审已经构造出两份“修改raw→重算summary→重签manifest”的内部自洽artifact，证明自签名哈希链
不能独自证明GPU物理执行真实性。S4-4已有`SELF_CONSISTENT / ANCHORED_AUTHENTIC / OFFLINE_UNATTESTABLE`
骨架，但还缺外部主体所有权、challenge发行顺序、witness响应和签名/非签名模式的精确定义。

本计划冻结这些缺口。它不新增求解器IR，也不更改CROWN/IBP语义；它只定义formal evidence协议。S3外审批准前
仍不开放S4代码或计时。

## 1. 触发事实

### 1.1 S3反例

现有S3 v2 probe的10类攻击会修改raw或派生字段、重签外层manifest，但保留至少一项派生语义不一致，因此被
replay拒绝。执行方影子预审另造两类更强攻击：

1. 把全部dynamic allocated从`13,824 B`改为`0 B`，再从新raw重算summary和manifest；
2. 修改一条P latency，更新该行median、全局summary和manifest。

两者都被self-check接受。这是预期的信任边界，不是一个能靠增加固定headline常数修复的问题。

### 1.2 S4既有设计

S4-4已经正确冻结：

- artifact内部只允许`SELF_CONSISTENT`；
- 外部anchor参与后才允许`ANCHORED_AUTHENTIC`；
- process freshness不能由offline JSON证明，T19必须输出`OFFLINE_UNATTESTABLE`；
- 96-case registry按external anchor、frozen protocol、raw semantic derivation与execution evidence分层。

本计划不推翻上述设计，而是把“谁发行anchor、何时发行、谁运行、谁签、签名究竟证明什么”补全。

## 2. 外部机制调研与采用边界

### 2.1 in-toto Statement

[in-toto Statement v1](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)用subject digest绑定
被证明对象，并用`predicateType`区分声明语义。BoundFlow可复用该三层思想：

```text
subject       = final manifest + semantic root
predicateType = boundflow execution witness schema URI
predicate     = challenge、执行环境、runner、结果与审计判定
```

但subject digest只标识对象，不证明predicate内容为真。

### 2.2 DSSE

[DSSE](https://github.com/secure-systems-lab/dsse)把payload type与原始payload bytes一起签名，避免类型混淆；
其规范同时明确把key management、身份绑定、payload语义和verification policy留给外部系统。因此BoundFlow可以把
witness装进DSSE envelope，但不能写成“用了DSSE所以benchmark真实”。

### 2.3 SLSA provenance

[SLSA provenance](https://slsa.dev/spec/v1.2-rc2/build-provenance)区分外部参数、resolved dependencies和
run details，并把builder身份视为信任边界。BoundFlow采用同样划分：

- audit challenge是external parameters；
- source/model/protocol/replayer digest是resolved dependencies；
- worker/process/GPU/时间与输出root是run details；
- auditor身份与其公钥/平台是独立trust policy。

### 2.4 透明日志

[Sigstore Rekor](https://docs.sigstore.dev/logging/overview/)提供append-only透明日志和inclusion proof。它能证明某个
已签声明在某时进入日志且之后未被静默改写，但仍不能证明声明中的GPU latency真实。因此Rekor只作为可选的
publication/time-order强化，不是S3/S4 correctness或performance的必要条件。

### 2.5 GPU硬件attestation

[NVIDIA Attestation](https://docs.nvidia.com/attestation/index.html)能对受支持GPU的硬件/固件状态形成加密证明；官方
quick start要求H100或更新且具备Confidential Computing能力。当前实验机是GeForce RTX 4060 Laptop，不满足该
正式路径。即使未来换到受支持设备，硬件attestation主要证明设备/固件状态，也不能自动证明BoundFlow用户态代码、
实验协议和latency统计正确。

## 3. 威胁模型

### 3.1 角色

```text
E = executor：实现代码、生成formal artifact
A = auditor：冻结问题、发challenge、独立重算/运行、给verdict
R = reader：论文审稿人或第三方artifact evaluator
H = hardware/platform attester：可选，不是当前环境能力
```

### 3.2 必须覆盖的攻击者

| 攻击者能力 | self-check能否防 | 需要的额外机制 |
|---|---|---|
| 只改一个文件、不重签 | 能 | manifest/file digest |
| 改raw并只重签外层 | 部分能 | raw-derived semantic replay |
| 改raw、summary、manifest全链 | 不能 | 独立anchor或fresh witness |
| 换source/model/protocol全套重跑 | 不能 | auditor冻结external parameters |
| 复用旧artifact冒充新运行 | 不能可靠防 | auditor challenge + auditor-controlled launch |
| executor见到challenge后伪造完整run | challenge单独不能防 | auditor亲自运行或可信builder/attester |
| auditor身份被冒充 | 不能 | 外部固定公钥/平台identity与threshold policy |
| 签名后删除历史版本 | 不能 | 可选透明日志/append-only exchange |

### 3.3 非目标

- 不声称防御拥有整台机器root权限且同时控制auditor的攻击者；
- 不把PID、start tick、CUDA event或nonce当成不可伪造硬件证明；
- 不引入硬件attestation作为RTX 4060上的强制依赖；
- 不让证据系统进入solver hot path或performance计时区间。

## 4. 四级保证格

### E0：`SELF_CONSISTENT`

证明：

- artifact tree/manifest完整；
- raw可确定性导出summary；
- source/model/protocol声明与artifact内部一致；
- semantic invariants、claim flags和tamper registry通过。

不证明：

- raw来自真实执行；
- 声明的source/model就是审计委托指定对象；
- fresh subprocess或GPU身份不可伪造。

### E1：`CHALLENGE_BOUND`

在E0基础上，artifact绑定外部审计方在运行前发行的不可预测256-bit nonce、冻结输入和有效期。

它能证明该artifact响应某个先前未出现的challenge，降低预计算/旧artifact重放风险；若executor仍能任意伪造所有
run details，它仍不能证明物理执行发生。

### E2：`INDEPENDENTLY_WITNESSED`

在E0基础上，由独立于executor的主体实际控制或见证执行。分两种合法profile：

1. `E2-DIRECT`：auditor在其控制的shell/process中亲自启动exact source的fresh run并独立重算；auditor的
   launch control本身提供freshness，不强制runner已经实现E1 challenge字段；
2. `E2-DELEGATED`：受外部信任的builder代执行；此时必须先有E1 challenge，并且executor不能访问builder的
   witness签名身份；
3. `E2-THRESHOLD`：两个或更多独立functionary分别签署同一run witness，满足预注册threshold policy。

S3/S4当前正式closure所需最低等级为E2。若外审无法现场运行，只能保留
`PENDING-INDEPENDENT-WITNESS`，不能把E0/E1写成validated authenticity。

### E3：`HARDWARE_ATTESTED`

在E2基础上，受支持平台提供nonce-bound硬件/固件/TEE evidence并由独立service验证。当前RTX 4060环境不可达，
不作为ASPLOS'27必要门槛；未来使用H100+/Blackwell时也必须另外验证用户态source和benchmark语义。

## 5. 两类新artifact schema

它们是证据schema，不是BoundFlow求解器IR。

### 5.1 `BFExternalAuditChallengeV1`

```text
schema_version
challenge_id
challenge_nonce_sha256
issued_at_utc
not_before_utc
expires_at_utc
auditor_identity
auditor_identity_mode
expected_source_revision
expected_protocol_sha256
expected_model_sha256
expected_property_sha256
expected_config_sha256
expected_replayer_blob_sha256
expected_worker_topology
expected_claim_flags
required_assurance_level
challenge_payload_sha256
signature_mode
signature_or_exchange_binding
```

规则：

- nonce由auditor用CSPRNG产生，最少256 bit；artifact只保存hash，外部challenge保存原值；
- challenge必须在exact source/result delivery冻结之后、audit fresh run启动之前发行；
- source/protocol/model等不是executor回填，而由auditor从request选择；
- expiry后runner必须fail closed；
- 同一challenge_id/nonce只能消费一次；失败run不能re-arm为formal；
- `signature_mode=none`时必须写`procedural-exchange-only`，不得使用`cryptographic`措辞。

### 5.2 `BFExternalExecutionWitnessV1`

```text
schema_version
challenge_payload_sha256
challenge_nonce_sha256
witness_identity
witness_identity_mode
witness_role
launch_authority
source_revision
protocol/model/property/config/replayer digests
command_template_sha256
environment_allowlist
gpu_identity_receipt
run_started_utc
run_finished_utc
worker_count_and_exit_summary
final_manifest_sha256
semantic_root_sha256
independent_recompute_sha256
replay_result
tamper_result_and_scope
freshness_result
assurance_level_achieved
limitations
witness_payload_sha256
signature_mode
signature_or_exchange_binding
```

`tamper_result_and_scope`必须区分：

```text
INTEGRITY_REJECTED
SEMANTIC_INCONSISTENCY_REJECTED
EXTERNAL_PARAMETER_MISMATCH_REJECTED
COHERENT_FULL_RESIGN_ACCEPTED_AT_E0
OFFLINE_UNATTESTABLE
```

禁止只写`fully resigned rejected`。

## 6. 签名与身份策略

### 6.1 当前最低可执行模式：procedural exchange

当前外审模型未必持有独立私钥，仓库也没有cosign/Rekor CLI。S4新协议的最低模式冻结为：

1. challenge作为DocOps audit request的外部输入；
2. auditor通过工具亲自启动fresh run；
3. auditor用stdlib-only脚本独立重算；
4. audit report记录challenge hash、manifest、semantic root与命令证据；
5. exchange状态机与Git commit记录顺序。

该模式可形成E2 procedural witness，但必须披露同机、同账号、同文件系统限制；不能称cryptographic attestation。

S3是迁移前的legacy artifact，runner没有challenge字段。若auditor亲自控制全新目录和18-worker launch并独立重算，
可记录为`E2-DIRECT-LEGACY`；它没有E1 challenge-binding，不得倒填或伪造challenge字段。S4起不再使用该例外。

### 6.2 可选签名模式：SSHSIG/DSSE

若外审主体有独立Ed25519密钥：

- 公钥fingerprint必须在challenge发行前由用户/审计策略固定；
- witness payload可用OpenSSH `ssh-keygen -Y sign`的独立namespace签名；
- 或装入DSSE envelope，payload type固定为BoundFlow witness schema；
- verifier必须从artifact外部读取allowed signer，不信任artifact内携带的新公钥；
- executor自己的GitHub/SSH密钥签名只证明executor发言，不能升级为independent witness。

### 6.3 可选公开模式：Sigstore/Rekor

论文artifact release阶段可把已验证的DSSE/in-toto statement写入透明日志并保存inclusion proof。该步骤证明发布顺序和
不可静默改写，不替代auditor-controlled execution。当前不安装工具、不触发外部写入，除非用户另行授权。

## 7. 状态机与所有权

### 7.1 合法状态

```text
DRAFT
  -> DELIVERY_FROZEN
  -> CHALLENGE_ISSUED
  -> CHALLENGE_CONSUMED
  -> RUN_STARTED
  -> RUN_SEALED
  -> SELF_CHECK_PASSED
  -> INDEPENDENT_RECOMPUTE_PASSED
  -> WITNESS_RECORDED
  -> ANCHORED_REPLAY_PASSED
  -> AUDIT_VERDICT_SUBMITTED
```

`AUDIT_VERDICT_SUBMITTED`之后才能由executor响应finding并关闭exchange；不能让runner自行写`APPROVED`。

### 7.2 必须拒绝

- challenge早于delivery/source freeze；
- challenge缺nonce、过期、重复消费或expected digest不全；
- executor生成challenge并同时自称independent auditor；
- run在challenge发行前已经完成；
- witness的launch authority与实际工具所有者不一致；
- 只保存artifact内challenge副本而没有external binding；
- E0结果宣称E2/E3；
- 签名公钥来自同一个未锚定artifact；
- 删除failed attempt或用resume补齐worker；
- witness在summary/manifest之前生成却声称绑定最终结果；
- exchange未approved就升级authoritative claim。

### 7.3 失败状态

```text
CHALLENGE_REJECTED
RUN_FAILED_PRESERVED
SELF_CHECK_FAILED
INDEPENDENT_RECOMPUTE_FAILED
WITNESS_IDENTITY_UNTRUSTED
OFFLINE_UNATTESTABLE
AUDIT_REJECTED
```

任何失败都不能原地覆盖；新尝试使用新challenge_id、全新目录和新round。

## 8. S3立即回灌

### 8.1 不改formal artifact

S3 r001已经交付，旧request/delivery和formal artifact保持不可变。当前只做解释修正：

- “10/10 fully outer-resigned rejected”降精度为“10/10 outer-manifest-resigned、derived-semantics-
  inconsistent attacks rejected”；
- 两类coherent full resign明确为E0可接受；
- v2结果保持`PENDING-EXTERNAL-AUDIT`；
- 外审必须亲自fresh run才可形成`E2-DIRECT-LEGACY` procedural witness；不得事后倒填E1 challenge。

### 8.2 S3外审最小动作

1. 读取原request/delivery、execution receipt和shadow preaudit；
2. 从formal raw独立重算AC3/AC4；
3. 亲自启动至少一套冻结v2 18-worker协议，或明确说明无法现场运行；
4. 比较fresh run与formal的correctness、counter、memory和order-level gates；
5. 自建一类coherent full resign并确认其应在E0接受、在external anchor/E2语义下不构成真实run；
6. 在audit report中明确S3-SHADOW-F1 severity与所达到assurance level。

若第3项未执行，外审仍可审计artifact内部闭包，但不能用tamper probe单独关闭执行真实性。

## 9. S4-4集成修订

### 9.1 保留

- 33 subprocess、raw-first、sidecar/index；
- 16-node/36-edge seal DAG；
- semantic root与final manifest；
- 96-case layered tamper；
- external anchor在artifact外；
- T19=`OFFLINE_UNATTESTABLE`。

### 9.2 增加

现有seal DAG增加两个**外部流程节点**，不改artifact内部16-node hash：

```text
audit_challenge
execution_witness
```

跨域依赖为：

```text
delivery/source/protocol/model -> audit_challenge
audit_challenge -> audit fresh run
fresh final_manifest + semantic_root + recompute -> execution_witness
external_anchor + execution_witness -> anchored_replay_record
anchored_replay_record -> audit verdict
```

S4的`external_anchor`从“executor提交后Git里的一份digest表”收紧为“auditor预先冻结challenge + 运行后witness”组合。
旧anchor schema仍可作为expected-input表，但单独最多E1，不能命名为independent authenticity。

### 9.3 不进入性能区间

challenge读取、source/environment snapshot、签名、日志flush、manifest、replay和witness全部位于benchmark计时区间外。
worker内部仍只记录冻结的计时边界；不能为了attestation改变N/D/P路径的同步次数。

## 10. 新tamper与negative registry

在S4原96类之上建议增加证据协议层测试；这些不是第97类solver semantic tamper，而是独立witness suite：

```text
W01 missing_challenge
W02 malformed_nonce
W03 expired_challenge
W04 challenge_reuse
W05 challenge_after_run
W06 source_digest_mismatch
W07 protocol_digest_mismatch
W08 model_property_config_mismatch
W09 replayer_digest_mismatch
W10 worker_topology_mismatch
W11 executor_self_claims_independent
W12 untrusted_signer
W13 artifact_embedded_key_autoselect
W14 signature_namespace_confusion
W15 witness_before_manifest
W16 witness_manifest_mismatch
W17 witness_semantic_root_mismatch
W18 independent_recompute_mismatch
W19 E0_overclaimed_as_E2
W20 E2_overclaimed_as_E3
W21 failed_attempt_overwrite
W22 unsigned_witness_claims_crypto
W23 coherent_full_resign_scope_lie
W24 transparency_inclusion_overclaims_truth
```

期望：W01—W23 fail closed；W24拒绝claim升级但允许透明日志inclusion本身通过。

## 11. 确定性验证矩阵

### 11.1 schema/unit

- canonical challenge/witness round-trip；
- unknown required field拒绝；
- datetime统一UTC、禁止NaN/Infinity；
- 256-bit nonce长度与重复检测；
- challenge payload hash不含signature自身；
- witness payload hash不含signature自身；
- subject digest与manifest/semantic root逐位一致；
- assurance level只能单调升级。

### 11.2 process/integration

- parent在challenge后spawn全新worker；
- 运行前校验source/protocol/model；
- 失败attempt保存且不能resume；
- external challenge与witness不进入artifact manifest信任域；
- auditor fresh目录与executor formal目录完全分离；
- stdlib independent recompute不import BoundFlow validator；
- anchored replay必须显式传外部anchor+witness。

### 11.3 claim

- E0只能写self-consistent；
- E1只能写challenge-bound；
- E2才允许independently-witnessed；
- E3缺硬件token时稳定拒绝；
- 任一level都不自动升级same-solver、query、10x或ASPLOS-ready。

## 12. 实施顺序

S3外审批准后、S4-4代码开放时，按以下提交切分：

1. `docs: freeze evidence assurance and witness protocol`；
2. `feat(artifact): add challenge and witness schemas`；
3. `feat(artifact): add self-check challenge-bound and witnessed modes`；
4. `test(artifact): add W01-W24 witness negative suite`；
5. `feat(audit): add procedural external-run receipt`；
6. `docs: bind S4-4 anchor to challenge and witness`；
7. 可选：`feat(attestation): add sshsig or DSSE witness envelope`；
8. 可选且需授权：`ops(attestation): publish transparency inclusion proof`。

SSHSIG/DSSE/Rekor不与核心schema第一提交捆绑，避免工具链依赖阻塞correctness closure。

## 13. GO / STOP

### GO

- S3外审明确裁决F1并记录assurance level；
- S4 formal artifact先达到E0；
- external challenge在audit run前发行；
- auditor亲自launch并独立recompute，达到E2；
- exchange approve后才升级closure。

### STOP

- 仍用“fully resigned”笼统描述不同攻击层；
- 只把anchor复制进artifact内部；
- executor签自己的witness并宣称独立；
- nonce存在但auditor没有控制launch；
- 使用Git commit/透明日志存在性证明benchmark数值真实；
- 在RTX 4060上声称GPU hardware attestation；
- 为了过tamper gate硬编码headline数字；
- S3外审前实现S4代码或开放timing。

## 14. 当前下一动作

本计划只完成设计冻结。当前仍应把S3 r001、执行回执、影子预审和本计划交给外部模型。外审需对
`S3-SHADOW-F1`作正式severity判定，并说明实际达到E0、E1还是E2。外审前`.docops/s.md`继续保持：

```text
st: s03
stat: active
health: green
blk: none
next: external-audit-asplos27-s3-optimizer-runtime
```
