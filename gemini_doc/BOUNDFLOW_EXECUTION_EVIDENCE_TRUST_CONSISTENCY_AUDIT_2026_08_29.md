---
status: consistency-audit-complete-code-closed
date: 2026-08-29
type: documentation-consistency-audit
topic: boundflow
slug: execution-evidence-trust-consistency-audit
stage: s03-s04
execution-authority: false
code-change-open: false
formal-run-open: false
external-audit-verdict: pending
performance-claimed: false
---

# BoundFlow S3/S4执行证据信任模型一致性审计

## 0. 结论

新冻结的E0/E1/E2/E3模型与S4-4原有`SELF_CONSISTENT / external anchor / OFFLINE_UNATTESTABLE`
方向一致，但全仓仍有三类旧措辞会让外审得到冲突指令：

1. S3把“重签manifest但未同步修复派生summary”的攻击简称为`fully outer-resigned`；
2. S4 readiness把executor提交后由Git/DocOps绑定的digest表单独称为`ANCHORED_AUTHENTIC`，没有定义独立主体；
3. S4设计外审handoff仍只问anchor是否足够，没有强制核对challenge发行、launch authority、witness与assurance level。

本轮通过逐文档supersession note、权威阅读顺序和外审问题修订消除这些冲突。历史数字、raw、artifact、exchange、
96-case registry和16-node/36-edge内部seal DAG均不改。

## 1. 权威概念

### 1.1 当前唯一保证分级

```text
E0 SELF_CONSISTENT
E1 CHALLENGE_BOUND
E2 INDEPENDENTLY_WITNESSED
E3 HARDWARE_ATTESTED
```

- E0证明artifact内部闭包；
- E1证明artifact响应外部预发行challenge，但不证明executor没有伪造run；
- E2要求auditor-controlled fresh launch或独立trusted builder/functionary；
- E3要求受支持硬件attestation，当前RTX 4060不可达。

S3 legacy runner没有challenge字段，外审亲自启动18-worker协议时只能写`E2-DIRECT-LEGACY`，不得倒填E1。

### 1.2 当前唯一tamper命名

禁止无scope使用“fully resigned”。必须选择：

```text
outer-manifest-resigned-derived-inconsistent
coherent-full-resign-self-consistent
external-parameter-mismatch
offline-freshness-unattestable
independently-witnessed-fresh-run
```

### 1.3 当前唯一外部真实性owner

```text
expected-input anchor alone = E1 upper bound
auditor challenge + controlled/delegated run + independent recompute + witness = E2
```

executor自己的Git commit、SSH key、DocOps event或artifact内anchor均不能单独成为independent witness。

## 2. 权威阅读顺序

### 2.1 S3外审

1. 原DocOps exchange `request.md`与`r001/delivery.md`：不可变历史合同；
2. `BOUNDFLOW_ASPLOS27_S3_EXECUTOR_SHADOW_PREAUDIT_2026_08_29.md`：executor反例与open finding；
3. `BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_AND_WITNESS_PLAN_2026_08_29.md`：当前保证模型；
4. `BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_SUPPLEMENT_F1_2026_08_29.md`：外审强制动作；
5. formal closure/execution receipt：数值与历史命令证据，tamper措辞受2—4限定。

### 2.2 S4设计/未来实现

1. `BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_AND_WITNESS_PLAN_2026_08_29.md`：challenge/witness owner；
2. `BOUNDFLOW_ASPLOS27_S4_4_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`：artifact内部施工合同；
3. `BOUNDFLOW_ASPLOS27_S4_DESIGN_EXTERNAL_AUDIT_HANDOFF_2026_08_28.md`：修订后的设计外审入口；
4. S4-4 readiness：source/codec/size诊断继续有效，旧anchor充分性语义被1取代；
5. S4-4 blueprint：历史总门禁，71-case等旧细节由construction package取代。

## 3. 冲突审计

| ID | 旧语义 | 风险 | 当前处置 |
|---|---|---|---|
| TC-01 | S3 prereg要求10类`fully outer-resigned` | 把外层重签误读为全链自洽重签 | 顶部修订为outer-manifest-resigned-derived-inconsistent |
| TC-02 | S3 closure/receipt写10/10 fully | 读者可能推断physical authenticity | 已加shadow finding与scope note |
| TC-03 | S3 immutable exchange仍有旧措辞 | 不能事后改r001 | 以supplement作为正式增补，不改历史 |
| TC-04 | S4 readiness：anchor exact即ANCHORED_AUTHENTIC | executor仍控制anchor内容 | 顶部标注最多E1，E2需witness |
| TC-05 | S4 blueprint：external anchor后anchored replay | 没有challenge/identity时序 | 顶部标注由witness plan补全 |
| TC-06 | S4 construction 16-node DAG | 新节点可能破坏冻结hash | challenge/witness定义为artifact外流程节点，内部hash不改 |
| TC-07 | S4 handoff只问anchor | 外审可能approve不完整trust boundary | 增加E0—E3、challenge/witness和launch authority问题 |
| TC-08 | 95/96 tamper都写fully | coherent全链重生成的拒绝owner不清 | 继续按enforcement layer拒绝；E0接受与E2真实性分开 |
| TC-09 | Git/DocOps被称外部 | 同repo不等于独立主体 | 明确只固定顺序/expected inputs，不自动提供independence |
| TC-10 | nonce被当freshness证明 | executor可读nonce后伪造 | nonce只抗旧artifact重放，E2仍需独立launch/witness |
| TC-11 | 签名被当truth | signer可能就是executor | 签名只认证发言主体，trust policy外置 |
| TC-12 | RTX 4060硬件证明 | 官方路径不适配当前GPU | E3稳定不可达，不作为S3/S4门槛 |

## 4. 逐文档状态

### Active authority

| 文档 | 当前职责 |
|---|---|
| trust and witness plan | E0—E3、schema、状态机、negative owner |
| S3 shadow preaudit | F1/F2与executor fresh证据 |
| S3 audit supplement | r001增补审计合同 |
| S4-4 construction package | 33-worker、raw、seal DAG、96-case artifact施工 |
| S4 design audit handoff | 当前设计审计入口 |
| claims map | 对论文claim的最终限制 |

### Active facts, superseded interpretation

| 文档 | 保留 | 被取代 |
|---|---|---|
| S3 prereg | 18 worker、六order、门槛 | `fully`攻击命名 |
| S3 closure | 数值、receipt、v1 NO-GO | tamper可证明真实性的隐含解释 |
| S3 execution receipt | 命令输出 | 10/10的广义保证 |
| S4-4 readiness | source closure、codec、size、fault拓扑 | anchor单独达到independent authenticity |
| S4-4 blueprint | 总目标和历史演进 | 71-case及无witness anchor流程 |

### Immutable historical input

- S3 exchange r001 request/delivery；
- formal v1/v2 artifacts与failed attempts；
- 所有已提交raw与manifest。

这些不因措辞修订被改写；新模型通过增补材料解释其保证范围。

## 5. 机械一致性规则

当前S3/S4权威文档应满足：

1. 任一出现`fully outer-resigned`的active S3文档都必须包含shadow preaudit或trust plan链接；
2. 任一把external anchor作为active closure条件的S4文档都必须包含challenge/witness supersession note；
3. claims map必须同时包含`PENDING-EXTERNAL-AUDIT`、`E2`和S4 closed边界；
4. design audit handoff必读顺序必须包含trust plan与S3 supplement；
5. design audit输出必须要求achieved assurance level；
6. exchange目录保持零diff、无executor生成audit文件；
7. 代码、测试、artifact保持零diff；
8. 所有新文档`performance-claimed=false`。

## 6. 外审判定口径

外审不能只回答“hash对不对”，还必须回答：

- 谁选择source/model/protocol；
- 谁发行challenge；
- 谁控制fresh launch；
- 谁独立重算；
- 谁签witness，public key从哪里信任；
- 结果达到E0/E1/E2/E3哪一层；
- 若没有E2，哪些claim继续pending；
- coherent full resign在E0被接受是否被正确披露。

## 7. 本轮边界

本轮只收口文档语义：

- 不修改S3 exchange、formal artifact、代码或测试；
- 不实施challenge/witness schema；
- 不安装签名/透明日志工具；
- 不开放S4 implementation、formal或timing；
- 不新增performance、correctness、same-solver、query、10x或ASPLOS-ready claim。

唯一下一动作仍是S3外审。
