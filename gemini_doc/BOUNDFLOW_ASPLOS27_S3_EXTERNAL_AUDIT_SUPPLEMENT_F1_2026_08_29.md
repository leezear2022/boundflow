---
status: supplemental-request-ready-for-external-audit
date: 2026-08-29
type: external-audit-supplement
topic: boundflow
slug: asplos27-s3-external-audit-supplement-f1
exchange-task: asplos27-s3-optimizer-runtime-20260828
exchange-round: 1
external-audit-verdict: pending
execution-authority: false
code-change-open: false
performance-claimed: false
---

# BoundFlow ASPLOS'27 S3 外审增补请求：F1执行真实性边界

## 0. 使用方式

本文件是既有DocOps exchange r001的补充输入，不改写`request.md`、`delivery.md`、formal artifact或历史raw。外审者
必须把原request作为AC1—AC7主合同，同时把本文件用于裁决执行方主动披露的`S3-SHADOW-F1`。

## 1. 必读材料

1. `.docops/exchange/asplos27-s3-optimizer-runtime-20260828/request.md`；
2. `.docops/exchange/asplos27-s3-optimizer-runtime-20260828/r001/delivery.md`；
3. `BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_HANDOFF_2026_08_28.md`；
4. `BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_EXECUTION_RECEIPT_2026_08_29.md`；
5. `BOUNDFLOW_ASPLOS27_S3_EXECUTOR_SHADOW_PREAUDIT_2026_08_29.md`；
6. `BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_AND_WITNESS_PLAN_2026_08_29.md`。

不得把4—6视为外部证据；它们只是executor披露、反例和审计问题清单。

## 2. 新finding

### `S3-SHADOW-F1`

现有10类probe均会在重签外层manifest后保留某项raw/summary/receipt不一致，因此semantic replay拒绝。executor另造：

- memory coherent resign：修改全部raw memory headline、重算summary与manifest；
- latency coherent resign：修改raw latency及对应median、重算summary与manifest。

两者通过self-check。请外审独立重建至少一种，不得只采信executor描述。

## 3. 外审必须区分的结论

```text
E0 SELF_CONSISTENT
  raw/summary/manifest内部闭包成立

E1 CHALLENGE_BOUND
  对审计方先前发行的输入/nonce作出响应

E2 INDEPENDENTLY_WITNESSED
  外审者控制fresh launch并独立重算，或challenge-bound独立builder见证

E3 HARDWARE_ATTESTED
  受支持硬件/固件attestation；当前RTX 4060不可用
```

请明确本轮证据实际达到哪一级，不得把E0 probe写成E2/E3。S3 runner尚无challenge字段；外审者亲自控制launch
可形成`E2-DIRECT-LEGACY`，但不得倒填或声称同时达到E1。

## 4. 强制现场动作

### 4.1 独立raw重算

用只import Python标准库的独立脚本完成原AC3/AC4。脚本不得调用artifact validator的derive/replay helper。

### 4.2 coherent full resign攻击

在临时副本中：

1. 修改至少一个具有headline影响的raw字段；
2. 同步重算该raw行的所有派生字段；
3. 从修改后的完整raw重新生成summary；
4. 重写file inventory和manifest hash；
5. 运行self-check；
6. 记录它被接受还是拒绝，以及正确的保证层解释。

如果被接受，不得直接判定正式raw是伪造的；应判定self-check不能证明physical execution authenticity。

### 4.3 auditor-controlled fresh run

外审者应亲自在其工具控制的shell中从空临时目录启动冻结v2协议，不能让executor预先提供新artifact。要求：

- source/protocol/model/capture digest先核对；
- 18个fresh subprocess，不resume、不筛选；
- 原执行顺序、3 replicate/order、sample/warmup与15秒恢复间隔不变；
- 保存所有worker exit/log/raw；
- 独立重算correctness、receipt、memory、P/N与P/D gates；
- 披露fresh结果与formal结果的差异，不要求latency逐位相同。

若资源或时间导致无法完成，必须写`E2_NOT_ESTABLISHED`并说明是否仍可依据其他独立证据approve；不得用原formal
replay替代fresh run。

## 5. 必答问题

1. `S3-SHADOW-F1`是blocker、major、minor还是info？
2. “fully outer-resigned”是否必须在closure/claims中降精度？
3. coherent full resign被E0接受是否符合离线replayer正确威胁模型？
4. auditor-controlled fresh run是否成功，达到了`E2-DIRECT-LEGACY`吗？
5. formal v2的`3.243894x`是否仍可关闭为固定P-anchor local optimizer result？
6. v1 `0.759540x` worst NO-GO是否完整保留？
7. 是否只开放S4 implementation/correctness，继续关闭formal/timing/performance？
8. S4-4是否必须实施challenge+witness后才能形成formal external closure？

## 6. Verdict规则

### 可approve的最低条件

- 原AC1—AC7除措辞精度外成立；
- coherent full resign的接受被正确限定为E0；
- 外审完成auditor-controlled fresh run和独立重算，形成`E2-DIRECT-LEGACY`；
- authoritative claims保持PENDING直到exchange批准；
- S4只开放implementation/correctness。

### approve-with-minor/major correction

若数值和fresh run成立，但必须修正“fully resigned”措辞，可要求executor在respond/close阶段统一更正权威文档；不能
事后改写原artifact或r001 request。

### reject/block

- fresh run不能复现语义/receipt/memory基本门禁；
- source/protocol/model identity不成立；
- formal raw存在未披露筛选/resume；
- executor用自签anchor冒充独立witness；
- claim已越界到same-solver/query/cross-model/10x。

## 7. 输出要求

正式audit report必须包含：

- verdict和blocker/major/minor/info计数；
- AC1—AC7逐项PASS/FAIL；
- `S3-SHADOW-F1`稳定finding与severity；
- stdlib重算脚本/输出；
- coherent full resign攻击步骤与结果；
- fresh run source、worker数、摘要、replay和限制；
- achieved assurance level；
- 对S4下一门禁的明确判定。

提交后按DocOps exchange流程生成正式`audit.md/audit.json`，不要由executor代写approve。
