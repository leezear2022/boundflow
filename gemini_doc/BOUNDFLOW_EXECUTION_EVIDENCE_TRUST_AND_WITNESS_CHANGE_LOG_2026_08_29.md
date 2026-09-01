# BoundFlow 执行证据可信度与外部见证修改记录

status: documentation-complete-code-closed
date: 2026-08-29
stage: s03-s04
performance-claimed: false

## 1. 起因

S3 executor shadow pre-audit证明：攻击者若同步修改raw、重算summary并重签manifest，artifact仍可保持内部自洽。
原S3“fully outer-resigned tamper”措辞没有区分内部一致性与执行真实性。

## 2. 新增

- 新增`BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_AND_WITNESS_PLAN_2026_08_29.md`；
- 新增`BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_SUPPLEMENT_F1_2026_08_29.md`，把F1独立攻击、fresh run与
  assurance-level判定变成外审强制输入；
- 新增`BOUNDFLOW_EXECUTION_EVIDENCE_TRUST_CONSISTENCY_AUDIT_2026_08_29.md`，逐文档核对S3/S4旧术语、
  anchor owner与权威阅读顺序；
- 冻结E0 self-consistent、E1 challenge-bound、E2 independently-witnessed、E3 hardware-attested四级保证；
- 冻结`BFExternalAuditChallengeV1`与`BFExternalExecutionWitnessV1`；
- 冻结challenge→fresh run→independent recompute→witness→audit verdict状态机；
- 冻结W01—W24 evidence negative suite；
- 明确当前RTX 4060不走NVIDIA H100+硬件attestation路径；
- 明确DSSE、in-toto、SLSA与Rekor分别解决封装、subject/predicate、provenance和透明日志问题，但均不单独
  证明benchmark数据真实。

## 3. 纠正

- S3 10/10结果改称“outer-manifest-resigned、derived-semantics-inconsistent attacks rejected”；
- coherent full resign在E0可被接受，不再伪称semantic validator可以证明物理freshness；
- S4已有external anchor单独最多形成E1；只有auditor-controlled fresh run与independent recompute才形成E2；
- executor自签名、Git commit或透明日志inclusion不得冒充独立见证；
- nonce只抗预计算/重放，不抗能伪造完整run的executor。

## 4. 同步文档

- `BOUNDFLOW_ASPLOS27_S3_FORMAL_CLOSURE_2026_08_28.md`：增加证据保证范围修订；
- `BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_HANDOFF_2026_08_28.md`：增加影子finding补充入口；
- `BOUNDFLOW_ASPLOS27_S3_EXTERNAL_AUDIT_EXECUTION_RECEIPT_2026_08_29.md`：限定10/10措辞；
- `BOUNDFLOW_ASPLOS27_S3_OPTIMIZER_RUNTIME_PREREG_2026_08_28.md`与S3 change log：标记历史tamper命名；
- S4-4 blueprint/readiness：把anchor单独充分性降为E1并指向challenge/witness owner；
- `BOUNDFLOW_ASPLOS27_S4_4_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`：增加challenge/witness
  所有权修订；
- `BOUNDFLOW_ASPLOS27_S4_DESIGN_EXTERNAL_AUDIT_HANDOFF_2026_08_28.md`：升级v19、37份必读、104/103项问题；
- S4 change log：记录内部DAG不变与外部流程owner；
- `asplos_claims_map.md`与`README.md`：同步PENDING-EXTERNAL-AUDIT边界。

exchange r001、formal artifact、代码、测试与历史raw均未修改。

## 5. Claim边界

本轮不新增correctness、performance、same-solver、complete-query、10x、memory或ASPLOS-ready claim；不开放S4代码、
formal或timing。唯一下一动作仍为S3外审。
