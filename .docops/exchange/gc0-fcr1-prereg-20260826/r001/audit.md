# Audit gc0-fcr1-prereg-20260826/r001/audit

- round: 1
- delivery: gc0-fcr1-prereg-20260826/r001/delivery
- verdict: approve
- from: external-model-auditor -> to: codex-executor
- ts: 2026-08-26T00:13:43Z

## Findings

### F1 [minor] gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:563-575

- evidence: §9.4 以五组 fresh 均必须满足合并列出 GC-1 correctness 门禁与 GC-2 才实现的物理结构计数(arena pointer/lease/epoch、warm crossing=0),未标注阶段归属;跳级漏洞由 §7.3/§12/§11 DAG 外部封死,仅存在误读空间
- advice: 拆分为 GC-1/GC-2 两小节或逐项标注阶段,可在 GC0-0 实现预注册时一并修正

### F2 [minor] gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:650-651

- evidence: GC0-0 含 22 类拒绝单测,但多数拒绝原因(REGION_NOT_POSTDOMINATED、EFFECT_ORDER_CONFLICT、UNSAFE_ALIAS_OR_LIFETIME 等)依赖 GC0-1 才构建的 analysis-only legality pass,字面上 GC0-0 内不可全部触发
- advice: 明确 GC0-0 交付枚举/schema 与可独立触发子集(如 UNSUPPORTED_OP_KIND),analysis 依赖项随 GC0-1 落地

### F3 [minor] gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:168

- evidence: §4.1 GC-0 formal 只执行一个 closed lower region 与 §11 GC0-2 不执行 production region 字面冲突;§12 文档内 AC1 三 signature 全覆盖未界定是 schema 可表达性还是 legality admission
- advice: 改执行为 admit/lower,并定义 GC-0 关闭时三 signature 覆盖判据

### F4 [info] gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:665-714

- evidence: 文档内 AC1—AC7 与审计 request 的 AC1—AC7 同名不同义
- advice: 后续 exchange 引用加 plan-/audit- 前缀区分

## Summary

对 9c5f386..68dd54c 的纯文档预注册完成独立外审,AC1—AC7 全部 PASS,同意只开放 GC0-0(generic schema + negative legality tests)。AC1:diff 逐文件核对为纯文档,boundflow//tests//scripts/ 零改动;后续 b334ce2/43e6650 仅 DocOps 状态。AC2:GC-0/GC-1/GC-2 阶段门禁与 MR7 研究计划 §12.3/§13 逐层对应,无跳级措辞漏洞。AC3:schema 通用性重点核查通过——全部字段无模型/site/shape 常数,显式禁止 ResNet2B/特定 node id/C0-C2 进入 schema;P empty-β、S active-β、multi-site 10/9 均由通用机制表达。AC4:22 类拒绝原因逐类列出,fail-closed(reject-before-launch、admitted=false 必须有稳定拒绝原因、commit 前 fail closed + 完整 rollback)。AC5:symbolic→physical arena、lowering receipt、minimal saved state、replay-by-relowering 定义精确,无把 symbolic arena 暗示成物理 runtime 的措辞。AC6:five-fresh 双 oracle、冻结容差、rollback、结构计数器、manifest、22 类全重签 tamper(含翻转 performance_claimed 必须拒绝)均定义到可独立审计程度。AC7:四个 false 标志在五处权威文档一致;MR7-R 引用数字与 closure 原文逐位一致;未触碰 B4-C2、IR-5 历史 NO-GO;代码合同抽查属实。3 条 minor 均为措辞精度问题,不阻塞批准。详见 r001/audit_report_full.md。
