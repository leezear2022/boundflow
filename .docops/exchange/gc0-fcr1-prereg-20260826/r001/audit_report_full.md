# GC-0/FCR-1 Verification Graph ABI Correctness 预注册独立审计报告(Round 1)

- task: gc0-fcr1-prereg-20260826
- auditor: external-model-auditor(独立外部审计,非执行方)
- base commit: `9c5f3867c657078cb6ba980a613b686c5a08f2d2`
- result commit(under audit): `68dd54c332f6d9a46640a70cdb83bbd4fb81c3f8`
- 审计时 HEAD: `43e6650`(其后仅两个 DocOps 状态提交)
- 审计性质: 纯文档预注册审计;不假设、不验证任何实现/正确性/计时/加速/query/queue/ASPLOS 就绪
- date: 2026-08-26

## 总体 verdict: **approve**

七项验收标准(AC1—AC7)全部 PASS,无 blocker、无 major finding;3 条 minor、1 条 info,
均为措辞精度问题,不构成跳级或 claim 漂移漏洞,不阻塞"只开放 GC0-0"。
**同意只开放 GC0-0(generic schema + negative legality tests)**,开放范围与
MR7 研究计划、MR7-R closure、四份权威文档的表述一致。

claim boundary 无漂移:`implementation_open=false`、`timing_open=false`、
`performance_claimed=false`、`ASPLOS-ready=false` 在主文档与四份权威文档中一致成立。

---

## AC1 身份与范围 — PASS

证据:

- `git diff --name-only 9c5f386..68dd54c` 仅 9 个文件,全部为 `.docops/` 状态与
  `gemini_doc/` 文档;`git diff 9c5f386..68dd54c -- boundflow/ tests/ scripts/` 为空,
  确认无任何实现/raw/timing 代码。
- diff 统计:908 insertions / 5 deletions;主体为新增 794 行计划文档与 44 行 changelog;
  MR7 研究计划仅改 5 行状态块;README/claims map/memo/current_status 各加一段状态块。
- 68dd54c 之后两个提交 `b334ce2`(deliver GC0 prereg audit)与
  `43e6650`(mark GC0 audit wait blocked)经 `git show --stat` 确认只触及
  `.docops/exchange/**`、`.docops/ev.jsonl`、`.docops/s.md`,纯 DocOps 流程状态。
- 四份权威文档状态一致:claims map(顶部 GC-0/FCR-1 claim 边界块)、memo(顶部块)、
  current_status_after_pr13(顶部块)、README(顶部"最新执行"块)均写明
  `PREREGISTERED-NOT-IMPLEMENTED-NOT-RUN-GC0-FCR1`、待外审、三个 false 标志。
- 主文档 §1.2 的"现有代码事实"与代码合同核对一致:
  `boundflow/ir/differentiable_lower_region.py:185` 确有 `provider_start_node != "/49"` 硬编码检查;
  `boundflow/ir/r3_bounded_arena.py:190-191,232-233` 确有 `domain_count=6`/`spec_count=1` 固定;
  `boundflow/ir/task_v1.py:188-221` 证实 `TaskIRUnit` 已有 typed IO、external dependency、
  memory effect、backend binding;`boundflow/ir/plan.py:654,1144` 证实
  `PlanTemplate/PlanInstance`;`boundflow/runtime/task_backend_dispatch.py:38` 证实
  `BackendDispatchKey`。预注册对旧类型局限的描述(`/49`、6-ReLU、lower-only)属实,无粉饰。

## AC2 阶段门禁 — PASS

证据:

- 主文档 §1.3(L77-79):GC-0 只做 graph/legality/lowering/arena ABI 与 replay;GC-1 才做
  guarded rewrite/closed region/custom VJP correctness;GC-2 才做真实 physical
  arena/prepared runtime 与结构计数。与 MR7 研究计划 §12.3(L745-748)四切片定义及
  §13 门禁表(L757-759)逐层对应,无冲突。
- 反跳级措辞闭环:§0(L24)"只有独立外审批准本文后,GC0-0 实现才可开始";
  §9.1.3(L527)"阶段间必须先关闭和外审";§11(L648)"每阶段关闭和外审后才允许写下一阶段";
  §11(L662)"不能合并为一个事后优化提交";§13.1 GO 链每级只开放下一级的**预注册**而非实现
  (L726"只开放 GC-1 correctness 预注册,不开放 GC-1 实现、GC-2 或 timing")。
- §12(L667-668)三层 acceptance 不可跳级:GC-0 由文档内 AC1—AC3 关闭,GC-2/FCR-1 最终关闭
  才要求文档内 AC1—AC7 全部满足。未发现可直接进入 GC-1 的措辞漏洞。
- MR7 研究计划本轮 5 行改动仅为状态块更新(见 diff),明确"批准后只开放 GC0-0 通用 schema
  与 negative legality tests",与本次审计开放问题一致。
- minor finding F-1、F-3 涉及 §9.4/§4.1 的阶段措辞精度,见 findings,均不构成跳级漏洞。

## AC3 schema 通用性 — PASS(重点项)

证据:

- §4.1—4.6 的 `VerificationProgram/Region/Value/Op/EffectToken/VJPContract` 冻结字段
  (L143-285)全部为语义/拓扑/数值策略字段,无模型名、节点名、站点序号、shape 常数、
  domain/spec 计数;§4.1(L166)明确"program ID 来源于 semantic content,不来源于 model
  filename";§4.1(L140-141)禁止稳定 ID 含 object id/绝对路径/进程号/随机名。
- 硬编码禁令显式成文:§2(L93)"不允许在 graph schema 中出现 ResNet2B、`/49`、`25/Conv_8`、
  `31/Gemm_14`、C0/C1/C2";§8.2(L504)"P/S/C0/C1/C2 只属于测试 instance;schema、rule、
  lowering 和 runtime 代码不得引用这些名字";§13.2 NO-GO(L751)"schema 或执行路径仍硬编码
  ResNet/site"即停止。全文 grep 确认这些名字只出现在禁止条款与测试 instance 语境。
- 三类实例可表达性:P empty-β 由 §4.3(L218)"empty β 必须是 first-class `shape[...,0]` 或显式
  absent value,不能伪造 dense zero tensor"表达;S active-β 由 §4.3 的
  `representation=sparse-location`、`axis_roles=beta-slot`、β location/sign/split/history 角色
  (L210-216)表达;multi-site 10/9 由 §4.5 effect token(optimizer-state/split-history/
  commit-state,L254-265)与 ordinal 字段(L267)表达。均为通用 schema 机制而非特判。
- §4.4 op vocabulary(L236-248)为 α-CROWN 类验证器的通用 op 类(relaxation、propagation、
  concretization 等),且 L250"未列入 vocabulary 的 op 必须在 admission 前拒绝"保证封闭性。
- §5.1 首批 rule ID(L309-315)按语义功能命名,不含 site;`V-H1` 的处理(L317-318)明确
  GC-0 formal 不授予其 rewrite/performance claim,边界清楚。

## AC4 合法性分析 — PASS

证据:

- analysis-only 定义明确:§5.2(L322)"不编译、不执行、不计时";输出 schema(L323-337)含
  五类 witness(external_use/effect_order/alias/dense_escape/vjp)与 `analysis_hash`。
- 可证伪性:L339"`admitted=true` 时 rejection 必须为空且所有 witness 完整;`false` 时至少一个
  稳定拒绝原因"——双向判据,可独立检验。
- 22 类稳定拒绝原因逐类列出(§5.3,L343-366),覆盖 op vocabulary、shape/dtype/layout、
  external use、postdominator、state version、effect order、α/β/polarity/endpoint、residual
  bias token、alias/lifetime、dense-A escape、VJP owner、higher-order、queue/termination
  crossing、runtime fallback、receipt identity。
- witness 定义充分性:closed-world 判据在 §4.2(L193-194)给出三个必要条件;postdominator
  witness 是 region schema 字段(L188);effect 排序规则在 §4.5(L267-269)"write 必须单 writer
  且拓扑排序";queue/termination 不得隐式忽略(L268-269)。
- fail-closed 边界:§4.2(L194)`fallback_policy="reject-before-launch"`,"运行中不得回退";
  §7.2(L457)"任一错误在 commit 前 fail closed,provider state 完整 rollback";
  §13.2(L749-758)列出 8 类立即停止条件。无静默降级措辞。

## AC5 lowering/arena/runtime — PASS

证据:

- §6.1(L370-373)lowering request 绑定全链 hash 且"不得携带 runtime tensor payload"、输入必须
  是 admitted region;§6.2/6.3 Relax/TIR ABI 的参数顺序、返回结构、PrimFunc 声明项均冻结
  (L377-410),且明确"实际 device tensor 不允许通过 Python list/dict 动态返回"。
- §6.4 `LoweringReceiptV1`(L414-422)绑定 program→module 九级 hash 加 target/toolchain
  identity,并强制 `timing_recorded=false, performance_claimed=false` 写入 receipt 本身。
- replay-by-relowering 精确:L425"replay 侧必须重新构造 graph、重新做 legality、重新
  lower/compile 并逐层比 hash;不能只校验 receipt 格式";§10.2(L615)"只比较 JSON、只重算
  outer digest 或只检查 module receipt 格式均不合格"。
- symbolic vs physical 边界无混淆:§7.3(L461-462)"GC-0 只冻结这些字段并验证 symbolic
  plan/identity,不得声称真实 runtime 已满足";§11 GC0-2(L652-653)"symbolic arena/lease/epoch
  identity、semantic replay;不执行 production region";真实 arena/prepared runtime 显式属于
  GC2-1(L659)。§7 标题虽含 "Physical arena",但 §1.3/§7.3/§11 三处把实现阶段锁死在 GC-2。
- minimal saved state:§4.6(L284-285)白名单式列举允许保存项,并禁止跨层 dense A 与 PyTorch
  autograd history;§4.6(L281-282)`dense-A escape policy=forbid`、`mutation policy=none-inside-vjp`。

## AC6 协议 — PASS

证据:

- five-fresh dual-oracle:§9.1.6-7(L530-532)五组 fresh process、固定 `PCM/CPM/PCM/CPM/PCM`
  顺序、M 不计时;§9.2(L537-544)两层独立 oracle(production 冻结 exact call + float64 无
  autograd 手写 closed expression),且 L543-544"candidate 不得调用 production oracle 或
  closed-form helper;replay 不得信任 raw 中已有 reference tensor"。
- 冻结容差:§9.3(L548-560)`atol=2e-4/rtol=1e-5/sign exact/NaN-Inf 即败`,离散量逐项 exact,
  10 lower/9 mutation 全轨迹逐步比较,evaluation 5 注入失败后全状态 exact rollback;
  L561"不得因 observed diff 较小而事后收紧/放宽容差"。
- rollback:§7.2(L457)、§8.2.3(L502)、§9.3(L559)三处定义一致。
- 结构计数器:§7.3(L463-473)与 §9.4(L563-575)列出 submission 10/9、crossing/allocation/
  PyTorch-op=0、dense-A=0/0、identity 不漂移,且 L473 要求 internal kernel 数单独披露,
  防止"一次 submission 伪写为一个 kernel"。
- artifact manifest:§10.1(L581-600)逐文件列出 13 项,manifest 绑定全部 payload SHA-256、
  source/generator/三外部仓库 commit 与运行环境,禁止本机路径。
- 22 类 fully re-signed tamper 逐类列出(§10.3,L619-642),每类"修改内层 payload 后重算所有
  外层 digest",要求 `22/22 rejected`,且 L644 禁止依赖未重签 digest 轻易拒绝;第 22 类直接
  把 `timing_recorded/performance_claimed` 翻转为真纳入 tamper,claim 边界本身纳入防篡改。

## AC7 claim 边界 — PASS

证据:

- 主文档 header(L7-9):`timing-open: false`、`performance-claimed: false`、
  `implementation-open: false(须先完成独立外审)`;§0.6(L21)把三个 false 列入冻结项;
  §12 文档内 AC7(L714)补 `ASPLOS-ready=false`。
- 四份权威文档(claims map/memo/current_status/README 的本轮新增块)均复述三个 false 与
  "待独立外审",claims map 另明确"不得 claim graph compiler 已实现、旧 bridge parity、
  query/queue 收益或 ASPLOS-ready"。五处表述一致,无漂移。
- 与历史 NO-GO 无冲突:§1.1(L39)明确 `1.91213674x` 是要求而非已得速度,与 MR7-R closure
  L84-85 口径一致;§2(L86-95)禁止 CUDA Graph/MetaSchedule/query/queue 外推与借用
  CIBC 12.7951x 等局部数字;claims map 中 B4-C2 hard NO-GO(L2127-2129)与 IR-5 final
  performance NO-GO(L581-585)均未被本轮触碰或升级。
- MR7-R 数值引用全部与 closure 原文一致:`20.333052%`、`24.683788 ms`、`5/5`、
  `1.91213674x`、57 launch/约 540 crossing(closure L78-85)。

---

## Findings

severity 取值:blocker / major / minor / info。本次无 blocker、无 major。

### F-1 minor — §9.4 结构门禁清单未标注阶段归属

- path: `gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:563-575`
- evidence: §9.4 以"五组 fresh 均必须满足"开头,列出含 arena pointer/slice offset/lease/epoch
  保持、warm per-op crossing/dynamic allocation/PyTorch tensor op=0 在内的结构计数;但按
  §1.3(L77-79)与 §11(L656-660),five-fresh 双 oracle 属 GC1-2,真实 arena/结构计数属
  GC2-1/GC2-2。§9.4 单一清单把两个阶段的门禁合并陈述,未逐项标注适用阶段。
- 影响评估: 不构成跳级漏洞——§7.3(L475)"不能达到上述结构目标,GC-2/FCR-1 correctness 不得
  关闭"、§12(L667-668)分层 acceptance 与 §11 的 DAG 从外部封死了跳级;仅存在误读空间
  (GC1-2 被要求物理 runtime 计数,或反向以为结构计数只查一次)。
- advice: 将 §9.4 拆为"GC-1 correctness gates"与"GC-2 structural gates"两小节,或逐项标注阶段;
  可在 GC0-0 实现预注册时一并修正,不阻塞本轮批准。

### F-2 minor — GC0-0 的"22 类拒绝单测"与 GC0-1 的 analysis pass 存在顺序张力

- path: `gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:650-651`
  (对照 §5.3 L343-366)
- evidence: §11 第 1 刀 `GC0-0 schema` 含"22 类拒绝单测",但 22 类中多数拒绝原因
  (如 `REGION_NOT_POSTDOMINATED`、`EFFECT_ORDER_CONFLICT`、`UNSAFE_ALIAS_OR_LIFETIME`、
  `DENSE_A_ESCAPE`)需要第 2 刀 GC0-1 才构建的 analysis-only legality pass 才能真实触发。
- 影响评估: 不阻塞开放 GC0-0——schema 类型与 admission 级拒绝(如 `UNSUPPORTED_OP_KIND`)
  可在 GC0-0 内完成;但"22 类拒绝单测全部落在 GC0-0"字面上不可达。
- advice: 在 GC0-0 实现预注册中明确:GC0-0 交付 22 类原因的枚举/schema 与可独立触发子集的
  负向测试,依赖 analysis 的拒绝测试随 GC0-1 落地;或把 GC0-0/GC0-1 的测试归属显式重述。

### F-3 minor — §4.1"GC-0 formal 只执行一个 closed lower region"措辞与"不执行"条款张力

- path: `gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:168`
  (对照 §11 L652-653、§12 L670-675)
- evidence: §4.1 写"GC-0 formal 只执行一个 closed lower region",而 §11 GC0-2 明确"不执行
  production region";同时 §12 文档内 AC1 要求 GC-0 关闭时"P empty-β、S active-β、
  multi-site 10/9 三 signature 全覆盖",未界定"覆盖"是 schema 可表达性还是 legality admission。
- 影响评估: 语义可从上下文推断(GC-0 阶段只 admit/lower 单一 closed lower region formal
  instance,不运行),但"执行"一词与"不执行 production region"字面冲突;"全覆盖"判据未定义。
- advice: 将"执行"改为"处理/admit/lower",并在 §12 文档内 AC1 注明 GC-0 关闭时三 signature 的
  覆盖判据(建议:schema 可表达 + 至少 P signature 全链 admit/lower/replay)。

### F-4 info — 文档内 AC1—AC7 与审计 request 的 AC1—AC7 编号同名不同义

- path: `gemini_doc/BOUNDFLOW_GC0_FCR1_VERIFICATION_GRAPH_ABI_CORRECTNESS_PLAN_2026_08_26.md:665-714`
  vs `.docops/exchange/gc0-fcr1-prereg-20260826/request.md:20-26`
- evidence: 两套 AC 编号内容完全不同(文档内 AC1=预注册与范围;request AC1=身份与范围)。
- advice: 后续 exchange 引用时建议加限定词(如"plan-AC3"vs"audit-AC3"),避免交叉引用混淆。
  本报告已全部按 request 口径标注。

---

## 不可现场复核项(声明)

1. 22 类拒绝原因与 22 类 tamper 的**可实现性**(目前仅有枚举定义,无实现可验);
2. 冻结容差 `atol=2e-4/rtol=1e-5` 与未来真实数值行为的适配性(实现前不可验);
3. schema 实现后是否仍无隐藏硬编码(本轮只能确认文档层面禁令完备并有 NO-GO 兜底);
4. `.docops/ev.jsonl` 中 delivery 声称的 `gc0 prereg deterministic content/path/scope checks`
   的具体脚本内容(ev 流有 `ch`/`va` 记录与 2 条 `va` pass,但检查脚本本体不在 diff 内)。

## 结论

预注册文档在身份、阶段门禁、schema 通用性、合法性分析、lowering/arena ABI、formal 协议与
claim 边界七个维度均完整、自洽、可证伪;无 blocker/major。**approve,同意只开放 GC0-0
(通用 schema + negative legality tests)**。建议执行方在 GC0-0 实现预注册时吸收 F-1/F-2/F-3
三条措辞修正。
