# GC0-0 Generic Verification Graph Schema 独立审计报告(Round 1)

- task: gc0-0-schema-20260826
- auditor: external-model-auditor(独立外部审计,非执行方)
- base commit: `ad23d86ddd2d8dc95b4ad4dd74d6a02710a34bce`
- result commit(under audit): `07f02fe`(feat(ir): add GC0 verification graph schema)
- 审计时 HEAD: `2a20205`(docs(docops): deliver GC0-0 schema audit,纯 DocOps 交接提交)
- 分支: `feat/rvir-v4-production-state-ownership-v1`
- date: 2026-08-26
- 审计性质: 只判断 GC0-0 schema 关闭边界;不假设 capture/analysis/lowering/runtime/timing/performance/ASPLOS 就绪

## 总体 verdict: **approve**

Audit-AC1—AC7 全部 PASS,无 blocker、无 major;1 条 minor(措辞/标注精度)、1 条 info(测试覆盖完备性)。
所有可机械核对的事实均从源码与测试独立重算(枚举脚本 + 自建 tamper 变体 + 独立回归),
未采信 changelog 摘要。**同意关闭 GC0-0 schema;唯一后继是 GC0-1 预注册(不是实现)。**

claim boundary 无漂移:实现、文档、四份权威文档(memo/claims map/current_status/README)与
MR7 研究计划一致声明 `timing_open=false`、`performance_claimed=false`、GC0-1 仍关闭、
批准后只开放 GC0-1 预注册。无性能数字(changelog 的 685.82s 是全量测试墙钟时间,非性能 claim)。

---

## Audit-AC1 身份、父关闭与范围 — PASS

证据:

- `git log` 确认 `07f02fe` 的父提交为 `ad23d86`(docs(docops): close GC0 prereg audit);
  `ad23d86` 即上轮 GC-0/FCR-1 预注册外审(approve)的关闭提交,父关闭链成立。
  `07f02fe` 之后仅 `2a20205` 一个 DocOps 交接提交。
- `git diff --stat ad23d86..07f02fe`:12 个文件,+3014/-39。代码侧仅新增
  `boundflow/ir/verification_graph.py`(+1856)与 `tests/test_gc0_verification_graph_schema.py`(+906);
  其余为 `.docops/`(ev.jsonl、s.md)与 `gemini_doc/`(GC0-0 changelog、预注册主文档 4 条 finding 修正、
  GC-0 changelog、MR7 计划、claims map、memo、current_status、README)。
- `git diff ad23d86..07f02fe -- boundflow/ scripts/` 除新增 verification_graph.py 外为空;
  `git diff` 对其余 tests/ 为空。无 capture/analysis/lowering/runtime/timing/production 改动。
- `grep -rn verification_graph boundflow/ scripts/`(排除测试)无任何引用:该模块是叶子,
  未被任何 production 路径 import。
- 工作区脏文件仅 `.docops/ev.jsonl`、exchange `state.json`(流程自动)与未跟踪的
  `docs/CIBC_for_DAC.pdf`(用户所有,未触碰),与任务声明一致。

## Audit-AC2 类型化 schema 与 22 类拒绝分区 — PASS(独立枚举)

审计方独立脚本(/tmp/gc00_audit.py,未入仓库)结果:

- `VerificationRejectionReason` 枚举恰 22 项,且名称与顺序同预注册主文档 §5.3(L347-368)
  逐项一致(审计方从文档独立转写比对,非从 changelog)。
- `GC0_DIRECT_REJECTION_REASONS` 恰 15 项;`GC01_ANALYSIS_REJECTION_REASONS` 恰 7 项
  (由"全集 − direct"派生,L211-215);两集合 disjoint 且 union 恰为 22。PASS。
- 源码 AST 枚举 `_reject(...)` 触发点:19 类 reason 有直接字面触发点;
  `REGION_NOT_POSTDOMINATED` 经 `verification_graph.py:912-916` 的条件表达式触发
  (朴素 AST 扫描漏计,已由 tamper 变体 V7 行为证实);
  真正从不触发的只有 `RESIDUAL_BIAS_TOKEN_UNCLOSED` 与 `QUEUE_OR_TERMINATION_EFFECT_CROSSED`
  (grep 全文仅出现于枚举定义 L183/L188)——两者均属 analysis-only 分区,与"留给 GC0-1"一致。
- fail-closed exception identity:所有拒绝经 `_reject` 抛出
  `VerificationGraphValidationError(ValueError)`,携带 `.reason`(枚举)与 `.detail`,
  消息格式 `"{reason}: {detail}"`(L270-280);自建触发验证类型/结构稳定。PASS。
- 5 个 analysis-only 分区 reason(DENSE_A_ESCAPE、EFFECT_ORDER_CONFLICT、
  REGION_EXTERNAL_USE、REGION_NOT_POSTDOMINATED、UNSAFE_ALIAS_OR_LIFETIME)在 validate()
  中存在**浅层触发点**(policy 字符串、字段一致性、closed-world witness 存在性等构造/恒等检查)。
  对照预注册 §11 GC0-0 范围("无需 analysis pass 即可独立触发的 constructor/identity/
  fallback/polarity/VJP 负例";仅"依赖拓扑、postdominator、effect-order 或 alias analysis 的
  negative graph"留给 GC0-1):这些浅层触发全部是 constructor/identity 级 fail-closed 检查,
  不构成 analysis pass,也不声称 analysis 已执行。**定性:预注册允许的浅层直接形式,非 finding;
  相关措辞精度问题见 F-1(minor)。**

## Audit-AC3 模型/site/shape 通用性 — PASS(重点项)

- 对照预注册 §2(L93)与 §8.2(L504)禁令,审计方独立扫描实现源码:
  `ResNet/ResNet2B//49/25/Conv_8/31/Gemm_14/"C0"/"C1"/"C2"` 零命中;schema 字段
  (Program/Region/Value/Op/Effect/VJP/Rule/Registry/LegalityResult/Module)全部为语义/拓扑/
  数值策略字段,无模型名、site 序号或冻结 shape 常数(fixture 的 6/1/8 等只存在于测试文件)。
- 三个 fixture(empty-β Conv、active-β Linear、multi-Conv 10/9)经审计方独立脚本逐一
  canonical round-trip:`from_canonical_json(canonical_json())` 还原对象相等、stable_hash 一致;
  empty-β 由 `shape=(D,0)`+`present=false`+`sparse-location` 表达(L412-493 的 beta_empty 判定),
  active-β 由 location/sign/history 三元组与 β gradient owner 表达,10/9 由通用
  COARSE_COMMIT attributes(`evaluation_count=10/mutation_count=9`)表达——均为通用 schema 机制,
  源码中无任何按 fixture 特判的分支。
- 无任何 production 执行路径被触及(见 AC1 import 检查与 diff 核对)。

## Audit-AC4 负向门禁与 launch 不可能 — PASS

- 测试文件枚举:16 个 reason 出现在断言中;15 个 GC0 direct reason **每个**至少有一个
  negative 测试映射到稳定 reason(`_assert_reason` 逐条核对);唯一多测的是
  DENSE_A_ESCAPE(analysis 分区,VJP policy 字符串浅层直接形式,测试名为
  `test_gc0_direct_vjp_region_and_identity_rejections_are_fail_closed`,见 F-1)。
- 7 个 analysis-only reason 中 6 个未被测试;唯一被测的 DENSE_A_ESCAPE 是构造级检查,
  无任何文档声称 analysis pass 已执行——无虚假声称。
- `LegalityResultV1`:`admitted=false` 且无 reason → 拒绝(L1136-1141);`admitted=true` 缺
  witness 或带 reason → 拒绝(L1126-1135);审计方自建 admitted-缺-witness 变体验证拒绝。PASS。
- launch 不可能:模块无任何 execute/launch/run/compile/apply/rewrite/lower 入口
  (dir() 扫描 + 全文精读);`VerificationFallbackPolicy` 只有 `REJECT_BEFORE_LAUNCH` 一个成员;
  `execution_enabled/timing_recorded/performance_claimed` 任一为真即拒绝(registry L1063-1067、
  module L1299-1304),审计方自建 `execution_enabled=True` 变体验证 fail closed。

## Audit-AC5 canonical identity 与 tamper — PASS(独立重算)

- `_canonical` = `json.dumps(sort_keys=True, separators=(",",":"), allow_nan=False)`(L284);
  同输入两次 dump 完全一致(三 fixture 各验证);NaN 属性在 freeze(TypeError)与 dump
  (ValueError)两层均 fail closed。
- leaf/program/registry/module 四级 hash 均由审计方用 hashlib 独立重算并与 `stable_hash()`
  一致(三 fixture × 三级 = 9 组重算全过);identity binding:同一构造 → 同一 hash,
  任一字段变化(如 `semantic_version` "1"→"2")→ module hash 变化(变体 V8)。
- strict module round-trip:`from_canonical_json` 要求字节级 canonical 相等 + 全量 revalidate
  (L1451-1464),三 fixture 全过。
- 审计方自建 tamper 变体:
  - V1 非 canonical 空白 → RECEIPT_IDENTITY_MISMATCH;
  - V2 全重签 `performance_claimed=true`(所有 digest 重算)→ RUNTIME_FALLBACK_REQUIRED;
  - V3 全重签删一条 registry 规则(冻结 8 条集合)→ RECEIPT_IDENTITY_MISMATCH;
  - V5 program 内层改动但保留过期 program_hash → RECEIPT_IDENTITY_MISMATCH;
  - V7 全重签删除 closed-world postdominator witness → REGION_NOT_POSTDOMINATED;
  - V6 全重签但 validation 一致的改动(dtype float32→float64 全链一致)→ 被接受为**新 identity**
    (hash 不同)——这是 canonical identity 的正确语义:防篡改 = 身份必变 + 语义规则必重验,
    与 delivery "canonical receipts are identities" 的口径一致。

## Audit-AC6 rule registry 与依赖 — PASS

- `build_gc0_rule_registry_v1()` 恰产出冻结 8 条 rule,rule_id 集合与
  `REQUIRED_VERIFICATION_RULE_IDS_V1` 精确相等(独立枚举);registry validate 强制
  rule 集合恰为该冻结集(L1056-1061)。
- 非可执行:registry/module 无任何执行入口(dir() 扫描零命中);`execution_enabled` 默认 False
  且置真即拒绝;无隐藏 builder 执行路径(`replacement_builder_id` 只是字符串标识)。
- `verification_graph.py` import 仅 `__future__/dataclasses/enum/hashlib/json/math/typing`,
  零 backend/runtime/timing/torch/tvm 依赖(AST 枚举,另测试 L690-713 亦有同向门禁)。
- `perf_counter/cuda.Event/torch.compile/time.time` 在源码中零命中。

## Audit-AC7 回归、静态检查与上轮 findings — PASS

审计方独立重跑(conda env boundflow,先 `source env.sh`):

- targeted:`pytest tests/test_gc0_verification_graph_schema.py -q` → **11 passed**(9 函数含
  参数化 3,与声明一致)。
- related:审计方自选 IR 相关 8 文件(gc0 + bound_ir×2 + box_perturbation + cibc_ibp×2 + env×2)
  → **49 passed**;delivery 的 "related-54" 未记录确切命令,无法逐字复现(见"不可现场复核项"),
  但被全量覆盖。
- 全量:`pytest tests -q -rs` → **1832 passed, 3 skipped, 6 warnings in 684.39s**,
  与 changelog 逐项一致;3 个 skip 原因(-rs 实测):TVM 可用时跳过 allow-no-tvm 重复编译冒烟
  (1)、两个冻结 VNN-COMP checkout 不可用(2),均为既有环境边界,与 changelog 描述一致。
- black `--check` 2 files clean;mypy `--follow-imports=skip` 2 files clean;
  pylint 10.00/10;`git diff --check ad23d86..07f02fe` clean。
- `dol exchange validate gc0-0-schema-20260826` → ok;`dol lint --soft` → ok。
- 上轮 3 minor + 1 info 处置逐条核对(预注册主文档 diff ad23d86..07f02fe):
  - F-1(§9.4 阶段合并):已拆为 9.4.1 GC-0 / 9.4.2 GC-1 / 9.4.3 GC-2 三小节。**已关闭。**
  - F-2(22 类不可全在 GC0-0 触发):§11 第 1 刀改写为"完整 22 类 enum/schema + 无需 analysis
    的直接子集;依赖拓扑/postdominator/effect-order/alias 的 negative graph 留到 GC0-1";
    Plan-AC2 同步改写。**已关闭。**
  - F-3(§4.1/§11"执行"措辞):§4.1 改为"schema construction、admit/lower ABI 与 canonical
    replay,不执行 production region";Plan-AC1 补三 signature 覆盖判据(GC0-0 只要求 schema
    construction + canonical round-trip)。**已关闭。**
  - F-4(AC 同名不同义):文档内验收改名 `Plan-AC1—Plan-AC7`,并成文"外部 exchange 统一写作
    Audit-AC,禁止重名"。**已关闭。**
- 无 claim 漂移、未越权开放 GC0-1 或性能:四份权威文档新增块口径一致("批准后只开放 GC0-1
  预注册,不得直接实现 GC0-1")。

---

## Findings

severity 取值:blocker / major / minor / info。本次无 blocker、无 major。

### F-1 minor — analysis 分区的 DENSE_A_ESCAPE 出现在名为 "direct" 的测试与 changelog 直接门禁清单中

- path: `tests/test_gc0_verification_graph_schema.py:828`(`test_gc0_direct_vjp_region_and_identity_rejections_are_fail_closed` 断言 DENSE_A_ESCAPE);
  `gemini_doc/BOUNDFLOW_GC0_0_VERIFICATION_GRAPH_SCHEMA_CHANGELOG_2026_08_26.md` §5("VJP
  owner/saved-state/higher-order/dense escape"列入直接负向门禁)对照同文 §3(dense escape 列入
  7 类 analysis-only)
- evidence: 审计方枚举确认 `DENSE_A_ESCAPE ∈ GC01_ANALYSIS_REJECTION_REASONS`(不在 15 类
  direct 常量中),但其实际触发是 VJP contract 的 `dense_a_escape_policy != "forbid"` 字符串检查
  (`verification_graph.py:823-827`)与 module 级 saved-value 检查(L1389-1400)——浅层直接形式。
  预注册 §11 GC0-0 范围显式包含 "VJP 负例",故行为本身在批准范围内;但测试命名与 changelog §5
  把一个 analysis 分区 reason 陈述为 direct 覆盖,可能误导 GC0-1 审计误判该项已完成分析级验证。
- 影响评估: 仅为标注精度问题;方向是 fail-closed 保守侧,不构成跳级或 claim 漂移漏洞。
- advice: 在 GC0-1 预注册中注明:analysis 分区 reason 可被 schema 级浅层检查以同名 reason 拒绝
  (如 DENSE_A_ESCAPE 的 policy 字符串形式),GC0-1 仍需交付其完整 analysis 触发链与 negative
  graph;或将 changelog §5 措辞改为 "dense escape policy(浅层形式)"。

### F-2 info — 3 个 analysis 分区 reason 的浅层拒绝分支无专项测试

- path: `boundflow/ir/verification_graph.py:594-598`(EFFECT_ORDER_CONFLICT,op 读写同一 effect
  token)、`verification_graph.py:484-488,908-918`(REGION_EXTERNAL_USE)、
  `verification_graph.py:903-907`(UNSAFE_ALIAS_OR_LIFETIME)
- evidence: 审计方枚举确认这三个 reason 在 validate() 中有浅层触发点,但测试文件从未断言它们
  (测试只覆盖 15 direct + DENSE_A_ESCAPE)。这些分支是 fail-closed 保守拒绝,不测不影响
  安全性,但属于未受回归保护的行为。
- advice: GC0-1 交付 analysis negative graph 时自然覆盖;若希望 GC0-0 内闭合,可加三条构造级
  负例(一条断言一个 reason 即可)。不阻塞本轮批准。

---

## 不可现场复核项(声明)

1. delivery 的 "pytest-related-54" 未记录确切命令/文件集,审计方无法逐字复现 54 这个数字;
   以自选 related 集(49 passed)与全量(1832 passed/3 skipped)替代验证,充分覆盖。
2. `VerificationGraphValidationError` 之外的构造期 TypeError(如 `_freeze_attribute` 对非
   canonical 属性)不属于 22 类 reason 语义,审计按"构造期类型错误不冒充 reason"理解,
   与预注册无冲突,但预注册未显式区分这两层。
3. 预注册 §10.3 的 "22/22 fully re-signed tamper rejected" 属 GC-0 整体 artifact 级门禁
   (GC0-3),本轮只现场验证 schema 级代表性变体(V1/V2/V3/V5/V7),未覆盖 lowering/arena/
   trajectory 类 tamper(其实现尚不存在)。

## 结论

GC0-0 交付与批准的开放边界精确吻合:通用 typed/canonical schema、22 类拒绝枚举的
15-direct/7-analysis 机械分区、三类通用 fixture 的 canonical round-trip、冻结非可执行 8 条
registry、fail-closed identity/fallback/VJP/状态/tamper 门禁,全部经独立重算确认;无 capture/
analysis/lowering/runtime/timing/production 改动,无性能 claim。**approve,同意关闭 GC0-0;
唯一后继是 GC0-1 预注册(不是 GC0-1 实现)。**
