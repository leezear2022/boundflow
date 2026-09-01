---
status: external-audit-report
date: 2026-08-30
type: external-audit-report
topic: boundflow
slug: asplos27-s4-0-mutable-state-admission-audit-r1
auditor: external-model (kimi-code CLI session, operator-supervised)
audit-head: 3057c15bb800214a07e12d5b7f9d3cafd89702ac
formal-source: b3afde87ad30a1dc14a31a2fa988c0e76b52dcb6
artifact: artifacts/asplos27-s4-admission/resnet2b-prop0-v1
verdict: approve-with-minor-correction
assurance-level-achieved: E2-DIRECT-LEGACY
performance-claimed: false
---

# BoundFlow ASPLOS'27 S4-0 mutable-state admission 外部审计报告（round 1）

## 0. Verdict

**approve-with-minor-correction**。

- blocker：0
- major：0
- minor：2（`S4-0-AUDIT-F1` 措辞降精度为强制项；`S4-0-AUDIT-F2` 报告归因措辞）
- info：2（`S4-0-AUDIT-F3` 无 challenge 字段的 E2-DIRECT-LEGACY 限定；`S4-0-AUDIT-F4` dol CLI 不在
  外审环境）

approve 的语义严格限定为：同意关闭 S4-0 correctness/admission，开放 S4-1A 的下一阶段预注册与实现
（六组 α 与 active β 转候选侧 persistent compressed buffers）。不表示任何 speedup、same-solver、
timing、complete-query、跨模型、10x 或 ASPLOS-ready。

强制修正项（executor 在 close 阶段执行，不得改写已交付 artifact）：

1. `gemini_doc/asplos_claims_map.md` 与 `gemini_doc/asplos_execution_memo_v1_0.md` 的 2026-08-30 S4-0
   条目中“fully outer-resigned tamper=10/10 rejected”/“10类全重签攻击均闭合”必须降精度为
   “10/10 outer+inner-resigned、derived-semantics-inconsistent 攻击拒绝；coherent full resign（同步修正
   全部派生语义）在 E0 下可接受，不证明物理执行真实性”。正式候选报告 §1/§5 同样补一句 E0 限定。
   依据：外审自建 coherent full resign 攻击被 self-check 接受（§4）。

修正完成后方可把状态升级为 `VALIDATED-S4-0-MUTABLE-STATE-ADMISSION`。

## 1. 审计输入与方法

读取：S4-0 外审交接、正式候选报告、S4-0 施工合同；亲读源码
`scripts/run_asplos27_s4_admission_worker.py`（396 行全读）、
`scripts/replay_asplos27_s4_0_admission_stdlib.py`（436 行全读）、
`boundflow/runtime/asplos27_s4_mutable_state_admission.py`（关键区段：strict owner 提取、lease 状态机、
两次 live capture、read-race 校验）、`scripts/run_asplos27_s4_0_admission_artifact.py`（282 行全读）。

外审独立产物（已归档 `artifacts/asplos27-s4-admission/audit-20260830/`）：

- `audit_s4_recompute.py`：stdlib-only 独立重算脚本（不 import boundflow/torch/replay 工具）；
- `audit_s4_attack.py`：coherent full resign 攻击（攻击构造允许复用 executor 的派生逻辑）。

## 2. AC1—AC7 逐项判定

### AC1 source identity：PASS

- 6/6 protocol code blob 与当前 HEAD（`3057c15`）`git show` 重算逐位一致；formal source=
  `b3afde8`，此后 code path 零漂移，工作树六文件 clean；
- 三个外部仓库 commit 外审在本机独立复核：alpha-beta-CROWN `e5c7e17...`、其 auto_LiRPA 子模块
  `5a098e8f...`、vnncomp2021 `90419aad...` 全部一致（protocol 中的 auto_lirpa_commit 指向 abcrown
  内嵌子模块，不是 boundflow/3rdparty 的 vendored 副本，外审已核实两者路径）；
- model `791aa24d...` 与 property `89edf066...` SHA256 从 vnncomp2021 原始文件重新散列一致；
- manifest 7 文件 digest 全对；manifest/protocol/summary/raw/admission 自哈希链全部由外审独立
  重算通过；全 artifact 扫描无本机绝对路径。

### AC2 real provider：PASS

- 5 行 raw、run ordinal 0—4 逐一核对；每行 `provider_core_intercept_count=1`、
  `provider_core_execute_count=0`、compute/update callback=0/0、candidate kernel/allocation=0/0、
  fallback/retry/mutation=0；
- worker 源码确认真实 provider：在 abcrown `stage_solve.update_bounds_core` exact-call 处截获
  `pre_result`，生产 flag（fix_interm_bounds/enable_decision_precompute/KfsbBranching 等）逐项强制，
  私有 sentinel `_AdmissionCaptured` 在 provider core 执行前终止；`sys.setprofile` 在提取窗口内计数
  provider callback；
- 真实容器/type 与 receipt 投影一致：`collections.defaultdict(default_factory=dict)` alpha、
  `builtins.dict` beta、nested dict、beta list、`auto_LiRPA.beta_crown.SparseBeta` entry、12 个 exact
  `torch.Tensor`（subclass 在 extractor 中显式拒绝）；
- 六 α leaf/requires_grad=false、六 β leaf/requires_grad=true 逐 slot 复核。

### AC3 arithmetic：PASS（外审独立重算，非读 summary）

从 5 行 raw 的 slots/shapes 独立重算：slot/path=6/12、alpha stored/active/preserved=
8496/4248/4248、beta slot=6、active beta=1 slot/6 elements、live tensor=12、elements/pass=8502、
bytes/pass=34008、capture passes=2、D2H=24 copies/68016 B——全部与 summary 一致；每 slot 的
active/preserved/beta 元素数与 shape 乘积逐项复核，12 条 path 互异且与 mutable_path_set_hash 绑定。

### AC4 lease/claim：PASS

5/5 行 lease 均为 OPEN(12 strong refs)→CLOSED(0)、single_transfer=false、buffer_prepare=0；
runtime 源码确认 lease 为进程内强引用、绑定 pid/thread/exact-call/stream、状态机
OPEN→TRANSFERRED→COMMITTING→终态、close 丢弃全部行、copy/pickle 全系禁止、转移失败即
FAILED_CLOSED；receipt/summary/worker 三层 claim flag（timing/performance/exclusivity/dense）全 false。

### AC5 negative：PASS

negative_registry.json：63 个唯一 nodeid、ordinal 连续、每case `exact_detail_and_reason_asserted=true`、
`fresh_pytest_case=true`、minimum 56 满足；63 个 nodeid 均在测试源文件中存在；
`logs/negative-pytest.txt` 记录 `63 passed`；registry SHA256 与 protocol 绑定一致。

### AC6 replay/tamper：PASS（E0 边界见 §4）

- 外审现场 stdlib replay（formal 与 fresh 两份 artifact）：均 PASS；
- 外审现场 tamper probe：10/10 rejected，逐类拒绝原因均为语义级（exact-call hash、count 算术、claim
  边界、slot 顺序、provider 投影/结构、lease、counters、registry、ordinal），非仅 manifest digest；
- 外审自建第 11 类 coherent full resign：见 §4，按预期在 E0 被接受。

### AC7 regression/claim：PASS

- targeted（外审现场）：`82 passed in 2.39s`（78 admission + 4 artifact closure）；
- 全量（外审现场）：`1966 passed, 3 skipped, 6 warnings in 715.20s`，3 skip 为既有 TVM/VNN-COMP
  环境边界；
- black：6 文件 unchanged；mypy（`--explicit-package-bases`）：6 文件 clean；pylint：逐文件 6/6
  10.00/10；`git diff --check`：PASS；
- claim 边界：claims map / memo / README / 报告均保持
  `FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-0`，无 VALIDATED 升级、无 timing/performance/
  same-solver 越界（除 §0 强制措辞修正项）；`dol` CLI 不在外审环境 PATH，dol lint 未独立复跑（F4）。

## 3. Auditor-controlled fresh run（E2-DIRECT-LEGACY）

外审在自己控制的 shell 中从空目录 `/tmp/boundflow-s4-audit-fresh-20260830/artifact` 启动完整生成：
5 个 fresh abcrown subprocess（外部 python `.venv`，torch 2.11.0+cu130）+ 63 节点 negative pytest，
run ordinal 0—4，无 resume/筛选。结果：

- fresh artifact 通过外审独立重算脚本全部检查与 executor stdlib replay；
- fresh 与 formal 交叉对比：slots（含 α/β live content hash）**逐位一致**，topology/
  construction-model/optimizer-policy/mutable-path-set hash 一致；formal 算术（8496/4248/4248 族）全部
  复现；negative 63 passed 复现；
- `production_plan_hash`/`snapshot_hash`/`oracle_mapping_provenance_hash`/
  `plan_binding_projection_hash` 在 formal 5 行互不相同、fresh 5 行互不相同且 fresh≠formal——外审
  溯源确认这些 hash 经 snapshot 全量 tensor metadata/history 内容散列，反映 per-process provider 捕获
  的正常数值变动，而 admitted α/β slot 内容跨全部 10 个进程位级一致（这是真实确定性执行的强证据）；
- 限制：S4-0 协议无 challenge/nonce 字段，本轮为 `E2-DIRECT-LEGACY`，不构成 E1；正式 artifact 的
  raw 字节本身仍为 E0（自洽），其真实性由 AC1 外部仓库核验 + 本 fresh run 共同支撑。

## 4. Coherent full resign 攻击（外审自建）

步骤（`audit_s4_attack.py`，临时副本 `/tmp/boundflow-s4-audit-attack-src`）：

1. 5 行 raw 的 `source.abcrown_commit` 全部改为伪造值 `"f"*40`，重算各行 source_hash/raw_hash；
2. protocol 同步伪造 source/source_hash/workers_jsonl_sha256/protocol_hash；
3. 从修改后 raw 重派生 summary（含 summary_hash）；
4. 重写 manifest 全部文件 digest 与自哈希；
5. 运行 stdlib replay：**ACCEPTED**。

正式 artifact 未受影响（raw SHA256 攻击前后均为 `9c6c99c3...`）。结论与 S3-SHADOW-F1 同类：离线
自签 self-check 只证明 E0 闭包；source identity 的真实保证来自外审对外部仓库的独立核验（AC1），不
来自 manifest。该边界与 trust plan 的 `COHERENT_FULL_RESIGN_ACCEPTED_AT_E0` 口径一致，S4-4 的
challenge+witness 将结构性覆盖。

## 5. Findings

### S4-0-AUDIT-F1 [minor / mandatory-before-close]

“fully outer-resigned tamper 10/10 rejected”/“10类全重签攻击均闭合”的笼统措辞在 claims map、memo
与正式报告 §1 中重现（S3 已要求同类降精度）。substance 无误——10 类 probe 确实拒绝语义不一致攻击，
但 shorthand 未标注 coherent full resign 在 E0 可接受。强制按 §0-1 降精度后方可升级 VALIDATED。

### S4-0-AUDIT-F2 [minor / advisory]

正式报告 §4 把 5 个 admission hash 的差异归因于“run identity 绑定字段”。严格说，exact-call identity
hash 是唯一 run-identity 绑定字段；plan/snapshot/oracle/plan-binding 四个 hash 的 per-process 差异来自
provider 捕获内容（snapshot 全量 tensor metadata/history）的正常数值变动，不是 run identity。建议
respond 阶段补一句精确归因；不影响任何数值结论（slots 跨 10 进程位级一致已独立确认）。

### S4-0-AUDIT-F3 [info]

S4-0 协议无 challenge/nonce 字段，本轮保证等级为 E2-DIRECT-LEGACY，无 E1。与 trust plan 的阶段划分
一致；S4-4 formal closure 不得再使用 legacy 例外。

### S4-0-AUDIT-F4 [info]

`dol` CLI 不在外审环境 PATH，dol lint/exchange validate 未独立复跑，由 executor 侧记录承担。

## 6. 不可现场复核项

- abcrown/auto_LiRPA provider 内部实现的完整正确性未逐行审计（commit 钉定 + intercept 点 + 生产 flag
  已核，provider 内部 semantics 信任其上流测试）；
- fresh run 与 formal 均在同一台 RTX 4060、同一账号下执行，同机限制按 trust plan §6.1 披露；
- `/tmp` 下 fresh artifact 与攻击副本易失，关键结论已全部内联于本报告，脚本已归档。

## 7. 下一门禁判定

- 开放：S4-1A 预注册与实现（六 α + active β → 候选侧 persistent compressed buffers），限
  implementation/correctness；
- 继续关闭：S4-1A timing/performance、same-solver speedup、complete-query、跨模型、10x、
  ASPLOS-ready；
- S4-4 formal external closure 前必须落地 challenge+witness（无 legacy 例外）。
