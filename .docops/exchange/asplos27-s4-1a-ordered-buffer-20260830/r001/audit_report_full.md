---
status: external-audit-report-full
date: 2026-08-31
type: external-audit-report
topic: boundflow
slug: asplos27-s4-1a-ordered-buffer-audit-r001
exchange-task: asplos27-s4-1a-ordered-buffer-20260830
exchange-round: 1
auditor: external-model (kimi-code CLI session, operator-supervised)
base-commit: 3dca00f
formal-source: bce26f0d8109f69d520dfe27a04fb9c2110b34b0
result-commit: f773370
audit-head: fa062b554446965c17b7589ebde0334c670f7e77
verdict: approve-with-minor-correction
assurance-level-achieved: E2-DIRECT-LEGACY
performance-claimed: false
---

# BoundFlow ASPLOS'27 S4-1A ordered buffer 外部审计完整报告（r001）

## 0. Verdict

**approve-with-minor-correction**。

- blocker：0
- major：0
- minor：3（F1 replay 未绑定 verification_reason；F2 mypy 3 errors 与 delivery 声明不符；
  F3 pylint 9.90/10 与声明 10.00 不符）
- info：3（F4 binary_index 顺序重排在 E0 下可接受但系语义空操作；F5 coherent full resign 的 E0 边界
  依预期存在；F6 dol CLI 不在外审环境）

批准语义严格限定为：同意关闭 `VALIDATED-S4-1A-ORDERED-BUFFER-PREPARE`，并另行开放 S4-1B0
implementation/correctness。不开放 CROWN numeric、optimizer、same-solver、memory/performance、
complete-query、10x、ASPLOS-ready；S4 timing 保持关闭。

强制修正项（executor 在 respond/close 阶段执行，不得改写已交付 artifact）：

1. 静态门禁口径修正（二选一）：(a) 修复 `scripts/run_asplos27_s4_1a_buffer_worker.py` 的 3 个 mypy
   错误并为 `boundflow/runtime/asplos27_s4_ordered_buffer_abi.py` 的惰性 `import tvm` 补
   `import-error` 头禁用（S2 既有约定），修复后重跑静态门禁并记录；(b) 或在权威文档把 delivery 的
   “black-mypy-pylint pass”降精度为实际口径（mypy 3 个类型级错误、pylint 9.90/10 E0401）。修复代码
   不影响已交付 artifact 的 replay（replay 不绑定 git code blob）。
2. 建议（不阻塞）：后续 hardening 提交把冻结合同的 fault→(detail_code, verification_reason) 映射加入
   stdlib replay 校验；不得在已交付 r001 上静默补丁。

## 1. 审计输入与方法

读取：exchange request/delivery、S4-1A 外审 handoff、施工合同、formal changelog；亲读
`boundflow/runtime/asplos27_s4_ordered_buffer_abi.py`（1211 行中全部资源顺序/ticket/lease/view/fault 区段）、
`scripts/run_asplos27_s4_1a_buffer_worker.py`、`scripts/replay_asplos27_s4_1a_buffer_stdlib.py`（445 行全读）、
`scripts/run_asplos27_s4_1a_buffer_artifact.py`（生成器入口与 CLI）。

外审独立产物（归档于 `artifacts/asplos27-s4-1a-buffer/audit-20260831/`）：

- `audit_s4_1a_recompute.py`：stdlib-only 独立重算（不 import boundflow/torch/tvm/numpy 及 executor
  replay 工具），对 formal 与 fresh 两份 artifact 各跑一遍；
- `audit_s4_1a_attack.py`：binary_index 重排攻击与 coherent full resign 攻击；verification_reason
  伪造攻击命令内联于本报告 §4。

环境：conda boundflow（torch 2.12.1+cu132）执行 audit/replay/test；外部 provider python=
`/home/lee/Codes/alpha-beta-CROWN/.venv/bin/python`（torch 2.11.0+cu130）；GPU RTX 4060 Laptop。

## 2. AC1—AC7 逐项判定

### AC1 source 与 scope：PASS

- 9/9 protocol code blob 以 `git show HEAD:<path>` 独立重算逐位一致（审计 HEAD=`fa062b5`，
  formal source=`bce26f0`，期间 code path 零漂移，工作树九文件 clean）；
- abcrown `e5c7e17...`、abcrown 内嵌 auto_LiRPA `5a098e8f...`、vnncomp2021 `90419aad...` 在本机
  独立复核一致；model `791aa24d...`、property `89edf066...` 重新散列一致；
- manifest 11 文件 digest 全对，自哈希链（raw/worker payload/receipt/protocol/summary/manifest）全部由
  外审独立重算通过；全 artifact（含 .bin sidecar 外的全部文本）扫描无本机绝对路径；
- scope：protocol 固定 `buffer_prepare=true`、`candidate_execute/mutation/timing/performance=false`；
  runtime 亲读确认无 CROWN evaluator、TIR launch、optimizer mutation、terminal 或 timing 路径
  （模块 docstring 与代码一致）。

### AC2 ordered physical owner：PASS

亲读确认：

- S4-0 ticket 单次消费：`begin_buffer_prepare` → ticket → resource owner 链，adoption/ticket/owner
  任一失败即 close  lease 并 FAILED_CLOSED；
- 资源顺序冻结且逐项强制：6 α parameter → 1 active β parameter（slot 5）→ 6 α gradient →
  1 β gradient → lower → upstream，共 16 buffer；5 个 empty β 只有 typed token（无物理分配）；
- 16 个 storage token（device/cdata/data_ptr/nbytes）互异强制（`CANDIDATE_STORAGE_ALIAS`），
  parameter/gradient 集合互斥、candidate/source 集合互斥；
- view key 含 storage identity（cdata/pointer/nbytes）+ tensor pointer + shape/stride/offset/dtype/
  device；`_view_key` 在 cache lookup 前拒绝 non-contiguous（`type is torch.Tensor` 且
  `is_contiguous()`，否则 reject）；
- `_validate_tensor` 强制 exact Tensor/float32/cuda/contiguous/offset=0/leaf/requires_grad 分角色；
- 初始化内容逐 buffer 校验：7 个 parameter clone 后 content hash 与 source 逐项比对，upstream 为
  全 -1.0 并与独立构造的期望值比对；
- fault 路径在 `except` 块结束后才构造 `S4MutableBufferPreparationError`，保证 `__context__ is None`，
  且 owner/ticket/adoption 全部 close、引用全部清空。

### AC3 5 fresh 正向与二进制语义：PASS（外审独立重算）

对 formal 与 fresh 两份 artifact 各自行重算（每份 5 正+7 故障，共 12 行）：

- 5/5 真实 production exact-call（`provider_core_intercept_count=1`、execute=0）；
- 每 run 8 组 source/candidate：外审直接读 `.bin` sidecar，按 index 偏移切片并独立 SHA256，
  与 index 记录逐位一致，且 source 与 candidate 逐字节相等；两份 artifact 各 40/40；
- parameter/gradient=`4,254` elements/`17,016 B`（从 16 个 descriptor 的 shape 独立求和）；
- storage/view=`16/16` 唯一、candidate logical=`34,080 B`（等于每个 .bin 的字节数，独立验证）；
- empty β token=5、physical=0；
- D2H 恒等式独立重算：source 24/68,016 + initialized candidate 8/17,040 = S4-1A 32/85,056；
  累计 = S4-0 24/68,016 + 32/85,056 = 56/153,072；D2D=7/17,016；
- close 后 candidate allocated delta：5/5 为 0（entry==exit）。

**关键交叉证据**：5 个 formal sidecar 与 5 个 fresh sidecar 的 SHA256 全部相同
（`78970266...`），即 10 个独立进程的 candidate buffer 内容逐字节一致——真实确定性执行的强证据。

### AC4 7 fresh 隔离故障：PASS

- 7 个 fault（parameter/gradient/output/view/roundtrip/receipt/adoption）分别在独立 fresh 进程执行
  （worker_sequence 与 ordinal 绑定，exact-call hash 外审独立重算一致）；
- detail_code 与冻结映射逐项一致；`context_is_none=true`、`allocated_delta=0`、
  `admission_state=TRANSFERRED`、fallback/retry/empty-cache=0 逐行核对；
- verification_reason 与冻结合同表逐项一致（外审人工比对）：output/adoption=
  `UNSAFE_ALIAS_OR_LIFETIME`，其余五类=`RECEIPT_IDENTITY_MISMATCH`；
- 外审亲启完整 fresh 生成（含全部 7 个 fault 进程），fresh raw 通过同一套独立重算；
- fault 行的 `traceback_sha256` 为 run 相关字段，不作为一致性要求（披露为边界）。

### AC5 unit、replay 与 tamper：PASS（附 F1）

- negative registry：77 个唯一 nodeid、全部存在于测试源码、逐 case
  `exact_detail_and_reason_asserted=true`、targeted_result=pass、minimum 68 满足；
- unit（外审现场）：80 passed；artifact+unit（外审现场）：84 passed；
- stdlib replay（外审现场，formal 与 fresh 各一）：PASS；该脚本确认不 import
  BoundFlow/PyTorch/TVM/Numpy（全文件亲读，仅 stdlib）；
- tamper probe（外审现场）：10/10 rejected，全部 `semantic_recompute_rejected=true`；
- 外审自建未注册攻击：
  - binary_index 组内顺序交换 + coherent 重签：**接受**，但经分析属语义空操作（每个 index 项自带
    offset/hash，顺序不是合同语义），记 info（F4）；
  - fault `verification_reason` 伪造 + coherent 重签：**接受**，证明 replay 未把冻结 reason 纳入校验，
    记 minor（F1）；当前正式 artifact 的 7 个 reason 由外审人工核对与冻结表一致，且 77 项 negative
    测试在代码层同时断言 detail 与 reason，运行时无实际缺口；
  - coherent full resign（伪造 abcrown commit + 全链重签）：**接受**，E0 边界如预期（F5，info，
    delivery risks 已主动披露）；
- 三次攻击后正式 artifact raw SHA256 复核未变（`05dd316d...`）。

### AC6 回归与静态门禁：PASS-with-minor-discrepancy（F2/F3）

- 全量（外审现场）：`2050 passed, 3 skipped, 6 warnings in 720.38s`，3 skip 为既有 TVM/VNN-COMP
  环境边界，与声明一致；
- Black：7 文件 unchanged；`git diff --check`：PASS；
- mypy（S3 既有口径 `--explicit-package-bases`，7 个交付文件）：6/7 clean，但
  `scripts/run_asplos27_s4_1a_buffer_worker.py` 有 3 个错误（L245 None 赋值给 Tensor、L297/L301
  对类对象的 monkey-patch 赋值）——类型级问题，无运行时影响（worker 在 formal 与 fresh 均实际跑通），
  但与 delivery “black-mypy-pylint pass” 口径不符；
- pylint（逐文件）：6/7 为 10.00/10，`boundflow/runtime/asplos27_s4_ordered_buffer_abi.py` 为
  9.90/10（E0401 pylint 无法解析惰性 `import tvm`）；S2 既有约定是在文件头禁用 import-error，
  本文件未遵循；
- `dol` CLI 不在外审环境 PATH，dol lint 未独立复跑（F6）。

### AC7 判定与后继门禁：PASS

- claims map / memo / changelog 均保持 `FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1A`，无
  VALIDATED 升级、无 timing/performance/same-solver/complete-query/10x 越界；
- claims map 的 S4-1A 条目已主动使用“derived-semantics-inconsistent outer-resigned”精确措辞（S3 F1
  教训已被吸收）；S4-0 条目已含 E0 限定（上轮强制修正已落实）；
- artifact 各层 claim flag（performance/timing/exclusivity/crown-numeric/optimizer-trajectory/
  provider-mapping-stability）全 false，外审逐行核对。

## 3. Auditor-controlled fresh run（E2-DIRECT-LEGACY）

外审从自己控制的 shell、空目录 `/tmp/boundflow-s4-1a-audit-fresh-20260831/artifact` 启动完整生成器：
12 个 fresh 进程（5 正向 + 7 隔离故障，含全部 fault 重跑）+ 77 节点 negative pytest。fresh artifact
通过外审独立重算脚本全部检查与 executor stdlib replay；candidate 二进制与 formal 逐字节一致
（§2-AC3）。fresh run 的 `source_revision`=审计 HEAD `fa062b5`（formal source 之后仅文档/DocOps
提交，9/9 code blob 逐位一致）。协议无 challenge/nonce 字段，本轮为 `E2-DIRECT-LEGACY`，不构成 E1。

## 4. 不可现场复核项

- abcrown/auto_LiRPA provider 内部正确性未逐行审计（commit 钉定 + intercept 点已核）；
- fresh 与 formal 同机同账号执行，按 trust plan §6.1 披露；
- `/tmp` 下 fresh artifact 与攻击副本易失，关键结论已内联本报告，脚本已归档；
- fault 行 `traceback_sha256` 含 run 相关路径信息，仅作存在性/格式核对。

## 5. Findings 汇总

| ID | severity | 状态 | 内容 |
|---|---|---|---|
| S4-1A-AUDIT-F1 | minor | open（建议后续 hardening） | stdlib replay 未绑定 fault verification_reason；当前 artifact 人工核对合规 |
| S4-1A-AUDIT-F2 | minor | mandatory-before-close | worker 脚本 3 个 mypy 错误与 delivery 声明不符；修复或降精度 |
| S4-1A-AUDIT-F3 | minor | mandatory-before-close | ordered_buffer_abi pylint 9.90/10（E0401）与“10.00”声明不符；补 import-error 禁用或降精度 |
| S4-1A-AUDIT-F4 | info | closed | binary_index 顺序重排在 E0 可接受，系语义空操作 |
| S4-1A-AUDIT-F5 | info | closed-by-disclosure | coherent full resign 的 E0 边界依预期存在；delivery risks 已披露 |
| S4-1A-AUDIT-F6 | info | open | dol CLI 不在外审环境，dol lint 未独立复跑 |

## 6. 结论

AC1—AC7 全部 PASS（AC6 附两处 minor 口径修正）。同意外审关闭 S4-1A 为
`VALIDATED-S4-1A-ORDERED-BUFFER-PREPARE`（在 §0 强制修正落实后），并只开放 S4-1B0
implementation/correctness。S4-4 formal closure 仍须先落地 challenge+witness。
