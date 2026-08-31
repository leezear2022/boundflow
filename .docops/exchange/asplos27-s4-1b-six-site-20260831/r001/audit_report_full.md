---
status: external-audit-report-full
date: 2026-08-31
type: external-audit-report
topic: boundflow
slug: asplos27-s4-1b-six-site-audit-r001
exchange-task: asplos27-s4-1b-six-site-20260831
exchange-round: 1
auditor: external-model (kimi-code CLI session, operator-supervised)
base-commit: 2f03905
source-commit: 760fa0d
result-commit: 591621b
verdict: approve-with-minor-correction
assurance-level-achieved: E2-DIRECT-LEGACY (auditor-witnessed test execution)
performance-claimed: false
---

# BoundFlow ASPLOS'27 S4-1B 六站点 production correctness 外部审计完整报告（r001）

## 0. Verdict

**approve-with-minor-correction**。

- blocker：0；major：0；minor：1（F1 pylint 口径）；info：2（F2 无多进程 formal artifact 的见证口径；
  F3 dol CLI 不在外审环境）

approve 语义严格限定为：同意关闭 `VALIDATED-S4-1B-SIX-SITE-VALUE`（六站点 value
correctness/ownership），并只开放 S4-1C compressed gradient implementation/correctness。不开放
optimizer、timing、performance、same-solver、complete-query、10x、ASPLOS-ready。

强制修正项（close 前落实，二选一）：

1. 为 `boundflow/runtime/asplos27_s4_coefficient_selector_pass.py` 第 233 行的惰性 `import tvm` 在文件头
   补 `import-error` 禁用（S4-1B0 backend 文件已有的同一约定），使逐文件 pylint 回到 10.00/10；或
2. 在权威文档把“Pylint 10.00/10”降精度为实际口径（该文件 9.80/10，E0401，环境解析限制）。

该修正是纯注释/口径层面，不影响任何运行时行为与已审代码语义。

## 1. 审计输入与方法

读取：exchange request/delivery、施工合同、实现 changelog；亲读全部五个交付文件中的三个实现文件
（`asplos27_s4_coefficient_selector_pass.py` 602 行全读、`asplos27_s4_six_site_value.py` runtime 509
行全读、`backends/tvm/asplos27_s4_six_site_value.py` 806 行中 ABI/graph/compile 全区段）与两个测试
文件的关键区段。

本轮为实现正确性审计，无多进程 formal artifact（delivery 已披露）。保证来自：外审亲读源码 +
外审现场亲跑全部 GPU 专项/联合/全量测试（auditor-witnessed execution）+ 冻结合同不变性核验。

## 2. AC1—AC7 逐项判定

### AC1 范围与顺序：PASS

- 顺序：S4-1B0 外审关闭提交 `2f03905`（12:35）→ 本批实现 `760fa0d`（13:11），顺序成立；
- 五个交付文件在 `760fa0d` 与审计 HEAD 间逐字节一致；
- 交付文件无 S4-1C dα/dβ、optimizer、timing 或 performance 路径（全文扫描仅 claim-flag 常量与
  docstring 提及，compile_ms 为编译元数据、不入 receipt，沿用 S2 既有口径）；
- 施工合同冻结于 `b34bf67`（2026-08-29）；`591621b` 对其仅有 6 行顶部加注，明确不改历史预注册
  元数据与门禁；19-action/42+7 ABI/phase 顺序未事后修改。

### AC2 Pass A 真实生产边界：PASS

亲读 `capture_r31b2_production_selectors_v1` 与 owner：

- 19 个插入点冻结顺序逐行核对：A29 在 relu28 coefficient 前（linear14_right 之后）、A26/A20 分别
  位于 residual11/residual6 的 stage1 与 stage2 之间、A24/A18 在 stage2 之后、Ainput 在 conv0_right
  之后、box_concretize 之前；
- 六个 selector 由 `bind_compiled_sources` 预绑定 source/output DLPack view 的编译 TIR kernel 写入；
  `adopt_selectors` 强制 `compiled_pack_launch_count==6 且 eager_pack_count==0`——production 路径
  eager pack=0 是结构性强制，不是观测巧合；
- endpoint 合法值 {-128,-1,0,1}、binary {-128,0,1}；nonfinite → -128（exponent-bit 判断，TIR 与
  eager 参考一致）；ternary ±0 → 0；stream/phase/单次性逐 action 强制，任何违反即 poison 不可重试。

### AC3 Pass B 编译图：PASS

- 从 Relax IR 构建源码独立清点 49 参数：42 read（input lower/upper、endpoint selector、6 站点
  lower/upper/alpha/map、5 个 sign、全部 weight/bias、linear 参数）+ 7 caller-owned write
  （selected_input_target + 6 个 V target）；active α 为 `[D,W]`（如 alpha17=(6,164)）；empty β 与
  site31 的 α/map/sign 不在参数表；
- 图结构：6 Conv（weight0/2/4/5/8/10）、1 Gemm（weight14 matmul+linear）、1 ternary input select
  （call_tir_inplace inplace_indices=[3]）、5 selected-ReLU、6 persistent copy（各 call_tir_inplace），
  7 个写目标全部 inplace 到 caller-owned 参数；
- 编译期强制 cuDNN partition=4/call=6、TIR=12；source/partitioned/lowered/device source hash 均由
  实际 content 重算并在 validate 中复核；identity 篡改变体在单测中逐项拒绝。

### AC4 ownership 与运行时：PASS

- selected-input 与 coefficient arena 同 storage（`_storage_identity` 四元组相等强制），rebind 在
  selector adoption 后显式 phase 推进；
- V17/19/23/25/28/31 为单一 37,464-element（149,856 B）连续无洞 arena 的 narrow view（slot 表
  0→36864+600 覆盖）；
- 49 个 DLPack view 仅 prepare 创建；warm view count=0；result owner 为单个 VM tuple 引用
  （capacity=1），无无界 list；
- default stream 拒绝；Torch current stream、TVM-FFI raw stream、prepared stream 三者一致逐次强制；
  VM 单次调用后 6/6 output pointer 与 arena target 逐位相等，不等即 poison；
- receipt 只含 hash/计数/flag，不含 raw pointer/Tensor/NDArray/VM 对象。

### AC5 独立数值复核：PASS

- 外审现场亲跑真实冻结 ResNet2B 测试：新增专项 `9 passed in 10.67s`，含
  `test_s4_six_site_real_r31b2_capture_and_value_graph`（真实 R31B2 捕获 + 六站点图）；
- oracle 独立性确认：`_oracle` 纯 PyTorch（`functional.conv2d`/`torch.where`/`functional.linear`），
  不消费任何 TVM/TIR 结果；TIR 端另有 bit 级 classifier 不变量；容差 rtol=atol=2e-4；
- 每槽 shape、pointer owner（value_arena 偏移相等）与 selector 原始内容均由测试断言，外审复核其
  断言语义成立。

### AC6 负向与回归：PASS（附 F1）

- 负向：compiled identity 7 变体、selector pointer substitution、phase/order/stream/stream-ffi、alias、
  count/claim 篡改均 fail closed（测试亲跑通过）；Pass A owner poison 语义源码确认；
- 联合专项（外审现场）：`189 passed in 19.13s`；全量（外审现场）：`2082 passed, 3 skipped,
  6 warnings in 730.47s`；3 个 skip 理由现场核对为既有 TVM 重复编译与两项 VNN-COMP 冻结 checkout
  不可用，与历史口径一致；
- Black：5 文件 unchanged；mypy（`--explicit-package-bases`）：5 文件 clean；`git diff --check`：
  PASS；pylint：4/5 文件 10.00/10，`asplos27_s4_coefficient_selector_pass.py` 为 9.80/10
  （E0401 惰性 `import tvm`，未按 S4-1B0 约定加 `import-error` 头禁用）——见 F1；
- `dol` CLI 不在外审环境 PATH，dol lint 未独立复跑（F3）。

### AC7 claim 边界：PASS

- claims map/memo 均为 implementation/correctness candidate 口径，无 VALIDATED 升级；S4-1C、
  optimizer、timing、performance、same-solver、complete-query、10x、ASPLOS-ready 全部保持关闭；
- 本轮无多进程 formal artifact，无 E0 self-check 措辞需要降精度（delivery risks 已主动披露 E0
  coherent full-resign 边界）。

## 3. Findings 汇总

| ID | severity | 状态 | 内容 |
|---|---|---|---|
| S4-1B-AUDIT-F1 | minor | mandatory-before-close | coefficient_selector_pass.py pylint 9.80/10（E0401）与“Pylint 10.00”声明不符；补头禁用或降精度 |
| S4-1B-AUDIT-F2 | info | closed-by-disclosure | 本轮无多进程 formal artifact；正确性由外审亲跑测试见证；后续 S4 正式闭环仍需 challenge+witness |
| S4-1B-AUDIT-F3 | info | open | dol CLI 不在外审环境，dol lint 未独立复跑 |

## 4. 结论

AC1—AC7 全部 PASS（AC6 附 1 项 minor 口径修正）。同意在 F1 修正落实后关闭
`VALIDATED-S4-1B-SIX-SITE-VALUE`，并只开放 S4-1C compressed gradient
implementation/correctness。S4-4 formal closure 前必须落地 challenge+witness。
