---
status: external-audit-report-full
date: 2026-08-31
type: external-audit-report
topic: boundflow
slug: asplos27-s4-1b0-ternary-audit-r001
exchange-task: asplos27-s4-1b0-ternary-20260831
exchange-round: 1
auditor: external-model (kimi-code CLI session, operator-supervised)
base-commit: 20f57bb
formal-source: 4e2a26128a9a538ac64f222e8b82e92ea745d3b6
result-commit: 50c5ff642f0eb99150cc0b1bc01f414beda28ab2
audit-head: 97de3d3bc8fd676c47da74354a6276f9cbd22f08
verdict: approve
assurance-level-achieved: E2-DIRECT-LEGACY
performance-claimed: false
---

# BoundFlow ASPLOS'27 S4-1B0 ternary endpoint 外部审计完整报告（r001）

## 0. Verdict

**approve**。

- blocker：0；major：0；minor：0；info：3（F1 历史 9/10 缺口修复已验证；F2 coherent full resign 的
  E0 边界确认存在且已披露；F3 dol CLI 不在外审环境）

approve 语义严格限定为：同意关闭 `VALIDATED-S4-1B0-TERNARY-ENDPOINT`（isolated endpoint
correctness），并另行开放 S4-1B production implementation/correctness。不开放 timing、performance、
same-solver、complete-query、10x、ASPLOS-ready。本轮无强制修正项。

## 1. 审计输入与方法

读取：exchange request/delivery、外审交接、候选 changelog；亲读
`boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py`（TIR pack/select、policy 常量、20 个 stable
reason）、`scripts/replay_asplos27_s4_1b0_ternary_stdlib.py`（290 行全读）、
`scripts/run_asplos27_s4_1b0_ternary_worker.py` 与生成器入口、negative contract JSON。

外审独立产物（归档 `artifacts/asplos27-s4-1b0-ternary/audit-20260831/`）：

- `audit_s4_1b0_recompute.py`：stdlib-only 独立重算脚本，直接解析 `.bin`，不 import
  boundflow/torch/tvm 或 executor replay；
- LSB 攻击与 coherent full resign 攻击命令内联于本报告 §4/§5。

现场重跑：stdlib replay、tamper probe、targeted 22、全量 2073、static gates、auditor-controlled
11-process fresh generation。raw 独立重算：selector/selected bitwise/counts/cache/fault/hash 链。
源码/冻结证据审查：TIR 语义、20 reason、合同 JSON。

## 2. AC1—AC7 逐项判定

### AC1 source 与协议身份：PASS

- 7/7 protocol code blob 以 `git show HEAD:<path>` 独立重算逐位一致（审计 HEAD=`97de3d3`）；
- 外部 commit：abcrown `e5c7e17...`、auto_LiRPA `5a098e8f...`、vnncomp `90419aad...` 本机复核一致；
  model `791aa24d...`、property `89edf066...` 重新散列一致；
- 4 份合同/fixture JSON 由 protocol.dependencies 以 SHA256 绑定并核验在仓文件一致；
- construction hash、source capture（suffix-boundary.pt `217aaad0...`）绑定一致；
- manifest 15 文件 digest 全对、自哈希通过；全 artifact 文本扫描无本机绝对路径；
- 顺序：修复 commit `4e2a261`（2026-08-31 10:23:32 +0800）是先祖，formal artifact commit `50c5ff6`
  （10:42:22）在其后——正式 artifact 确实生成于修复之后。

### AC2 实现语义：PASS

- 亲读 TIR：pack 用 `reinterpret uint32` + exponent mask `0x7f800000` 判 nonfinite（无 `x==x`），
  ±0 经 `>0`/`<0` 双否归入 zero endpoint；select 按 1→lower、-1→upper、0→`(lower+upper)*0.5`
  （TIR f32 两步舍入），其余→canonical qNaN `0x7fc00000`（reinterpret 常量）；
- Python bit oracle（`ternary_pack_bit_oracle_v1`/`ternary_select_bit_oracle_v1`）为独立 reference；
  worker 另用 PyTorch `torch.where` 参考比对 TIR 输出——无 TIR 自比；
- 20 个 stable reason 与 `BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json` 集合
  逐字一致（无多余、无缺失）；
- spec 校验拒绝 legacy symbol 冲突、policy 漂移；无 timing API/performance flag 漂移（artifact 各层
  flag 全 false）。

### AC3 positive raw：PASS（外审独立重算，非采信 summary）

- 5 份 sidecar 逐字节一致：外审独立 SHA256 均为
  `a07aea90d2404b0e3c40f2af4aeaea169a1465b5feb24616c75cf882b5db5e6c`（313,344 B 每份）；
- 从 `.bin` 原始字节独立重算：selector 计数 `8689/9137/606/0`（positive/negative/zero/invalid）；
  18,432 个 selected 输出逐元素 bitwise exact（外审自己的分类器与中点舍入链，非 executor 代码）；
- `old_binary_zero_misclassified=606` 与 raw 回绑一致；
- 5 个 DLPack descriptor hash + `dlpack_pointer_exact=5`、module receipt hash 链（TIR JSON/device
  source/receipt 互绑）逐行核验。

### AC4 cache 与 fault：PASS

- cache 行：`events=["miss","hit"]`、compile/miss/hit/entry=1/1/1/1、same_module_receipt=true、
  tensor_retention=0；外审 fresh 生成中真实复现；
- 5 个 fault 各自由独立 fresh PID 真实触发，reason 按冻结顺序逐项一致（midpoint policy /
  device source / DLPACK identity / stream identity / invalid-selector-not-poisoned）；
  `context_is_none=true`、fallback/eager/native-shadow=0、reject-before-launch；
- 外审亲启完整 fresh 生成（11 个独立 PID），fresh 5 fault 全部复现。

### AC5 replay 与 tamper：PASS

- stdlib replay（外审现场，formal 与 fresh 各一）：PASS；确认其从 raw 语义重算（selector 位级、
  selected bitwise、counts、cross-run determinism 回绑），不是只验 hash；
- 注册 tamper（外审现场）：10/10 rejected，逐类语义原因；
- 外审自建攻击 1（历史 9/10 攻击面复测）：翻动 positive-00.bin 某 coefficient LSB（符号不变），
  并重绑该行 binary/index hash、summary sidecar 绑定与 manifest——**被
  “positive fresh-process binary determinism differs”拒绝**。`4e2a261` 的修复对历史缺口有效；
- 外审自建攻击 2（coherent full resign 伪造 abcrown commit + 全链重签）：**接受**——E0 边界如
  披露存在（tamper report 已含 `coherent_full_resign_e0_boundary_disclosed: true`），不构成缺陷；
- 两次攻击后正式 artifact 原 sidecar SHA256 复核未变。

### AC6 验证链：PASS

- targeted（外审现场）：`22 passed in 6.66s`；
- 全量（外审现场）：`2073 passed, 3 skipped, 6 warnings in 723.46s`，3 skip 为既有 TVM/VNN-COMP
  环境边界；
- Black：7 文件 unchanged；mypy（`--explicit-package-bases`）：7 文件 clean；pylint：逐文件 7/7
  10.00/10（S4-1A 的 F2/F3 类问题本轮不存在）；`git diff --check`：PASS；
- `dol` CLI 不在外审环境 PATH，dol lint/exchange validate 未独立复跑（F3）；executor 侧记录为
  PASS。

### AC7 claim 与后继：PASS

- claims map/memo/changelog 均为 `FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0`，无
  VALIDATED 升级、无 timing/performance 越界；S4-1A 的关闭与 S4-1B0 开放顺序正确；
- approve 后只关闭 isolated endpoint correctness 并另开 S4-1B production
  implementation/correctness。

## 3. Auditor-controlled fresh run（E2-DIRECT-LEGACY）

外审从空目录 `/tmp/boundflow-s4-1b0-audit-fresh-20260831/artifact` 亲启生成器：11 个独立 PID
（5 positive + 1 cache + 5 fault）。fresh summary 与 formal 除 `source_revision`（formal=
`4e2a261`、fresh=审计 HEAD `97de3d3`，期间仅文档/DocOps 提交，7/7 code blob 逐位一致）外全字段
一致，包括 module receipt hash、sidecar SHA256、selector 计数、fault reason 序列。协议无
challenge/nonce 字段，本轮为 `E2-DIRECT-LEGACY`，不构成 E1；S4-4 起不得再用 legacy 例外。

## 4. 历史 9/10 演练缺口处置核验

执行方披露的第一次演练 9/10（coefficient LSB 不改变符号 + replay 缺 5 份 sidecar 全同回绑）经外审
重点复测：§2-AC5 的 LSB 攻击精确命中该攻击面，被新 determinism 回绑拒绝。失败演练未进入正式
artifact，正式 artifact 从修复后的空目录重新生成（git 顺序证据见 AC1）。处置合规。

## 5. 不可现场复核项

- TVM 编译器内部的数值 lowering 正确性信任其 codegen 与 device source（device_source.cu 已入
  manifest 并绑定 receipt）；外审验证的是 bit 级输出语义与冻结合同的一致；
- fresh 与 formal 同机同账号，按 trust plan §6.1 披露；
- `/tmp` 下 fresh artifact 与攻击副本易失，关键结论已内联本报告；
- abcrown/auto_LiRPA 本轮仅作 fixture 来源（source capture），非执行路径。

## 6. Findings 汇总

| ID | severity | 状态 | 内容 |
|---|---|---|---|
| S4-1B0-AUDIT-F1 | info | closed-verified | 历史 9/10 缺口的 4e2a261 修复经外审 LSB 攻击复测有效 |
| S4-1B0-AUDIT-F2 | info | closed-by-disclosure | coherent full resign 在 E0 可接受，tamper report 与 delivery 已披露 |
| S4-1B0-AUDIT-F3 | info | open | dol CLI 不在外审环境，dol lint 未独立复跑 |

## 7. 结论

AC1—AC7 全部 PASS，无强制修正项。同意关闭 `VALIDATED-S4-1B0-TERNARY-ENDPOINT`，并只开放
S4-1B production implementation/correctness。S4-4 formal closure 前必须落地 challenge+witness。
