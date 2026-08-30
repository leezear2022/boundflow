# S4-0 mutable-state admission外部审计交接

> **归档说明**：本交接已由2026-08-30外审报告完成。AC6中的“fully outer-resigned”只表示10个同时重签内外
> hash、但仍与冻结派生语义不一致的攻击；不表示coherent full resign会被E0 self-check拒绝。外审自建coherent
> full resign已证明后者可接受，最终保证等级为`E2-DIRECT-LEGACY`。

请不要采信执行方summary数字；从正式raw、源码和stdlib独立脚本重算。审计对象为HEAD上的
`FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-0`，不是性能claim。

## 审计入口

- 报告：`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_FORMAL_CANDIDATE_REPORT_2026_08_30.md`
- 合同：`gemini_doc/BOUNDFLOW_ASPLOS27_S4_0_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`
- artifact：`artifacts/asplos27-s4-admission/resnet2b-prop0-v1`
- runtime：`boundflow/runtime/asplos27_s4_mutable_state_admission.py`
- worker/generator/replay/tamper：`scripts/run_asplos27_s4_admission_worker.py`、
  `scripts/run_asplos27_s4_0_admission_artifact.py`、`scripts/replay_asplos27_s4_0_admission_stdlib.py`、
  `scripts/probe_asplos27_s4_0_admission_tamper.py`

## Acceptance criteria

1. AC1 source identity：HEAD/formal source、六个code blob、三个外部仓库commit、model/property SHA256及manifest
   全部一致，artifact无本机路径；
2. AC2 real provider：5个fresh subprocess、run ordinal 0—4、每进程恰一次production intercept；真实容器/type/
   readiness与receipt投影一致，provider core/callback为0；
3. AC3 arithmetic：从slots和shape独立重算6/12、8496/4248/4248、1/6、8502/34008、24/68016；不得只读
   summary；
4. AC4 lease/claim：OPEN且12 strong refs→CLOSED且0，buffer/candidate/mutation/timing/performance/global
   exclusivity全部false或0；
5. AC5 negative：63个独立node确实逐项断言exact detail+reason，minimum 56成立；
6. AC6 replay/tamper：stdlib replay不import boundflow/torch，raw语义重算通过；10个inner+outer-resigned、
   derived-semantics-inconsistent变体均因语义而非只因manifest digest被拒绝；另测coherent full resign并按E0
   边界披露其可接受性；
7. AC7 regression/claim：targeted、全量、black/mypy/pylint/diff/DocOps复现；不得把formal candidate升级为
   VALIDATED、S4-1A或performance。

## 输出要求

请给出approve/reject、blocker/major/minor/info分级、每条AC的独立证据与不可现场复核项。approve只表示同意关闭
S4-0 correctness/admission，并开放S4-1A的下一阶段预注册；不表示任何speedup、same-solver或ASPLOS-ready。
