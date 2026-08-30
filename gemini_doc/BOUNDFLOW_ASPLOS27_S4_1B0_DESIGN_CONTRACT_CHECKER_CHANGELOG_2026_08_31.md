# S4-1B0 设计合同一致性检查器修改记录

date: 2026-08-31
stage: s04
execution-authority: false
production-code-changed: false
formal-run-open: false
timing-open: false
performance-claimed: false

## 1. 修改目的

S4-1A 外审仍未返回，S4-1B0 production、tests、worker、generator、replay、tamper 和 formal 均保持关闭。
此前冻结的 IEEE fixture、negative contract、ABI contract 与 formal artifact contract 各自可解析，但缺少一个
跨文件的机器检查，不能及时发现 dependency SHA、reason partition、worker topology、storage ledger 或 claim
边界之间的漂移。

## 2. 新增内容

新增纯标准库、只读脚本：

```text
scripts/check_asplos27_s4_1b0_design_contracts_stdlib.py
```

它不 import BoundFlow、Torch、TVM、TVM-FFI 或 NumPy，也不构建 module、启动 GPU、生成 artifact 或修改
production。检查范围包括：

1. 四个 schema 与共同 construction-model hash；
2. dependency asset 的实际 SHA256；
3. 16 个 pack 与 16 个 select 位语义 fixture，包括 signed zero、subnormal、nonfinite、canonical qNaN；
4. 两个 midpoint reassociation 反例；
5. 20 个 stable reason、16 个 test layout 与五组 fault ordinal 的无重叠全集；
6. negative/ABI 的 cache key、exclude、failure policy 一致性；
7. 五个 caller-owned tensor、六次 argument occurrence、DLPack 与 storage 账；
8. `5 positive + 1 cache + 5 fault = 11` worker topology；
9. `313,344 B` positive sidecar 及 selector/selected 字节账；
10. warm receipt、tamper、E0 authenticity boundary 与全部 false claim。

`--require-code-closed` 额外确认当前尚未出现 S4-1B0 backend/test/formal scripts/raw artifact，避免在
S4-1A 外审批准前越级施工。批准后仍可去掉该参数继续使用跨合同检查。

## 3. 边界

本修改只增强 design freeze 的机械一致性，不形成以下任何 claim：

```text
S4-1B0 implementation correctness
production evaluator binding
same-solver replacement
timing / performance
complete-query / 10x / ASPLOS-ready
```

当前下一动作仍是等待 S4-1A Round 1 外审；外审批准并由 executor 关闭 exchange 后，才允许按冻结施工包
开始 S4-1B0 implementation/correctness。

## 4. 实际验证

```text
checker SHA256 = bfd8f53c099e4513d7c51150093e3fd7cf0eec41df52cd5c695081a093c56c9c
positive cross-contract checks = 161 PASS
construction model hash = 5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a
stable reason / test layout = 20 / 16
fresh process topology = 11
positive sidecar bytes = 313344
code/formal closed paths = PASS
topology tamper = rejected
IEEE selector tamper = rejected
Black = PASS
Mypy = clean
Pylint = 10.00/10
git diff --check = PASS
```

两类负向探针仅在临时副本中修改合同：`11→12` worker total 与首个 nonfinite selector
`-128→1`，检查器均以非零退出拒绝。临时副本不作为正式 artifact，也不进入仓库。
