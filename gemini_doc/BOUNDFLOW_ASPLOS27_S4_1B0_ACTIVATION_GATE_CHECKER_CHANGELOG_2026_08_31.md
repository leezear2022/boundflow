# S4-1B0 激活门禁检查器修改记录

date: 2026-08-31
stage: s04
production-code-changed: false
formal-run-open: false
timing-open: false
performance-claimed: false

## 1. 目的

S4-1A 外审仍未返回。此前虽然已经冻结 S4-1B0 施工合同并建立 200 项跨合同检查，但“什么时候允许开始
implementation/correctness”仍需人工同时判断 exchange、audit、closure、DocOps 与 Git 状态。新增纯标准库、
只读检查器：

```text
scripts/check_asplos27_s4_1b0_activation_gate_stdlib.py
```

它不 import BoundFlow、Torch、TVM、TVM-FFI 或 NumPy，不修改 exchange，不生成 production/formal 文件。

## 2. 状态机

返回语义固定为：

```text
exit 0 / PROCEED = S4-1B0 implementation/correctness 可以激活
exit 3 / WAIT    = 外审或 executor close 尚未完成，保持等待
exit 1 / ERROR   = 权威状态互相矛盾，禁止激活
```

`approved` 不是 GO。必须同时满足：

1. S4-1A exchange=`closed`；
2. `approved_round` 为正整数且等于当前 round；
3. 对应 `audit.md/json`、`closure.md/json` 全部存在并注册在 exchange docs；
4. audit verdict=`approve`，且没有 blocker/major finding；
5. closure resolution=`approved`，task/round/doc identity 一致；
6. DocOps 为 active/green s04，旧 S4-1A blocker 已移除，next 包含 `s4-1b0`；
7. 分支正确，S4-1A formal、S4-1B0合同和construction-root checker提交均为HEAD祖先；
8. S4-1B0 200项设计检查通过，future production/formal路径仍未出现。

即使返回PROCEED，也只开放implementation/correctness；formal、timing、performance仍固定false。

## 3. 验证计划

- 内存状态机覆盖ready/approved/closed/closed-no-round/round-mismatch五类；
- 当前真实仓库必须返回WAIT、reason=`external-audit-s4-1a-pending`、exit=3；
- 人工构造的矛盾closed状态必须返回ERROR；
- Black、Mypy、Pylint、diff与DocOps lint全部通过。

本文件只记录门禁自动化，不解除当前 `external-audit-s4-1a-pending`。

## 4. 实际验证

```text
checker SHA256 = 572d3ea9918b4f1705c01ed3f3a1504d57ce5442283283da7a55cfa6dac54acd
state-machine self-test = 5/5 PASS
real repository status = WAIT
real repository reason = external-audit-s4-1a-pending
real repository exit code = 3
design prerequisite = 200 PASS
implementation_authority = false
formal_authority = false
timing_authority = false
performance_claimed = false
Black = PASS
Mypy = clean (activation + design checker)
Pylint = 10.00/10 (activation + design checker)
git diff --check = PASS
```

当前真实检查同时绑定分支、HEAD祖先、exchange round、DocOps blocker/next与200项设计合同。返回WAIT是当前
正确结果，不属于测试失败；只有外审approved后再由executor执行exchange close并同步DocOps，才可能进入PROCEED。
