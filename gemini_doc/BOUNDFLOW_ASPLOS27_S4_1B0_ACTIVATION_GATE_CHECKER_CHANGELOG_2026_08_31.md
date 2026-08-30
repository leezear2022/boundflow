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

## 5. 第二批：闭合 delivery/audit/closure 内容链

第一批只核对JSON身份与文件存在，尚未验证三份Markdown是否与各自JSON中的`md_sha256`一致。第二批增加：

1. request、delivery、audit、closure的Markdown与JSON八个文件全部存在；
2. delivery task/doc/round/from/to 与 exchange executor/auditor一致；
3. audit task/doc/round/from/to/verdict 与 delivery link一致；
4. closure task/doc/round/approved_round/from/resolution一致；
5. delivery/audit/closure 三个`md_sha256`均从实际Markdown独立重算；
6. exchange docs同时注册request/delivery/audit/closure，且无重复项；
7. approved audit仍不得含blocker/major finding。

内存自测新增一个完整合法closed exchange和一个只篡改`audit.md`的变体：前者通过，后者必须以
`audit-md-sha256-mismatch`拒绝。该测试只使用自动清理的临时目录，不接触真实exchange。

## 6. 第二批验证结果

```text
checker SHA256 = f643270b9975fedb7e5f1bed24e345810243eaf4edcf5c4dd551453421bb26ca
classifier state cases = 5/5 PASS
closed exchange content-chain cases = 2/2 PASS
total self-test = 7/7 PASS
audit Markdown tamper = rejected
real repository status = WAIT
real repository exit code = 3
implementation/formal/timing authority = false/false/false
Mypy = clean
Pylint = 10.00/10
git diff --check = PASS
```

真实exchange仍无audit/closure，本批没有模拟其已批准，也没有解除DocOps blocker。

## 7. 第三批：source与publication边界

PROCEED额外要求：

1. request JSON的task/doc/type/round/executor/auditor与真实`request.md` SHA一致；
2. delivery `result_commit`可解析，包含冻结S4-1A formal基线`f773370`，且自身为HEAD祖先；
3. HEAD与tracking upstream ahead/behind均为0；
4. S4-1A exchange、DocOps state、四份S4-1B0合同、施工包和两个checker组成的critical path无tracked或
   untracked修改。

外审若要求修复，后续delivery可以是`f773370`的合法后继，不要求永远逐位等于旧commit。WAIT状态只披露
publication/dirty信息；PROCEED才把不同步或critical dirty升级为ERROR，避免开发检查器本身时误报。

```text
checker SHA256 = 93e20eb86dec46955d0a0822b1a3a4dbd080aff5f7ceb10cb006a1907a2c3eee
self-test = 10/10 PASS
  classifier = 5
  closed content chain = 2
  publication clean/ahead/dirty = 3
request Markdown SHA256 = a3fb97491d039cde0bfe9dd0d3c564fa22c36e8998e9eac9ccfa2d459cd32740
delivery declared commit = f773370
delivery resolved/baseline = f7733702cad8519cb32433ea759ce63c905f1539
pre-edit HEAD/upstream divergence = 0/0
real repository status = WAIT / exit 3
Mypy = clean
Pylint = 10.00/10
git diff --check = PASS
```

本批仍只增强激活检查，不修改production、formal或performance claim。
