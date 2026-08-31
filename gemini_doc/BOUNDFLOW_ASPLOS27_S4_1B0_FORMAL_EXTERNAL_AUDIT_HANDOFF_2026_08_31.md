---
status: ready-for-external-audit
date: 2026-08-31
stage: s04
performance-claimed: false
---

# S4-1B0 ternary endpoint formal 外审交接

## 1. 请求结论

请独立审计 S4-1B0 isolated ternary endpoint implementation 与 formal correctness artifact，并给出：

- `approve`：允许关闭为`VALIDATED-S4-1B0-TERNARY-ENDPOINT`，后继最多开放S4-1B
  implementation/correctness；
- `changes_requested`：列出blocker/major/minor与可复核证据；
- `reject`：若发现语义、来源身份或证据链不可修复偏离。

本轮没有timing或performance authority。不得把局部endpoint correctness升级为same-solver、complete-query、
10x或ASPLOS-ready claim。

## 2. 审计基线

- branch：`feat/rvir-v4-production-state-ownership-v1`；
- formal tool/source revision：`4e2a26128a9a538ac64f222e8b82e92ea745d3b6`；
- endpoint implementation：`f61e917`（`f6df7ee`实现，`f61e917`补全20 reason显式测试绑定）；
- formal tooling：`6bfa948`；
- replay determinism hardening：`4e2a261`；
- endpoint backend：`boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py`；
- endpoint unit：`tests/test_asplos27_s4_ternary_endpoint.py`；
- artifact：`artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1`；
- tamper report：`artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1-tamper-report.json`。

## 3. 本轮实现边界

实现的是backend-local fixed lowering，不是新solver IR，也不是production evaluator：

```text
coefficient + lower + upper + caller-owned selector + caller-owned selected
  → TIR pack: IEEE exponent-bit classifier (+1/-1/0/-128)
  → TIR select: lower / upper / f32((lower + upper) * 0.5) / canonical qNaN
```

关键合同：

- nonfinite必须通过IEEE-754 exponent bits判断，不能使用会被CUDA化简的`x == x`；
- `coefficient == ±0`均归zero endpoint；
- midpoint严格按`f32((lower + upper) * f32(0.5))`顺序；
- invalid selected必须是canonical qNaN `0x7fc00000`；
- module cache绑定完整source/TIR/schedule/device/ABI身份并在命中时重新校验source；
- prepared probe固定5个caller-owned DLPack view、当前device/stream双向核对、2次launch；
- 所有错误fail closed，禁止fallback/eager/native shadow；
- production evaluator ticket、S4-1A buffer alias、timing全部未接入。

## 4. Artifact结构与声明数字

正式拓扑：5 positive + 1 cache + 5 fault，11个fresh PID。

每份positive sidecar固定：

- coefficient：18,432 × float32 = 73,728 B；
- lower：73,728 B；
- upper：73,728 B；
- selector：18,432 × int8 = 18,432 B；
- selected：73,728 B；
- 合计：313,344 B。

5份sidecar SHA256均为
`a07aea90d2404b0e3c40f2af4aeaea169a1465b5feb24616c75cf882b5db5e6c`。从raw重算的selector计数是
`8689/9137/606/0`，selected逐元素bitwise exact。cache为`miss/hit`且compile/miss/hit/entry=
`1/1/1/1`。五类fault reason按顺序为：

1. `TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH`；
2. `TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH`；
3. `TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH`；
4. `TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH`；
5. `TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED`。

## 5. 独立复核入口

```bash
python scripts/replay_asplos27_s4_1b0_ternary_stdlib.py \
  artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1

python scripts/probe_asplos27_s4_1b0_ternary_tamper.py \
  artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1

conda run -n boundflow pytest -q \
  tests/test_asplos27_s4_ternary_endpoint.py \
  tests/test_asplos27_s4_1b0_ternary_artifact.py
```

replay只使用Python标准库，不import BoundFlow/Torch/TVM。建议审计方另写stdlib脚本，直接从5份`.bin`
独立重算classifier、midpoint和selected bit pattern，不采信`summary.json`里的数字。

## 6. 必审AC

### AC1：source与协议身份

- HEAD/远端/branch、code blob、四份合同、construction hash、fixture、model/property和三个外部commit独立一致；
- artifact无本机绝对路径；
- formal artifact确实在`4e2a261`已发布后生成。

### AC2：实现语义

- 亲读TIR确认bit classifier、signed zero、midpoint顺序、canonical qNaN；
- PyTorch oracle只作独立reference，不存在TIR自比；
- 20 stable reason与冻结negative contract逐字一致；
- 无timing API或performance flag漂移。

### AC3：positive raw

- 5份sidecar逐字节一致；
- 从raw独立重算`8689/9137/606/0`；
- 18,432个selected output逐元素bitwise exact；
- 5个DLPack storage/pointer身份与module receipt合法。

### AC4：cache与fault

- cache fresh process真实产生miss→hit且只编译一次；
- 5个fault均由真实worker触发，reason/detail、reject-before-launch、fallback/eager/native-shadow=0；
- fault退出后device/stream/storage cleanup不漂移。

### AC5：replay与tamper

- stdlib replay必须从raw重算，不得只验hash；
- 10类coherent outer-resigned攻击全部被派生语义、跨fresh determinism或claim边界拒绝；
- coherent full-resign E0边界仍存在且已披露，不得写成不可伪造证明。

### AC6：验证链

- targeted现场应为`22 passed`；
- full suite记录为`2073 passed, 3 skipped`，审计方可复跑或说明未复跑边界；
- Black/mypy clean、Pylint `10.00/10`、`git diff --check`、DocOps lint通过。

### AC7：claim与后继

- 当前只能是`FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0`；
- approve后也只能关闭isolated endpoint correctness；
- S4-1B production implementation/correctness必须另开；
- timing、performance、same-solver、complete-query、10x、ASPLOS-ready保持关闭。

## 7. 已披露失败演练

第一次临时演练为`9/10`：coefficient LSB变化未改变符号，同时replay遗漏5份positive binary全同回绑。
该缺口在正式生成前由`4e2a261`修正；之后从空临时目录重跑为`10/10`，再从空正式路径生成artifact。
请把这项修正作为重点攻击面，而不是忽略第一次失败。

## 8. Executor本地结果

- activation gate：79项实现检查、192项设计检查，`PROCEED`；
- stdlib replay：PASS；
- tamper：`10/10 rejected`；
- targeted：`22 passed`；
- full：`2073 passed, 3 skipped, 6 warnings in 715.10s`；
- Black：clean；mypy：5个formal文件逐文件clean；Pylint：`10.00/10`；
- manifest SHA256：`95a65429e4b59c0554c04f62b1b91f8538bc699bac809972aed173b009c43d76`；
- summary SHA256：`49f77c581bc9d96423da0e6e4e47da9714a6d8048c01160a5d2980c480c3244f`；
- tamper report SHA256：`26f27868e04e7c218fdb6b988cb40c4b61b0b162490d07c3e37593ca9f82b4d5`。

外审报告请按AC1—AC7逐项给出PASS/FAIL、blocker/major/minor/info，并明确哪些结果是现场重跑、哪些是
raw独立重算、哪些只做源码/冻结证据审查。
