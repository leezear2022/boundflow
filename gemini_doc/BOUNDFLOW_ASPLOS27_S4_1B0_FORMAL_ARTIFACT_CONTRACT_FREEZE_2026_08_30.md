---
status: design-formal-artifact-contract-frozen-code-closed
date: 2026-08-30
type: formal-artifact-contract-freeze
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
---

# S4-1B0 formal artifact 合同冻结

## 1. 目的

在不开放代码的前提下，将 S4-1B0 future formal 的目录、worker、binary raw、manifest、stdlib replay、tamper
和外审状态冻结成 machine-readable 合同：

```text
gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_FORMAL_ARTIFACT_CONTRACT_V1_2026_08_30.json
SHA256 = 7b4069fbb9dd8851ffe41606da5953a1e88c002160e58127bd51f900239c4aa6
```

该合同复用已经通过 S4-1A 的 raw-first/hash-chain 模式，但不复制 S4-1A 的 buffer payload。

## 2. Artifact topology

```text
5 positive fresh workers
1 cache miss->hit fresh worker
5 isolated fault fresh workers
= 11 fresh processes / 11 JSONL rows
```

每个positive sidecar包含：

```text
coefficient float32 = 73,728 B
lower       float32 = 73,728 B
upper       float32 = 73,728 B
selector    int8    = 18,432 B
selected    float32 = 73,728 B
total                = 313,344 B
```

所有raw先由worker落盘，parent只有在11行完整、顺序正确且sidecar齐备后才能生成summary/manifest。partial/resume拒绝。

## 3. Module 与 cache evidence

artifact必须保存：

- unscheduled TIR JSON；
- scheduled TIR JSON；
- CUDA device source；
- immutable module receipt；
- fresh cache 的miss→hit observation。

外审可以现场重编并比较三个hash；stdlib replay只重算保存内容、receipt和hash-chain，不冒充硬件真实性证明。

## 4. Fault partition

20个stable reason被五个fresh fault worker无重叠覆盖：

```text
1-4   classifier/policy
5-10  cache/source
11-16 descriptor/DLPack
17-18 stream/launch
19-20 invalid-selector/claim
```

20个reason全部由unit tests覆盖并映射到唯一fault category；formal只启动5个fresh fault worker，每类执行一个冻结的
代表reason，不能在一个进程中串行触发整组reason后冒充隔离故障。每个formal fault都必须记录stable reason、cleanup、
fallback/retry/native-shadow边界。

## 5. Replay 与 tamper

production positive fixture的invalid count为0，因此canonical qNaN不能从positive row得到非空证明；该证据由IEEE fixture
与invalid-selector representative fault提供。stdlib replay禁止import BoundFlow、Torch、TVM、TVM-FFI或NumPy；它从raw重算位语义、class count、sidecar offset/hash、
cache、fault partition、summary和claims。

10类tamper全部修改语义payload并重签所有外层hash；仍须因semantic recomputation被10/10拒绝。coherent full resign仍属于
E0边界，必须披露，不能宣称artifact自证真实硬件来源。

## 6. 状态边界

内部formal完成后只允许：

```text
FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0
FORMAL-NO-GO-S4-1B0-TERNARY-ENDPOINT
```

外审前禁止写`VALIDATED-S4-1B0`。当前S4-1A外审仍未返回，因此本合同不开放worker/generator/replay/tamper代码。

## 7. 当前门禁

```text
formal artifact design contract = frozen
artifact scripts/tests/raw       = absent
S4-1A external audit             = pending
S4-1B0 implementation/formal     = closed
timing/performance               = closed
```
