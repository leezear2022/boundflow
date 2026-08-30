---
status: design-abi-contract-frozen-code-closed
date: 2026-08-30
type: abi-contract-freeze
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
---

# S4-1B0 build/cache/runtime ABI 合同冻结

## 1. 目的

将施工合同第4—7节转成machine-readable ABI，使实现阶段可以逐层生成dataclass、validator和receipt，不再从长文档
人工复制字段。

```text
gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_ABI_CONTRACT_V1_2026_08_30.json
SHA256 = 5f977528d32790abbbf07cb4f95f44b6f0f73e83f532c8b6caa696d662b8b481
```

它绑定此前的 IEEE fixture 和 negative contract，但仍是设计资产，不被当前 production import。

## 2. 分层

ABI明确分开：

1. `TernaryEndpointBuildSpecV1`：纯编译语义；
2. `TernaryEndpointScheduleSpecV1`：固定elementwise schedule；
3. `CompiledTernaryEndpointV1`：保存可重哈希的IR JSON与device source；
4. `TernaryEndpointModuleReceiptV1`：immutable compiled identity；
5. `TernaryEndpointModuleCacheV1`：compile-before-known key与atomic entry；
6. `TernaryEndpointCacheObservationV1`：mutable hit/miss计数；
7. `PreparedTernaryEndpointProbeV1`：五个caller-owned tensor/view；
8. warm launch receipt：只含O(1) host事实；
9. formal observation：显式同步后才能保存raw与class counts。

## 3. 关键防错

- `device_source_hash`是compile输出，不得循环进入首次cache lookup key；
- hit必须从保存的IR/source重哈希，不能只验证hash格式；
- mutable cache counts不得改变module identity；
- caller不能提供device/stream identity；
- warm run不创建DLPack view，也不同步统计selector类别；
- 施工包中的grouped字段会规范化为独立字段，例如`fallback/eager/native_shadow`展开为三个`*_count`，映射必须进入ABI；
- generic builder只要求`numel>0`，不能硬编码ResNet2B的18,432；
- isolated selected output与selector为distinct caller-owned storage，不提前声称production alias；
- build/backend lowering不新增顶层compiler IR。

## 4. 实现消费顺序

S4-1A外审批准并关闭后：

1. 先从ABI生成spec/schedule与canonical hash测试；
2. 再实现两个TIR builder和compiled module content rehash；
3. 再实现cache receipt/observation分层；
4. 最后实现Prepared probe和warm receipt；
5. 使用IEEE fixture做数值测试、negative contract做失败测试；
6. 三份machine-readable设计资产全部通过后才生成formal artifact。

## 5. 当前门禁

```text
machine-readable ABI     = frozen
production consumer      = absent
S4-1A external audit     = pending
S4-1B0 implementation    = closed
formal/timing/performance = closed
```
