---
status: design-fixture-frozen-code-closed
date: 2026-08-30
type: fixture-freeze
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
---

# S4-1B0 IEEE 位级 fixture 冻结

## 1. 目的

将 S4-1B0 施工合同中的 float32/selector 语义从自然语言转为 machine-readable design fixture，使外审批准后
的第一批 CPU oracle、TIR 和 negative tests 使用同一组 raw bits，而不是各自重新选择边界样本。

fixture：

```text
gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IEEE_BIT_FIXTURES_V1_2026_08_30.json
SHA256 = 33d0aa1db3380fdc30ff7a48eaa4d5d98f93806583493da40b5ddee4031e15d7
```

它仍属于设计资料，不被当前 production/test import，也不是 formal artifact。

## 2. 覆盖范围

fixture 冻结：

- 16 个 pack raw-bit case：NaN payload、Inf、max finite、normal、subnormal、signed zero；
- 16 个 select case：三种合法 selector、三种非法 selector、midpoint overflow/underflow/signed-zero、payload preservation；
- 两个禁止 midpoint reassociation 的逐位反例；
- 固定 shape module 对错误 numel 的 reject-before-launch 期望；
- future ResNet2B production fixture 的 `8689/9137/606/0` design-time inventory；
- 全部 claim flag 为 false。

## 3. 现场来源

所有 pack/select expected bits 都由当前 TVM/CUDA 临时 module 在 RTX 4060 Laptop / SM89 上现场执行后冻结，
并与 IEEE bit oracle逐项一致。临时 module 没有写入仓库，也没有复用未来 production symbol。

关键结果：

```text
pack exact = 16/16
select exact = 16/16
canonical invalid output = 0x7fc00000
midpoint reassociation counterexample = 2/2
DLPack pointer exact = 5/5
```

## 4. 后续消费规则

外审批准并激活 S4-1B0 后：

1. CPU/stdlib bit oracle必须先消费完整 32 cases；
2. TIR GPU test必须从raw int32 bits构造float32 view，不能经Python float重建NaN payload；
3. implementation不得把fixture中的16写成generic builder常量；formal production spec仍为18,432；
4. `8689/9137/606/0`只用于真实production fixture核对，不得进入generic validator；
5. 若任何 expected bits需要修改，必须升级schema版本并保留v1，不得静默改写。

## 5. 门禁

```text
design fixture frozen       = true
production/test consumer    = absent
S4-1A external audit        = pending
S4-1B0 implementation       = closed
formal/timing/performance   = closed
```
