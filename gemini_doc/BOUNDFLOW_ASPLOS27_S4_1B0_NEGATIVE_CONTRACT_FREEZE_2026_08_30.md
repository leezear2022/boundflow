---
status: design-negative-contract-frozen-code-closed
date: 2026-08-30
type: negative-contract-freeze
stage: s04
execution-authority: false
code-change-open: false
formal-run-open: false
timing-open: false
performance-claimed: false
---

# S4-1B0 negative/cache/formal 合同冻结

## 1. 目的

将施工合同第 8、9、10 节转为 machine-readable negative contract，确保获批后的 backend、cache、prepared
probe、test 和 artifact generator共享同一 reason registry与worker topology。

```text
gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json
SHA256 = 460415b9c4b415812edb5b89f1039e483eb24d1ee8b5cb486ca6a014c38557aa
```

该文件是设计合同，不是当前 production registry。

## 2. 冻结内容

- 20 个 stable reason，ordinal 与施工合同逐项一致；
- `spec -> build -> compile/cache -> prepare -> launch/observation` phase 分类；
- 16 个冻结 test layout 名称；
- compile 前 cache key 输入与明确排除项；
- build/schedule/module/cache/descriptor/launch/formal 七层 identity；
- `5 positive + 1 cache + 5 fault = 11` fresh-process formal topology；
- production evaluator、S4-1A ticket、optimizer、timing和performance全部false。

## 3. 使用规则

1. 实现中的reason常量必须从同一registry构造，测试不得各自拼字符串；
2. 每个reason至少有一个直接negative或明确的组合触发证据；
3. cache lookup不得把compile输出`device_source_hash`循环放进首次key；
4. cache hit必须重哈希cached source/module，不得只检查64-char格式；
5. mutable hit/miss/compile计数不得污染immutable module receipt；
6. formal worker必须fresh-process、raw-first，partial/resume一律拒绝；
7. 该合同发生不兼容变化时升级schema，不静默改写v1。

## 4. 当前门禁

```text
negative design contract = frozen
runtime registry         = absent
tests/artifact           = absent
S4-1A external audit     = pending
S4-1B0 implementation    = closed
timing/performance       = closed
```
