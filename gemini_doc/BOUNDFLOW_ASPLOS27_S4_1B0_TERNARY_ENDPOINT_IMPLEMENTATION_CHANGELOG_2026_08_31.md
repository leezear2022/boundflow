---
status: implementation-correctness-candidate
date: 2026-08-31
stage: s04
topic: asplos27-s4-1b0-ternary-endpoint
performance-claimed: false
formal-validated: false
same-solver-claimed: false
---

# S4-1B0 三值 input endpoint 实现与正确性修改记录

## 1. 结论

S4-1B0 的独立 backend/correctness 切片已实现，但尚未生成 11-process formal artifact，也未经过下一轮
外审。本轮状态只能写作：

```text
IMPLEMENTED-CORRECTNESS-CANDIDATE-S4-1B0-TERNARY-ENDPOINT
```

它不是 production evaluator correctness、same-solver performance 或 ASPLOS-ready claim。

## 2. 新增文件

- `boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py`
- `tests/test_asplos27_s4_ternary_endpoint.py`

没有修改旧 S2/R31B2 binary endpoint 模块，没有新增顶层 IR，也没有接入 S4-1A ticket、S4 evaluator、
optimizer 或 timing 路径。

## 3. 实现内容

### 3.1 位级语义

- pack 直接 reinterpret float32 bits，以 exponent mask `0x7f800000` 识别 NaN/Inf；
- finite positive/negative/±0 分别编码 `+1/-1/0`，nonfinite 编码 `-128`；
- select 的 `+1/-1/0` 分别选择 lower/upper/`(lower+upper)*float32(0.5)`；
- 非法 selector 输出固定 payload `0x7fc00000`；
- midpoint 操作顺序没有重结合，保留 max-finite 与 min-subnormal 两个逐位反例。

### 3.2 编译身份与 cache

- generic `numel>0` build spec 和冻结 schedule spec；
- 两个独立 TIR symbol，经同一 CUDA module 编译；
- compiled object 保存 unscheduled/scheduled IR JSON、device source 及三份 content hash；
- immutable module receipt 与 mutable cache observation 分层；
- cache key 只绑定 compile 前可知事实，device source hash 不形成循环 key；
- hit 重新哈希 IR JSON、device source 与 receipt；失败不发布半条 cache entry，也不 fallback；
- 代码从 dataclass 字段、常量和 scope 事实重建 construction model，SHA256 必须等于
  `5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a`。

### 3.3 prepared correctness probe

- caller owns coefficient/lower/upper/selector/selected 五个 CUDA tensor；
- prepare 创建 5 个 DLPack view，逐个 roundtrip 核对 pointer/shape/stride/dtype/device；
- 五个 storage 必须互不 alias；
- current device、SM capability 与 current PyTorch/TVM-FFI stream 均 fail-closed；
- warm run 恰好一次 pack + 一次 select，6 个 argument occurrence；
- warm receipt 不记录 tensor、content count、timing 或 performance；
- formal-only poison validator 只允许在 caller 显式 synchronize 后使用。

### 3.4 负路径

冻结的 20 个 stable reason 与 negative contract 顺序逐项一致。测试实际覆盖 schema/policy/midpoint/
nonfinite、symbol/legacy/TIR/source/cache、shape/dtype/device/layout/alias/DLPack、stream/launch、invalid
selector 与 claim flag 边界。

## 4. 确定性验证

在 `conda activate boundflow && source env.sh`、RTX 4060 Laptop / SM89 环境执行：

```text
pytest -q tests/test_asplos27_s4_ternary_endpoint.py
  19 passed

pytest -q tests/test_asplos27_s4_ternary_endpoint.py \
  tests/test_asplos27_s4_ordered_buffer_abi.py
  99 passed

pytest -q tests
  2070 passed, 3 skipped

python scripts/check_asplos27_s4_1b0_design_contracts_stdlib.py
  PASS / 192 checks / construction hash exact

mypy <backend> <test>
  clean

pylint <backend> <test>
  10.00/10

black <backend> <test>
  clean
```

GPU 正向测试包含 frozen 16-case IEEE fixture、cache miss→hit、non-default stream、5/5 DLPack pointer、
canonical qNaN payload 与 caller-owned output。这里没有计时，也不产生性能数字。

## 5. 下一门禁

下一步以本实现提交为 immutable source identity，新建 S4-1B0 formal activation check；只有它确认 code blob、
合同、分支/上游和 claim 边界一致后，才生成冻结的 5 positive + 1 cache + 5 fault 共 11 个 fresh process
artifact。formal artifact 通过本地 replay/tamper 后仍只能标记
`FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0`。

S4-1B production arena alias、evaluator、optimizer、timing、performance、same-solver、complete-query 与 10x
仍全部关闭。
