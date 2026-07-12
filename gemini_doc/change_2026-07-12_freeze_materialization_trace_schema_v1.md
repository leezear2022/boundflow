# 变更记录：冻结 Materialization Trace Schema v1

## 修改

- schema 固定为 `boundflow.materialization/v1`，补齐 query/event identity、solver phase、来源、
  operator tree、spec/domain axes、logical lifetime 与 autograd/α/β 标记。
- 分离 logical materialized bytes、observed allocation delta、CUDA peak allocated/reserved。
- trace context 可选捕获 CUDA allocator；默认关闭，正常 timing path 不创建 event。
- 预留 α、β、intermediate bound、weight 与 operator state 的独立 byte categories。
- ReLU 当前 barrier 明确标为 persistent；不再使用含糊的 `lifetime`/`dense_bytes` 名称。

## 边界

本批只冻结可观测性口径，不改变 ReLU/CROWN 数学，不实现 SignSplit operator，不生成性能
结论。trace-on 只用于机制分析。

## 验证

- schema/event/summary/context isolation，以及 CROWN、α、αβ、DAG 定向回归：22 passed；
- `pylint boundflow/runtime/materialization.py`：10.00/10；
- 全量：164 passed、1 个预期 skip；`git diff --check` 通过。
