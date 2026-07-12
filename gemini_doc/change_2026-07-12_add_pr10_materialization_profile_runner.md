# 变更记录：增加 PR-10 materialization profile runner

## 修改

- 新增 MLP chain、CNN chain、residual block、add+concat DAG 和三 BasicBlock mini-ResNet。
- 同一 query 分开运行 trace-off timing/peak 与 trace-on mechanism characterization。
- trace-off 内部继续分离 warm latency 与 peak：latency 重复不清 allocator cache，peak 使用清空
  cache 后的独立单次执行，避免前序矩阵污染 reserved memory。
- 覆盖 CROWN、α-CROWN、固定 split αβ-CROWN，以及 spec/domain batch 扫描。
- fixed split 按域分别从该域 ambiguous ReLU 中选择，禁止把第一个域的 split 广播到其它域；
  多域 αβ 使用 per-domain 参数。
- 输出 `raw.jsonl`、`normalized.csv`、`manifest.json`；fail/OOM 不丢弃。
- 独立记录 coefficient logical bytes 与 α/β/intermediate/weight/operator state bytes。

## 口径

当前 domain batch 是合成固定域批 `synthetic_fixed_domain_batch`，不宣称来自真实 BaB 搜索树；
真实 solver domain replay 仍需后续独立证据。
trace-on latency 不进入性能字段；正式 latency 和 CUDA peak 只来自 trace-off 路径。

## 验证

- CPU 单 query smoke 检查 JSONL、CSV、manifest 和 trace-on/off 隔离；
- clean full GPU profile：5 workload × 3 method × 4 spec × 3 domain，共 180/180 `ok`；
- 全量 pytest、Pylint 和 diff check。

结果与不可过度声明的边界见 `gemini_doc/pr10_materialization_profile_summary_2026_07_12.md`。
