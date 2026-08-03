# RVIR 在线原始证据 replay v2 变更记录

日期：2026-08-03

GitHub：PR [#7](https://github.com/leezear2022/boundflow/pull/7)

## 起因

外部审计同意 RVIR 以 VALIDATED-REDUCED 关闭，同时给出 minor M4：v1 artifact 的
`online_execution.json` 只冻结 377 次在线执行的摘要，原始 `queries.jsonl` 与
`typed_ir.jsonl` 未进入 artifact；第三方因此不能独立重放 parent 顺序与逐调用 IR hash。

## 原始证据恢复

- 重新 checkout αβ-CROWN 固定 commit
  `e5c7e17bf0488843acb77b7519f59876717a49f4` 及其 auto_LiRPA submodule；
- 使用仓库上游 fixture `simple_mlp.onnx` / `robustness_mlp.vnnlib`，CPU、30 秒、
  `bab + typed-ir + baseline-first` 重新运行；
- 得到 query / dispatched / completed = `377/377/377`、347 parent links、380 visited
  domains、final lower `tensor(-0.18902308)`；
- 新生成 `queries.jsonl` 的 SHA256 为 `d1bd3ee0…d6141a`，`typed_ir.jsonl` 为
  `bee3a69d…a9a9cdb`，与 v1 摘要记录的原始 source digests 完全相同。

## v2 artifact

新增 `artifacts/rvir/rvir-cpu-correctness-v2-20260803/`：

- 保留 394 条历史 activation admission 与已审计 ResNet semantics；
- 新增 `online_queries.jsonl`：377 条 adapter v2 query 原文；
- 新增 `online_typed_ir.jsonl`：377 条完成的 typed execution record 原文；
- `online_execution.json` 同时冻结 result/baseline BaB projection；
- manifest 对五个 payload 文件逐一记录 SHA256。

Fresh replay 现在会：

1. 校验全部 artifact digest；
2. 重编译 394 条历史 admission；
3. 校验在线 query/record ID 与 sequence 一一对应；
4. 校验 parent 必须先于 child、377 条全部 completed；
5. 对 377 条在线 query 重新编译 Bound/PlanTemplate/PlanInstance/Task/Schedule，并逐行比较
   五层 IR hash；
6. 从原始行重新统计 root/parent、phase/method、requested output，并与摘要比较；
7. 复核 observer on/off status、visited domains 与 final-lower projection 相等。

v1 artifact 保持不可变，replay 继续兼容其历史 schema。

## 边界

- 这是 external exact-call 的 CPU correctness/integration 与证据可重放增强；
- fused replacement coverage 仍为 `0/394`；
- 不形成 CUDA、latency、throughput 或 ASPLOS-ready claim；
- ResNet 本轮复用并重新校验已审计 semantics payload，没有冒充原始 αβ-CROWN rerun。

## 当前验证

- v1 fresh replay：PASS；
- v2 fresh replay：PASS；
- v2 online raw SHA 与 v1 source digests：2/2 完全一致；
- rehashed tamper probes：伪造 schedule hash 与 parent 顺序均被语义 replay 拒绝；
- 专项测试：`4 passed in 27.10s`；
- 全量回归：`460 passed, 37 skipped, 5 warnings in 76.72s`；
- Black：通过；Mypy：clean；Pylint：10.00/10。
