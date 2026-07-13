# 2026-07-13：PR-12D general-DAG 与 CUDA stream correctness closure

## 背景与修正

对提交 `d3b3913` 的高强度复审发现两个 correctness blocker：

1. `Affine → ReLU` matcher 未检查 affine output fanout；fused 后跳过 affine 会丢弃其它
   `adjoints[h]` contribution。最小双分支反例的 bound diff 约 `6.98`，且 10000/10000
   随机样本违反 fused lower bound。
2. executor 只进入 `torch.cuda.stream()`，没有设置 TVM-FFI current stream。default stream
   延迟反例中，custom stream 立即消费输出的误差约 `2.94`。

因此先前的 PR-12D PASS 被撤回；本提交只修 correctness，不运行性能 Pareto 或 final held-out。

## 修复

### Single-consumer fusion contract

- matcher 只为 affine output 的唯一 consumer 是目标 ReLU 时生成 step；
- runtime 重新计算 consumer set，不信任外部 plan；
- 若 `adjoints[affine_output]` 已存在额外贡献，则不消费 affine，走原路径；
- fanout、重复 step、越界/stale step 全部确定性 fallback。

当前 v1 不实现 partial-consume。fanout DAG 的正确语义是 **不 fuse、保持 sound**，不是宣称
一般 fanout 已被融合。

### 完整 ExecutionStep contract

step 新增 graph fingerprint，并验证：

- graph/version identity；
- affine/relu index 与 adjacency；
- `kind` 与实际 op family；
- consumed outputs；
- boundary representation；
- internal materialization policy；
- backend id；
- single-consumer 与 step 唯一性。

任一字段不匹配均 fallback，不抛出后继续执行 stale fused plan。

### Pre-materialization capability filter

新增静态 descriptor。executor 先根据 method/grad/α/β/split、device/dtype/layout、shape 和
Conv attrs 判断 capability，之后才物化 dense boundary A。unsupported grouped Conv 的 trace 中
不再出现 `fused_region_dense_boundary`。

### Torch–TVM stream bridge

TVM executor 改用：

```python
with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
    compiled(...)
```

这同时设置 Torch current stream 与 TVM-FFI environment stream。回归测试在 default stream
排入延迟任务，在 custom stream 调 executor 并立即消费输出，只同步 custom stream；结果与
Torch reference 对齐。另有 stream-handle identity 断言。

## 新增 correctness 覆盖

- fanout MLP 最小反例：matcher 生成 0 step，最终 bounds 与 dense 相同，2048 个样本 sound；
- 伪造 fanout step：runtime 防御性拒绝；
- stale fingerprint、错误 kind/backend/boundary/internal-policy：全部 fallback；
- α、β、requires-grad、split：descriptor/executor run 均不调用；
- groups=2 Conv：在 fused A materialization 前 fallback；
- multi-block mini-ResNet：stem、2 个 residual blocks、stride-2 projection、branch merge 与
  ReLU-output fanout，3 个合法 fused regions，最终 bounds 对齐；
- non-default CUDA stream race 与 FFI/Torch stream id 对齐。

## 验证

```text
PR-12D runtime integration file：23 passed
PR-12/CNN/DAG/ReLU 专项：86 passed
全量：299 passed、1 skipped
mypy boundflow/runtime/fused_crown.py：success
pylint boundflow/runtime/fused_crown.py：10.00/10
Black check（新增/重写文件）：通过
git diff --check：通过
```

## 阶段判定

```text
PR-12A Backend candidate schema:       PASS
PR-12B Fused Linear foundation:        PASS
PR-12C Fused Conv2d foundation:        PASS
PR-12D Correctness closure:            PASS (single-consumer fusion; fanout fallback)
PR-12E Formal performance/Pareto:      PENDING
PR-12F Planner held-out validation:    PENDING
PR-12 Overall:                         IN PROGRESS
PR-13:                                 BLOCKED
```

下一步才允许建立公平 runtime/network benchmark。Planner production auto-selection、正式
latency-memory Pareto、compile amortization 和冻结 final held-out 仍未完成，不得由本次
correctness PASS 外推性能结论。
