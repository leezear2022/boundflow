# 变更记录：2026-04-08 Phase 7A PR-11——shared CROWN benchmark 与轻量观测

**日期**: 2026-04-08  
**类型**: perf / benchmark / observability  
**范围**: shared CROWN runtime benchmark、dense fallback 计数、结果摘要文档

---

## 动机

PR-10 和 shared CROWN layout-only support 已经把正确性与结构化路径打通，但“是否真的带来收益、剩余 dense 点在哪”还没有一份最小可复现证据。

PR-11 的目标不是立刻继续扩 `LinearOperator` algebra，而是先把这两个问题钉死：

- structured ReLU backward 相对旧 dense ReLU barrier 的实际时间表现
- layout-only `permute/reshape` 路径相对 dense materialize 的实际时间表现
- shared CROWN 主路径里 `split_pos_neg()` 还剩哪些 operator 会落回 dense

## 主要改动

- 新增：`scripts/bench_phase7a_shared_crown_path_attribution.py`
  - 提供 4 个固定 workload：
    - `relu_heavy_mlp`
    - `residual_relu_mlp`
    - `concat_relu_mlp`
    - `permute_reshape_linear`
  - 对每个 workload 输出：
    - `structured_ms_p50`
    - `baseline_ms_p50`
    - `speedup = baseline / structured`
    - `counts_structured`
    - `counts_baseline`
  - 观测方式采用脚本侧 monkeypatch，不改 solver public API：
    - 包装 `linear_operator._split_pos_neg_dense(...)` 统计 dense fallback 次数与 operator 类型
    - 包装 `_backprop_relu_step(...)` 与 `_backprop_permute_step(...)` 统计 backward 命中情况
  - baseline 同样采用脚本侧 patch：
    - `dense_relu`：强制回退到 dense ReLU backward 参考实现
    - `dense_layout`：强制把 `permute` backward materialize 成 `DenseLinearOperator`

- 新增：`tests/test_phase7a_pr11_shared_crown_bench.py`
  - 断言三类 `relu_barrier` workload 下，structured 路径与 dense ReLU baseline 数值一致
  - 断言 `permute_reshape_linear` 下，structured 路径与 dense layout baseline 数值一致
  - 锁定 bench stdout JSON schema 与计数字段
  - 显式验证 `RightMatmulLinearOperator` 仍是 `split_pos_neg_dense` 的主热点之一

- 新增：`gemini_doc/phase7a_pr11_shared_crown_benchmark_summary.md`
  - 记录 CUDA 主口径 bench 命令、本机结果表与结论摘要

- 更新：`docs/change_log.md`
  - 追加 PR-11 总账条目

## 结果摘要

CUDA (`RTX 4090`) / `--profile bench --warmup 5 --iters 20` 下：

- `permute_reshape_linear`：
  - structured layout-only path 相对 dense layout barrier 为 `1.11x`
- `relu_heavy_mlp` / `residual_relu_mlp` / `concat_relu_mlp`：
  - structured ReLU backward 相对 dense ReLU barrier 约为 `0.65x~0.66x`
  - 当前实现下还没有 latency 正收益
- dense fallback 热点：
  - `relu_heavy_mlp`、`residual_relu_mlp`：主要是 `RightMatmulLinearOperator`
  - `concat_relu_mlp`：`RightMatmulLinearOperator` + `SliceInputLinearOperator`

这说明 PR-11 已经把“收益是否存在”和“下一步该优化谁”两件事都收敛成了可复现结论。

## 影响面

- 不改 `run_crown_ibp_mlp()` / `alpha_crown.py` / `alpha_beta_crown.py` / `bab.py` 的 public API。
- 不引入新的 runtime 返回值或 debug flag。
- benchmark 与计数逻辑全部留在脚本层，方便后续继续复用同一口径对 PR-12 做前后对照。

## 验证

已执行：

```bash
conda run -n boundflow python -m pytest -q tests/test_phase7a_pr11_shared_crown_bench.py

conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cpu \
  --profile smoke \
  --workloads all \
  --warmup 1 \
  --iters 1

conda run --no-capture-output -n boundflow python \
  scripts/bench_phase7a_shared_crown_path_attribution.py \
  --device cuda \
  --profile bench \
  --workloads all \
  --warmup 5 \
  --iters 20
```

结果：

- `tests/test_phase7a_pr11_shared_crown_bench.py`: `3 passed in 0.76s`
- CPU smoke bench：脚本正常输出 4 行 workload JSON
- CUDA bench：
  - `relu_heavy_mlp`: `3.936 ms` vs `2.610 ms`
  - `residual_relu_mlp`: `2.406 ms` vs `1.574 ms`
  - `concat_relu_mlp`: `3.457 ms` vs `2.246 ms`
  - `permute_reshape_linear`: `1.320 ms` vs `1.471 ms`
