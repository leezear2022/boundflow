# 变更记录：Phase 6G PR-4（BaB/αβ）——microbench 开关矩阵 + 计数器口径固化（可归因收益）

## 动机

Phase 6G 的主线是 **不动语义，只做系统化收益**。在 PR-3A/3B/3C 落地后（node eval cache、分支复用 forward trace、batch infeasible partial prune），需要一个 reviewer-proof 的 microbench 来把吞吐增益 **拆成可归因的对照**：

- “开 cache 带来多少 oracle call 降幅？”
- “开 branch hint 是否真的消掉二次 forward？”
- “开 batch infeasible prune 是否减少被评估的节点数？”

因此本 PR 把现有 node-batch bench 升级为 **开关矩阵**，并把关键计数器固定输出到 stdout JSON（便于回归与论文口径复现）。

## 本次改动

- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - 新增 3 个开关维度（开关矩阵）：
    - `enable_node_eval_cache: {0,1}`
    - `use_branch_hint: {0,1}`
    - `enable_batch_infeasible_prune: {0,1}`
  - 输出结构调整为：
    - `rows`: 每个开关组合一行（包含 batch/serial 的 p50 以及计数器）
    - `meta`: 固定输出 bench 配置（device/dtype/K/S/steps/lr/dup_frac/infeasible_frac/warmup/iters/timer/torch_version 等）
  - 固化计数器（每行都输出，值可为 0）：
    - `oracle_calls`
    - `forward_trace_calls`
    - `cache_hits/cache_misses/cache_hit_rate`
    - `pruned_infeasible_count`
    - `evaluated_nodes_count`
  - 计数器插桩采用 monkeypatch（仅统计，不影响 runtime 语义）：
    - 统计 `run_alpha_beta_crown_mlp` 调用次数与每次评估的 node 数（通过 `spec.center.shape[0]`）
    - 统计 oracle forward trace 与 branch pick fallback forward trace 的调用次数
  - workload 采用“bound + branch pick”的合成路径（更贴近真实 BaB 循环），并提供：
    - `--dup-frac` 控制 split 重复比例（驱动 cache hit）
    - `--infeasible-frac` 控制固定 infeasible split 比例（驱动 prune）

## 如何验证

```bash
# sanity：输出 stdout JSON（rows/meta）并包含固定计数器字段
python scripts/bench_phase6g_bab_node_batch_throughput.py \
  --device cpu --dtype float32 \
  --nodes 32 --node-batch-size 8 --specs 16 \
  --steps 0 --warmup 1 --iters 3
```

## 备注

- 本 PR 仅增强 bench 的“口径与可归因性”，不改变 BaB/αβ 的任何求界语义。
- stderr 会输出一条口径说明：`serial_ms_*` 包含 Python loop + branch-pick overhead；stdout 仅输出 JSON，便于脚本化解析。

