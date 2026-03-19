# 变更记录：Phase 6H PR-7（Phase 6 收尾）——E2E tail latency（p90/p99）+ timeout 计数 + schema v2 + torch_benchmark 兼容修复

## 动机

Phase 6 到 6H PR-4/PR-5/PR-6 已经把“端到端 time-to-verify → JSONL/CSV/fig → 一键 runner → AE README/claims → 测试收集卫生”闭环跑通。

最后一个 reviewer/AE 常见追问点是 **尾部延迟与异常样本是否被透明记录**：

- 只有 `p50` 容易掩盖 tail 行为；
- 计时路径（尤其 `torch.utils.benchmark`）在不同 PyTorch 版本下的统计字段不稳定，容易出现脚本“能跑但元信息报错”的尴尬；
- 若未来引入 timeout，需要明确 `timeouts_count/valid_runs_count`，避免 silently drop。

因此本 PR 在 **不改 runtime 求界语义** 的前提下，把 E2E bench 的统计口径加固到 “p50+p90(+p99) + 有效样本计数 + best-effort timeout”，并升级 schema_version。

## 本次改动

### 1) E2E bench 输出升级（schema v2）

- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta.schema_version` 从 `phase6h_e2e_v1` 升级为 `phase6h_e2e_v2`。
  - 每个开关组合 row 新增：
    - `batch_ms_p90 / serial_ms_p90`（以及 `*_ms_p99`）
    - `speedup_p90 / speedup_p99`
    - `batch_runs_count/batch_valid_runs_count/batch_timeouts_count`（serial 同理）
  - 计时参数新增：
    - `--timeout-s`：perf_counter 路径的 best-effort 单次 run timeout（默认 0 禁用）
    - `--torch-benchmark-repeats`：torch_benchmark 路径重复多次 measurement 以估计 p90/p99
  - 语义不变，但可比性更严格：
    - `comparable=1` 需要 **verdict 一致 + 无 timeout + 有有效样本**；否则 `note_code/note` 标明不可比原因。

### 2) report/plot 同步消费新字段

- 更新：`scripts/report_phase6h_e2e.py`
  - CSV 增加 `time_ms_p90/time_ms_p99` 与 `runs_count/valid_runs_count/timeouts_count`。
  - `summary.md` 主表展示 `time_ms_p50/time_ms_p90` 与 `speedup_p50/speedup_p90`。
  - 修复：sweep 失败记录（`rows=[]`）现在也能在 summary 里显示（从 raw payload meta 读取）。

- 更新：`scripts/plot_phase6h_e2e.py`
  - 新增 `*_speedup_p90.png`（p90 speedup 主图），保持旧版 p50 图不变。

### 3) torch_benchmark 兼容修复（避免脚本在 PyTorch 2.8+ 上崩）

- 更新：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - 将 `Measurement.number` 替换为 `Measurement.number_per_run`（PyTorch 2.8+ 的实际字段）。

### 4) schema 钉子更新

- 更新：`tests/test_phase6h_bench_e2e_schema.py`
- 更新：`tests/test_phase6h_report_csv_schema.py`
  - 断言新增字段存在，避免后续扩展造成 silent break。

### 5) AE README 同步口径（p50 → p50+p90）

- 更新：`gemini_doc/ae_readme_phase6h.md`
  - Claim C1/C2 中增加 `p90` 字段与 `*_speedup_p90.png` 的引用，避免文档与 schema 不一致。

## 如何验证

```bash
# 核心 schema 回归
python -m pytest -q \
  tests/test_phase6h_bench_e2e_schema.py \
  tests/test_phase6h_report_csv_schema.py \
  tests/test_phase6h_plot_smoke.py

# 手动：跑一次 E2E bench（8 个开关组合）
python scripts/bench_phase6h_bab_e2e_time_to_verify.py \
  --device cpu --dtype float32 --workload 1d_relu \
  --oracle alpha_beta --steps 0 --max-nodes 64 --node-batch-size 8 \
  --warmup 1 --iters 5
```

## 备注

- `--timeout-s` 默认关闭；开启时仅做 best-effort（依赖 `SIGALRM`），主要用于 “异常 case 不让 sweep 无声卡死” 的工程兜底。
- 本 PR 只增强可观测性与工件口径，不改变 αβ oracle / BaB 的求界语义与 verdict 定义。
